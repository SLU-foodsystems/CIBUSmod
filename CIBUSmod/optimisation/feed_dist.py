from itertools import product
import warnings
import sys
import re as regex

import pandas as pd
import numpy as np

import cvxpy
import scipy

from .. import (
    Regions,
    DemandAndConversions,
    CropProduction,
    FeedMgmt,
    ParameterRetriever,
)

from ..mgmt_modules.feed_mgmt.feed_mgmt_feeddist import FeedDistFeedMgmt
from .geo_dist import GeoDistributor

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned, inv_dict, extend_index, index_to_multi, multiindex_product
from ..utils.data_attr import DataAttr
from ..main_modules.animal_herd import concat_herds

from .indexed_matrix import IndexedMatrix
from .utils import Constraint, feed_demands_to_crop_demands

from typing import Literal


class FeedDistributor(GeoDistributor):
    """Class that handles the distribution of animals (and their feed rations) and
    crops across regions for a given demand and a number of constraints by minimising
    deviation from an initial distribution of crop areas and animal heads (x0).

    Parameters
    ----------
    par : ParameterRetriever object
    regions : Regions object
    demand : DemandAndConversions object
    crops : CropProduction object
    herds : pandas.Series of AnimalHerd objects
    feed_mgmt : FeedMgmt object
    par : ParameterRetriever object
    """

    module_name = "FeedDistributor"
    feed_mgmt_type = FeedDistFeedMgmt

    def allocate_feed_demand(self):
        """
        Save the feed consumption stored in x_fds on the data_attr of each herd based on
        the number of animals in x_ani.
        """
        if self.x is None:
            return

        concatenated_herds = concat_herds(self.herds)
        heads_total = concatenated_herds.data_attr.get("heads")

        for herd in (h for h in self.herds if h.has_feed_demand()):

            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            # Get the total number of animals for this (sp, br, ps, ss)-tuple across all
            # herds.
            h_total = heads_total.loc[slice(None), (sp, br, ss, ps, slice(None))]
            # Keep only (ps, ani)
            h_total.columns = h_total.columns.droplevel(
                ["species", "breed", "sub_system"]
            )
            # Get the number of animals in this herd object, which will be the same
            # value for most herds (but not non-cows in CattleHerd)
            h_heads = herd.data_attr.get("heads")

            matching_zeroes = h_heads[h_total == 0].replace({np.nan: 0}) == 0
            assert matching_zeroes.all().all(), (
                f"Total number of heads is recorded as zero, despite non-zero heads in herd: (sp={sp}, br={br}, ss={ss}, ps={ps}) "
            )

            # Compute the ratio between the two, where we (thanks to the assertion above) can safely replace 0 with 1 in the denominator to avoid division by zero
            ratio = (h_heads / h_total.replace({0: 1})).sort_index(axis=1).sort_index()

            if sp != "cattle":
                non_zero_ratio = ratio[ratio != 0]
                all_ratio_one_or_zero = (
                    (non_zero_ratio.replace({np.nan: 1}) == 1).all().all()
                )
                assert all_ratio_one_or_zero, (
                    "All ratios for non-cattle should be either 0 or 1"
                )
            try:
                feed_demands = (
                    self.x["fds"]
                    .to_frame("feed_amount")
                    .loc[(slice(None), slice(None), sp, br, ps, ss, slice(None)), :]
                    .reset_index()
                    .pivot(
                        columns=["prod_system", "animal", "feed"],
                        index="region",
                        values="feed_amount",
                    )
                    .sort_index(axis=1)
                    .sort_index()
                )
            except KeyError:
                # If KeyError there are no feeds to distribute to this AnimalHerd
                continue

            adjusted_feed_demands = (ratio.T * feed_demands.T).T
            # Drop cols where feed is nan, which may arise when multiplying with ratio
            adjusted_feed_demands = adjusted_feed_demands.loc[
                :, adjusted_feed_demands.columns.get_level_values("feed").notna()
            ]

            # Ensure that wherever h_heads is zero, feed_demands must be zero.
            heads_long = h_heads.T.stack("region")
            for (ps, ani, re), _ in heads_long[heads_long == 1]:
                feed_demands_slice = feed_demands.loc[re, (ps, ani, slice(None))]
                assert (feed_demands_slice == 0).all(), (
                    "Wherever we have zero animals, we should also have 0 feed"
                )

            herd.data_attr.add(
                adjusted_feed_demands,
                name="feed.demand",
                unit="kg DM/year",
                orig="FeedDist",
                desc="Demand for feed",
            )

    def allocate_feed_crop_prod_demand(self):
        par = self.feed_mgmt.par
        par.clear()

        pss = set([herd.prod_system for herd in self.herds])
        feed_to_crop_prod: pd.DataFrame = par.get_unique(["feed", "crop_prod"])
        # Copy feed -> cp mapping for each production system, with new col 'prod_system'
        feed_to_crop_prod = pd.concat(
            [feed_to_crop_prod.assign(prod_system=ps) for ps in pss]
        )

        par.set(
            feed=feed_to_crop_prod["feed"].to_list(),
            crop_prod=feed_to_crop_prod["crop_prod"].to_list(),
            prod_system=feed_to_crop_prod["prod_system"].to_list(),
        )
        feed_to_crop_prod["feed_to_prod"] = par.get("feed_to_prod")
        feed_to_crop_prod["share_domestic"] = par.get("share_domestic") / 100
        feed_to_crop_prod = feed_to_crop_prod.set_index(
            ["prod_system", "feed", "crop_prod"]
        )

        for herd in (h for h in self.herds if h.has_feed_demand()):

            feed_demands = herd.data_attr.get("feed.demand")

            # Compute the demand for crop_products yielded by the feed demands
            feed_crop_prod_demands = feed_demands_to_crop_demands(
                feed_demands, feed_to_crop_prod
            )

            herd.data_attr.add(
                feed_crop_prod_demands,
                name="feed.crop_product_demand",
                unit="kg/year",
                orig="FeedDist",
                desc="Demand for crop products for feed",
            )

    def get_x0(self):
        super().get_x0()

        idx_list = []
        for i,herd in enumerate(self.herds):

            # Get feeds in herd
            if "f_feed" in herd.par.data.index.names:
                feed_idx = pd.Index(
                    herd.par.get_unique("feed", qry = "parameter.isin(['share_in_ration','min_share_in_ration','max_share_in_ration'])"),
                    name = 'feed'
                )
            else:
                # Empty index if 'f_feed' filter column not in data
                feed_idx =  pd.Index([], name = 'feed')

            # Create full feed index for herd
            idx = multiindex_product(
                [self.herds.index[[i]]] +
                [herd.index] +
                [feed_idx] +
                [pd.Index(herd.animals, name='animal')]
            )

            idx_list += [idx]

        # Create combined feed index for all herds
        full_feed_idx = (
            idx_list[0]
            .append(list(idx_list[1:]))
            .reorder_levels([
                "feed",
                "animal",
                "species",
                "breed",
                "prod_system",
                "sub_system",
                "region",
            ])
        )

        # Add to x index dict
        self.x_idx['fds'] = full_feed_idx.copy()

        # Add feeds to x0
        self.x0['fds'] = pd.Series(data=0, index=full_feed_idx.copy())

        # Add feeds to x0 index
        self.x0_idx['fds'] = full_feed_idx.copy()

    def calculate_scaling_factors(
        self, scale_power: float = 0.0, cutoff_percentile: float = 99.0
    ):
        super().calculate_scaling_factors(scale_power, cutoff_percentile)
        # No scaling factors for feeds
        self.scale_f["fds"].iloc[:] = 0

    def make_C1(self):
        """Creates C1: A1 @ x == b1

        Main constraint to ensure that production exactly meets demand. Crop and animal
        products without any demand remain unconstrained. Demand is calculated in the
        'DemandAndConversions' module.
        """

        # Animal product demand
        A1_1 = self.make_A1_1()
        # Crop product demand
        A1_2 = self.make_A1_2()
        # Feed demand
        A1_3 = self.make_A1_3(
            row_idx = self.D_idx["crp"],
            prod_type = "crop_prod"
        )

        # Stack matrices
        A1 = scipy.sparse.vstack(
            [
                scipy.sparse.hstack(
                    [
                        A1_1.M,  # Animal heads to animal products
                        scipy.sparse.csc_array((A1_1.M.shape[0], A1_2.M.shape[1])),
                        scipy.sparse.csc_array((A1_1.M.shape[0], A1_3.M.shape[1])),
                    ]
                ),
                scipy.sparse.hstack(
                    [
                        scipy.sparse.csc_array((A1_2.M.shape[0], A1_1.M.shape[1])),
                        A1_2.M,  # Crops to crop products
                        A1_3.M,  # Feeds to (negative) crop products
                    ]
                ),
            ],
            format="csc",
        )

        A1 = IndexedMatrix(
            matrix=A1,
            row_idx={"ani": A1_1.rows, "crp": A1_2.rows},
            col_idx={"ani": A1_1.cols, "crp": A1_2.cols, "fds": A1_3.cols},
        )
        b1 = np.concatenate([self.D["ani"].values, self.D["crp"].values])

        # Append constraint
        self.constraints.update(
            {
                "C1: A1 @ x == b1": {
                    "left": lambda x, A1, b1: A1.M @ x,
                    "right": lambda A1, b1: b1,
                    "rel": "==",
                    "pars": {"A1": A1, "b1": b1},
                }
            }
        )

    def make_C2(self):
        """
        Creates C2: A2 @ x >= 0

        Constrain the share of feed demand for different crop products that must be met
        regionally. The minimum share is set via the parameter 'share_regional' in the
        'FeedMgmt' module and can differ for different animals.
        """

        factors = self._get_feed_to_prod_factors(index=True)
        factors_with_reg_share = factors[factors["share_regional"] > 0]
        row_idx = pd.MultiIndex.from_product(
            [
                self.x_idx["fds"].unique("prod_system"),
                factors_with_reg_share.index.unique("crop_prod"),
                self.x_idx["fds"].unique("region"),
            ],
            names=["prod_system", "crop_prod", "region"],
        )

        # We do not need the x_animal part for this
        Z_ani = scipy.sparse.csc_array((len(row_idx), len(self.x_idx["ani"])))
        # Production of crop products
        A2_1 = self.make_A2_1(row_idx)
        # Regional feed demand for crop products
        A2_2 = self.make_A2_2(row_idx, factors_with_reg_share)

        # Stack matrices
        A2 = IndexedMatrix(
            matrix=scipy.sparse.hstack([Z_ani, A2_1.M, A2_2.M], format="csc"),
            row_idx=A2_2.rows,
            col_idx={"ani": self.x_idx["ani"], "crp": A2_1.cols, "fds": A2_2.cols},
        )

        # Append constraint
        self.constraints["C2: A2 @ x >= 0"] = {
            "left": lambda x, A2: A2.M @ x,
            "right": lambda A2: 0,
            "rel": ">=",
            "pars": {"A2": A2},
        }

    def make_C5(self):
        """Creates C5: A5 @ x <= 0

        Constrain the maximum share of a crop product demand for feed that can be
        supplied by a particular crop group. This constraint is used to e.g. constrain
        the share of 'grazing' that can be supplied by 'semi-natural grasslands', but
        can also be used to constrain e.g. share of wheat for feed from winter/spring
        varieties.

        The maximum share is set via the parameter 'max_crop_in_crop_prod' in the
        'FeedMgmt' module and can differ for different animals.
        """

        cps_cgs: pd.DataFrame = self.feed_mgmt.par.get_unique(
            ["crop_prod", "crop_group"],
            qry='parameter == "max_crop_in_crop_prod"',
        )

        # Create the row index (cp,cg,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (cp, cg, ps, re)
                for cp, cg in cps_cgs.values
                for ps in self.x_idx["ani"].unique("prod_system")
                for re in self.x_idx["ani"].unique("region")
            ],
            names=["crop_prod", "crop_group", "prod_system", "region"],
        )

        # Production of crop products
        A5_1 = self.make_A5_1(row_idx)
        # Maximum supply of crop product(s) for a given feed
        A5_2 = self.make_A5_2(row_idx)
        Z_ani = scipy.sparse.csc_array((A5_1.M.shape[0], len(self.x_idx["ani"])))

        # Stack matrices
        A5 = scipy.sparse.hstack([Z_ani, A5_1.M, A5_2.M], format="csc")

        A5 = IndexedMatrix(
            matrix=A5,
            row_idx=A5_2.rows,
            col_idx={"ani": self.x_idx["ani"], "crp": A5_1.cols, "fds": A5_2.cols},
        )

        # Append constraint
        self.constraints.update(
            {
                "C5: A5 @ x <= 0": {
                    "left": lambda x, A5: A5.M @ x,
                    "right": lambda A5: 0,
                    "rel": "<=",
                    "pars": {"A5": A5},
                }
            }
        )

    def make_C7(self) -> None:
        """
        Creates C7: Drops variables

        Constrain crops to certain regions based on minimum growing degree days (GDD5).
        The minimum GDD5 for different crops is set with the parameter 'min_GDD5' in the
        'CropProduction' module. The number of GDD5 in each region is defined by the
        parameter 'GDD' in the 'Regions' module.

        This constraint also indirectly constrains animals with regional demand for
        crops that can't be grown in a region.
        """

        # This constraint is not implemented as a constraint in the solver but instead
        # drops variables representing crops or animals that can't be present in a
        # region.

        # IMPORTANT: This must be run after all other constraints have been defined!

        ani_idx = self.x_idx["ani"]
        crp_idx = self.x_idx["crp"]
        fds_idx = self.x_idx["fds"]

        # Get allowed crop-region combinations (i.e. region GDD5 >= min_GDD5 for crop)
        self.crops.par.clear()
        self.regions.par.clear()
        sel_crp = crp_idx[
            self.regions.par.get("GDD5", **crp_idx.to_frame().to_dict("list"))
            >= self.crops.par.get("min_GDD5", **crp_idx.to_frame().to_dict("list"))
        ]

        # Get positions of variables to keep for
        n_ani = len(ani_idx)
        n_crp = len(crp_idx)
        isel_ani = list(range(0, n_ani))
        isel_crp = [crp_idx.get_loc(s) + n_ani for s in sel_crp]
        isel_fds = list(range(n_ani + n_crp, n_ani + n_crp + len(fds_idx)))
        isel = isel_ani + isel_crp + isel_fds

        # Store short index (i.e. index of variables after dropping)
        self.x_idx_short = {"ani": ani_idx, "crp": sel_crp, "fds": fds_idx}

        # Drop variables from objective and constraint matrices
        for mat in self.matrices().values():
            if mat.M.shape[1] > len(isel):
                mat.M = mat.M[:, isel]
                mat.cols["ani"] = ani_idx.copy()
                mat.cols["crp"] = sel_crp.copy()
                mat.cols["fds"] = fds_idx.copy()

        return None

    def make_C8(
        self,
        C8_crp: pd.Series | None = None,
        C8_ani: pd.Series | None = None,
        C8_fds: pd.Series | None = None,
        C8_rel: list[str] | Literal["==", ">=", "<="] = "==",
        C8_tol: float = 1e-4,
    ):
        """Creates C8: A8 @ x <rel> b8

        Flexible constraint that constrains crop areas and/or animal numbers corresponding
        to the given Series in relation to the Series' values. Constraints can be eiter
        equality or max/min. Equality constraints (C8_rel = '==') are implemented as min
        and max constraints with a relative tolerance of +/- C8_tol.

        Multiple constraints can be created by supplying lists as parameters.

        Parameters
        ----------
        C8_crp : (list of) pandas.Series
            Crop areas to constrain to (index must match self.x_idx['crp'])
        C8_ani : (list of) pandas.Series
            Animal numbers to constrain to (index must match self.x_idx['ani'])
        C8_fds : (list of) pandas.Series
            Feed numbers to constrain to (index must match self.x_idx['fds'])
        C8_rel : (list of) string
            Type of constraint. Equality ('=='), minimum ('>=') or maximum ('<=')
        C8_tol : (list of) float

        Returns
        -------
        None
        """

        pars = {
            "C8_crp": C8_crp,
            "C8_ani": C8_ani,
            "C8_fds": C8_fds,
            "C8_rel": C8_rel,
            "C8_tol": C8_tol,
        }
        pars_len = {p: len(pars[p]) if isinstance(pars[p], list) else 0 for p in pars}
        pars_len_max = max(max(pars_len.values()), 1)

        if any([p_len > 1 and p_len < pars_len_max for p_len in pars_len.values()]):
            raise ValueError("Supplied lists must have the same length")

        # Align lists, e.g. ensuring that C8_rel is a list of relations rather than one
        for p in pars:
            if pars_len[p] < pars_len_max:
                if pars_len[p] == 1:
                    pars[p] = pars[p] * pars_len_max
                else:
                    pars[p] = [pars[p]] * pars_len_max

        if all([v is None for k in ["C8_crp", "C8_ani", "C8_fds"] for v in pars[k]]):
            raise ValueError(
                "At least one of 'C8_crp' or 'C8_ani' must be given to use constraint C8"
            )
        if any([v not in ["==", ">=", "<="] for v in pars["C8_rel"]]):
            raise ValueError("All 'C8_rel' must be one of '==', '>=' or '<='")

        if "==" in pars["C8_rel"] and pars["C8_tol"] is None:
            raise ValueError(
                "The C8_tol parameter was missing, but is required for C8 equality constraints."
            )

        # Get number of previously defined C8 constraints
        try:
            n_def = (
                max(
                    [
                        int(regex.search(r"_(\d+)", s).group(1))
                        for s in self.constraints.keys()
                        if "C8" in s
                    ]
                )
                + 1
            )
        except Exception:
            n_def = 0

        for i in range(pars_len_max):
            # Make matrix (A8)
            A8 = self.make_A8(
                C8_ani=pars["C8_ani"][i],
                C8_crp=pars["C8_crp"][i],
                C8_fds=pars["C8_fds"][i],
            )

            # Make right hand vector (b8)
            b8 = np.concatenate(
                [
                    pars[k][i].values
                    for k in ["C8_ani", "C8_crp", "C8_fds"]
                    if pars[k][i] is not None
                ]
            )

            rel = pars["C8_rel"][i]

            # Append constraint
            if rel == "==":
                tol = pars["C8_tol"][i]

                # Lower bound
                self.constraints.update(
                    {
                        f"C8_{str(i + n_def)}(low): A8 @ x >= b8 * (1-tol)": {
                            "left": lambda x, A8, b8, tol: A8.M @ x,
                            "right": lambda A8, b8, tol: b8 * (1 - tol),
                            "rel": ">=",
                            "pars": {"A8": A8, "b8": b8, "tol": tol},
                        }
                    }
                )
                # Upper bound
                self.constraints.update(
                    {
                        f"C8_{str(i + n_def)}(upp): A8 @ x <= b8 * (1+tol)": {
                            "left": lambda x, A8, b8, tol: A8.M @ x,
                            "right": lambda A8, b8, tol: b8 * (1 + tol),
                            "rel": "<=",
                            "pars": {"A8": A8, "b8": b8, "tol": tol},
                        }
                    }
                )

            else:
                self.constraints.update(
                    {
                        f"C8_{str(i + n_def)}: A8 @ x {rel} b8": {
                            "left": lambda x, A8, b8: A8.M @ x,
                            "right": lambda A8, b8: b8,
                            "rel": rel,
                            "pars": {"A8": A8, "b8": b8},
                        }
                    }
                )

        return None

    def make_C9(
        self,
        C9_crp: pd.Series | None = None,
        C9_ani: pd.Series | None = None,
        C9_fds: pd.Series | None = None,
        C9_rel: str = "==",
        C9_tol: float = 1e-4,
    ):
        """Creates C9: A9 @ x <rel> b9

        Flexible constraint that constrains the sum of crop areas and/or animal numbers
        corresponding to the index of passed Series in relation to the sum of given
        Series. Constraints can be either equality or max/min. Equality constraints
        (C9_rel = '==') are implemented as min and max constraints with a relative
        tolerance of +/- C9_tol.

        Multiple constraints can be created by supplying lists as parameters.

        Parameters
        ----------
        C9_crp : (list of) pandas.Series
            Crop areas to constrain sum of (index must match self.x_idx['crp'])
        C9_ani : (list of) pandas.Series
            Animal numbers to constrain sum of (index must match self.x_idx['ani'])
        C9_fds : (list of) pandas.Series
            Animal numbers to constrain sum of (index must match self.x_idx['fds'])
        C9_rel : (list of) string
            Type of constraint. Equality ('=='), minimum ('>=') or maximum ('<=')
        C9_tol : (list of) float

        Returns
        -------
        None
        """

        pars = {
            "C9_crp": C9_crp,
            "C9_ani": C9_ani,
            "C9_fds": C9_fds,
            "C9_rel": C9_rel,
            "C9_tol": C9_tol,
        }
        pars_len = {p: len(pars[p]) if isinstance(pars[p], list) else 0 for p in pars}
        pars_len_max = max(max(pars_len.values()), 1)

        if any([x > 1 and x < pars_len_max for x in pars_len.values()]):
            raise ValueError("Supplied lists must have the same length")

        # Align lists
        for p in pars:
            if pars_len[p] >= pars_len_max:
                continue
            if pars_len[p] == 1:
                pars[p] = pars[p] * pars_len_max
            else:
                pars[p] = [pars[p]] * pars_len_max

        if all([v is None for k in ["C9_ani", "C9_crp", "C9_fds"] for v in pars[k]]):
            raise ValueError(
                "At least one of 'C9_crp', 'C9_ani' or 'C9_fds' must be given to use constraint C9"
            )
        if any([v not in ["==", ">=", "<="] for v in pars["C9_rel"]]):
            raise ValueError("All 'C9_rel' must be one of '==', '>=' or '<='")

        # Get number of previously defined C9 constraints
        try:
            n_def = (
                max(
                    [
                        int(regex.search(r"_(\d+)", s).group(1))
                        for s in self.constraints.keys()
                        if "C9" in s
                    ]
                )
                + 1
            )
        except Exception:
            n_def = 0

        for i in range(pars_len_max):
            if all([pars[k] is None for k in ["C9_ani", "C9_crp", "C9_fds"]]):
                raise ValueError(
                    "The constraints 'C9_ani', C9_crp', and 'C9_fds' were all None"
                )

            # Make matrix (A9)
            A9 = self.make_A9(
                pars["C9_ani"][i],
                pars["C9_crp"][i],
                pars["C9_fds"][i],
            )

            # Make right hand vector (b9)
            b9 = sum(
                [
                    pars[k][i].sum() if pars[k][i] is not None else 0
                    for k in ["C9_ani", "C9_crp", "C9_fds"]
                ]
            )

            rel = pars["C9_rel"][i]

            # Append constraint
            if rel == "==":
                tol = pars["C9_tol"][i]

                # Lower bound
                self.constraints.update(
                    {
                        f"C9_{str(i + n_def)}(low): A9 @ x >= b9 * (1-tol)": {
                            "left": lambda x, A9, b9, tol: A9.M @ x,
                            "right": lambda A9, b9, tol: b9 * (1 - tol),
                            "rel": ">=",
                            "pars": {"A9": A9, "b9": b9, "tol": tol},
                        }
                    }
                )
                # Upper bound
                self.constraints.update(
                    {
                        f"C9_{str(i + n_def)}(upp): A9 @ x <= b9 * (1+tol)": {
                            "left": lambda x, A9, b9, tol: A9.M @ x,
                            "right": lambda A9, b9, tol: b9 * (1 + tol),
                            "rel": "<=",
                            "pars": {"A9": A9, "b9": b9, "tol": tol},
                        }
                    }
                )
            else:
                self.constraints.update(
                    {
                        f"C9_{str(i + n_def)}: A9 @ x {rel} b9": {
                            "left": lambda x, A9, b9: A9.M @ x,
                            "right": lambda A9, b9: b9,
                            "rel": rel,
                            "pars": {"A9": A9, "b9": b9},
                        }
                    }
                )

    def make_C10(self):
        """
        Ensure the production of by-products meets the demand required by feeds.
        """
        # Fetch the ammount of generated byproducts as calculated in the DemandsAndConversions-module
        D_byprod = self.demand.data_attr.get("by_products")

        # Subtract food uses of by-products
        used_byprod = self.demand.data_attr.get("by_prod_demand").sum(axis=1)
        if used_byprod.shape[0] > 0:
            D_byprod = D_byprod - used_byprod.astype(float).reindex(
                D_byprod.index
            ).fillna(0)

        # Union of byproducts used as feed and those originally in D_byprod
        all_byprods = set(
            self.feed_mgmt.par.get_unique("by_prod", qry="parameter == 'feed_to_prod'")
        ) | set(
            D_byprod.index.unique("by_prod"),
        )
        # Make sure any byproducts used as feed but not in D_byprod are included
        complete_idx = pd.MultiIndex.from_product(
            [D_byprod.index.unique("prod_system"), all_byprods],
            names=["prod_system", "by_prod"],
        )
        D_byprod = D_byprod.reindex(complete_idx, fill_value=0)

        # Construct the mapping of feed-products to by-products
        A10_fds = self.make_A1_3(
            row_idx = D_byprod.index,
            prod_type = "by_prod"
        )

        # Zero matrices
        Z_ani = scipy.sparse.csc_array((A10_fds.M.shape[0], len(self.x_idx["ani"])))
        Z_crp = scipy.sparse.csc_array((A10_fds.M.shape[0], len(self.x_idx["crp"])))

        M = IndexedMatrix(
            scipy.sparse.hstack([Z_ani, Z_crp, A10_fds.M], format="csc"),
            D_byprod.index,
            {
                "ani": self.x_idx["ani"],
                "crp": self.x_idx["crp"],
                "fds": self.x_idx["fds"],
            },
        )

        self.constraints.update(
            {
                "C10: A10 @ x + D >= 0": {
                    "left": lambda x, A10, D: A10.M @ x + D,
                    "right": lambda A10, D: 0,
                    "rel": ">=",
                    "pars": {"A10": M, "D": D_byprod},
                }
            }
        )

    def make_C11(self):
        """
        Ensure that the feed amounts comply with the feed rations.
        """

        def with_zeroes(A11: None | IndexedMatrix):
            """Helper function to add 0s for the animal- and crop parts"""
            if A11 is None:
                return None
            Z_ani = scipy.sparse.csc_array((A11.M.shape[0], len(self.x_idx["ani"])))
            Z_crp = scipy.sparse.csc_array((A11.M.shape[0], len(self.x_idx["crp"])))
            return IndexedMatrix(
                scipy.sparse.hstack([Z_ani, Z_crp, A11.M], format="csc"),
                col_idx={
                    "ani": self.x_idx["ani"],
                    "crp": self.x_idx["crp"],
                    "fds": self.x_idx["fds"],
                },
                row_idx=A11.rows,
            )

        # Create A-matrices for each of the parameters. If there is no data for any
        # given parameter, it will be ignored
        A11_eq = with_zeroes(self.make_A11("share_in_ration"))
        A11_min = with_zeroes(self.make_A11("min_share_in_ration"))
        A11_max = with_zeroes(self.make_A11("max_share_in_ration"))

        C11s: dict[str, Constraint] = {}

        if A11_min is not None:
            C11s["C11 (min): A11 @ x >= 0"] = {
                "left": lambda x, A11: A11.M @ x,
                "right": lambda A11: 0,
                "rel": ">=",
                "pars": {"A11": A11_min},
            }

        if A11_eq is not None:
            C11s["C11 (eq): A11 @ x == 0"] = {
                "left": lambda x, A11: A11.M @ x,
                "right": lambda A11: 0,
                "rel": "==",
                "pars": {"A11": A11_eq},
            }

        if A11_max is not None:
            C11s["C11 (max): A11 @ x <= 0"] = {
                "left": lambda x, A11: A11.M @ x,
                "right": lambda A11: 0,
                "rel": "<=",
                "pars": {"A11": A11_max},
            }

        self.constraints.update(C11s)

    def make_C12(self):
        """
        Ensure the nutrient demands are met by the animals- and feed configuration.
        """
        self.feed_mgmt.par.clear()
        # We manually add "DM" here, as it's not a value in feed_par
        feed_pars = ["DM", *self.feed_mgmt.par.get_unique("feed_par")]

        row_idx = extend_index(
            index=self.x_idx["fds"].droplevel("feed").unique(),
            names=["feed_par"],
            levels=[feed_pars],
            mode="prepend",
        )

        def make_A12(rel):
            A12_1 = self.make_A12_1(row_idx, rel)
            # Drop rows where we lack constriants
            A12_1.prune_rows()
            # Create an A12_2 which only contains the rows for which we have data for
            # the given herd, animal and feed_par.
            A12_2 = self.make_A12_2(A12_1.rows)
            Z_crp = scipy.sparse.csc_array((len(A12_1.rows), len(self.x_idx["crp"])))

            if A12_1.shape[0] == 0:
                return None

            return IndexedMatrix(
                scipy.sparse.hstack([A12_1.M, Z_crp, A12_2.M]),
                row_idx=A12_1.rows,
                col_idx={
                    "ani": self.x_idx["ani"],
                    "crp": self.x_idx["crp"],
                    "fds": self.x_idx["fds"],
                },
            )

        A12_min = make_A12("min")
        A12_eq = make_A12("eq")
        A12_max = make_A12("max")

        if A12_min is not None:
            self.constraints["C12 (min): A12 @ x >= 0"] = {
                "left": lambda x, A12: A12.M @ x,
                "right": lambda A12: 0,
                "rel": ">=",
                "pars": {"A12": A12_min},
            }
        if A12_eq is not None:
            self.constraints["C12 (eq): A12 @ x == 0"] = {
                "left": lambda x, A12: A12.M @ x,
                "right": lambda A12: 0,
                "rel": "==",
                "pars": {"A12": A12_eq},
            }
        if A12_max is not None:
            self.constraints["C12 (max): A12 @ x <= 0"] = {
                "left": lambda x, A12: A12.M @ x,
                "right": lambda A12: 0,
                "rel": "<=",
                "pars": {"A12": A12_max},
            }

    def make_C13(self):
        """
        Constrain feed parameters in as a share of dry matter (DM), using the parameter
        'feed_req_of_DM_min' and 'feed_req_of_DM_max' on herds.
        """
        A13_fds_min = self.make_A13("min")
        A13_fds_max = self.make_A13("max")

        n_cols_ani = len(self.x_idx["ani"])
        n_cols_crp = len(self.x_idx["crp"])

        def _A13(A13_fds: IndexedMatrix):
            """Helper function to fill the remaining matrix with zeroes."""
            M = scipy.sparse.hstack(
                [
                    scipy.sparse.csc_array((A13_fds.shape[0], n_cols_ani)),
                    scipy.sparse.csc_array((A13_fds.shape[0], n_cols_crp)),
                    A13_fds.M,
                ],
                format="csc",
            )
            return IndexedMatrix(
                M,
                row_idx=A13_fds.rows,
                col_idx={
                    "crp": self.x_idx["crp"],
                    "ani": self.x_idx["ani"],
                    "fds": self.x_idx["fds"],
                },
            )

        A13_min = _A13(A13_fds_min)
        A13_max = _A13(A13_fds_max)

        if A13_min.shape[0] > 0:
            self.constraints["C13 (min): A13 @ x >= 0"] = {
                "left": lambda x, A13: A13.M @ x,
                "right": lambda A13: 0,
                "rel": ">=",
                "pars": {"A13": A13_min},
            }

        if A13_max.shape[0] > 0:
            self.constraints["C13 (max): A13 @ x <= 0"] = {
                "left": lambda x, A13: A13.M @ x,
                "right": lambda A13: 0,
                "rel": "<=",
                "pars": {"A13": A13_max},
            }

    def make_C14(self):
        """
        Ensure that the imports of feed do not exceed the max_total_imported parameter.

        The A-matrix maps feeds to their import-shares, so that we get the volume we
        import, while the B matrix contains the ceiling of max imported volume. This is
        done per prod_sys and feed product, for both main- and by-products.
        """
        try:
            b14_cp = self.make_b14("crop_prod")
            b14_by = self.make_b14("by_prod")
        except (ValueError, KeyError):
            warnings.warn(
                "C14 enabled, but b14 could not be built. This is likely because no feeds had max_total_import defined. Thus, C14 was ignored."
            )
            return

        b14 = np.concatenate([np.array(b14_cp.values), np.array(b14_by.values)])

        A14_cp = self.make_A1_3(b14_cp.index, "crop_prod", domestic=False)
        A14_by = self.make_A1_3(b14_by.index, "by_prod", domestic=False)

        n_rows = A14_cp.shape[0] + A14_by.shape[0]
        A14 = scipy.sparse.hstack(
            [
                scipy.sparse.csc_array((n_rows, len(self.x_idx["ani"]))),
                scipy.sparse.csc_array((n_rows, len(self.x_idx["crp"]))),
                scipy.sparse.vstack([A14_cp.M, A14_by.M], format="csc"),
            ],
            format="csc",
        )

        A14 = IndexedMatrix(
            A14,
            row_idx=A14_cp.rows.append(A14_by.rows),
            col_idx={
                "crp": self.x_idx["crp"],
                "ani": self.x_idx["ani"],
                "fds": self.x_idx["fds"],
            },
        )

        self.constraints.update(
            {
                "C14: A14 @ x >= 0": {
                    "left": lambda x, A14, b14: A14.M @ x - b14,
                    "right": lambda A14, b14: 0,
                    "rel": "<=",
                    "pars": {"A14": A14, "b14": b14},
                }
            }
        )

    def make_C15(self):
        """
        Ensure that generated by-products cover regional feed demand for by-products 
        according to 'share_regional' parameter in FeedMgmt module.

        By-products are assumed to be generated in the same region as the animals/crops
        producing main products from which the by-products are produced.
        """

        # Get feed to by-product conversion factors
        fe_to_bp = self._get_feed_to_prod_factors('by_prod')
        
        # Make unique MultiIndex of (prod_system, by_prod) where
        # share_domestic and share_regional > 0
        ps_bp_idx = (
            fe_to_bp
            .loc[(fe_to_bp['share_domestic']>0) & (fe_to_bp['share_regional']>0)]
            .set_index(['prod_system','by_prod'])
            .index.unique()
        )
        # Get regions
        re_idx = self.x_idx['crp'].unique('region')
        
        # Make row index (prod_system, by_prod, region)
        row_idx = multiindex_product([ps_bp_idx, re_idx]).sort_values()
        
        # Map animal numbers --> Generated by-products
        A15_1 = self.make_A15_1(row_idx)
        # Map crop areas --> Generated by-products
        A15_2 = self.make_A15_2(row_idx)
        # Map feeds --> Used by-products
        A15_3 = self.make_A15_3(row_idx)
        
        # Construct final matrix
        A15 = IndexedMatrix(
            scipy.sparse.hstack([A15_1.M, A15_2.M, A15_3.M]),
            row_idx=row_idx,
            col_idx={
                "ani": self.x_idx["ani"],
                "crp": self.x_idx["crp"],
                "fds": self.x_idx["fds"],
            },
        )
        
        # Append constraint
        self.constraints["C15: A15 @ x >= 0"] = {
                "left": lambda x, A15: A15.M @ x,
                "right": lambda A15: 0,
                "rel": ">=",
                "pars": {"A15": A15},
            }

    def make_C16(self):

        # Get demand for crop residues
        D = self.demand.data_attr.get("crop_resid_demand").sum(axis=1)
        # Create index for feed demand for crop residues
        feed_to_resid_idx = multiindex_product([
            self.x_idx['ani'].unique("prod_system"),
            self.feed_mgmt.par.get_unique(["crop_resid","feed"], qry="parameter == 'feed_to_prod'")
            .set_index(["crop_resid","feed"]).index
        ])
        
        # Create row index as union of crop residue demand
        # from DemandAndConversions and feed demand from FeedMgmt
        row_idx = D.index.union(
            feed_to_resid_idx.droplevel("feed")
        ).unique()
        
        # Get feeds used for bedding
        feeds_for_bedding = (
            self.manure_mgmt
            .par.get_unique(
                "feed",
                qry="parameter == 'bedding_material_use'"
            )
        )
        
        # Make sure there are no "feed" used for bedding that are not
        # connected to a crop crop residue in the FeedMgmt module
        # NOTE: Potentially we may want to allow feeds
        # connected to other products (i.e. crop_prod, by_prod)
        assert set(feeds_for_bedding) - set(feed_to_resid_idx.get_level_values("feed")) == set(), \
        "Feed used for bedding not connected to crop residue in FeedMgmt module"
        
        # ani (bedding) --> crop_resid
        A16_1 = self.make_A16_1(row_idx, feeds_for_bedding)
        # crop --> crop_resid
        A16_2 = self.make_A16_2(row_idx)
        # feed --> crop_resid
        A16_3 = self.make_A1_3(row_idx, "crop_resid")
        
        # Stack matrices
        A16 = IndexedMatrix(
            scipy.sparse.hstack([A16_1.M, A16_2.M, A16_3.M]),
            row_idx=row_idx,
            col_idx=self.x_idx.copy(),
        )
        b16 = D.reindex(row_idx, fill_value=0)
        
        # Append constraint
        self.constraints.update(
            {
                "C16: A16 @ x >= b16": {
                    "left": lambda x, A16, b16: A16.M @ x,
                    "right": lambda A16, b16: b16,
                    "rel": ">=",
                    "pars": {"A16": A16, "b16": b16},
                }
            }
        )

    def make_P1(self):
        # x['ani'] --> x0['ani']
        P1_1 = self.make_P1_1()
        # x['crp'] --> x0['crp']
        P1_2 = self.make_P1_2()
        # x['fds'] --> 0
        P1_3 = self.make_P1_3()

        # P1 needs to be a square matrix, with each side the length of x
        P1 = scipy.sparse.vstack(
            [
                scipy.sparse.hstack(
                    [
                        P1_1.M,
                        scipy.sparse.csc_array((P1_1.M.shape[0], P1_2.M.shape[1])),
                        scipy.sparse.csc_array((P1_1.M.shape[0], P1_3.M.shape[1])),
                    ]
                ),
                scipy.sparse.hstack(
                    [
                        scipy.sparse.csc_array((P1_2.M.shape[0], P1_1.M.shape[1])),
                        P1_2.M,
                        scipy.sparse.csc_array((P1_2.M.shape[0], P1_3.M.shape[1])),
                    ]
                ),
                scipy.sparse.hstack(
                    [
                        scipy.sparse.csc_array((P1_3.M.shape[0], P1_1.M.shape[1])),
                        scipy.sparse.csc_array((P1_3.M.shape[0], P1_2.M.shape[1])),
                        P1_3.M,
                    ]
                ),
            ],
            format="csc",
        )

        return IndexedMatrix(
            matrix=P1,
            row_idx={"ani": P1_1.rows, "crp": P1_2.rows},
            col_idx={"ani": P1_1.cols, "crp": P1_2.cols},
        )

    def make_A1_2(self):
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx["crp"]
        # Get col index from crop production (cr,ps,re)
        col_idx = self.x_idx["crp"]

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        net_production = self._get_crop_production()

        merged = net_production.merge(
            row_idx_df, on=["prod_system", "crop_prod"]
        ).merge(col_idx_df, on=["crop", "prod_system", "region"])

        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_frame(
            merged, row_idx, col_idx, values_name="net_production"
        )

    def make_A1_3(
            self,
            row_idx:pd.MultiIndex,
            prod_type:Literal["crop_prod", "by_prod", "crop_resid"],
            domestic:bool = True
        ):
        """
        Matrix that converts feeds to products at a national level.
        The values are negative to as this represents a demand

        Used in constraints C1, C10, C14 and C16
        """
        # Get col index from feed demands (f,sp,br,ps,ss,re)
        col_idx = self.x_idx["fds"]

        # Convert row- and col indices to dataframes to prepare for merging
        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        # Get the factors, and perform the multiplication now for ease when merging
        factors = self._get_feed_to_prod_factors(prod_type)
        # Negative values (*-1) to indicate a 'demand' rather than production of cps
        if domestic:
            # Map domestic share
            factors["feed_to_prod"] *= -1 * factors["share_domestic"]
        else:
            # Map imported share
            factors["feed_to_prod"] *= -1 * (1-factors["share_domestic"])

        # Merge the row_idx with factors, and the result of that with the col_idx
        merged = row_idx_df.merge(factors, on=["prod_system", prod_type]).merge(
            col_idx_df,
            on=["feed", "animal", "species", "breed", "prod_system", "sub_system"],
        )

        return IndexedMatrix.from_frame(
            merged, row_idx=row_idx, col_idx=col_idx, values_name="feed_to_prod"
        )

    def make_A2_1(self, row_idx: pd.MultiIndex):
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx["crp"]

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        # dataframe with index (crop, prod_system, region) and multi-index columns
        # (crop_prod)
        crop_production: pd.DataFrame = self.crops.data_attr.get("production")
        # Get the crop_products in the index - anything else we can ignore
        crop_products: pd.Index = row_idx.unique("crop_prod")

        # Filter crop production to only produced crops (cp, ps)
        produced_crops = crop_production.index[
            crop_production[crop_products].sum(axis=1) > 0
        ].unique()
        # ... and filter crop production based on this
        filtered_production = crop_production.loc[produced_crops].reset_index()
        # Reshape production DataFrame to long format, with the index:
        #   crop, prod_system, region, crop_prod, production
        production_long = (
            filtered_production.melt(
                id_vars=["crop", "prod_system", "region"],
                value_vars=crop_products,
                var_name="crop_prod",
                value_name="production",
            ).dropna(subset=["production"])  # Exclude missing production values
        )

        # Merge production with `row_idx` and `col_idx` to get row and column indices
        merged = production_long.merge(
            row_idx_df, on=["prod_system", "crop_prod", "region"]
        ).merge(col_idx_df, on=["crop", "prod_system", "region"])

        return IndexedMatrix.from_frame(
            merged, row_idx, col_idx, values_name="production"
        )

    def make_A2_2(self, row_idx: pd.MultiIndex, factors_with_reg_share: pd.DataFrame):
        # Get col index from feed consumption (feed,animal,species,breed,prod_system,sub_system,region)
        col_idx = self.x_idx["fds"]
        # Create DataFrames from indices for merging
        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        # Construct series with index (feed, crop_prod) by multiplying together columns
        # Note: Multiply by -1 to indicate demand
        factors = (
            -1
            * factors_with_reg_share["feed_to_prod"]
            * factors_with_reg_share["share_domestic"]
            * factors_with_reg_share["share_regional"]
        ).reset_index(name="values")

        # Merge factors with row and column indices
        merged = row_idx_df.merge(factors, on=["prod_system", "crop_prod"]).merge(
            col_idx_df, on=col_idx.names
        )

        return IndexedMatrix.from_frame(merged, row_idx, col_idx)

    def make_A5_1(self, row_idx):
        """
        Maps the crops in x to their respective crop products, generating the total
        amount of each crop product (with crop group) produced in each production system
        and region.
        """
        # Get map crop_group --> crop(s)
        map_cg_cr = inv_dict(self.par.get_rel("crop", "crop_group"))

        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx["crp"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        for cp, cg, ps in row_idx.droplevel("region").unique():
            # Get crop(s)
            cr = map_cg_cr[cg]

            res = (
                self.crops.data_attr.get("production")
                .loc[(cr, ps, slice(None)), (cp)]
                .fillna(0)
            )

            # Store values and row/col nr
            val.extend(res.values)
            row_nr.extend(
                [
                    row_idx.get_loc((cp, cg, ps, re))
                    for re in res.index.get_level_values("region")
                ]
            )
            col_nr.extend([col_idx.get_loc((cr, ps, re)) for cr, ps, re in res.index])

        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_coordinates(
            (val, (row_nr, col_nr)),
            row_idx,
            col_idx,
        )

    def make_A5_2(self, row_idx):
        """
        Maps the feed products to their respective maximum share of crop products
        generated.
        """
        # Get crop product and crop combinations where there is a constraint for maximum inclusion
        cps_cgs = self.feed_mgmt.par.get_unique(
            ["crop_prod", "crop_group"],
            qry='parameter == "max_crop_in_crop_prod"',
        )

        # Get the factors mapping feeds to crop products.
        fds_to_cps_factors = self._get_feed_to_prod_factors(index=True)
        # Pick only the domestic share, as this constraint does not apply to imported crops.
        # Convert it to a long-format dataframe (w/o index) so that we can merge it
        fds_to_cps_factors = (
            (fds_to_cps_factors["feed_to_prod"] * fds_to_cps_factors["share_domestic"])
            .to_frame(name="feed_to_dom_crop_prod")
            .reset_index()
        )

        # All regions
        regions = self.x_idx["ani"].unique("region")

        # Create the row index (cp,cg,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (cp, cg, ps, re)
                for cp, cg in cps_cgs.values
                for ps in self.x_idx["ani"].unique("prod_system")
                for re in regions
            ],
            names=["crop_prod", "crop_group", "prod_system", "region"],
        )

        # Get col index from feeds (f,ani,sp,br,ps,ss,re)
        col_idx = self.x_idx["fds"]

        # Create one large DataFrame from herds to use with vectorized operations
        retrieve_df = (
            pd.DataFrame(
                {
                    "species": [herd.species for herd in self.herds],
                    "breed": [herd.breed for herd in self.herds],
                    "prod_system": [herd.prod_system for herd in self.herds],
                    "sub_system": [herd.sub_system for herd in self.herds],
                    "animal": [herd.animals for herd in self.herds],
                }
            )
            # explode it on animal so that we get one row per animal.
            .explode(column="animal")
            # and then cross-merge with cps_cgs to get the cartesian product,
            # The new len (of rows) is then = len(old) x len(cps_cgs)
            .merge(cps_cgs, how="cross")
        )
        # Append a dummy-column so that we can pivot without losing the column index
        retrieve_df["value"] = np.nan
        # Pivot to get a wide matrix, as get_from_frame cannot use only a series
        retrieve_df = retrieve_df.pivot(
            index=["species", "breed", "prod_system", "sub_system", "animal"],
            columns=["crop_prod", "crop_group"],
            values="value",
        )

        self.feed_mgmt.par.clear()
        # Get all max_crop_in_crop_prod values for all herds at the same time,
        # and format to a long-format so that we can merge it later
        max_crop_values = (
            (
                self.feed_mgmt.par.get_from_frame(
                    "max_crop_in_crop_prod", retrieve_df, warn_if_nan=False
                )
                / 100
            )
            .stack(level=["crop_prod", "crop_group"], future_stack=True)
            .reset_index()
            .rename(columns={0: "max_crop_in_crop_prod"})
            .dropna()  # Drop any nan-values, as we don't need them
        )

        # Merge the mappings from feed product -> (domestic) crop products with the max_crop_values
        values_df = max_crop_values.merge(
            fds_to_cps_factors,
            on=["species", "breed", "prod_system", "sub_system", "animal", "crop_prod"],
        )
        values_df["values"] = (
            -1 * values_df["feed_to_dom_crop_prod"] * values_df["max_crop_in_crop_prod"]
        )
        # Drop now duplicate values to avoid confusion
        values_df = values_df.drop(
            columns=["feed_to_dom_crop_prod", "max_crop_in_crop_prod"]
        )

        # Create dataframes for row- and col indices so that we can merge with the data
        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        # Merge with row- and col index to match values with their respective col/row i
        merged = values_df.merge(
            row_idx_df, on=["prod_system", "crop_prod", "crop_group"]
        ).merge(col_idx_df, on=col_idx.names)

        return IndexedMatrix.from_frame(merged, row_idx, col_idx)

    def make_A8(self, C8_ani, C8_crp, C8_fds):
        # Store the row_index for ani, crp and fds
        row_idx = {}
        # Get col indexes from x_idx for ani, crp and fds
        col_idx = {k: v.copy() for k, v in self.x_idx.items()}

        # Vertically stack identity matrices for each of the categories that are not
        # None, and fill horizontally with zeroes.
        MS = []
        for label, A in [("ani", C8_ani), ("crp", C8_crp), ("fds", C8_fds)]:
            if A is None:
                continue
            # Assign index
            row_idx[label] = A.index
            # Create identity matrix from col_idx
            n = len(col_idx[label])
            M = scipy.sparse.identity(n, format="csc")
            # Drop rows to match row index
            sel_rows = [col_idx[label].get_loc(i) for i in row_idx[label]]
            M = M[sel_rows, :]
            # Create zero matrix and hstack
            Z_ani = scipy.sparse.csc_array((M.shape[0], len(col_idx["ani"])))
            Z_crp = scipy.sparse.csc_array((M.shape[0], len(col_idx["crp"])))
            Z_fds = scipy.sparse.csc_array((M.shape[0], len(col_idx["fds"])))
            # Extend M with zeroes
            if label == "ani":
                M = scipy.sparse.hstack([M, Z_crp, Z_fds], format="csc")
            elif label == "crp":
                M = scipy.sparse.hstack([Z_ani, M, Z_fds], format="csc")
            else:  # label == "fds"
                M = scipy.sparse.hstack([Z_ani, Z_crp, M], format="csc")

            MS.append(M)

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(scipy.sparse.vstack(MS), row_idx, col_idx)

        return M

    def make_A9(self, C9_ani, C9_crp, C9_fds):
        # No row index, only one row
        row_idx = None
        # Get col index (cr,ps,re), (sp,br,ps,ss,re), (f,ani,br,ps,ss,re)
        col_idx = self.x_idx.copy()

        MS = []
        for label, A in [("ani", C9_ani), ("crp", C9_crp), ("fds", C9_fds)]:
            if A is not None:
                # Create a 1-by-len(col_idx) matrix and set cols corresponding to
                # index to 1
                M = scipy.sparse.lil_array((1, len(col_idx[label])))
                sel_cols = [col_idx[label].get_loc(i) for i in A.index]
                M[:, sel_cols] = 1
                MS.append(M)
            else:
                # Append zero matrix
                Z = scipy.sparse.csc_array((1, len(col_idx[label])))
                MS.append(Z)

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(scipy.sparse.hstack(MS, format="csc"), row_idx, col_idx)

        return M

    def make_A11(self, param: str) -> None | IndexedMatrix:
        row_idx = self.x_idx["fds"]
        col_idx = self.x_idx["fds"]

        herd_dfs = []
        for herd in self.herds:
            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            p = herd.par
            p.clear()

            try:
                feeds = p.get_unique(["feed"], qry=f'parameter=="{param}"').feed.tolist()
            except KeyError:
                # Empty list if 'f_feed' filter column not in data
                feeds = []

            # Where there are no 'constraints' for this parameter- and herd combo, we
            # do not need to add anything to the df.
            if len(feeds) == 0:
                continue

            retrieve_df = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [(ani, sp, br, ps, ss) for ani in herd.animals],
                    names=["animal", "species", "breed", "prod_system", "sub_system"],
                ),
                columns=pd.Index(feeds, name="feed"),
            )
            df = p.get_from_frame(param, retrieve_df, warn_if_nan=False).stack("feed")

            herd_dfs.append(df)

        # No values for the given parameter defined, in which case we can return a None
        if len(herd_dfs) == 0:
            return None

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        shares = (
            pd.concat(herd_dfs).reorder_levels(
                [n for n in self.x_idx["fds"].names if n != "region"]
            )
            / 100
        ).to_frame(name="share")

        # Get loss factors which we want to multiply with every (non-zero) element.
        # Loss factors are needed here because x_fds concerns feed _demand_, whereas
        # the constraint concerns feed _consumption_. Rename feed index level for later merge
        losses = self._get_losses_factors(shape="long").rename_axis(index={'feed':'feed_c'})

        # Note: Join first on shares, then on col_idx for a significantly faster merge
        merged_df = row_idx_df.merge(
            shares.reset_index(), on=shares.index.names
        ).merge(
            col_idx_df,
            on=[cname for cname in row_idx.names if cname != "feed"],
            suffixes=("", "_c"),
        ).merge(
            losses.reset_index(), on=losses.index.names
        )

        merged_df["values"] = (
            np.where(
                merged_df["feed"] == merged_df["feed_c"],
                1 - merged_df["share"],
                -merged_df["share"],
            )
            * merged_df["losses_factor"]
        )

        return IndexedMatrix.from_frame(merged_df, row_idx, col_idx)

    def make_A12_1(self, row_idx: pd.MultiIndex, rel: Literal["min", "eq", "max"]):
        """
        Map animals to their respective feed requirements
        """
        col_idx = self.x_idx["ani"]

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        dfs = []

        for herd in self.herds:
            # Note: negative values to indicate demand
            feed_req = -herd.data_attr.get(f"feed_req_{rel}")
            data = (
                feed_req.unstack("region")
                .reset_index()
                .rename(
                    columns={
                        0: "feed_req",
                        "prod_system": "feed_ps",
                    }
                )
            )
            data["species"] = herd.species
            data["breed"] = herd.breed
            data["herd_ps"] = herd.prod_system
            data["herd_ss"] = herd.sub_system
            dfs.append(data)

        merged_df = (
            pd.concat(dfs)
            .merge(
                row_idx_df,
                # TODO: Note the use of 'herd_ss' here. Should we include 'feed_ss'?
                left_on=[
                    "feed_par",
                    "animal",
                    "species",
                    "breed",
                    "feed_ps",
                    "herd_ss",
                    "region",
                ],
                right_on=[
                    "feed_par",
                    "animal",
                    "species",
                    "breed",
                    "prod_system",
                    "sub_system",
                    "region",
                ],
            )
            .merge(
                col_idx_df,
                left_on=["species", "breed", "herd_ps", "herd_ss", "region"],
                right_on=["species", "breed", "prod_system", "sub_system", "region"],
            )
        )

        return IndexedMatrix.from_frame(
            merged_df, row_idx, col_idx, values_name="feed_req"
        )

    def make_A12_2(self, row_idx: pd.MultiIndex):
        """
        Map feeds to their respective feed parameters (e.g. fat, energy, etc. contents),
        adjusted for storage- and feeding losses
        """
        col_idx = self.x_idx["fds"]
        row_idx_df = pd.DataFrame(range(len(row_idx)), index=row_idx, columns=["row_i"])
        col_idx_df = pd.DataFrame(range(len(col_idx)), index=col_idx, columns=["col_i"])

        joined_df = row_idx_df.join(col_idx_df)

        def perc_to_change_factor(df):
            return (100 - df) / 100

        # Set filters to get feed composition and losses
        self.feed_mgmt.par.clear()
        self.feed_mgmt.par.set(**joined_df.index.to_frame().to_dict('list'))

        # Get feed composition
        joined_df["feed_composition"] = np.nan_to_num(
            self.feed_mgmt.par.get('feed_composition', warn_if_nan=False) # <-- Warn for NaN???
        )
        joined_df.loc[joined_df.index.get_level_values('feed_par') == 'DM', "feed_composition"] = 1 # Dry matter is allways 1

        joined_df["loss_factor"] = (
            perc_to_change_factor(self.feed_mgmt.par.get('storage_losses')) *
            perc_to_change_factor(self.feed_mgmt.par.get('feeding_losses'))
        )
        joined_df["values"] = joined_df["feed_composition"] * joined_df["loss_factor"]
        joined_df = joined_df.reset_index(drop=True)

        return IndexedMatrix.from_frame(joined_df, row_idx, col_idx)

    def make_b14(self, prod_type: Literal["crop_prod", "by_prod"]) -> pd.Series:
        """
        Demand vector of the max total import values for each feed
        """
        self.feed_mgmt.par.clear()

        par = "max_total_imported"

        pss_cps = self.feed_mgmt.par.get_unique(['prod_system',prod_type], qry=f'parameter=="{par}"')

        data = pd.Series(
            self.feed_mgmt.par.get(par, **pss_cps.to_dict('list')),
            index = pss_cps.set_index(['prod_system',prod_type]).index
        ) * 1_000 # tonnes --> kg

        return data

    def make_A13(self, rel_type: Literal["min", "max"]):
        """
        Maps x_fds to values ensuring that feed_req_of_DM_{min,max} are met.

        Each row maps to one feed_parameter constraint (e.g. min 5% fat of DM) in one
        animal system (ani, sp, br, ps, ss, re). Each value in that row maps the feed
        given in the corresponding column to one of two values:

        - 0, if the animal system is not the same, else
        - (p - k) * l, where:
            - p is the factor of the given feed_par in this feed for this animal sys,
            - k is the min/max share of the feed_par in relation to the DM
            - l is the factor (e.g. 5% loss -> 0.95) for storage- and feeding losses
        """

        col_idx = self.x_idx["fds"]

        feed_pars = set()
        herd_dfs = {}

        for herd in self.herds:
            data = herd.data_attr.get(f"feed_req_of_DM_{rel_type}")

            # Where there are no 'constraints' for this parameter- and herd combo, we
            # do not need to add anything to the df.
            if data.empty:
                continue
            # Keep track of which
            feed_pars.update(data.columns.unique("feed_par"))
            # prod_system already in data attribute, hence not here.
            herd_dfs[(herd.species, herd.breed, herd.sub_system)] = data.T.stack(
                "region"
            )

        if len(herd_dfs) == 0:
            row_idx = pd.MultiIndex.from_tuples(
                [],
                names=[
                    "feed_par",
                    "animal",
                    "species",
                    "breed",
                    "prod_system",
                    "sub_system",
                    "region",
                ],
            )
            return IndexedMatrix(
                scipy.sparse.csc_array((0, len(col_idx))), row_idx, col_idx
            )

        herds_df = (
            pd.concat(herd_dfs, names=["species", "breed", "sub_system"])
            .to_frame(name="feed_req_of_DM")
            .reset_index()
        )

        # We base our row-idx on x_fds, but without the 'feed' level
        _base_row_idx = self.x_idx["fds"].droplevel("feed")
        # ... and then multiply in each feed_par that we want to look at
        row_idx = extend_index(
            levels=[feed_pars], names=["feed_par"], index=_base_row_idx, mode="prepend"
        )

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        merged = (
            herds_df
            # Merge on row_idx to get a full index matching (feed_par, sp, br, ps, ss, ani, region) -> feed_req_of_dm
            .merge(row_idx_df, on=row_idx.names)
            # merge with col_idx to add feed to the rows
            .merge(
                col_idx_df,
                on=[
                    "animal",
                    "species",
                    "breed",
                    "prod_system",
                    "sub_system",
                    "region",
                ],
            )
        )

        def perc_to_change_factor(df):
            return (100 - df) / 100

        # Set filters to get feed composition and losses
        self.feed_mgmt.par.clear()
        self.feed_mgmt.par.set(
            **merged[[
                'feed',
                'animal',
                'species',
                'breed',
                'prod_system',
                'sub_system',
                'region',
                'feed_par'
                ]].to_dict('list')
        )

        # Get feed composition
        merged['feed_to_par_factor'] = np.nan_to_num(
            self.feed_mgmt.par.get('feed_composition', warn_if_nan=False) # <-- Warn for NaN???
        )

        # Get feed losses
        merged['losses_factor'] = (
            perc_to_change_factor(self.feed_mgmt.par.get('storage_losses')) *
            perc_to_change_factor(self.feed_mgmt.par.get('feeding_losses'))
        )

        merged["values"] = (
            merged["losses_factor"] * (merged["feed_to_par_factor"] - merged["feed_req_of_DM"])
        )

        return IndexedMatrix.from_frame(merged, row_idx, col_idx)
    
    def make_A15_1(self,row_idx):
        """
        Create a matrix mapping animal numbers to generated by-products based
        on data attribute 'by_prod_per_animal_prod' calculated in the 
        DemandAndConversions module.
        """

        # Get col index from crop production (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]
        
        # Get by-products generated per animal product
        by_per_ap = self.demand.data_attr.get('by_prod_per_animal_prod').sort_index()
        
        val = []
        row_nr = []
        col_nr = []
        
        for herd in self.herds:
        
            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system
        
            # Go through animal products
            for ap in herd.data_attr.get("production").columns.unique("animal_prod"):
                # Go through output production systems
                for ops in herd.data_attr.get("production").columns.unique("prod_system"):
                    # Get by-products produced from animal product and
                    # corresponding conversion factors
                    try:
                        bps_f = by_per_ap.loc[(ops,sp,ap)]
                    except KeyError:
                        # If not by-products generated, continue
                        continue
                    for bp in bps_f.index:
                        if (ops, bp) in row_idx:
                            # Get production of by-product bp from animal product (ap) and output production system (ops) per head
                            # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
                            # and region (re)
                            res = (
                                herd.data_attr.get("production")
                                .loc[:, (ops, slice(None), ap)]
                                .sum(axis=1) * bps_f[bp]
                            )
        
                            if (res != 0).any():
                                # Store values and row/col nr
                                val.extend(res.values)
                                row_nr.extend([row_idx.get_loc((ops, bp, re)) for re in res.index])
                                col_nr.extend(
                                    [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                                )
                                
        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_coordinates((val, (row_nr, col_nr)), row_idx, col_idx)
    
    def make_A15_2(self,row_idx):
        """
        Create a matrix mapping crop areas to generated by-products based
        on data attribute 'by_prod_per_crop_prod' calculated in the 
        DemandAndConversions module.
        """

        # Get col index from crop production (cr,ps,re)
        col_idx = self.x_idx["crp"]
        
        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        # Get net crop production and by-products generated per crop product
        net_production = self._get_crop_production()
        by_per_cp = self.demand.data_attr.get('by_prod_per_crop_prod').reset_index(name='by_prod_per_crop_prod')
        
        if len(
            set(by_per_cp.set_index(['prod_system','by_prod']).index) & 
            set(row_idx.droplevel('region'))
        ) == 0:
            # If by-products generated by crops not in row_idx return zero matrix
            return IndexedMatrix.zeros(
                row_idx, col_idx
            )
        
        # Calculate ammount of by-products generated per crop areas
        by_production = net_production.merge(
            by_per_cp, on=["prod_system","crop_prod"]
        )
        by_production["values"] = (
            by_production["net_production"] * by_production["by_prod_per_crop_prod"]
        )
        by_production = by_production.loc[by_production["values"] != 0] # Drop zero rows
        
        # Merge to get row and col index numbers for constructing matrix
        merged = by_production.merge(
            row_idx_df,
            on=["prod_system", "by_prod","region"]
        ).merge(
            col_idx_df,
            on=["crop", "prod_system", "region"]
        )
        
        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_frame(
            merged, row_idx, col_idx
        )
    
    def make_A15_3(self,row_idx):
        """
        Create a matrix mapping feeds to regional demand for by-products
        based on the parameters 'feed_to_prod', 'share_domestic' and 'share_regional'
        in the FeedMgmt module.
        """

        # Get col index from crop production (fe,an,sp,br,ps,ss,re)
        col_idx = self.x_idx["fds"]
        
        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")
        
        fe_to_bp = self._get_feed_to_prod_factors('by_prod')
        fe_to_bp = fe_to_bp.loc[(fe_to_bp['share_domestic']>0) & (fe_to_bp['share_regional']>0)]
        
        merged = fe_to_bp.merge(
            row_idx_df,
            on = ["prod_system","by_prod"]
        ).merge(
            col_idx_df,
            on = ["feed","animal","species","breed","prod_system","sub_system","region"]
        )
        
        merged["values"] = (
            merged["feed_to_prod"]
            * merged["share_domestic"]
            * merged["share_regional"]
            * (-1)
        )
        
        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_frame(
            merged, row_idx, col_idx
        )

    def make_A16_1(self, row_idx, feeds):
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]
        
        self.manure_mgmt.par.clear()
        self.feed_mgmt.par.clear()
        
        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []
        
        # Some convenience functions for getting parameters
        def sfv(**kwargs):
            self.manure_mgmt.par.set(**kwargs)
            self.feed_mgmt.par.set(**kwargs)
        gfm = self.manure_mgmt.par.get_from_frame
        gff = self.feed_mgmt.par.get_from_frame
        
        # Go through animal herds
        for herd in self.herds:
        
            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system
        
            # Set filter values
            sfv(
                species = sp,
                breed = br,
                sub_system = ss
            )
        
            # Go through animal products
            for ops, cr in row_idx:
        
                # Set filter values
                sfv(
                    prod_system = ops,
                    crop_resid = cr,
                )
        
                # Get animal heads in ops
                heads = herd.data_attr.get('heads').loc[:,ops]
                # Add feeds to column index
                df = (
                    # Columns must be MultiIndex
                    index_to_multi(heads, axis=1)
                    .reindex(
                        pd.MultiIndex.from_product([heads.columns, feeds], names=["animal","feed"]),
                        axis=1
                    )
                )
        
                # Sum of (heads * bedding material use per head * feed -> crop_resid factor * share domestic)
                # --> Domestic demand for crop_resid per defining animal (x)
                # Negative sign
                res = (
                    df
                    .mul(gfm("bedding_material_use",df))
                    .mul(gff("feed_to_prod",df))
                    .mul(gff("share_domestic",df)/100)
                    .sum(axis=1)
                )
        
                # We only need to store anything if there is a demand for bedding materials
                if (res>0).any():
                    # Store values and row/col nr
                    val.extend(-res.values) # negative sign
                    row_nr.extend([row_idx.get_loc((ops, cr))] * len(res))
                    col_nr.extend(
                        [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                    )
        
        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_coordinates((val, (row_nr, col_nr)), row_idx, col_idx)

    def make_A16_2(self, row_idx):

        # Get index for columns
        col_idx = self.x_idx["crp"]
        
        # Create DataFrame with correct shape
        df = pd.DataFrame(
            1,
            columns = col_idx,
            index = row_idx
        )
        
        # Calculate harvestable crop residues generated per crop
        df = (
            df
            # Get generated above ground crop residues
            .mul(
                self.crops.data_attr.get('crop_residues').loc[:,"above ground"],
                axis=1
            )
            # Factor in share of crop residues harvestable
            .mul(
                self.crop_residue_mgmt.par
                .get_from_frame("crop_resid_harvestable", df.droplevel("prod_system"))
            )
        )
        
        # Set to Zero where prod_system in columns and rows do not match.
        # I.e. crops produce crop residues of the same production system
        mask = (
            np.array(
                df.index.get_level_values('prod_system')
            )[:, None]
            ==
            np.array(
                df.columns.get_level_values('prod_system')
            )[None, :]
        )
        df = df.where(mask, 0)
        
        # Create IndexedMatrix
        return IndexedMatrix(
            scipy.sparse.csc_array(df),
            row_idx,
            col_idx
        )

    def make_P1_1(self):
        """
        Creates the P_{1,1} matrix, which is an identity matrix of size len(x0['ani'])
        as x0['ani'].index == x['ani'].index
        """
        # Get row index from x0['ani'] (sp,br,ps,re)
        row_idx = self.x0_idx["ani"]
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]

        # Data and corresponding row/col numbers for constructing matrix
        val = [1] * len(col_idx)
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((sp, br, ps, re)) for sp, br, ps, _, re in col_idx]

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix.from_coordinates(
            (val, (row_nr, col_nr)),
            row_idx,
            col_idx,
        )

        return M

    def make_P1_2(self):
        """
        Creates the P_{1,2} matrix, which is an identity matrix of size len(x0['crp']) as x0['crp'].index == x['crp'].index
        """
        # Get row index from x0['crp'] (cr,ps,re)
        row_idx = self.x0_idx["crp"]
        # Get row index from x['crp'] (cr,ps,re)
        col_idx = self.x_idx["crp"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = [1] * len(col_idx)
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((cr, ps, re)) for cr, ps, re in col_idx]

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix.from_coordinates(
            (val, (row_nr, col_nr)),
            row_idx,
            col_idx,
        )

        return M

    def make_P1_3(self):
        """
        Create the P_{1,3} matrix, which is a zero-value matrix covering x_feed.
        This, as we do not want to include the feeds in the optimisation target.
        """

        # Get row index from x0['fds'] (f,ani,sp,br,ps,ss,re)
        row_idx = self.x0_idx["fds"]
        # Get row index from x['fds'] (f,ani,sp,br,ps,ss,re)
        col_idx = self.x_idx["fds"]

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.csc_array((len(row_idx), len(col_idx))),
            row_idx,
            col_idx,
        )
        return M
    
    def _get_crop_production(self):
        """
        Get DataFrame mapping crops to net production (i.e. production - seeds)
        of crop products
        """
        def get_long(name: str) -> pd.DataFrame:
            return (
                self.crops.data_attr.get(name)
                .stack("crop_prod")
                .reset_index()
                .rename(columns={0: name})
            )

        # Get the crop_production and seed demand, and convert both to long format
        crop_production = get_long("production")
        seed_demand = get_long("seed_demand")

        net_production = crop_production.merge(
            seed_demand,
            how="left",  # keep the rows where that have no seed demand.
            on=["crop", "prod_system", "region", "crop_prod"],
        )

        net_production["net_production"] = net_production[
            "production"
        ] - net_production["seed_demand"].fillna(0)
        net_production = net_production[
            ["crop", "prod_system", "region", "crop_prod", "net_production"]
        ]

        return net_production

    def _get_feed_to_prod_factors(
        self,
        prod_type: Literal["crop_prod", "by_prod", "crop_resid"] = "crop_prod",
        index: bool = False,
    ) -> pd.DataFrame:
        """
        Get a DataFrame mapping feed products to crop products, with their respective
        conversion factor as well as respective import- and regional shares.
        """
        feed_par = self.feed_mgmt.par
        feed_par.clear()

        feed_to_prod: pd.DataFrame = feed_par.get_unique(["feed", prod_type])
        row_idx = self.x_idx["fds"].droplevel("region").unique().to_frame(index=False)
        df = row_idx.merge(feed_to_prod, on="feed")

        filters = {cname: df[cname].to_list() for cname in df.columns}
        params = ["feed_to_prod", "share_domestic", "share_regional"]
        for param in params:
            df[param] = feed_par.get(param, **filters)

        # Shares are in % (e.g. 42), but we want them as fraction (e.g. 0.42)
        for p in ["share_domestic", "share_regional"]:
            df[p] = df[p] / 100

        if index:
            df = df.set_index(
                ["feed", "animal", "species", "breed", "prod_system", "sub_system"]
                + [prod_type]
            )

        return df

    def _get_losses_factors(
        self, shape: Literal["wide", "long"] = "wide"
    ) -> pd.DataFrame:
        """
        Get change factors to account for losses in feeds.

        Returns
        -------
        pd.DataFrame: with row-index corresonding to x_fds (except for region and feed)
            and single-level column-index "feed".
        """

        # Get all losses
        losses_retrieve_df = pd.DataFrame(
            columns=self.x_idx["fds"].unique("feed"),
            # Note: Feed losses can't differ by region
            index=self.x_idx["fds"].droplevel(["region", "feed"]).unique(),
        )

        def perc_to_change_factor(df):
            return (100 - df) / 100

        self.feed_mgmt.par.clear()
        df = perc_to_change_factor(
            self.feed_mgmt.par.get_from_frame("storage_losses", losses_retrieve_df)
        ) * perc_to_change_factor(
            self.feed_mgmt.par.get_from_frame("feeding_losses", losses_retrieve_df)
        )

        assert shape in ["wide", "long"]

        if shape == "long":
            df = df.stack("feed").to_frame(name="losses_factor")

        return df

    def _get_feed_compositions(self, shape: Literal["wide", "long"] = "wide"):
        """
        Get the feed compositions, i.e. the ratio of a certan feed_par (e.g. ME, energy)
        that is present in a given feed for a given species per unit of dry-mass.

        DM is included and set to 1 for each feed.

        index: feed, species
        columns: feed_par

        Returns
        -------
        pd.DataFrame
        """
        # Get all feeds for which we have a feed_composition
        feeds = self.feed_mgmt.par.get_unique(
            "feed", qry="parameter=='feed_composition'"
        ).tolist()
        feed_pars = [*self.feed_mgmt.par.get_unique("feed_par")]

        base_idx = self.x_idx["fds"]
        mask = base_idx.get_level_values("feed").isin(feeds)
        row_idx = base_idx if all(mask) else base_idx[mask]

        data = self.feed_mgmt.par.get_from_frame(
            "feed_composition",
            # Retrieve df
            pd.DataFrame(
                index=row_idx,
                columns=pd.Index(feed_pars, name="feed_par"),
                dtype=float,
            ),
            warn_if_nan=False,
        ).sort_index()
        # Add a column called 'DM' with value 1 everywhere (all feeds defined on DM basis)
        data = data.assign(DM=1)

        if shape == "long":
            data = data.stack("feed_par").to_frame(name="feed_to_par_factor")

        return data

    def adjust_crop_allocation(self):
        # NOTE: THIS METHOD DOES NOT WORK WITH FeedDistributor
        # currently as the attribute 'feed.max_supply_from_crop_group'
        # is not set with the FeedDistFeedMgmt
        print('Crop allocation not adjusted ', end='')
        return None
