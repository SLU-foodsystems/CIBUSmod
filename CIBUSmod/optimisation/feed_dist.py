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
from ..utils.misc import multiply_aligned, inv_dict, extend_index
from ..utils.data_attr import DataAttr
from ..main_modules.animal_herd import concat_herds

from .indexed_matrix import IndexedMatrix
from .utils import Constraint, make_cvxpy_constraint, feed_demands_to_crop_demands

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

        for herd in self.herds:
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

        for herd in self.herds:
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

        # Add feeds to x index
        self.x_idx['fds'] = pd.MultiIndex.from_tuples(
            [
                (f, ani, sp, br, ps, ss, re)
                for (sp, br, ps, ss) in self.herds.index
                for re in self.herds[(sp, br, ps, ss)].index
                for f in 
                    self.herds[(sp, br, ps, ss)].par
                    .get_unique("feed", qry = "parameter.isin(['share_in_ration','min_share_in_ration','max_share_in_ration'])")
                for ani in self.herds[(sp, br, ps, ss)].animals
            ],
            names = [
                "feed",
                "animal",
                "species",
                "breed",
                "prod_system",
                "sub_system",
                "region",
            ],
        )

        # Add feeds to x0
        self.x0['fds'] = pd.Series(data=0, index=self.x_idx["fds"])

        # Add feeds to x0 index
        self.x0_idx['fds'] = self.x_idx["fds"].copy()

    def calculate_scaling_factors(
        self, scale_power: float = 0.0, cutoff_percentile: float = 99.0
    ):
        """Calculates scaling factor to apply to x and x0 in objective O1 as f = rn * sf
        where rn is a factor normalising all features (i.e. distinct land uses and
        animal species) to the same range and fs is a scaling factor calculated as
        fs = ( mean(x0 * rn) / (x0 * rn) ) ^ scale_power. A cutoff that limits the
        maximum scaling factor to a certain percentile is implemented to avoid that
        crops/animals with x0 close to or equal to zero are effectively removed from the
        solution space.
        """

        scale_f = {key: df.copy() for key, df in zip(self.x0.keys(), self.x0.values())}

        # First all fetures (i.e. land uses and animal species) are normalised to the same range
        # (0 - max) as land use = cropland
        norm_max = (
            self.x0["crp"]
            .rename(self.crops.par.get_rel("crop", "land_use"))
            .loc["cropland"]
            .max()
        )
        # We then compute the range(?) for each group
        rn = pd.concat(
            [
                self.x0["ani"]
                .groupby("species")
                .transform(lambda x: (1 / x.max()) if x.max() > 0 else norm_max)
                * norm_max,
                self.x0["crp"]
                .rename(self.crops.par.get_rel("crop", "land_use"))
                .groupby("crop")
                .transform(lambda x: (1 / x.max()) if x.max() > 0 else norm_max)
                * norm_max,
            ]
        )

        # Compute sf without taking fds into account, as we don't want it to affect the
        # means.
        x0 = pd.concat([self.x0["ani"], self.x0["crp"]]) * rn.values
        sf = x0.mean() / x0
        cutoff_value = np.percentile(sf.loc[sf != np.inf], cutoff_percentile)
        sf.loc[sf > cutoff_value] = cutoff_value
        sf = sf**scale_power

        f = rn.values * sf.values
        assert np.isfinite(f).all(), "Non-finite values encountered in scaling factors"

        (n_ani, n_crp) = (len(scale_f["ani"]), len(scale_f["crp"]))
        # Write back the values to scale_f
        scale_f["ani"].iloc[:] = f[:n_ani]
        scale_f["crp"].iloc[:] = f[n_ani : n_ani + n_crp]
        scale_f["fds"].iloc[:] = 0
        self.scale_f = scale_f

    def define_cvx_problem(self):
        # Apply scaling factors to x0
        x0s = cvxpy.Constant(
            np.concatenate(
                [
                    (self.x0[k] * self.scale_f[k]).reindex(self.x0_idx[k])
                    for k in ["ani", "crp", "fds"]
                ]
            )
        )

        lvls = ["species", "breed", "prod_system", "region", "sub_system"]
        # Get scaling factors for x
        sf = cvxpy.Constant(
            np.concatenate(
                [
                    self.scale_f["ani"]
                    .reindex(self.x_idx["ani"].reorder_levels(lvls))
                    .reindex(self.x_idx_short["ani"].reorder_levels(lvls)),
                    self.scale_f["crp"].reindex(self.x_idx_short["crp"]),
                    self.scale_f["fds"].reindex(self.x_idx_short["fds"]),
                ]
            )
        )

        n = (
            len(self.x_idx_short["ani"])
            + len(self.x_idx_short["crp"])
            + len(self.x_idx_short["fds"])
        )
        x = cvxpy.Variable(n, nonneg=True)

        objective = cvxpy.Minimize(
            cvxpy.sum_squares((self.P1.M @ cvxpy.multiply(sf, x)) - x0s)
        )

        # Append constraints
        constraints = [
            make_cvxpy_constraint(cons, x) for cons in self.constraints.values()
        ]

        # Define problem
        self.problem = cvxpy.Problem(objective=objective, constraints=constraints)

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
        A1_3 = self.make_A1_3()

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

    def make_C3(self):
        """Creates C3: A3 @ x <= b3

        Constrain the maximum area per 'land_use' in each region. The maximum area is
        set relative to areas in x0 via the parameter 'max_land_use_factor' in the
        'Regions' module. A 'max_land_use_factor' of 1 implies that areas can't exceed
        current areas in each region.

        If the Regions setting 'max_land_use_from_scenario_x0' is set to True (default
        is False) maximum land use changes if there are changes to 'x0_crops' in a
        scenario, otherwise default values of 'x0_crops' are used irrespective of any
        changes in scenarios.
        """

        A3 = self.make_A3()

        b3 = np.array(
            [
                self.regions.data_attr.get("max_land_use").loc[x[1], x[0]]
                for x in A3.rows
            ]
        )

        # Append constraint
        self.constraints.update(
            {
                "C3: A3 @ x <= b3": {
                    "left": lambda x, A3, b3: A3.M @ x,
                    "right": lambda A3, b3: b3,
                    "rel": "<=",
                    "pars": {"A3": A3, "b3": b3},
                }
            }
        )

    def make_C4(self):
        """Creates C4: A4 @ x <= 0

        Constrain the maximum share of defining animal heads per species, breed and
        prod_system belonging to a given sub_system on national level.

        The maximum share is set via the parameter 'max_share_sub_system' in the
        respective 'AnimalHerd' modules and can differ by breed.
        """

        A4 = self.make_A4()

        if A4.shape[0] == 0:
            warnings.warn(
                "When building C4, the resulting A4 matrix turned out empty. C4 thus ignored."
            )
            return

        # Append constraint
        self.constraints.update(
            {
                "C4: A4 @ x <= 0": {
                    "left": lambda x, A4: A4.M @ x,
                    "right": lambda A4: 0,
                    "rel": "<=",
                    "pars": {"A4": A4},
                }
            }
        )

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

    def make_C6(self):
        """Creates C6: A6 @ x <= 0

        Constrain the minimum and/or maximum share of cropland devoted to a given crop
        group in a given region in a given production system. The maximum share is set
        on 'crop_group' level via the parameters 'min_in_rot' and 'max_in_rot' in the
        'CropProduction' module.

        Note: This constraint only applies to crops with 'cropland' as 'land_use' in the
        relation tables.
        """

        # Note to future:
        # - Deal with crops assumed not to be in rotation by putting 0 in the matrix

        for minmax in ["min", "max"]:
            if (
                self.crops.par.get_unique(
                    "crop_group", qry=f'parameter == "{minmax}_in_rot"'
                ).shape[0]
                > 0
            ):
                A6 = self.make_A6(minmax)

                sign = "<=" if minmax == "max" else ">="

                # Append constraint
                self.constraints.update(
                    {
                        f"C6_{minmax}: A6 @ x {sign} 0": {
                            "left": lambda x, A6: A6.M @ x,
                            "right": lambda A6: 0,
                            "rel": sign,
                            "pars": {"A6": A6},
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
        A10_fds = self.make_A10_1(D_byprod.index)

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
                "C10: A10 @ x - D <= 0": {
                    "left": lambda x, A10, D: A10.M @ x - D,
                    "right": lambda A10, D: 0,
                    "rel": "<=",
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

        # Build first a 'large' A12_2, from which we later slice smaller versions
        # depending on which rows are present in A12_1.
        A12_2_complete = self.make_A12_2(row_idx)

        def make_A12(rel):
            A12_1 = self.make_A12_1(row_idx, rel)
            # Drop rows where we lack constriants
            A12_1.prune_rows()
            # Create an A12_2 which only contains the rows for which we have data for
            # the given herd, animal and feed_par.
            A12_2 = IndexedMatrix.align_rows(A12_2_complete, A12_1)
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

        A14_cp = self.make_A14("crop_prod", b14_cp.index)
        A14_by = self.make_A14("by_prod", b14_by.index)

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

    def make_A1_1(self):
        # Get row index from animal product demand vector (ps,sp,ap)
        row_idx = self.D_idx["ani"]
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        # Go through animal herds
        for herd in self.herds:
            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            # Go through animal products
            for ap in herd.data_attr.get("production").columns.unique("animal_prod"):
                # Go through output production systems
                for ops in herd.data_attr.get("production").columns.unique(
                    "prod_system"
                ):
                    if (ops, sp, ap) in row_idx:
                        # Get production of animal product (ap) from output production system (ops) per head
                        # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
                        # and region (re)
                        res = (
                            herd.data_attr.get("production")
                            .loc[:, (ops, slice(None), ap)]
                            .sum(axis=1)
                        )

                        # Store values and row/col nr
                        val.extend(res.values)
                        row_nr.extend([row_idx.get_loc((ops, sp, ap))] * len(res))
                        col_nr.extend(
                            [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                        )

        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_coordinates((val, (row_nr, col_nr)), row_idx, col_idx)

    def make_A1_2(self):
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx["crp"]
        # Get col index from crop production (cr,ps,re)
        col_idx = self.x_idx["crp"]

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

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

        merged = net_production.merge(
            row_idx_df, on=["prod_system", "crop_prod"]
        ).merge(col_idx_df, on=["crop", "prod_system", "region"])

        # Create Compressed Sparse Column matrix
        return IndexedMatrix.from_frame(
            merged, row_idx, col_idx, values_name="net_production"
        )

    def make_A1_3(self):
        """
        Matrix that converts feed products to crop products at a national level.
        The values are negative to subtract from the other crop products produces, for
        the A1 matrix to yield the net amount of crop products on multiplication with x.
        """
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx["crp"]
        # Get col index from feed demands (f,sp,br,ps,ss,re)
        col_idx = self.x_idx["fds"]

        # Convert row- and col indices to dataframes to prepare for merging
        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        # Get the factors, and perform the multiplication now for ease when merging
        factors = self._get_feed_to_prod_factors("crop_prod")
        # Negative values (*-1) to indicate a 'demand' rather than production of cps
        factors["feed_to_prod"] *= -1 * factors["share_domestic"]

        # Merge the row_idx with factors, and the result of that with the col_idx
        merged = row_idx_df.merge(factors, on=["prod_system", "crop_prod"]).merge(
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

    def make_A3(self):
        # Get land uses to constrain
        land_uses = self.regions.data_attr.get("max_land_use").columns
        # Get row index from land uses and regions (lu,re)
        row_idx = pd.MultiIndex.from_product(
            [land_uses, self.x_idx["crp"].unique("region")],
            names=["land_use", "region"],
        )
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx["crp"]

        # Get dict for translating crop --> land use
        rel = self.par.get_rel("crop", "land_use")

        # Data and corresponding row/col numbers for constructing matrix
        val = [1 if rel[cr] == lu else 0 for lu in land_uses for cr, _, _ in col_idx]
        col_nr = list(range(len(col_idx))) * len(land_uses)
        row_nr = [row_idx.get_loc((lu, re)) for lu in land_uses for _, _, re in col_idx]

        M = scipy.sparse.coo_array(
            (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
        ).tocsc()
        # Zero matrix
        Z_ani = scipy.sparse.csc_array((M.shape[0], len(self.x_idx["ani"])))
        Z_fds = scipy.sparse.csc_array((M.shape[0], len(self.x_idx["fds"])))

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([Z_ani, M, Z_fds], format="csc"),
            row_idx,
            {"ani": self.x_idx["ani"], "crp": col_idx, "fds": self.x_idx["fds"]},
        )

        return M

    def make_A4(self):
        # Get row index from animal herds with max in sub_system constraint (sp,br,ps,ss)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (h.species, h.breed, h.prod_system, h.sub_system)
                for h in self.herds
                if "max_share_sub_system"
                in h.par.data.index.get_level_values("parameter")
                and h.sub_system
                in h.par.get_unique(
                    "sub_system", qry='parameter == "max_share_sub_system"'
                )
            ],
            names=["species", "breed", "prod_system", "sub_system"],
        )

        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        for sp, br, ps, ss in row_idx:
            h = self.herds.loc[sp, br, ps, ss]
            h.par.clear()

            f = (
                h.par.get(
                    "max_share_sub_system",
                    species=sp,
                    breed=br,
                    prod_system=ps,
                    sub_system=ss,
                )
                / 100
            )

            try:
                len(f)
            except TypeError:
                pass
            finally:
                f = float(f[0])

            vls = [
                0
                if (sp != sp_) | (br != br_) | (ps != ps_)
                else ((1 - f) if ss == ss_ else -f)
                for sp_, br_, ps_, ss_, _ in col_idx
            ]
            cns = list(range(len(col_idx)))
            rns = [row_idx.get_loc((sp, br, ps, ss)) for _ in col_idx]

            val.extend(vls)
            col_nr.extend(cns)
            row_nr.extend(rns)

        M = scipy.sparse.coo_array(
            (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
        ).tocsc()
        # Zero matrices for crops and feeds
        Z_crp = scipy.sparse.csc_array((M.shape[0], len(self.x_idx["crp"])))
        Z_fds = scipy.sparse.csc_array((M.shape[0], len(self.x_idx["fds"])))

        # Create Compressed Sparse Column matrix
        return IndexedMatrix(
            scipy.sparse.hstack([M, Z_crp, Z_fds], format="csc"),
            row_idx,
            {"ani": self.x_idx["ani"], "crp": col_idx, "fds": self.x_idx["fds"]},
        )

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

    def make_A6(self, minmax: Literal["min", "max"]):
        self.crops.par.clear()

        # Get crop groups with min/max inclusion in rotation constraint
        cgs = self.crops.par.get_unique(
            "crop_group", qry=f'parameter == "{minmax}_in_rot"'
        )
        pss = self.x_idx["crp"].unique("prod_system")
        res = self.x_idx["crp"].unique("region")

        # Get row index from (cg,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [(cg, ps, re) for cg in cgs for ps in pss for re in res],
            names=["crop_group", "prod_system", "region"],
        )
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx["crp"]

        # Get dict for translating crop --> land use
        lu_rel = self.par.get_rel("crop", "land_use")
        # Get dict for translating crop --> crop group
        cg_rel = self.par.get_rel("crop", "crop_group")

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        for cg, ps in row_idx.droplevel("region").unique():
            # Create Series of 'min/max_in_rot' factors for crop_group = cg
            # and prod_system = ps for each region
            f = pd.Series(
                self.crops.par.get(
                    f"{minmax}_in_rot", crop_group=cg, prod_system=ps, region=list(res)
                )
                / 100,
                index=res,
            )

            vls = [
                0
                if (ps != ps_) | (lu_rel[cr] != "cropland")
                else ((1 - f.loc[re]) if cg_rel[cr] == cg else -f.loc[re])
                for cr, ps_, re in col_idx
            ]
            cns = list(range(len(col_idx)))
            rns = [row_idx.get_loc((cg, ps, re)) for _, _, re in col_idx]

            val.extend(vls)
            col_nr.extend(cns)
            row_nr.extend(rns)

        M = scipy.sparse.coo_array(
            (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
        ).tocsc()
        Z_ani = scipy.sparse.csc_array((M.shape[0], len(self.x_idx["ani"])))
        Z_fds = scipy.sparse.csc_array((M.shape[0], len(self.x_idx["fds"])))

        # Create Compressed Sparse Column matrix
        return IndexedMatrix(
            scipy.sparse.hstack([Z_ani, M, Z_fds], format="csc"),
            row_idx,
            {"ani": self.x_idx["ani"], "crp": col_idx, "fds": self.x_idx["fds"]},
        )

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

    def make_A10_1(self, D_idx: pd.MultiIndex) -> IndexedMatrix:
        """
        Create a matrix mapping feeds to domestic by-products based on the parameters
        'share_domestic' and 'feed_to_prod' in feed_mgmt par.
        """
        # Get row index from byproducts demand vector (prod_sys, by_prod)
        row_idx = D_idx
        # Get col index from feed demands (f,sp,br,ps,ss,re)
        col_idx = self.x_idx["fds"]

        feed_to_byprod = self._get_feed_to_prod_factors("by_prod", index=True)
        # Store only the conversion of feed to (domestic share of) byprods, and drop any
        # zero-value rows. Reset the index to prepare for merging.
        feed_to_byprod_long = (
            (feed_to_byprod["feed_to_prod"] * feed_to_byprod["share_domestic"])
            .to_frame(name="values")
            .replace({0: np.nan})
            .dropna()
            .reset_index()
        )

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        merged_df = feed_to_byprod_long.merge(
            row_idx_df, on=["by_prod", "prod_system"]
        ).merge(col_idx_df, on=[cname for cname in col_idx.names if cname != "region"])

        return IndexedMatrix.from_frame(merged_df, row_idx, col_idx)

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

            feeds = p.get_unique(["feed"], qry=f'parameter=="{param}"').feed.tolist()

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

        loss_factors = self._get_losses_factors(shape="long")
        feed_compositions = self._get_feed_compositions(shape="long").dropna()

        values = feed_compositions.join(loss_factors)
        values = (values["losses_factor"] * values["feed_to_par_factor"]).to_frame(
            name="values"
        )

        joined_df = values.join(row_idx_df).join(col_idx_df)
        # Drop all levels in the index except "row_i". This is significantly (10x)
        # faster than doing reset_index() directly, for some reason.
        cols_to_drop = [x for x in joined_df.index.names if x != "row_i"]
        joined_df = joined_df.reset_index(cols_to_drop, drop=True)

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

        # Get loss factors and composition for all feeds
        losses_factors = self._get_losses_factors(shape="long").reset_index()
        feed_compositions = self._get_feed_compositions(shape="long").reset_index()

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
            # Now merge with feed_compositions to map feeds to feed_pars. how="left" to set a default value of 0
            .merge(
                feed_compositions,
                how="left",
                on=[
                    "feed_par",
                    "feed",
                    "animal",
                    "species",
                    "breed",
                    "prod_system",
                    "sub_system",
                ],
            )
            .merge(
                losses_factors,
                how="left",
                on=["feed", "animal", "species", "breed", "prod_system", "sub_system"],
            )
            .fillna(0)
        )

        merged["values"] = (
            merged["losses_factor"] * (merged["feed_to_par_factor"] - merged["feed_req_of_DM"])
        )
        return IndexedMatrix.from_frame(merged, row_idx, col_idx)

    def make_A14(
        self, prod_type: Literal["crop_prod", "by_prod"], row_idx: pd.Index
    ) -> IndexedMatrix:
        """
        Make a matrix mapping the feeds to their total imported amounts of crop
        products. The row-index is ["prod_system", prod_type].
        """
        self.feed_mgmt.par.clear()
        col_idx = self.x_idx["fds"]

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        feed_to_prod = self._get_feed_to_prod_factors(prod_type, index=True)
        feed_to_prod = (
            feed_to_prod["feed_to_prod"] * (1 - feed_to_prod["share_domestic"])
        ).reset_index(name="feed_to_imp_prod")

        merged = feed_to_prod.merge(row_idx_df, on=["prod_system", prod_type]).merge(
            col_idx_df,
            on=["animal", "species", "breed", "prod_system", "sub_system", "feed"],
        )

        return IndexedMatrix.from_frame(
            merged, row_idx, col_idx, values_name="feed_to_imp_prod"
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

    def _get_feed_to_prod_factors(
        self,
        crop_prod_type: Literal["crop_prod", "by_prod"] = "crop_prod",
        index: bool = False,
    ) -> pd.DataFrame:
        """
        Get a DataFrame mapping feed products to crop products, with their respective
        conversion factor as well as respective import- and regional shares.

        Depending on the 'detailed_index' parameter, the function will fetch data on a
        (ps, feed, crop_or_by_prod) level (if False) or (feed, ani, sp, br, ps, ss) (if
        True)-level of detail.
        """
        feed_par = self.feed_mgmt.par
        feed_par.clear()

        feed_to_prod: pd.DataFrame = feed_par.get_unique(["feed", crop_prod_type])
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
                + [crop_prod_type]
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
        print('Crop allocation not asdjusted')
        return None
