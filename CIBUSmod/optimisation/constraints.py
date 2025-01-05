import warnings
from typing import Literal

import pandas as pd
import numpy as np
import scipy
import re as regex
from .utils import Constraint

from ..utils.misc import inv_dict, extend_index
from .indexed_matrix import IndexedMatrix

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
                    scipy.sparse.csc_matrix((A1_1.M.shape[0], A1_2.M.shape[1])),
                    scipy.sparse.csc_matrix((A1_1.M.shape[0], A1_3.M.shape[1])),
                ]
            ),
            scipy.sparse.hstack(
                [
                    scipy.sparse.csc_matrix((A1_2.M.shape[0], A1_1.M.shape[1])),
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
    Z_ani = scipy.sparse.csc_matrix((len(row_idx), len(self.x_idx["ani"])))
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
    Z_ani = scipy.sparse.csc_matrix((A5_1.M.shape[0], len(self.x_idx["ani"])))

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

    if all([pars[k].isnull.all() for k in ["C8_crp", "C8_ani", "C8_fds"]]):
        raise ValueError(
            "At least one of 'C8_crp' or 'C8_ani' must be given to use constraint C8"
        )
    if any([v not in ["==", ">=", "<="] for v in pars["C8_rel"]]):
        raise ValueError("All 'C8_rel' must be one of '==', '>=' or '<='")

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
        # TODO: Should we not fill with zeroes otherwise?
        b8 = np.concatenate(
            [
                pars[k][i].values
                for k in ["C8_ani", "C8_crp", "C8_fds"]
                if pars[k][k] is not None
            ]
        )

        rel = pars["C8_rel"][i]

        # Append constraint
        if rel == "==":
            tol = pars["C8_tol"][i]

            # Lower bound
            self.constraints.update(
                {
                    f"C8_{str(i+n_def)}(low): A8 @ x >= b8 * (1-tol)": {
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
                    f"C8_{str(i+n_def)}(upp): A8 @ x <= b8 * (1+tol)": {
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
                    f"C8_{str(i+n_def)}: A8 @ x {rel} b8": {
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

    if all([pars[k].isnull().all() for k in ["C9_ani", "C9_crp", "C9_fds"]]):
        raise ValueError(
            "At least one of 'C9_crp' or 'C9_ani' must be given to use constraint C9"
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
                    f"C9_{str(i+n_def)}(low): A9 @ x >= b9 * (1-tol)": {
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
                    f"C9_{str(i+n_def)}(upp): A9 @ x <= b9 * (1+tol)": {
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
                    f"C9_{str(i+n_def)}: A9 @ x {rel} b9": {
                        "left": lambda x, A9, b9: A9.M @ x,
                        "right": lambda A9, b9: b9,
                        "rel": rel,
                        "pars": {"A9": A9, "b9": b9},
                    }
                }
            )

def make_C11(self):
    """
    Ensure that the feed amounts comply with the feed rations.
    """

    MERGE_EQ_AS_MIN_MAX = False

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
    # given parameter, we will not add that constraint.
    A11_eq = self.make_A11("share_in_ration")
    A11_min = self.make_A11("min_share_in_ration")
    A11_max = self.make_A11("max_share_in_ration")

    def combine_A11s(
        A11_eq: IndexedMatrix, A11_minmax: None | IndexedMatrix, mul_factor: float
    ):
        if A11_minmax is None:
            A11_min = A11_eq.copy()
            A11_min.M = A11_min.M.multiply(mul_factor)
            return A11_min

        if mul_factor <= 1:
            M = scipy.sparse.csc_matrix.maximum(
                A11_minmax.M,
                A11_eq.M.multiply(mul_factor),
            )
        else:
            M = scipy.sparse.csc_matrix.minimum(
                A11_minmax.M,
                A11_eq.M.multiply(mul_factor),
            )

        return IndexedMatrix(
            M,
            A11_minmax.rows,
            A11_minmax.cols,
        )

    if MERGE_EQ_AS_MIN_MAX and A11_eq is not None:
        tol = 0.01
        A11_min = combine_A11s(A11_eq, A11_min, 1 - tol)
        A11_max = combine_A11s(A11_eq, A11_max, 1 + tol)
        A11_eq = None

    A11_eq = with_zeroes(A11_eq)
    A11_min = with_zeroes(A11_min)
    A11_max = with_zeroes(A11_max)

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
        Z_crp = scipy.sparse.csc_matrix((len(A12_1.rows), len(self.x_idx["crp"])))

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
    Ensure that the imports of feed do not exceed the max_total_imported parameter.

    The A-matrix maps feeds to their import-shares, so that we get the volume we
    import, while the B matrix contains the ceiling of max imported volume. This is
    done per prod_sys and feed product, for both main- and by-products.
    """
    try:
        b13_cp = self.make_b13("crop_prod")
        b13_by = self.make_b13("by_prod")
    except ValueError:
        warnings.warn(
            "C13 enabled, but b13 could not be built. This is likely because no feeds had max_total_import defined. Thus, C13 was ignored."
        )
        return

    b13 = np.concatenate([np.array(b13_cp.values), np.array(b13_by.values)])

    A13_cp = self.make_A13("crop_prod", b13_cp.index)
    A13_by = self.make_A13("by_prod", b13_by.index)

    n_rows = A13_cp.shape[0] + A13_by.shape[0]
    A13 = scipy.sparse.hstack(
        [
            scipy.sparse.csc_matrix((n_rows, len(self.x_idx["ani"]))),
            scipy.sparse.csc_matrix((n_rows, len(self.x_idx["crp"]))),
            scipy.sparse.vstack([A13_cp.M, A13_by.M], format="csc"),
        ],
        format="csc",
    )

    self.constraints.update(
        {
            "C13: A13 @ x >= 0": {
                "left": lambda x, A13, b13: A13 @ x - b13,
                "right": lambda **kwargs: 0,
                "rel": "<=",
                "pars": {"A13": A13, "b13": b13},
            }
        }
    )

def make_C14(self):
    """
    Constrain feed parameters in relation to DM through the data_attr
    "feed_req_of_DM_{min,max}" set on herds.
    """
    A14_fds_min = self.make_A14("min")
    A14_fds_max = self.make_A14("max")

    n_cols_ani = len(self.x_idx["ani"])
    n_cols_crp = len(self.x_idx["crp"])

    def _A14(A14_fds: IndexedMatrix):
        M = scipy.sparse.hstack(
            [
                scipy.sparse.csc_matrix((A14_fds.shape[0], n_cols_ani)),
                scipy.sparse.csc_matrix((A14_fds.shape[0], n_cols_crp)),
                A14_fds.M,
            ],
            format="csc",
        )
        return IndexedMatrix(
            M,
            row_idx=A14_fds.rows,
            col_idx={
                "crp": self.x_idx["crp"],
                "ani": self.x_idx["ani"],
                "fds": self.x_idx["fds"],
            },
        )

    A14_min = _A14(A14_fds_min)
    A14_max = _A14(A14_fds_max)

    if A14_min.shape[0] > 0:
        self.constraints["C14 (min): A14 @ x >= 0"] = {
            "left": lambda x, A14: A14.M @ x,
            "right": lambda A14: 0,
            "rel": ">=",
            "pars": {"A14": A14_min},
        }

    if A14_max.shape[0] > 0:
        self.constraints["C14 (max): A14 @ x <= 0"] = {
            "left": lambda x, A14: A14.M @ x,
            "right": lambda A14: 0,
            "rel": "<=",
            "pars": {"A14": A14_max},
        }

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
                    scipy.sparse.csc_matrix((P1_1.M.shape[0], P1_2.M.shape[1])),
                    scipy.sparse.csc_matrix((P1_1.M.shape[0], P1_3.M.shape[1])),
                ]
            ),
            scipy.sparse.hstack(
                [
                    scipy.sparse.csc_matrix((P1_2.M.shape[0], P1_1.M.shape[1])),
                    P1_2.M,
                    scipy.sparse.csc_matrix((P1_2.M.shape[0], P1_3.M.shape[1])),
                ]
            ),
            scipy.sparse.hstack(
                [
                    scipy.sparse.csc_matrix((P1_3.M.shape[0], P1_1.M.shape[1])),
                    scipy.sparse.csc_matrix((P1_3.M.shape[0], P1_2.M.shape[1])),
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
    Z_ani = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["ani"])))
    Z_fds = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["fds"])))

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
    Z_crp = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["crp"])))
    Z_fds = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["fds"])))

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
    Z_ani = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["ani"])))
    Z_fds = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["fds"])))

    # Create Compressed Sparse Column matrix
    return IndexedMatrix(
        scipy.sparse.hstack([Z_ani, M, Z_fds], format="csc"),
        row_idx,
        {"ani": self.x_idx["ani"], "crp": col_idx, "fds": self.x_idx["fds"]},
    )

def make_A8(self, C8_ani, C8_crp, C8_fds):
    # Get row index (cr,ps,re), (sp,br,ps,ss,re), (f,ani,sp,br,ps,ss,re)
    row_idx = {}
    # Get col index (cr,ps,re), (sp,br,ps,ss,re), (f,ani,sp,br,ps,ss,re)
    col_idx = {k: v.copy() for k, v in self.x_idx.items()}

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
        Z_ani = scipy.sparse.csc_matrix((M.shape[0], len(col_idx["ani"])))
        Z_crp = scipy.sparse.csc_matrix((M.shape[0], len(col_idx["crp"])))
        Z_fds = scipy.sparse.csc_matrix((M.shape[0], len(col_idx["fds"])))
        # Extend M with zeroes
        if label == "ani":
            M = scipy.sparse.hstack([M, Z_crp, Z_fds], format="csc")
        elif label == "crp":
            M = scipy.sparse.hstack([Z_ani, M, Z_fds], format="csc")
        else:  # ac == fds
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
            M = scipy.sparse.lil_matrix((1, len(col_idx[label])))
            sel_cols = [col_idx[label].get_loc(i) for i in A.index]
            M[:, sel_cols] = 1
            MS.append(M)
        else:
            # Append zero matrix
            Z = scipy.sparse.csc_matrix((1, len(col_idx[label])))
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
    Z_ani = scipy.sparse.csc_matrix((A10_fds.M.shape[0], len(self.x_idx["ani"])))
    Z_crp = scipy.sparse.csc_matrix((A10_fds.M.shape[0], len(self.x_idx["crp"])))

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

    row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
    col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

    loss_factors = self._get_losses_factors(shape="long").reset_index()

    feed_compositions_long = (
        self._get_feed_compositions(shape="long").reset_index().dropna()
    )

    data = loss_factors.merge(
        feed_compositions_long,
        on=[
            "feed",
            "animal",
            "species",
            "breed",
            "prod_system",
            "sub_system",
        ],
    )
    data["values"] = data["losses_factor"] * data["feed_to_par_factor"]
    data = data.drop(columns=["losses_factor", "feed_to_par_factor"])

    merged_df = data.merge(
        row_idx_df,
        on=["feed_par", "animal", "species", "breed", "prod_system", "sub_system"],
    ).merge(col_idx_df, on=col_idx.names)

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

    shares_df = (
        (
            pd.concat(herd_dfs).reorder_levels(
                [n for n in self.x_idx["fds"].names if n != "region"]
            )
            / 100
        )
        .to_frame(name="share")
        .reset_index()
    )

    row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
    col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

    # Get loss factors which we want to multiply with every (non-zero) element.
    # This, because we in x_fds have feed demand, whereas the constraint regards
    # the feed consumption.
    losses = self._get_losses_factors(shape="long").reset_index()

    # Note: ca 85% of execution time for this function spent here
    merged_df = (
        row_idx_df.merge(
            col_idx_df,
            on=[cname for cname in row_idx.names if cname != "feed"],
            suffixes=("", "_c"),
        )
        .merge(
            shares_df,
            on=[cname for cname in shares_df.columns if cname != "share"],
        )
        .merge(
            losses,
            on=[cname for cname in losses.columns if cname != "losses_factor"],
        )
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

def make_b13(self, prod_type: Literal["crop_prod", "by_prod"]) -> pd.Series:
    """
    Demand vector of the max total import values for each feed
    """
    self.feed_mgmt.par.clear()

    par = "max_total_imported"

    pss = self.x_idx["crp"].unique("prod_system")
    cps = self.feed_mgmt.par.get_unique(prod_type, qry=f'parameter=="{par}"')

    data = self.feed_mgmt.par.get_from_frame(
        par,
        pd.DataFrame(
            columns=pd.Index(pss, name="prod_system"),
            index=pd.Index(cps, name=prod_type),
        ),
    ).unstack()

    assert isinstance(data, pd.Series)
    return data

def make_A13(
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

def make_A14(self, rel_type: Literal["min", "max"]):
    """
    Maps x_fds to values ensuring that feed_req_of_DM_{min/max} are met.

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
            scipy.sparse.csc_matrix((0, len(col_idx))), row_idx, col_idx
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

    # Get all feeds
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
        merged["feed_to_par_factor"] * merged["losses_factor"]
        - merged["feed_req_of_DM"]
    )
    return IndexedMatrix.from_frame(merged, row_idx, col_idx)

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
        scipy.sparse.csc_matrix((len(row_idx), len(col_idx))),
        row_idx,
        col_idx,
    )
    return M
