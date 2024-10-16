import warnings
import pandas as pd
import numpy as np

import cvxpy as cvxpy
import scipy

import re as regex

from .. import (
    Regions,
    DemandAndConversions,
    CropProduction,
    FeedMgmt,
    ParameterRetriever,
)

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned, inv_dict
from ..utils.data_attr import DataAttr
from ..main_modules.animal_herd import concat_herds

from typing import Callable, Literal, TypedDict


class Constraint(TypedDict):
    left: Callable
    right: Callable
    rel: Literal["==", ">=", "<="]
    pars: dict


def make_cvxpy_constraint(cons: Constraint, x: cvxpy.Variable) -> cvxpy.Constraint:
    """
    Convert a Constraint-dict to a cvxpy.Constraint instant
    """
    operators = {
        "==": lambda left, right: left == right,
        ">=": lambda left, right: left >= right,
        "<=": lambda left, right: left <= right,
    }
    left = cons["left"]
    right = cons["right"]
    rel = cons["rel"]
    pars = cons["pars"]

    return operators[rel](left(x, **pars), right(**pars))


class FeedDistributor:
    """Class that handles the distribution of animals, crops and feeds across regions for a given
    demand and a number of constraints by minimising deviation from an initial distribution
    of crop areas and animal heads (x0).

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

    success: bool
    constraints: dict[str, Constraint]
    objective: dict
    problem: None | cvxpy.Problem
    x: None | dict[str, pd.Series]

    def __init__(
        self,
        regions: Regions,
        demand: DemandAndConversions,
        crops: CropProduction,
        herds: pd.Series,
        feed_mgmt: FeedMgmt,
        par: ParameterRetriever,
    ):
        self.par = par
        self.data_attr = DataAttr(self)

        self.regions = regions
        self.demand = demand
        self.crops = crops
        self.herds = herds
        self.feed_mgmt = feed_mgmt

    def reset(self):
        self.x = None
        self.success = False
        self.constraints = dict()
        self.objective = dict()
        self.problem = None

    def make(
        self,
        use_cons: list | str,
        scale_power: float = 0.4,
        scale_cutoff_percentile: float = 99,
        verbose: bool = False,
        **kwargs,
    ):
        """Creates constraints and defines optimisation problem

        Parameters
        ----------
        use_cons : (list of) str
            List of numbers corresponding to the constraints to be used. For descriptions
            of each constraint see ?FeedDistributor.make_C<nr>
        scale_power : float, default 0.4
            Power used to calculate scaling factors for the optimisation.
            scale_power=0 -> minimise absolute difference in crop areas/animal numbers
            scale_power=1 -> minimise relative difference in crop areas/animal numbers
            See ?FeedDistributor.calculate_scaling_factors for details
        scale_cutoff_percentile : float (0-100), default 99
            Percentile cutoff for scaling factors. Should be <100 to avoid effectively
            removing crops/animals where x0 is close to zero from the solution.
            See ?FeedDistributor.calculate_scaling_factors for details
        verbose : bool, default False
            Print progress messages
        **kwargs
            Keyword agruments to be passed on to the FeedDistributor.make_C<nr> methods.
            These are on the form 'C<nr>_<arg>'.

        Returns
        -------
        None
        """

        vprint = verbose_init(verbose, id_str="FeedDistributor.make")

        # Reset problem definitions and solution
        self.reset()

        if not isinstance(use_cons, list):
            use_cons = [use_cons]
        use_cons = [str(e) for e in use_cons]

        # Make sure that C7 is handled last
        if "7" in use_cons:
            use_cons.append(use_cons.pop(use_cons.index("7")))

        vprint("Getting x0 and making indexes ...")
        self.make_x0()

        vprint("Creating demand vector ...")
        self.make_demand()

        vprint("Calculating scaling factors ...")
        # Calculate scaling factors
        self.calculate_scaling_factors(
            scale_power=scale_power, cutoff_percentile=scale_cutoff_percentile
        )

        # Make objective function(s)
        vprint("Making objective O1 ...")
        self.P1 = self.make_P1()

        # Make constraints
        for nr in use_cons:
            fun = getattr(self, f"make_C{nr}")
            vprint(f"Making constraint C{nr}...")
            fun(**{k: v for k, v in kwargs.items() if f"C{nr}" in k})

        # If C7 not included no variables are dropped
        if "7" not in use_cons:
            self.x_idx_short = {
                "ani": self.x_idx["ani"].copy(),
                "crp": self.x_idx["crp"].copy(),
            }

        vprint(type="end")

    def solve(
        self,
        # Default solver settings
        #
        # GUROBI
        # ------
        # This solver seem to work really well. Licence needed, but free academic licences are
        # available and easy to get.
        #
        # OSQP
        # ----
        # Settings for OSQP available at https://osqp.org/docs/interfaces/solver_settings.html
        # Using a too high tolerance (eps_abs, eps_rel) leads to large relative deviations
        # from x0 for crops with small areas, but a low tolerance increases time to find solution.
        solver_settings: dict | list = [
            {"solver": "GUROBI", "verbose": False},
            # {
            #     'solver': 'OSQP',
            #     'max_iter': 200000,
            #     'eps_abs': 5e-6,
            #     'eps_rel': 5e-6,
            #     'verbose': False
            # }
        ],
        apply_solution: bool = True,
        verbose: bool = False,
    ) -> None:
        """Solve optimisation problem

        Parameters
        ----------
        solver_settings : dict, list of dicts
            Dict of keyword arguments to be passed on to cvxpy.Problem.solve()
            If a list of dicts is supplied the method will move to the next dict
            of solver settings if the previous ones failed.
            If not supplied default values are used.
        apply_solution : bool, default True
            Update CropProduction and AnimalHerd objects according to the found
            solution via the the method FeedDistributor.apply_solution()
        verbose : bool, default False
            Print progress messages

        Returns
        -------
        None
        """

        vprint = verbose_init(verbose, id_str="FeedDistributor.solve")

        # If a list of alternative solver/settings is not supplied
        # make a one element list
        if not isinstance(solver_settings, list):
            solver_settings = [solver_settings]

        if self.problem is None:
            vprint("Defining problem ...")
            self.problem = self.get_cvx_problem()

        # Try to find a solution with (potentially) different solver/settings
        # If an optimal solution is found break and do not try next solver/settings
        for kwargs in solver_settings:
            solver = kwargs["solver"]
            try:
                vprint(f"Finding solution with '{solver}' ...")
                self.problem.solve(**kwargs)
            except Exception as e:
                self.x = None
                self.success = False
                vprint(f"Failed with {type(e).__name__}: {e}", type="msg")
                continue

            if self.problem.status and "optimal" in self.problem.status:
                self.success = True
                vprint(
                    f"Optimal solution found! Status: '{self.problem.status}', Iterations: {self.problem.solver_stats.num_iters}, Solver: '{self.problem.solver_stats.solver_name}'",
                    type="msg",
                )
                break
            else:
                self.success = False
                status = self.problem.status if self.problem.status else "None"
                try:
                    num_iters = self.problem.solver_stats.num_iters
                except Exception:
                    num_iters = "n/a"
                vprint(
                    f"No solution found! Status: '{status}', Iterations: {num_iters}",
                    type="msg",
                )

        # Check solution and print results
        if self.success:
            # DO SOME MORE FEASIBILITY CHECKS ON THE SOLUTION HERE??!!

            vprint("Retrieving solution ...")

            # Get and store optimal value for variable
            x = self.problem.variables()[0].value
            # Put xs on short index (!= index if C7 is used) and reindex
            self.x = {
                "ani": pd.Series(
                    x[: len(self.x_idx_short["ani"])], index=self.x_idx_short["ani"]
                ).reindex(self.x_idx["ani"], fill_value=0),
                "crp": pd.Series(
                    x[len(self.x_idx_short["ani"]) :], index=self.x_idx_short["crp"]
                ).reindex(self.x_idx["crp"], fill_value=0),
            }

            self.data_attr.add(
                self.x["crp"],
                name="x_crops",
                unit="ha or m2",
                orig="FeedDistributor",
                desc="Crop areas in solution in ha (or m2 for greenhouse crops)",
            )
            self.data_attr.add(
                self.x["ani"],
                name="x_animals",
                unit="heads",
                orig="FeedDistributor",
                desc='Number of "defining animal" heads in solution',
            )

            if apply_solution:
                vprint("Applying solution ...")
                self.apply_solution()

        else:
            self.x = None
            raise RuntimeError("No solution found!")

        vprint(type="end")

        return None

    def matrices(self):
        mats = {"OBJ.P1": self.P1}
        mats.update(
            {
                f'{cn[:cn.index(":")]}.{mn}': m
                for cn, c in self.constraints.items()
                for mn, m in c["pars"].items()
                if isinstance(m, IndexedMatrix)
            }
        )
        return mats

    def apply_solution(self, x=None):
        """Update CropProduction and AnimalHerds according to found solution"""

        if x is None:
            if self.x is None:
                raise Exception("Cannot apply_solution as x is not defined")
            x = self.x

        # Update CropProduction
        self.crops.scale(x["crp"])

        # Update AnimalHerds
        with warnings.catch_warnings():
            # Ignore pandas peformance warning. Performance not a problem
            # here but the issue could probably be solved by sorting x['ani']
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

            for h in self.herds:
                h.scale(
                    x["ani"].loc[(h.species, h.breed, h.prod_system, h.sub_system)],
                    x_is=h.x_is,
                )

        # Allocate crop production to uses
        self.allocate_crop_production_per_use()
        if "A5" in self.matrices():
            self.adjust_crop_allocation()

    def make_x0(self):
        # Define index for x
        self.x_idx = {
            "ani": pd.MultiIndex.from_tuples(
                [
                    (sp, br, ps, ss, re)
                    for (sp, br, ps, ss) in self.herds.index
                    for re in self.herds[(sp, br, ps, ss)].index
                ],
                names=["species", "breed", "prod_system", "sub_system", "region"],
            ),
            "crp": self.crops.index.copy(),
            "fds": pd.MultiIndex.from_tuples(
                [
                    (f, sp, br, ps, ss, re)
                    for (sp, br, ps, ss) in self.herds.index
                    for re in self.herds[(sp, br, ps, ss)].index
                    for f in self.herds[(sp, br, ps, ss)].par.get_unique("feed")
                ],
                names=[
                    "feed",
                    "species",
                    "breed",
                    "prod_system",
                    "sub_system",
                    "region",
                ],
            ),
        }

        # Get x0
        self.x0 = {
            "ani": self.regions.data_attr.get("x0_animals").copy(),
            "crp": self.regions.data_attr.get("x0_crops").copy(),
            "fds": pd.Series(data=0, index=self.x_idx["fds"]),
        }

        # Sort x0['ani'] to match x['ani']
        self.x0["ani"] = self.x0["ani"].loc[
            self.x_idx["ani"].droplevel("sub_system").unique()
        ]

        # Store x0 indexes
        self.x0_idx = {
            "ani": self.x0["ani"].index,
            "crp": self.x0["crp"].index,
            "fds": self.x_idx["fds"].copy(),
        }

    def make_demand(self):
        """
        Calculates and sets the demand-matrix (D) and its index (D_idx).
        """
        self.D = {
            "ani": self.demand.data_attr.get("animal_prod_demand").sum(axis=1),
            "crp": self.demand.data_attr.get("crop_prod_demand").sum(axis=1),
        }

        # Add rows for any domestically produced crop products used for feed or seed not already in crop product demand vector (D['crp'])
        self.feed_mgmt.par.clear()
        for cp in set(self.feed_mgmt.par.get_unique("crop_prod")) | set(
            self.crops.par.get_unique("crop_prod", qry='parameter == "seed"')
        ):
            for ps in self.D["crp"].index.get_level_values("prod_system").unique():
                idx = (ps, cp)
                if (
                    self.feed_mgmt.par.get(
                        "share_imported", crop_prod=cp, prod_system=ps
                    )
                    != 100
                ).any() & (idx not in self.D["crp"].index):
                    self.D["crp"][idx] = 0

        # Store indexes
        self.D_idx = {"ani": self.D["ani"].index, "crp": self.D["crp"].index}

    def calculate_scaling_factors(
        self, scale_power: float = 0.0, cutoff_percentile: float = 99.0
    ):
        """Calculates scaling factor to apply to x and x0 in objective O1 as f = rn * sf where rn is a factor
        normalising all features (i.e. distinct land uses and animal species) to the same range and fs is a
        scaling factor calculated as fs = ( mean(x0 * rn) / (x0 * rn) ) ^ scale_power. A cutoff that limits the
        maximum scaling factor to a certain percentile is implemented to avoid that crops/animals with x0 close
        to or equal to zero are effectively removed from the solution space.
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

        x0 = pd.concat((self.x0["ani"], self.x0["crp"])) * rn.values
        sf = x0.mean() / x0
        cutoff_value = np.percentile(sf.loc[sf != np.inf], cutoff_percentile)
        sf.loc[sf > cutoff_value] = cutoff_value
        sf = sf**scale_power

        f = rn.values * sf.values
        assert np.isfinite(f).all()

        scale_f["ani"].iloc[:] = f[: len(scale_f["ani"])]
        scale_f["crp"].iloc[:] = f[len(scale_f["ani"]) :]
        self.scale_f = scale_f

    def get_cvx_problem(self):
        # Apply scaling factors to x0
        x0s = cvxpy.Constant(
            np.concatenate(
                [
                    (self.x0[k] * self.scale_f[k]).reindex(self.x0_idx[k])
                    for k in ["ani", "crp"]
                ]
            )
        )

        # Get scaling factors for x
        sf = cvxpy.Constant(
            np.concatenate(
                [
                    (
                        self.scale_f["ani"]
                        .reindex(
                            self.x_idx["ani"].reorder_levels(
                                [
                                    "species",
                                    "breed",
                                    "prod_system",
                                    "region",
                                    "sub_system",
                                ]
                            )
                        )
                        .reindex(
                            self.x_idx_short["ani"].reorder_levels(
                                [
                                    "species",
                                    "breed",
                                    "prod_system",
                                    "region",
                                    "sub_system",
                                ]
                            )
                        )
                    ),
                    self.scale_f["crp"].reindex(self.x_idx_short["crp"]),
                ]
            )
        )

        n = len(self.x_idx_short["ani"]) + len(self.x_idx_short["crp"])
        x = cvxpy.Variable(n, nonneg=True)

        objective = cvxpy.Minimize(
            cvxpy.sum_squares((self.P1.M @ cvxpy.multiply(sf, x)) - x0s)
        )

        # Append constraints
        constraints = [
            make_cvxpy_constraint(cons, x) for cons in self.constraints.values()
        ]

        # Define problem
        return cvxpy.Problem(objective=objective, constraints=constraints)

    def make_C1(self):
        """Creates C1: A1 @ x == b1

        Main constraint to ensure that production exactly meets demand. Crop and animal products without any
        demand remain unconstrained. Demand is calculated in the 'DemandAndConversions' module.
        """

        # Animal product demand
        A1_1 = self.make_A1_1()
        # Feed demand
        A1_2 = self.make_A1_2()
        # Crop product demand
        A1_3 = self.make_A1_3()

        # Stack matrices
        A1 = scipy.sparse.vstack(
            [
                scipy.sparse.hstack(
                    [
                        A1_1.M,
                        scipy.sparse.csc_matrix((A1_1.M.shape[0], A1_3.M.shape[1])),
                    ]
                ),
                scipy.sparse.hstack([A1_2.M, A1_3.M]),
            ],
            format="csc",
        )

        A1 = IndexedMatrix(
            matrix=A1,
            row_idx={"ani": A1_1.rows, "crp": A1_2.rows},
            col_idx={"ani": A1_1.cols, "crp": A1_3.cols},
        )
        b1 = np.concatenate((self.D["ani"].values, self.D["crp"].values))

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
        """Creates C2: A2 @ x >= 0

        Constraint the share of feed demand for different crop products that must be met regionally.
        The minimum share is set via the parameter 'share_regional' in the 'FeedMgmt' module and can differ
        for different animals.
        """

        # Regional feed demand for crop products
        A2_1 = self.make_A2_1()
        # Production of crop products
        A2_2 = self.make_A2_2()

        # Stack matrices
        A2 = scipy.sparse.hstack([A2_1.M, A2_2.M], format="csc")

        A2 = IndexedMatrix(
            matrix=A2, row_idx=A2_1.rows, col_idx={"ani": A2_1.cols, "crp": A2_2.cols}
        )

        # Append constraint
        self.constraints.update(
            {
                "C2: A2 @ x >= 0": {
                    "left": lambda x, A2: A2.M @ x,
                    "right": lambda A2: 0,
                    "rel": ">=",
                    "pars": {"A2": A2},
                }
            }
        )

    def make_C3(self):
        """Creates C3: A3 @ x <= b3

        Constrain the maximum area per 'land_use' in each region. The maximum area is set relative to areas
        in x0 via the parameter 'max_land_use_factor' in the 'Regions' module. A 'max_land_use_factor' of 1
        implies that areas can't exceed current areas in each region.
        If the Regions setting 'max_land_use_from_scenario_x0' is set to True (default is False) maximum land
        use changes if there are changes to 'x0_crops' in a scenario, otherwise default values of 'x0_crops'
        are used irrespective of any changes in scenarios.
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

        Constrain the maximum share of defining animal heads per species, breed and prod_system belonging
        to a given sub_system on national level

        The maximum share is set via the parameter 'max_share_sub_system' in the respective 'AnimalHerd'
        modules and can differ by breed.
        """

        A4 = self.make_A4()

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

        Constrain the maximum share of a crop product demand for feed that can be supplied by a
        particular crop group. This constraint is used to e.g. constrain the share of 'grazing' that can be
        supplied by 'semi-natural grasslands', but can also be used to constrain e.g. share of wheat for
        feed from winter/spring variaties.

        The maximum share is set via the parameter 'max_crop_in_crop_prod' in the 'FeedMgmt'
        module and can differ for different animals.
        """

        # Maximum supply of crop product(s) from crop(s)
        A5_1 = self.make_A5_1()
        # Production of crop products
        A5_2 = self.make_A5_2()

        # Stack matrices
        A5 = scipy.sparse.hstack([A5_1.M, A5_2.M], format="csc")

        A5 = IndexedMatrix(
            matrix=A5, row_idx=A5_1.rows, col_idx={"ani": A5_1.cols, "crp": A5_2.cols}
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

        Constrain the maximum share of cropland devoted to a given crop group in a given region in a
        given production system. The maximum share is set on 'crop_group' level via the parameter
        'max_in_rot' in the 'CropProduction' module.

        Note: This constraint only applies to crops with 'cropland' as 'land_use' in the relation tables.
        """

        # Note to future:
        # - Would it be useful with a constraint for minimum share?
        # - Deal with crops assumed not to be in rotation by putting 0 in the matrix

        A6 = self.make_A6()

        # Append constraint
        self.constraints.update(
            {
                "C6: A6 @ x <= 0": {
                    "left": lambda x, A6: A6.M @ x,
                    "right": lambda A6: 0,
                    "rel": "<=",
                    "pars": {"A6": A6},
                }
            }
        )

    def make_C7(self) -> None:
        """Creates C7: Drops variables

        Constrain crops to certain regions based on minimum growing degree days (GDD5). The minimum
        GDD5 for different crops is set with the parameter 'min_GDD5' in the 'CropProduction' module.
        The number of GDD5 in each region is defined by the parameter 'GDD' in the 'Regions' module.

        This constraint also indirectly constrains animals with regional demand for crops that can't
        be grown in a region."""

        # This constraint is not implemented as a constraint in the solver but instead drops
        # variables representing crops or animals that can't be present in a region.
        # IMPORTANT: This must be run after all other constraints have been defined!

        # Index of crops
        cr_idx = self.x_idx["crp"]

        # Get allowed crop-region combinations (i.e. region GDD5 >= min_GDD5 for crop)
        self.crops.par.clear()
        self.regions.par.clear()
        sel_cr = cr_idx[
            self.regions.par.get("GDD5", **cr_idx.to_frame().to_dict("list"))
            >= self.crops.par.get("min_GDD5", **cr_idx.to_frame().to_dict("list"))
        ]

        # Index of animal herds
        an_idx = self.x_idx["ani"]

        # Get crop products that CAN be produced in region
        sel_cp = (
            self.crops.data_attr.get("production")
            .loc[sel_cr]
            .stack()
            .groupby(["crop_prod", "prod_system", "region"])
            .sum()
            .replace({0: np.nan})
            .dropna()
            .index
        )

        # Index of crop products
        cp_idx = (
            self.crops.data_attr.get("production")
            .stack()
            .droplevel("crop")
            .reorder_levels(["crop_prod", "prod_system", "region"])
            .index.unique()
        )
        # Get crop products that CAN'T be produced in region
        nsel_cp = cp_idx.difference(sel_cp)

        # List to populate with herds that can be in region
        # (i.e. with no regional demand for feeds that can't be
        # produced in the region)
        sel_an = []

        for h in self.herds:
            if h.data_attr.get("feed.regional_crop_product_demand").shape[1] > 0:
                # Get crop products with a regional feed demand
                nsel_cp2 = (
                    h.data_attr.get("feed.regional_crop_product_demand")
                    .stack(["prod_system", "crop_prod"])
                    .reorder_levels(["crop_prod", "prod_system", "region"])
                    .index
                )

                # Get regions where herd has a regional demand
                # for a feed that can't be grown.
                nsel_re = (
                    nsel_cp.intersection(nsel_cp2).get_level_values("region").unique()
                )

                # Get regions where herd CAN be present
                sel_re = h.index.difference(nsel_re)
            else:
                sel_re = h.index

            # Add herds allowed to animal selection
            sp = h.species
            br = h.breed
            ps = h.prod_system
            ss = h.sub_system
            sel_an += [(sp, br, ps, ss, re) for re in sel_re]
        # To pandas MultiIndex
        sel_an = pd.MultiIndex.from_tuples(
            sel_an, names=["species", "breed", "prod_system", "sub_system", "region"]
        )

        # Get variable positions not to drop
        isel_an = [an_idx.get_loc(s) for s in sel_an]
        isel_cr = [cr_idx.get_loc(s) + len(an_idx) for s in sel_cr]
        isel = isel_an + isel_cr

        # Store short index (i.e. index of variables after dropping)
        self.x_idx_short = {"ani": sel_an, "crp": sel_cr}

        # Drop variables from objective and constraint matrices
        for mat in self.matrices().values():
            if mat.M.shape[1] > len(isel):
                mat.M = mat.M[:, isel]
                mat.cols["ani"] = sel_an.copy()
                mat.cols["crp"] = sel_cr.copy()

        return None

    def make_C8(
        self,
        C8_crp: pd.Series | None = None,
        C8_ani: pd.Series | None = None,
        C8_rel: str = "==",
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
        C8_rel : (list of) string
            Type of constraint. Equality ('=='), minimum ('>=') or maximum ('<=')
        C8_tol : (list of) float

        Returns
        -------
        None
        """

        pars = {"C8_crp": C8_crp, "C8_ani": C8_ani, "C8_rel": C8_rel, "C8_tol": C8_tol}
        pars_len = {p: len(pars[p]) if isinstance(pars[p], list) else 0 for p in pars}
        pars_len_max = max(max(pars_len.values()), 1)

        if any([x > 1 and x < pars_len_max for x in pars_len.values()]):
            raise ValueError("Supplied lists must have the same length")

        # Align lists
        for p in pars:
            if pars_len[p] < pars_len_max:
                if pars_len[p] == 1:
                    pars[p] = pars[p] * pars_len_max
                else:
                    pars[p] = [pars[p]] * pars_len_max

        if all([v is None for v in pars["C8_crp"]]) and all(
            [v is None for v in pars["C8_ani"]]
        ):
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
            A8 = self.make_A8(pars["C8_crp"][i], pars["C8_ani"][i])

            # Make right hand vector (b8)
            if (pars["C8_crp"][i] is not None) & (pars["C8_ani"][i] is not None):
                b8 = np.concatenate(
                    (pars["C8_ani"][i].values, pars["C8_crp"][i].values)
                )
            elif pars["C8_crp"][i] is not None:
                b8 = pars["C8_crp"][i].values
            else:
                b8 = pars["C8_ani"][i].values

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
        C9_rel: str = "==",
        C9_tol: float = 1e-4,
    ):
        """Creates C9: A9 @ x <rel> b9

        Flexible constraint that constrains the sum of crop areas and/or animal numbers
        corresponding to the index of passed Series in relation to the sum of given Series.
        Constraints can be either equality or max/min. Equality constraints (C9_rel = '==')
        are implemented as min and max constraints with a relative tolerance of +/- C9_tol.

        Multiple constraints can be created by supplying lists as parameters.

        Parameters
        ----------
        C9_crp : (list of) pandas.Series
            Crop areas to constrain sum of (index must match self.x_idx['crp'])
        C9_ani : (list of) pandas.Series
            Animal numbers to constrain sum of (index must match self.x_idx['ani'])
        C9_rel : (list of) string
            Type of constraint. Equality ('=='), minimum ('>=') or maximum ('<=')
        C9_tol : (list of) float

        Returns
        -------
        None
        """

        pars = {"C9_crp": C9_crp, "C9_ani": C9_ani, "C9_rel": C9_rel, "C9_tol": C9_tol}
        pars_len = {p: len(pars[p]) if isinstance(pars[p], list) else 0 for p in pars}
        pars_len_max = max(max(pars_len.values()), 1)

        if any([x > 1 and x < pars_len_max for x in pars_len.values()]):
            raise ValueError("Supplied lists must have the same length")

        # Align lists
        for p in pars:
            if pars_len[p] < pars_len_max:
                if pars_len[p] == 1:
                    pars[p] = pars[p] * pars_len_max
                else:
                    pars[p] = [pars[p]] * pars_len_max

        if all([v is None for v in pars["C9_crp"]]) and all(
            [v is None for v in pars["C9_ani"]]
        ):
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
            if not ((pars["C9_crp"][i] is None) & (pars["C9_ani"][i] is None)):
                # Make matrix (A9)
                A9 = self.make_A9(pars["C9_crp"][i], pars["C9_ani"][i])

                # Make right hand vector (b9)
                b9 = sum(
                    [
                        pars["C9_ani"][i].sum() if pars["C9_ani"][i] is not None else 0,
                        pars["C9_crp"][i].sum() if pars["C9_crp"][i] is not None else 0,
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
            else:
                raise ValueError("Both 'C9_crp' and 'C9_ani' were None")

        return None

    def make_P1(self):
        # x['ani'] --> x0['ani']
        P1_1 = self.make_P1_1()
        # x['crp'] --> x0['crp']
        P1_2 = self.make_P1_2()

        P1 = scipy.sparse.vstack(
            [
                scipy.sparse.hstack(
                    [
                        P1_1.M,
                        scipy.sparse.csc_matrix((P1_1.M.shape[0], P1_2.M.shape[1])),
                    ]
                ),
                scipy.sparse.hstack(
                    [
                        scipy.sparse.csc_matrix((P1_2.M.shape[0], P1_1.M.shape[1])),
                        P1_2.M,
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
            for ap in (
                herd.data_attr.get("production")
                .columns.get_level_values("animal_prod")
                .unique()
            ):
                # Go through output production systems
                for ops in (
                    herd.data_attr.get("production")
                    .columns.get_level_values("prod_system")
                    .unique()
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
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A1_2(self):
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx["crp"]
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

            # Get crop products and production systems with domestic demand for feed
            opss_cps = (
                herd.data_attr.get("feed.crop_product_demand")
                .xs("domestic", level="origin", axis=1)
                # drop feeds with no (< 5e-6 kg) domestic demand
                .round(5)
                .replace({0: np.nan})
                .dropna(axis=1, how="all")
                .droplevel("animal", axis=1)
                .columns.unique()
                .values
            )

            # Go through crop products and production systems used as feed
            for ops, cp in opss_cps:
                # Get feed demand for crop product (cp) from output production system (ops) per head
                # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
                # and region (re)
                res = (
                    -herd.data_attr.get("feed.crop_product_demand")
                    .loc[:, ("domestic", ops, slice(None), cp)]
                    .sum(axis=1)
                )

                # Store values and row/col nr
                val.extend(res.values)
                row_nr.extend([row_idx.get_loc((ops, cp))] * len(res))
                col_nr.extend(
                    [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A1_3(self):
        # Get row index from crop product demand vector (ps,cp)
        row_idx = self.D_idx["crp"]
        # Get col index from crop production (cr,ps,re)
        col_idx = self.x_idx["crp"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        cps = self.crops.data_attr.get("production").columns

        for cr, ps in col_idx.droplevel("region").unique():
            for cp in cps:
                if (ps, cp) in row_idx:
                    # Get production of crop product (cp) minus seed demand
                    # from production system (ps) per area of crop (cr)
                    # in production system (ps) and region (re)
                    res = self.crops.data_attr.get("production").loc[
                        (cr, ps, slice(None)), (cp)
                    ] - (
                        self.crops.data_attr.get("seed_demand").loc[
                            (cr, ps, slice(None)), (cp)
                        ]
                        if cp in self.crops.data_attr.get("seed_demand").columns
                        else 0
                    )

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend([row_idx.get_loc((ps, cp))] * len(res))
                    col_nr.extend(
                        [
                            col_idx.get_loc((cr, ps, re))
                            for re in res.index.get_level_values("region")
                        ]
                    )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A2_1(self):
        # Get row index from feeds with a regional demand (cp,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (cp, ps, re)
                for cp in list(
                    set(
                        [
                            cp
                            for herd in self.herds
                            if herd.data_attr.get(
                                "feed.regional_crop_product_demand"
                            ).shape[1]
                            > 0
                            for cp in (
                                herd.data_attr.get("feed.regional_crop_product_demand")
                                .replace({0: np.nan})
                                .dropna(
                                    axis=1, how="all"
                                )  # drop feeds with no regional demand
                                .columns.get_level_values("crop_prod")
                            )
                        ]
                    )
                )
                for ps in self.x_idx["ani"].get_level_values("prod_system").unique()
                for re in self.x_idx["ani"].get_level_values("region").unique()
            ],
            names=["crop_prod", "prod_system", "region"],
        )
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        # Go through animal herds
        for herd in self.herds:
            # Skip herds where there are no regional demands for feeds
            if herd.data_attr.get("feed.regional_crop_product_demand").shape[1] <= 0:
                continue

            sp = herd.species
            br = herd.breed
            ps = herd.prod_system
            ss = herd.sub_system

            # Get crop products and production systems with regional demand for feed
            opss_cps = (
                herd.data_attr.get("feed.regional_crop_product_demand")
                .replace({0: np.nan})
                .dropna(axis=1, how="all")  # drop feeds with no regional demand
                .droplevel("animal", axis=1)
                .columns.unique()
                .values
            )

            # Go through crop products and production systems with regional demand for feed
            for ops, cp in opss_cps:
                # Get regional feed demand for crop product (cp) from output
                # production system (ops) per head of defining animal of species
                # (sp) and breed (br) in production system (ps), sub system (ss)
                # and region (re)
                res = -(
                    herd.data_attr.get("feed.regional_crop_product_demand")
                    .loc[:, (ops, slice(None), cp)]
                    .sum(axis=1)
                )

                # Store values and row/col nr
                val.extend(res.values)
                row_nr.extend([row_idx.get_loc((cp, ops, re)) for re in res.index])
                col_nr.extend(
                    [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A2_2(self):
        # Get row index from feeds with a regional demand (cp,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (cp, ps, re)
                for cp in list(
                    set(
                        [
                            cp
                            for herd in self.herds
                            if herd.data_attr.get(
                                "feed.regional_crop_product_demand"
                            ).shape[1]
                            > 0
                            for cp in (
                                herd.data_attr.get("feed.regional_crop_product_demand")
                                .replace({0: np.nan})
                                .dropna(
                                    axis=1, how="all"
                                )  # drop feeds with no regional demand
                                .columns.get_level_values("crop_prod")
                            )
                        ]
                    )
                )
                for ps in self.x_idx["ani"].get_level_values("prod_system").unique()
                for re in self.x_idx["ani"].get_level_values("region").unique()
            ],
            names=["crop_prod", "prod_system", "region"],
        )
        # Get col index from crops (cr,ps,re)
        col_idx = self.x_idx["crp"]

        row_idx_lookup = row_idx.droplevel("region").sort_values()

        # To store data and corresponding row/col numbers for constructing matrix
        val = []
        row_nr = []
        col_nr = []

        for cr, ps in (
            self.crops.data_attr.get("production")
            .index.droplevel("region")[
                self.crops.data_attr.get("production")[
                    row_idx.get_level_values("crop_prod").unique()
                ].sum(axis=1)
                > 0
            ]
            .unique()
        ):
            for cp in row_idx.get_level_values("crop_prod").unique():
                if (cp, ps) in row_idx_lookup:
                    # Get production of crop product (cp) from production system (ps) per area of crop (cr)
                    # in production system (ps) and region (re)
                    res = (
                        self.crops.data_attr.get("production")
                        .loc[(cr, ps, slice(None)), (cp)]
                        .fillna(0)
                    )

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend(
                        [
                            row_idx.get_loc((cp, ps, re))
                            for re in res.index.get_level_values("region")
                        ]
                    )
                    col_nr.extend(
                        [
                            col_idx.get_loc((cr, ps, re))
                            for re in res.index.get_level_values("region")
                        ]
                    )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A3(self):
        # Get land uses to constrain
        land_uses = self.regions.data_attr.get("max_land_use").columns
        # Get row index from land uses and regions (lu,re)
        row_idx = pd.MultiIndex.from_product(
            [land_uses, self.x_idx["crp"].get_level_values("region").unique()],
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
        Z = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["ani"])))  # Zero matrix

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([Z, M], format="csc"),
            row_idx,
            {"ani": self.x_idx["ani"], "crp": col_idx},
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
        Z = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["crp"])))  # Zero matrix

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([M, Z], format="csc"),
            row_idx,
            {"ani": self.x_idx["ani"], "crp": col_idx},
        )

        return M

    def make_A5_1(self):
        # Get crop product and crop combindations where there is a constraint for maximum inclusion
        cps_cgs = self.feed_mgmt.par.get_unique(
            ["crop_prod", "crop_group"], qry='parameter == "max_crop_in_crop_prod"'
        )

        # Get row index (cp,cg,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (cp, cg, ps, re)
                for cp, cg in cps_cgs.values
                for ps in self.x_idx["ani"].get_level_values("prod_system").unique()
                for re in self.x_idx["ani"].get_level_values("region").unique()
            ],
            names=["crop_prod", "crop_group", "prod_system", "region"],
        )
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

            # Check if herd has 'feed.max_supply_from_crop'
            if herd.data_attr.get("feed.max_supply_from_crop_group") is not None:
                # Go through production systems, crop products and crop combinations with a max supply constraint
                for ops, cp, cg in herd.data_attr.get(
                    "feed.max_supply_from_crop_group"
                ).columns:
                    # Get maximum supply of crop product (cp) from output production system (ops) in region (re)
                    # from crop_group (cg) per head of defining animal of species (sp) and breed (br) in production
                    # system (ps), sub system (ss) and region (re)
                    res = -herd.data_attr.get("feed.max_supply_from_crop_group").loc[
                        :, (ops, cp, cg)
                    ]

                    # Store values and row/col nr
                    val.extend(res.values)
                    row_nr.extend(
                        [row_idx.get_loc((cp, cg, ops, re)) for re in res.index]
                    )
                    col_nr.extend(
                        [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                    )

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A5_2(self):
        # Get crop product and crop_group combindations where there is a constraint for maximum inclusion
        cps_cgs = self.feed_mgmt.par.get_unique(
            ["crop_prod", "crop_group"], qry='parameter == "max_crop_in_crop_prod"'
        )

        # Get map crop_group --> crop(s)
        map_cg_cr = inv_dict(self.par.get_rel("crop", "crop_group"))

        # Get row index (cp,cg,ps,re)
        row_idx = pd.MultiIndex.from_tuples(
            [
                (cp, cg, ps, re)
                for cp, cg in cps_cgs.values
                for ps in self.x_idx["ani"].get_level_values("prod_system").unique()
                for re in self.x_idx["ani"].get_level_values("region").unique()
            ],
            names=["crop_prod", "crop_group", "prod_system", "region"],
        )
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
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_A6(self):
        self.crops.par.clear()

        # Get crop groups with max/min inclusion in rotation constraint
        cgs = self.crops.par.get_unique("crop_group", qry='parameter == "max_in_rot"')
        pss = self.x_idx["crp"].get_level_values("prod_system").unique()
        res = self.x_idx["crp"].get_level_values("region").unique()

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
            f = float(
                self.crops.par.get("max_in_rot", crop_group=cg, prod_system=ps) / 100
            )

            vls = [
                0
                if (ps != ps_) | (lu_rel[cr] != "cropland")
                else ((1 - f) if cg_rel[cr] == cg else -f)
                for cr, ps_, _ in col_idx
            ]
            cns = list(range(len(col_idx)))
            rns = [row_idx.get_loc((cg, ps, re)) for _, _, re in col_idx]

            val.extend(vls)
            col_nr.extend(cns)
            row_nr.extend(rns)

        M = scipy.sparse.coo_array(
            (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
        ).tocsc()
        Z = scipy.sparse.csc_matrix((M.shape[0], len(self.x_idx["ani"])))  # Zero matrix

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.hstack([Z, M], format="csc"),
            row_idx,
            {"ani": self.x_idx["ani"], "crp": col_idx},
        )

        return M

    def make_A8(self, C8_crp, C8_ani):
        # Get row index (cr,ps,re), (sp,br,ps,ss,re)
        row_idx = {}
        # Get col index (cr,ps,re), (sp,br,ps,ss,re)
        col_idx = {k: v.copy() for k, v in self.x_idx.items()}

        MS = []
        for ac in ["ani", "crp"]:
            if eval("C8_" + ac) is not None:
                row_idx[ac] = eval("C8_" + ac).index
                # Create identity matrix from col_idx
                n = len(col_idx[ac])
                M = scipy.sparse.identity(n, format="csc")
                # Drop rows to match row index
                sel_rows = [col_idx[ac].get_loc(i) for i in row_idx[ac]]
                M = M[sel_rows, :]
                # Create zero matrix and hstack
                if ac == "ani":
                    Z = scipy.sparse.csc_matrix((M.shape[0], len(col_idx["crp"])))
                    MS.append(scipy.sparse.hstack([M, Z], format="csc"))
                else:
                    Z = scipy.sparse.csc_matrix((M.shape[0], len(col_idx["ani"])))
                    MS.append(scipy.sparse.hstack([Z, M], format="csc"))

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(scipy.sparse.vstack(MS), row_idx, col_idx)

        return M

    def make_A9(self, C9_crp, C9_ani):
        # No row index, only one row
        row_idx = None
        # Get col index (cr,ps,re), (sp,br,ps,ss,re)
        col_idx = self.x_idx.copy()

        MS = []
        for ac in ["ani", "crp"]:
            if eval("C9_" + ac) is not None:
                idx = eval("C9_" + ac).index
                # Create a 1-by-len(col_idx) matrix
                # and set cols corresponding to index to 1
                M = scipy.sparse.lil_matrix((1, len(col_idx[ac])))
                sel_cols = [col_idx[ac].get_loc(i) for i in idx]
                M[:, sel_cols] = 1
                MS.append(M)
            else:
                # Append zero matrix
                Z = scipy.sparse.csc_matrix((1, len(col_idx[ac])))
                MS.append(Z)

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(scipy.sparse.hstack(MS, format="csc"), row_idx, col_idx)

        return M

    def make_P1_1(self):
        # Get row index from x0['ani'] (sp,br,ps,re)
        row_idx = self.x0_idx["ani"]
        # Get col index from animal herds (sp,br,ps,ss,re)
        col_idx = self.x_idx["ani"]

        # Data and corresponding row/col numbers for constructing matrix
        val = [1] * len(col_idx)
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((sp, br, ps, re)) for sp, br, ps, _, re in col_idx]

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def make_P1_2(self):
        """Creates P1,2 matrix. That is an identity matrix of size len(x0['crp']) as x0['crp'].index == x['crp'].index"""
        # Get row index from x0['crp'] (cr,ps,re)
        row_idx = self.x0_idx["crp"]
        # Get row index from x['crp'] (cr,ps,re)
        col_idx = self.x_idx["crp"]

        # To store data and corresponding row/col numbers for constructing matrix
        val = [1] * len(col_idx)
        col_nr = list(range(len(col_idx)))
        row_nr = [row_idx.get_loc((cr, ps, re)) for cr, ps, re in col_idx]

        # Create Compressed Sparse Column matrix
        M = IndexedMatrix(
            scipy.sparse.coo_array(
                (val, (row_nr, col_nr)), shape=(len(row_idx), len(col_idx))
            ).tocsc(),
            row_idx,
            col_idx,
        )

        return M

    def allocate_crop_production_per_use(self):
        """Allocate crop areas to different uses.
        Creates attriute 'production_per_use' in CropProduction"""

        # Get prouction per crop product
        prod = (
            self.crops.data_attr.get("production")
            .stack()
            .groupby(["region", "prod_system", "crop_prod"])
            .sum()
        )

        # Get concatenated herds
        con_herds = concat_herds(self.herds)

        # Get crop product demand for feed per region
        feed_demand = (
            con_herds.data_attr.get("feed.crop_product_demand")
            .xs("domestic", level="origin", axis=1)
            .T.groupby(["species", "breed", "sub_system", "prod_system", "crop_prod"])
            .sum()
            .T.stack(["prod_system", "crop_prod"])
            .reindex(prod.index)
            .fillna(0)
        )
        feed_demand.columns = feed_demand.columns.map(
            "feed ({0[0]}, {0[1]}, {0[2]})".format
        ).rename("demand")

        # Calculate feed demand met regionally as the maximum
        # share possible (i.e. regional crop areas are first
        # used to cater for regional feed demand befor national
        # demand for feed, food, etc.)
        regional_feed_demand = feed_demand.where(
            prod >= feed_demand.sum(axis=1),
            (feed_demand.mul(1 / feed_demand.sum(axis=1), axis=0).mul(prod, axis=0)),
        )

        prod_to_national = prod - regional_feed_demand.sum(axis=1)

        # Calculate remaining feed demand that needs to be supplied nationally
        national_feed_demand = (
            (feed_demand - regional_feed_demand)
            .groupby(["prod_system", "crop_prod"])
            .sum()
        )

        national_demand = (
            pd.concat(
                [
                    self.demand.data_attr.get("crop_prod_demand"),
                    self.crops.data_attr.get("seed_demand")
                    .groupby("prod_system")
                    .sum()
                    .stack()
                    .rename("seed"),
                    national_feed_demand,
                ],
                axis=1,
            )
            .fillna(0)
            .rename_axis("demand", axis=1)
        )

        national_demand_shares = (
            national_demand.transform(
                lambda x: x / x.sum() if x.sum() > 0 else 0, axis=1
            )
            .reindex(
                prod_to_national.reorder_levels(
                    ["prod_system", "crop_prod", "region"]
                ).index
            )
            .reorder_levels(["region", "prod_system", "crop_prod"])
        )

        # Calculate total demand (regional+national)
        total_demand = national_demand_shares.mul(
            prod_to_national, axis=0
        ) + regional_feed_demand.reindex(national_demand_shares.columns, axis=1).fillna(
            0
        )
        total_demand["none"] = prod - total_demand.sum(axis=1)
        # Set small negatives to zero
        assert total_demand.min().min() > -1e-6
        total_demand = total_demand.where(total_demand > 0, 0)

        # Calculate shares of total demand per use
        total_demand_shares = total_demand.mul(1 / prod, axis=0)
        # Assume 100% none demand for rows with NaNs (i.e. where prod==0)
        total_demand_shares.loc[:, "none"].fillna(1, inplace=True)
        total_demand_shares.fillna(0, inplace=True)

        assert np.isclose(total_demand_shares.sum(axis=1), 1).all()

        # Calculate crop production per use
        crop_production_per_use = (
            multiply_aligned(
                total_demand_shares.unstack()
                .reindex(
                    self.crops.data_attr.get("production")
                    .reorder_levels(["region", "prod_system", "crop"])
                    .index
                )
                .reorder_levels(["crop", "prod_system", "region"]),
                self.crops.data_attr.get("production"),
            )
            .T.groupby("demand")
            .sum()
            .T
        )

        # Add data attribute
        self.crops.data_attr.add(
            crop_production_per_use,
            name="production_per_use",
            unit="kg/year",
            orig="FeedDistributor",
            desc="Total crop production distributed across different uses (unreliable)",
        )

    def adjust_crop_allocation(self):
        """Adjust allocation of crop production to uses on FeedMgmt
        'max_crop_in_crop_prod' parameter used to e.g. limite the share of grazing that
        can be supplied from semi-natural grasslands for different animals"""

        # NOTE: THIS ALLOCATION PROCEDURE GENERATES UNRELIABLE RESULTS IN TERMS OF ALLOCATING
        # TOO MUCH OR LITTLE TO DIFFERENT ANIMAL HERDS. BALANCES ON REGION/ PRODUCTION SYSTEM
        # LEVEL ARE HOWEVER FINE. BUT INTERPRET RESULTS WITH CARE

        # Get crop production per use and create df for adjustments
        crop_production_per_use = self.crops.data_attr.get("production_per_use").copy()
        crop_production_per_use_adjusted = crop_production_per_use.copy()

        # Get map crop_group --> crop(s)
        map_cg_cr = inv_dict(self.par.get_rel("crop", "crop_group"))

        # Get concatenated herds
        con_herds = concat_herds(self.herds)

        # Get maximum inclusion of crops in crop_prod per animal herd
        max_feed_from_crop = (
            con_herds.data_attr.get("feed.max_supply_from_crop_group")
            .T.groupby(
                [
                    "species",
                    "breed",
                    "sub_system",
                    "prod_system",
                    "crop_prod",
                    "crop_group",
                ]
            )
            .sum()
            .T.stack(["prod_system", "crop_prod", "crop_group"])
            .fillna(0)
        ).reorder_levels(["crop_prod", "crop_group", "prod_system", "region"])
        max_feed_from_crop.columns = max_feed_from_crop.columns.map(
            "feed ({0[0]}, {0[1]}, {0[2]})".format
        ).rename("demand")

        # Get crop products with a max
        # feed from crop_groups constraints
        cps = max_feed_from_crop.index.get_level_values("crop_prod").unique()

        for cp in cps:
            # Get total demand for crop_prod per animal herd
            cp_demand_per_herd = (
                con_herds.data_attr.get("feed.crop_product_demand")
                .xs(("domestic", cp), level=("origin", "crop_prod"), axis=1)
                .T.groupby(["species", "breed", "sub_system", "prod_system"])
                .sum()
                .T.stack(["prod_system"])
                .fillna(0)
            ).reorder_levels(["prod_system", "region"])
            cp_demand_per_herd.columns = cp_demand_per_herd.columns.map(
                "feed ({0[0]}, {0[1]}, {0[2]})".format
            ).rename("demand")

            # Get constrained crop groups
            cgs = (
                max_feed_from_crop.loc[cp].index.get_level_values("crop_group").unique()
            )

            # Get constrained and unconstrained crops
            crs_cons = [cr for cg in cgs for cr in map_cg_cr[cg]]
            crs_uncons = self.crops.index.get_level_values("crop")[
                self.crops.data_attr.get("production").loc[:, cp] > 0
            ].unique()
            crs_uncons = [cr for cr in crs_uncons if cr not in crs_cons]

            # Go through constrained crop_groups and crops and update
            for cg in cgs:
                # Calculate allocation factors for constrained crop_group
                cg_allocation_factors = max_feed_from_crop.loc[cp, cg].transform(
                    lambda x: x / x.sum(), axis=1
                )

                # Get crops in constrained crop_group
                crs = map_cg_cr[cg]

                for cr in crs:
                    # Get total use of crop
                    total_use_of_cr = crop_production_per_use.loc[
                        cr, crop_production_per_use.columns.str.contains("feed")
                    ].sum(axis=1)
                    # Apply allocation factors
                    cr_allocated = cg_allocation_factors.mul(total_use_of_cr, axis=0)
                    # Update dataframe
                    crop_production_per_use_adjusted.update(
                        pd.concat({cr: cr_allocated}, names=["crop"]).fillna(0)
                    )

            # Get total use of unconstrained crops
            total_use_uncons_crs = (
                crop_production_per_use.loc[
                    crs_uncons, crop_production_per_use.columns.str.contains("feed")
                ]
                .sum(axis=1)
                .groupby(["prod_system", "region"])
                .sum()
            )
            # Get adjusted use of constrained crops per herd
            use_cons_crs_per_herd = (
                crop_production_per_use_adjusted.loc[
                    crs_cons, crop_production_per_use.columns.str.contains("feed")
                ]
                .groupby(["prod_system", "region"])
                .sum()
            )
            # Calculate allocation factors
            uncons_crs_allocation_factors = (
                cp_demand_per_herd - use_cons_crs_per_herd
            ).div(total_use_uncons_crs, axis=0)
            # Make sure rows sums to 1 (this shouldn't be needed... some problem here...)
            uncons_crs_allocation_factors = uncons_crs_allocation_factors.transform(
                lambda x: x / x.sum(), axis=1
            )
            # Go through unconstrained crops and update
            for cr in crs_uncons:
                # Get total use of crop
                total_use_of_cr = crop_production_per_use.loc[
                    cr, crop_production_per_use.columns.str.contains("feed")
                ].sum(axis=1)
                # Apply allocation factors
                cr_allocated = uncons_crs_allocation_factors.mul(
                    total_use_of_cr, axis=0
                )
                # Update dataframe
                crop_production_per_use_adjusted.update(
                    pd.concat({cr: cr_allocated}, names=["crop"]).fillna(0)
                )

        mini = crop_production_per_use_adjusted.min().min()
        if mini < -5:
            # Negatives!
            warnings.warn(
                f"Negatives of down to {mini} kg in adjusted crop allocation, these "
                + "were set to zero"
            )

        dif = abs(
            (
                crop_production_per_use_adjusted.sum(axis=1)
                - crop_production_per_use.sum(axis=1)
            )
        ).max()
        if dif > 5:
            # Dif from unadjusted
            warnings.warn(
                f"Adjusted crop allocation differed from unadjusted by up to {dif} kg"
            )

        # Set small negatives to zero
        crop_production_per_use_adjusted = crop_production_per_use_adjusted.where(
            crop_production_per_use_adjusted >= 0, 0
        )

        # Update data attribute
        self.crops.data_attr.add(
            crop_production_per_use_adjusted,
            name="production_per_use",
            unit="kg/year",
            orig="FeedDistributor",
            desc="Total crop production distributed across different uses (unreliable)",
        )


class IndexedMatrix:
    """Class to store pandas.Index/MultiIndex alongside a sparse
    matrix to keep track of things"""

    def __init__(self, matrix, row_idx, col_idx):
        self.M = matrix

        if isinstance(row_idx, list):
            levels = list(row_idx[0].names)
            for idx in row_idx:
                levels.extend([lvl for lvl in idx.names if lvl not in levels])
            print(levels)

        self.rows = row_idx
        self.cols = col_idx

    def eval(self, x):
        return pd.Series(self.M @ x, index=self.rows)
