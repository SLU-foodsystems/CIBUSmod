from itertools import product
import warnings
import sys

import pandas as pd
import numpy as np
import cvxpy

from .. import (
    Regions,
    DemandAndConversions,
    CropProduction,
    FeedMgmt,
    ParameterRetriever,
)

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr

from .indexed_matrix import IndexedMatrix
from .utils import Constraint, make_cvxpy_constraint


class FeedDistributor:
    """Class that handles the distribution of animals, crops and feeds across regions
    for a given demand and a number of constraints by minimising deviation from an
    initial distribution of crop areas and animal heads (x0).

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

    from .util_methods import (
        _get_feed_to_prod_factors,
        _get_losses_factors,
        _get_feed_compositions,
    )

    from .allocation_methods import (
        adjust_crop_allocation,
        allocate_crop_production_per_use,
        allocate_feed_demands,
        allocate_feed_crop_prod_demands,
    )

    from .constraints import (
        make_C1,
        make_C2,
        make_C3,
        make_C4,
        make_C5,
        make_C6,
        make_C7,
        make_C8,
        make_C9,
        make_C11,
        make_C12,
        make_C13,
        make_C14,
        make_P1,
        make_A1_1,
        make_A1_2,
        make_A1_3,
        make_A2_1,
        make_A2_2,
        make_A3,
        make_A4,
        make_A5_1,
        make_A5_2,
        make_A6,
        make_A8,
        make_A9,
        make_A10_1,
        make_C10,
        make_A12_1,
        make_A12_2,
        make_A11,
        make_b13,
        make_A13,
        make_A14,
        make_P1_1,
        make_P1_2,
        make_P1_3,
    )

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
            List of numbers corresponding to the constraints to be used. For
            descriptions of each constraint see ?FeedDistributor.make_C<nr>
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
            try:
                fun(**{k: v for k, v in kwargs.items() if f"C{nr}" in k})
            except Exception as e:
                print(
                    f"Exception raised when making constraint C{nr}.", file=sys.stderr
                )
                raise e

        # If C7 not included no variables are dropped
        if "7" not in use_cons:
            self.x_idx_short = {
                "ani": self.x_idx["ani"].copy(),
                "crp": self.x_idx["crp"].copy(),
                "fds": self.x_idx["fds"].copy(),
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
                print(e, file=sys.stderr)
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
            assert x is not None, "Could not fetch optimal value from problem."

            # Put xs on short index (!= index if C7 is used) and reindex
            n_ani_short = len(self.x_idx_short["ani"])
            n_crp_short = len(self.x_idx_short["crp"])
            self.x = {
                "ani": pd.Series(
                    x[:n_ani_short], index=self.x_idx_short["ani"]
                ).reindex(self.x_idx["ani"], fill_value=0),
                "crp": pd.Series(
                    x[n_ani_short : n_ani_short + n_crp_short],
                    index=self.x_idx_short["crp"],
                ).reindex(self.x_idx["crp"], fill_value=0),
                "fds": pd.Series(
                    x[n_ani_short + n_crp_short :], index=self.x_idx_short["fds"]
                ).reindex(self.x_idx["fds"], fill_value=0),
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
            self.data_attr.add(
                self.x["fds"],
                name="x_feeds",
                unit="kg DM",
                orig="FeedDistributor",
                desc="Total amount of feed for each animal system",
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

        self.allocate_feed_demands()
        self.allocate_feed_crop_prod_demands()

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
                    (f, ani, sp, br, ps, ss, re)
                    for (sp, br, ps, ss) in self.herds.index
                    for re in self.herds[(sp, br, ps, ss)].index
                    for f in self.herds[(sp, br, ps, ss)].par.get_unique("feed")
                    for ani in self.herds[(sp, br, ps, ss)].animals
                ],
                names=[
                    "feed",
                    "animal",
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

        # Add rows for any domestically produced crop products used for feed or seed not
        # already in crop product demand vector (D['crp'])
        # Get all crop-products from feed_mgmt and the seed crop products from Crops
        self.feed_mgmt.par.clear()
        cps = set(self.feed_mgmt.par.get_unique("crop_prod")) | set(
            self.crops.par.get_unique("crop_prod", qry='parameter == "seed"')
        )
        # ... and the prod_systems in the crop-demand vector
        pss = self.D["crp"].index.unique("prod_system")

        share_domestic = self.feed_mgmt.par.get_from_frame(
            "share_domestic",
            pd.DataFrame(
                index=pd.Index(pss, name="prod_system"),
                columns=pd.Index(cps, name="crop_prod"),
                dtype=float,
            ),
        )

        for ps, cp in product(pss, cps):
            # Skip if already assigned
            if (ps, cp) in self.D["crp"].index:
                continue
            if (share_domestic.loc[ps, cp] == 0).any():
                self.D["crp"][(ps, cp)] = 0

        # Store indexes
        self.D_idx = {"ani": self.D["ani"].index, "crp": self.D["crp"].index}

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

    def get_cvx_problem(self):
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
        return cvxpy.Problem(objective=objective, constraints=constraints)
