from typing import Callable, Literal, TypedDict

import cvxpy
import numpy as np
import pandas as pd

from .indexed_matrix import IndexedMatrix


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


def scale_constraints_by_row(constraints: dict, eps: float = 1e-12) -> dict:
    """
    Returns a COPY of 'constraints' (a dict of Constraint-dicts, as built by
    GeoDistributor/FeedDistributor.make() and stored in 'dist.constraints') where
    every constraint's row has been rescaled to a maximum absolute coefficient of 1.

    Why: within a single constraint row, coefficients routinely span many orders of
    magnitude (e.g. C11 ration-share rows mixing kg DM feed amounts with animal head
    counts, or C1 rows mixing hectares of cropland with kg of animal product). That
    range makes the KKT system an interior-point/barrier solver has to factor
    ill-conditioned, independent of anything about the objective -- this has been
    observed to make GUROBI fail with a generic SolverError (or an incorrect
    'infeasible' report) on large, tightly-constrained problems, even though a
    feasible point provably exists. Dividing an entire row -- both its matrix
    coefficients AND its right-hand side -- by the same positive scalar changes
    nothing about which x satisfy '[row] <rel> [rhs]', so this is a pure
    numerical-conditioning fix, not an approximation: the feasible region and the
    optimal x (up to solver tolerance) are unchanged.

    How: for each constraint, 'left'/'right' are replaced with wrapped versions that
    multiply whatever the ORIGINAL left/right functions return by a fixed per-row
    scale vector -- e.g. 'wrapped_left(x, **pars) = row_scale * orig_left(x, **pars)'.
    This is equivalent to (but implemented without touching) rescaling the
    underlying IndexedMatrix/right-hand-side arrays directly, and is robust to
    whatever a particular constraint's left/right actually compute (e.g. C14's
    'left' adds a right-hand-side-like term instead of putting it in 'right' --
    scaling both sides by the same factor is still correct regardless of exactly
    how a constraint's algebra is arranged, since it distributes over addition).

    Crucially, 'pars' itself (and so the constraint's underlying IndexedMatrix and
    any right-hand-side array) is left COMPLETELY untouched -- only new 'left'/
    'right' closures are added to a shallow COPY of each constraint dict, and
    'constraints' itself is not mutated. This matters because code elsewhere (e.g.
    CIBUSmod.utils.maximise_nutrient_supply's constraint-rebuilding helpers) reads
    'pars' directly to get PHYSICAL, unscaled quantities (e.g. a demand floor to
    compare against production); if scaling mutated 'pars' in place, all of that
    downstream code would silently start working with scaled, physically
    meaningless numbers.

    Because of that, this function is NOT idempotent-safe to apply to its own
    output's 'pars' (calling it twice on results derived from the same 'pars' would
    scale twice) -- it must be called exactly once, as the LAST step before handing
    constraints to cvxpy.Problem(), on whatever the FINAL constraint dict for a
    solve is (i.e. after any row-slicing/rebuilding, such as
    FeedDistributor.make_C7()'s column-pruning or
    CIBUSmod.utils.maximise_nutrient_supply's C1/C10/C15 rebuilding, has already
    happened).

    Parameters
    ----------
    constraints : dict
        Constraint-dicts keyed by name, e.g. 'dist.constraints'. Each value must
        have exactly one IndexedMatrix entry in its 'pars' (true for every
        constraint currently built by GeoDistributor/FeedDistributor/
        CIBUSmod.utils.maximise_nutrient_supply -- a constraint with zero or more
        than one raises ValueError rather than silently guessing which to scale
        by).
    eps : float, default 1e-12
        Rows whose largest absolute coefficient is below this are left unscaled
        (scale factor 1) rather than divided by (near) zero -- relevant for
        all-zero rows, which can occur e.g. for a by-product/region combination
        with no actual supply or demand routed through it.

    Returns
    -------
    dict
        A new dict, same keys as 'constraints', each value a shallow copy with
        'left'/'right' replaced by row-scaled wrappers ('rel' and 'pars' shared
        with the original, unmodified).
    """

    scaled = {}
    for name, cons in constraints.items():
        pars = cons["pars"]
        mats = [v for v in pars.values() if isinstance(v, IndexedMatrix)]
        if len(mats) != 1:
            raise ValueError(
                f"scale_constraints_by_row(): constraint '{name}' has "
                f"{len(mats)} IndexedMatrix entries in its 'pars' (expected "
                "exactly 1). This function's row-scaling assumes each constraint "
                "has a single coefficient matrix to scale by -- extend it "
                "explicitly (rather than guessing which matrix to use) if a "
                "constraint with a different shape is introduced."
            )

        M_abs = mats[0].M.tocsr(copy=True)
        M_abs.data = np.abs(M_abs.data)
        row_max = np.asarray(M_abs.max(axis=1).todense()).flatten()
        row_scale = np.ones_like(row_max)
        np.divide(1.0, row_max, out=row_scale, where=row_max > eps)

        def _wrap(orig_left=cons["left"], orig_right=cons["right"], row_scale=row_scale):
            # cvxpy.multiply(), NOT plain '*': for two 1-D arrays cvxpy's '*'
            # operator means matrix (i.e. dot-product) multiplication, not
            # elementwise -- using it here would silently collapse each
            # constraint's rows into a single scalar instead of scaling each row
            # independently. cvxpy.multiply() is the explicit elementwise op and
            # works the same whether 'x' ends up a cvxpy Variable/Expression
            # (production use) or a plain numpy array.
            def wrapped_left(x, **pars):
                return cvxpy.multiply(row_scale, orig_left(x, **pars))

            def wrapped_right(**pars):
                return cvxpy.multiply(row_scale, orig_right(**pars))

            return wrapped_left, wrapped_right

        wrapped_left, wrapped_right = _wrap()
        scaled[name] = {
            "left": wrapped_left,
            "right": wrapped_right,
            "rel": cons["rel"],
            "pars": pars,
        }

    return scaled


def feed_demands_to_crop_demands(
    feed_demands: pd.DataFrame, feed_to_crop_products: pd.DataFrame
):
    """
    Calculate the demand of crop products (per origin: domestic and imported) from the
    feed demands and a mapping from feeds to crop products and with share of domestic.
    """
    # Ensure that feed_to_crop_products has the expected columns
    if (
        "feed_to_prod" not in feed_to_crop_products.columns
        or "share_domestic" not in feed_to_crop_products.columns
    ):
        raise ValueError(
            "Expected feed_to_crop_products dataframe to have columns feed_to_prod and share_domestic."
        )

    # Put the data in a long format so we can merge it
    feed_demands_long = feed_demands.stack(
        level=["prod_system", "animal", "feed"], future_stack=True
    ).reset_index()

    # Keep only the relevant columns
    feed_demands_long.columns = [
        "region",
        "prod_system",
        "animal",
        "feed",
        "base_demand",
    ]

    feed_to_cp = dict(map(lambda x: x[1:3], feed_to_crop_products.index.values))
    feed_to_crop_products = feed_to_crop_products.reset_index()

    # Bring in the two other columns that we need, so that we can multiply with the
    # demand
    merged = feed_demands_long.merge(
        feed_to_crop_products,
        on=["prod_system", "feed"],
    )

    # Map each feed -> crop_product
    merged["crop_prod"] = merged["feed"].replace(feed_to_cp).drop(columns=["feed"])
    # Calculate the new values by multiplying demand with import shares
    merged["demand_imported"] = (
        merged["base_demand"] * merged["feed_to_prod"] * (1 - merged["share_domestic"])
    )
    merged["demand_domestic"] = (
        merged["base_demand"] * merged["feed_to_prod"] * merged["share_domestic"]
    )

    # Pivot to reshape back to a wide format with desired columns, and do this for both
    # domestic and imported values
    return pd.concat(
        [
            pd.concat(
                {  # We add a new level, "origin" to the column index
                    origin: merged.pivot_table(
                        index="region",
                        columns=["prod_system", "animal", "crop_prod"],
                        values=f"demand_{origin}",
                        aggfunc="sum",
                    ),
                },
                names=["origin"],
                axis=1,
            )
            for origin in ["domestic", "imported"]
        ],
        axis=1,
    )
