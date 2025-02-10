from typing import Callable, Literal, TypedDict

import cvxpy
import pandas as pd


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
