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
    Calculate the demand for crop products from the feed demands per origin (domestic
    and imported).
    """
    # Ensure we don't have duplicates in the 'feed' column, otherwise we can get
    # inconsistent results
    if not feed_to_crop_products.reset_index()["feed"].is_unique:
        _feeds = feed_to_crop_products.reset_index()["feed"]
        duplicates = ", ".join(_feeds[_feeds.duplicated()].tolist())
        msg = f"Ambigious mapping of feed_to_crop_products: duplicate feed(s) {duplicates}"
        raise ValueError(msg)

    # Ensure the feed_to_crop_products has the two columns feed_to_prod and
    # share_imported
    if any(
        [
            c not in feed_to_crop_products.columns
            for c in ["feed_to_prod", "share_imported"]
        ]
    ):
        raise ValueError(
            "Expected feed_to_crop_products dataframe to have columns named feed_to_prod and share_imported."
        )

    # Put the data in a long format so we can merge it
    feed_demands_long = feed_demands.stack(
        level=["prod_system", "animal", "feed"], future_stack=True
    ).reset_index()

    feed_demands_long.columns = [
        "region",
        "prod_system",
        "animal",
        "feed",
        "base_demand",
    ]

    # Bring in the two other columns that we need, so that we can multiply with the
    # demand
    merged = feed_demands_long.merge(
        feed_to_crop_products[["feed_to_prod", "share_imported"]], on="feed"
    )
    # Map each feed -> crop_product
    merged["feed"] = merged["feed"].replace(dict(feed_to_crop_products.index.values))
    # Rename 'feed' as it now contains crop_products
    merged = merged.rename(columns={"feed": "crop_prod"})
    # Calculate the new values by multiplying demand with import shares
    merged["demand_imported"] = (
        merged["base_demand"] * merged["feed_to_prod"] * merged["share_imported"]
    )
    merged["demand_domestic"] = (
        merged["base_demand"] * merged["feed_to_prod"] * (1 - merged["share_imported"])
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
