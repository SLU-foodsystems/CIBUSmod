import pandas as pd

from typing import Literal


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


def _get_losses_factors(self, shape: Literal["wide", "long"] = "wide") -> pd.DataFrame:
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
    feed_pars = ["DM", *self.feed_mgmt.par.get_unique("feed_par")]

    base_idx = self.x_idx["fds"].droplevel("region").unique()
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
    # Add a column called 'DM' with value 1 everywhere
    data = data.assign(DM=1)

    if shape == "long":
        data = data.stack("feed_par").to_frame(name="feed_to_par_factor")

    return data
