import warnings
import pandas as pd
import numpy as np

from ..utils.misc import multiply_aligned, inv_dict
from ..main_modules.animal_herd import concat_herds

def allocate_crop_production_per_use(self):
    """Allocate crop areas to different uses.
    Creates attribute 'production_per_use' in CropProduction"""

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
        .T.stack(["prod_system", "crop_prod"], future_stack=True)
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
    total_demand_shares.loc[:, "none"] = total_demand_shares.loc[:, "none"].fillna(
        1
    )
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
    """
    Adjust allocation of crop production to uses on FeedMgmt 'max_crop_in_crop_prod'
    parameter used to e.g. limit the share of grazing that can be supplied from
    semi-natural grasslands for different animals
    """

    # NOTE: THIS ALLOCATION PROCEDURE GENERATES UNRELIABLE RESULTS IN TERMS OF
    # ALLOCATING TOO MUCH OR LITTLE TO DIFFERENT ANIMAL HERDS. BALANCES ON REGION/
    # PRODUCTION SYSTEM LEVEL ARE HOWEVER FINE. BUT INTERPRET RESULTS WITH CARE

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
    cps = max_feed_from_crop.index.unique("crop_prod")

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
        cgs = max_feed_from_crop.loc[cp].index.unique("crop_group")

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

def allocate_feed_demands(self):
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
        assert matching_zeroes.all().all(), f"Total number of heads is recorded as zero, despite non-zero heads in herd: (sp={sp}, br={br}, ss={ss}, ps={ps}) "

        # Compute the ratio between the two, where we (thanks to the assertion above) can safely replace 0 with 1 in the denominator to avoid division by zero
        ratio = (h_heads / h_total.replace({0: 1})).sort_index(axis=1).sort_index()

        if sp != "cattle":
            non_zero_ratio = ratio[ratio != 0]
            all_ratio_one_or_zero = (
                (non_zero_ratio.replace({np.nan: 1}) == 1).all().all()
            )
            assert (
                all_ratio_one_or_zero
            ), "All ratios for non-cattle should be either 0 or 1"

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

        # Ensure that wherever h_heads is zero, feed_demands must be zero.
        heads_long = h_heads.T.stack("region")
        for (ps, ani, re), _ in heads_long[heads_long == 1]:
            feed_demands_slice = feed_demands.loc[re, (ps, ani, slice(None))]
            assert (
                feed_demands_slice == 0
            ).all(), "Wherever we have zero animals, we should also have 0 feed"

        herd.data_attr.add(
            adjusted_feed_demands,
            name="feed.demand",
            unit="kg DM/year",
            orig="FeedDist",
            desc="Demand for feed",
        )

def allocate_feed_crop_prod_demands(self):
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
