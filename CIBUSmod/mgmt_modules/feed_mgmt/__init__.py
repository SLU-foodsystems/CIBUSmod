from abc import ABC, abstractmethod

from typing import Literal
import warnings
import pandas as pd
import numpy as np

from CIBUSmod.main_modules.animal_herd import AnimalHerd
from CIBUSmod.utils.retriever import ParameterRetriever
from CIBUSmod.utils.misc import multiply_aligned, fix_herds

class FeedMgmt(ABC):
    """
    Class that that calculates amount of crop products ('crop_prod'), by-products ('by_prod') and
    crop residues ('crop_resid') needed for a certain demand feed 'feed' accounting far all losses
    between harvest/production and final consumption by the animals. It also calculates nutrient
    characteristics of rations and methane emissions associated with feed digestion.

    Parameters
    ----------
    herds : (pandas.Series of) AnimalHerd object(s)
    par : ParameterRetriever object
    type : str, default 'GeoDist'
        Should match the optimisation module used
        'GeoDist' for GeoDistributor or
        'FeedDist' for FeedDistributor
    
    Note: This is a factory base-class to instantiate a concrete implementation of either a
    GeoDistFeedMgmt object (type='GeoDist') or a FeedDistFeedMgmt object (type='FeedDist')
    """

    def __new__(cls, herds, par, type="GeoDist"):
        if cls is FeedMgmt:
            if type == "GeoDist":
                from .feed_mgmt_geodist import GeoDistFeedMgmt
                cls = GeoDistFeedMgmt
            elif type == "FeedDist":
                from .feed_mgmt_feeddist import FeedDistFeedMgmt
                cls = FeedDistFeedMgmt
            else:
                raise ValueError(
                    "Received unexpected type {type}, expected one of 'FeedDist' and 'GeoDist'."
                )
        return object.__new__(cls)

    def __init__(self, herds: AnimalHerd | list | pd.Series, par: ParameterRetriever, type: str = "GeoDist"):
        self.par = par
        self.herds: pd.Series = fix_herds(herds)
        self.index = list(self.herds)[0].index

    @abstractmethod
    def calculate(self, verbose=False):
        pass

    def calculate_ration_characteristics(self):
        """Calculates ration characteristics"""
        for herd in (h for h in self.herds if h.has_feed_demand()):

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(species=herd.species, breed=herd.breed)

            feed_DM = herd.data_attr.get("feed.consumption")
            # Get dry matter intake [kg DM]
            ration_DM = feed_DM.T.groupby(["prod_system", "animal"]).sum().T

            pars = ["N", "P", "K", "ASH", "GE"]

            if herd.species == "cattle":
                pars += ["DE", "rough", "fat"]
            elif herd.species in ["sheep","goats","horses","pigs"]:
                pars += ["DE"]
            elif herd.species == "poultry":
                pars += ["AME"]

            for par in pars:
                res = (
                    (
                        self.par.get_from_frame(
                            "feed_composition", feed_DM, feed_par=par
                        )
                        * feed_DM
                    )
                    .T.groupby(["prod_system", "animal"])
                    .sum()
                    .T
                    / ration_DM.replace({0: np.nan})  # To avoid div by 0 error
                )

                # Add data attribute
                herd.data_attr.add(
                    res,
                    name="feed.ration_" + par,
                    unit="MJ/kg DM" if par in ["GE", "DE"] else "*/kg DM",
                    orig="FeedMgmt",
                    desc="Ration " + par + " content",
                    scalable=False,
                )

    def calculate_product_demand(
        self, prod_type: Literal["crop_prod", "crop_resid", "by_prod"]
    ):
        # Get Series of crop/by- products or crop residues with feed as index
        prs = self.par.get_unique(["feed", prod_type]).set_index("feed")[prod_type]

        if (prod_type != "crop_prod") and (
            len(self.par.get_unique(prod_type, 'parameter == "share_domestic"')) > 0
        ):
            from .feed_mgmt_feeddist import FeedDistFeedMgmt
            if not isinstance(self, FeedDistFeedMgmt):
                warnings.warn(
                    f"FeedMgmt: Specified domestic shares for '{prod_type}' will have no effect! \n"
                    + "For by-products, imports are handled in the ByProductMgmt module and for crop residues, imports are not allowed."
                )

        # Create list for feeds not linked to a product
        not_linked_feeds = []

        for herd in (h for h in self.herds if h.has_feed_demand()):

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species=herd.species, breed=herd.breed, sub_system=herd.sub_system
            )

            # Construct retrieve dataframe columns
            df_cols = []
            for ps, ani, fe in herd.data_attr.get("feed.demand").columns:
                if fe in prs.index:
                    for pr in prs[fe] if not isinstance(prs[fe], str) else [prs[fe]]:
                        df_cols += [(ps, ani, pr, fe)]
                else:
                    not_linked_feeds += [fe]

            retrieve_df = pd.DataFrame(
                index=herd.index,
                columns=pd.MultiIndex.from_tuples(
                    df_cols, names=["prod_system", "animal", prod_type, "feed"]
                ),
                dtype=float,
            )

            if retrieve_df.shape[1] > 0:
                # Get factors feed --> product
                feed_to_prod = self.par.get_from_frame("feed_to_prod", retrieve_df)

                if prod_type == "crop_prod":
                    # Get import shares
                    feed_to_imp_prod = (
                        feed_to_prod
                        * (100 - self.par.get_from_frame("share_domestic", retrieve_df))
                        / 100
                    )
                    feed_to_dom_prod = feed_to_prod - feed_to_imp_prod

                    # Calculate domestic and imported crop product demand
                    dom_prod = (
                        multiply_aligned(
                            feed_to_dom_prod, herd.data_attr.get("feed.demand")
                        )
                        .T.groupby(["prod_system", "animal", prod_type], sort=False)
                        .sum()
                        .T
                    )
                    imp_prod = (
                        multiply_aligned(
                            feed_to_imp_prod, herd.data_attr.get("feed.demand")
                        )
                        .T.groupby(["prod_system", "animal", prod_type], sort=False)
                        .sum()
                        .T
                    )

                    # Combine
                    result_df = pd.concat(
                        [
                            pd.concat(
                                {"domestic": (dom_prod)}, names=["origin"], axis=1
                            ),
                            pd.concat(
                                {"imported": (imp_prod)}, names=["origin"], axis=1
                            ),
                        ],
                        axis=1,
                    )

                    # Calculate regional demand for crop products
                    feed_to_reg_prod = (
                        feed_to_dom_prod
                        * self.par.get_from_frame("share_regional", retrieve_df)
                        / 100
                    )
                    result_df_reg = (
                        multiply_aligned(
                            feed_to_reg_prod, herd.data_attr.get("feed.demand")
                        )
                        .T.groupby(["prod_system", "animal", prod_type], sort=False)
                        .sum()
                        .T
                    )
                else:
                    # Calculate demand
                    result_df = (
                        multiply_aligned(
                            feed_to_prod, herd.data_attr.get("feed.demand")
                        )
                        .T.groupby(["prod_system", "animal", prod_type], sort=False)
                        .sum()
                        .T
                    )
            else:
                # Empty result dataframes
                result_df = pd.DataFrame(
                    index=herd.index,
                    columns=pd.MultiIndex.from_tuples(
                        [],
                        names=["origin", "prod_system", "animal", prod_type]
                        if prod_type == "crop_prod"
                        else ["prod_system", "animal", prod_type],
                    ),
                    dtype=float,
                )
                result_df_reg = pd.DataFrame(
                    index=herd.index,
                    columns=pd.MultiIndex.from_tuples(
                        [], names=["prod_system", "animal", prod_type]
                    ),
                    dtype=float,
                )

            # Add data attributes (drop zero cols)
            result_df = result_df.loc[:, result_df.sum() > 0]
            prod_demand_key = {
                "crop_prod": "feed.crop_product_demand",
                "crop_resid": "feed.crop_residue_demand",
                "by_prod": "feed.by_product_demand",
            }[prod_type]
            herd.data_attr.add(
                result_df,
                name=prod_demand_key,
                unit="kg DM/year" if prod_type == "crop_resid" else "kg/year",
                orig="FeedMgmt",
                desc=f'Demand for {"crop products" if prod_type == "crop_prod" else "by-products" if prod_type=="by_prod" else "crop residues"} for feed',
            )

            if prod_type == "crop_prod":
                result_df_reg = result_df_reg.loc[:, result_df_reg.sum() > 0]
                herd.data_attr.add(
                    result_df_reg,
                    name="feed.regional_crop_product_demand",
                    unit="kg/year",
                    orig="FeedMgmt",
                    desc="Demand for crop products for feed that must be supplied regionally",
                )

        return not_linked_feeds

    def calculate_max_crop_in_crop_prod(self):
        # Get crop_groups to handle
        cgs = self.par.get_unique(
            ["crop_group", "crop_prod"], qry='parameter == "max_crop_in_crop_prod"'
        ).set_index("crop_prod")["crop_group"]

        for herd in (h for h in self.herds if h.has_feed_demand()):

            # If all empty, skip the calculations and just add a None
            if (
                not herd.data_attr.get("feed.crop_product_demand")
                .columns.get_level_values("crop_prod")
                .isin(cgs.index)
                .any()
            ):
                # Add data attribute
                herd.data_attr.add(
                    None,
                    name="feed.max_supply_from_crop_group",
                    unit="kg/year",
                    orig="FeedMgmt",
                    desc="Maximum supply of feed from crop_group. Used in GeoDistributor constraint",
                )
                continue

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species=herd.species, breed=herd.breed, sub_system=herd.sub_system
            )

            # Get crop product demand supplied by crops in crs
            df = herd.data_attr.get("feed.crop_product_demand").loc[
                :, pd.IndexSlice["domestic", :, :, cgs.index]
            ]

            # Append crops to column index
            df.columns = pd.MultiIndex.from_tuples(
                map(lambda x: (x + tuple([cgs[x[-1]]])), df.columns),
                names=df.columns.names + ["crop_group"],
            )

            # Calculate maximum supply of crop_prod from crop
            res = (
                df
                * self.par.get_from_frame(
                    "max_crop_in_crop_prod",
                    df,
                    species=herd.species,
                    breed=herd.breed,
                    sub_system=herd.sub_system,
                )
            ).T.groupby(["prod_system", "crop_prod", "crop_group"]).sum().T / 100

            # Add data attribute
            herd.data_attr.add(
                res,
                name="feed.max_supply_from_crop_group",
                unit="kg/year",
                orig="FeedMgmt",
                desc="Maximum supply of feed from crop_group. Used in GeoDistributor constraint",
            )

    def calculate_feed_by_product_generation(self):
        """Calculate by-products generated as a result of feed demand.

        For crop_prod and crop_resid type feeds the domestic share is known from the
        'share_domestic' parameter and is applied directly. Results are stored in
        'feed.by_product_generation' on each herd (scalable=True so GeoDistributor
        scaling is propagated automatically).

        For by_prod type feeds in GeoDistributor mode 'share_domestic' is not defined
        in FeedMgmt (imports are handled post-hoc by ByProductMgmt). Generation is
        therefore stored separately in 'feed.by_product_generation_from_byprod_feed'
        indexed by [prod_system, animal, by_prod_generated, by_prod_feed] with 100%
        domestic assumed. ByProductMgmt scales this by the actual domestic fraction
        from the supply-demand balance.

        In FeedDistributor mode 'share_domestic' is defined for all feed types so all
        generation goes into 'feed.by_product_generation'.
        """
        from .feed_mgmt_feeddist import FeedDistFeedMgmt
        is_feeddist = isinstance(self, FeedDistFeedMgmt)

        # Get all (feed, by_prod_generated) pairs defined for feed_to_by_prod
        bpg_index = self.par.get_unique(
            ["feed", "by_prod"], 'parameter == "feed_to_by_prod"'
        )

        empty_adj = lambda herd: pd.DataFrame(
            index=herd.index,
            columns=pd.MultiIndex.from_tuples([], names=["prod_system", "animal", "by_prod"]),
            dtype=float,
        )
        empty_unadj = lambda herd: pd.DataFrame(
            index=herd.index,
            columns=pd.MultiIndex.from_tuples(
                [], names=["prod_system", "animal", "by_prod_generated", "by_prod_source"]
            ),
            dtype=float,
        )

        if len(bpg_index) == 0:
            for herd in (h for h in self.herds if h.has_feed_demand()):
                herd.data_attr.add(
                    empty_adj(herd), name="feed.by_product_generation",
                    unit="kg/year", orig="FeedMgmt",
                    desc="By-products generated from feed demand",
                    scalable=True,
                )
                if not is_feeddist:
                    herd.data_attr.add(
                        empty_unadj(herd),
                        name="feed.by_product_generation_from_byprod_feed",
                        unit="kg/year", orig="FeedMgmt",
                        desc="By-products generated from by_prod feed demand (domestic fraction not applied)",
                        scalable=True,
                    )
            return

        bpg = bpg_index.set_index("feed")["by_prod"]

        # Feeds linked to a by_prod source via feed_to_prod (no share_domestic in GeoDist).
        # Maps feed name → source by_prod name for the domestic fraction lookup in ByProductMgmt.
        _feed_to_prod_df = self.par.get_unique(["feed", "by_prod"], 'parameter == "feed_to_prod"')
        byprod_type_feeds = dict(zip(_feed_to_prod_df["feed"], _feed_to_prod_df["by_prod"]))

        for herd in (h for h in self.herds if h.has_feed_demand()):
            self.par.clear()
            self.par.set(species=herd.species, breed=herd.breed, sub_system=herd.sub_system)

            adj_cols = []    # crop_prod/crop_resid feeds (share_domestic applies)
            unadj_cols = []  # by_prod feeds in GeoDist (no share_domestic)

            for ps, ani, fe in herd.data_attr.get("feed.demand").columns:
                if fe not in bpg.index:
                    continue
                for bp_gen in (bpg[fe] if not isinstance(bpg[fe], str) else [bpg[fe]]):
                    if is_feeddist or (fe not in byprod_type_feeds):
                        adj_cols.append((ps, ani, bp_gen, fe))
                    else:
                        # Keep feed name in the column tuple so get_from_frame can look up
                        # feed_to_by_prod; the feed→source_by_prod remapping happens after
                        # the multiplication step below.
                        unadj_cols.append((ps, ani, bp_gen, fe))

            # --- Adjusted generation: share_domestic applied ---
            adj_df = pd.DataFrame(
                index=herd.index,
                columns=pd.MultiIndex.from_tuples(
                    adj_cols, names=["prod_system", "animal", "by_prod", "feed"]
                ),
                dtype=float,
            )
            if adj_df.shape[1] > 0:
                factors = self.par.get_from_frame("feed_to_by_prod", adj_df)

                # Fetch share_domestic without the by_prod level: get_from_frame
                # passes ALL column levels as filters, and share_domestic has no
                # f_by_prod filter column, which would produce NaN.
                share_dom_retrieve = pd.DataFrame(
                    index=herd.index,
                    columns=pd.MultiIndex.from_tuples(
                        list(dict.fromkeys(
                            (ps, ani, fe) for ps, ani, _bp, fe in adj_cols
                        )),
                        names=["prod_system", "animal", "feed"],
                    ),
                    dtype=float,
                )
                share_dom_raw = self.par.get_from_frame("share_domestic", share_dom_retrieve) / 100
                # Broadcast to adj_df column structure
                share_dom = pd.DataFrame(
                    {col: share_dom_raw[(col[0], col[1], col[3])] for col in adj_df.columns},
                    index=herd.index,
                )
                share_dom.columns = adj_df.columns

                adj_result = (
                    multiply_aligned(factors * share_dom, herd.data_attr.get("feed.demand"))
                    .T.groupby(["prod_system", "animal", "by_prod"], sort=False).sum().T
                )
                adj_result = adj_result.loc[:, adj_result.sum() > 0]
            else:
                adj_result = empty_adj(herd)

            herd.data_attr.add(
                adj_result, name="feed.by_product_generation",
                unit="kg/year", orig="FeedMgmt",
                desc="By-products generated from feed demand",
                scalable=True,
            )

            if not is_feeddist:
                # --- Unadjusted generation: by_prod feeds, 100% domestic assumed ---
                unadj_df = pd.DataFrame(
                    index=herd.index,
                    columns=pd.MultiIndex.from_tuples(
                        unadj_cols, names=["prod_system", "animal", "by_prod", "feed"]
                    ),
                    dtype=float,
                )
                if unadj_df.shape[1] > 0:
                    unadj_factors = self.par.get_from_frame("feed_to_by_prod", unadj_df)
                    unadj_result = (
                        multiply_aligned(unadj_factors, herd.data_attr.get("feed.demand"))
                        .T.groupby(["prod_system", "animal", "by_prod", "feed"], sort=False)
                        .sum().T
                    )
                    # Replace feed names with the source by_prod names so ByProductMgmt
                    # can look up the correct domestic fraction from the supply-demand
                    # balance (e.g. "vegetable oils" → "rapeseed").
                    unadj_result.columns = pd.MultiIndex.from_tuples(
                        [
                            (ps, ani, bp_gen, byprod_type_feeds.get(fe, fe))
                            for ps, ani, bp_gen, fe in unadj_result.columns
                        ],
                        names=["prod_system", "animal", "by_prod_generated", "by_prod_source"],
                    )
                    unadj_result = (
                        unadj_result
                        .T.groupby(
                            ["prod_system", "animal", "by_prod_generated", "by_prod_source"],
                            sort=False,
                        ).sum().T
                    )
                    unadj_result = unadj_result.loc[:, unadj_result.sum() > 0]
                else:
                    unadj_result = empty_unadj(herd)

                herd.data_attr.add(
                    unadj_result,
                    name="feed.by_product_generation_from_byprod_feed",
                    unit="kg/year", orig="FeedMgmt",
                    desc="By-products generated from by_prod feed demand (domestic fraction not applied)",
                    scalable=True,
                )

    def calculate_enteric_methane(self):
        idx = pd.IndexSlice

        for herd in (h for h in self.herds if h.has_feed_demand()):

            self.par.set(
                species=herd.species, breed=herd.breed, sub_system=herd.sub_system
            )

            if herd.species in ["cattle", "sheep", "goats"]:
                CH4_specific_energy = 55.6  # [MJ/kg]

                # Get dry matter intake [kg DM]
                dry_matter_intake = (
                    herd.data_attr.get("feed.consumption")
                    .T.groupby(["prod_system", "animal"])
                    .sum()
                    .T
                )

                # Get gross energy intake [MJ]
                GE_intake = (
                    herd.data_attr.get("feed.ration_GE")
                    * herd.data_attr.get("feed.consumption")
                    .T.groupby(["prod_system", "animal"])
                    .sum()
                    .T
                )

                if herd.species == "cattle":
                    # Calculate Ym (i.e. % of gross energy intake resulting in mtehane
                    # emissions) based on method presented in <<Bertilsson (2016) Updating
                    # Swedish emission factors for cattle to be used for calculations of
                    # greenhouse gases>> which is used in the Swedish NIR.

                    # Get fat in ration [g/kg DM]
                    fat_in_ration = herd.data_attr.get("feed.ration_fat")
                    # Get concentrate share [% of DM]
                    concentrate_share = herd.data_attr.get("feed.ration_rough")

                    # Calculate Ym (i.e. % of gross energy intake resulting in mtehane emissions)
                    Ym = -0.046 * concentrate_share + 7.1379

                    # Update values using specific method for cows
                    Ym.update(
                        (
                            (
                                1.39 * dry_matter_intake
                                - 0.091
                                * fat_in_ration
                                * herd.data_attr.get("heads")
                                * 365.25
                            )
                            / GE_intake
                            * 100
                        ).xs("cows", level="animal", axis=1, drop_level=False)
                    )
                else:
                    # Get specified Ym factors per animal
                    Ym = self.par.get_from_frame(
                        "Ym_enteric", herd.data_attr.get("heads")
                    )

                # Calculate enteric methane emissions from Ym and GE intake [kg CH4]
                enteric_methane = GE_intake * Ym / 100 / CH4_specific_energy

            else:
                # Calculate enteric fermentation based on EFs [kg CH4/animal/year] per animal

                # Calculate enteric methane emissions [kg CH4]
                enteric_methane = herd.data_attr.get("heads") * self.par.get_from_frame(
                    "EF_enteric", herd.data_attr.get("heads")
                )

            # Adjust based on enteric methane adjustment factor
            enteric_methane = enteric_methane * self.par.get_from_frame(
                "enteric_adj", enteric_methane
            )

            # Add compound to column index
            enteric_methane.columns = pd.MultiIndex.from_tuples(
                [(ps, an, "CH4bio") for ps, an in enteric_methane.columns],
                names=["prod_system", "animal", "compound"],
            )

            # Store enteric methane emissions [kg CH4]
            herd.data_attr.add(
                enteric_methane.fillna(0),
                name="enteric_methane",
                unit="kg/year",
                orig="FeedMgmt",
                desc="Emissions of methane (CH4) from enteric fermentation",
            )

            self.par.clear()
