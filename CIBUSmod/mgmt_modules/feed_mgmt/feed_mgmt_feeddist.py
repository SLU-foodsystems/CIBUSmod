import warnings

from ...utils.verbose_print import verbose_init

from .feed_mgmt import FeedMgmt

class FeedDistFeedMgmt(FeedMgmt):
    """
    Class that that calculates amount of 'crop products' or 'by-products' needed for a
    certain demand of 'feed' accounting far all losses between harvest/prouction and
    final consumption by the animals.

    Parameters
    ----------
    herds: (pandas.Series of) AnimalHerd object(s)
    par: ParameterRetriever object
    """

    def calculate(self, verbose=False):
        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str="FeedMgmt")

        vprint("Calculating feed consumption and losses ...")
        self.calculate_consumption_and_losses()

        not_linked_feeds = []
        vprint("Calculating demand for crop products ...")
        not_linked_feeds += [self.calculate_product_demand(prod_type="crop_prod")]
        self.calculate_max_crop_in_crop_prod()

        vprint("Calculating demand for by-products ...")
        not_linked_feeds += [self.calculate_product_demand(prod_type="by_prod")]

        vprint("Calculating demand for crop residues ...")
        not_linked_feeds += [self.calculate_product_demand(prod_type="crop_resid")]

        # Check for feeds not linked to any product
        not_linked_feeds = list(set(not_linked_feeds[0]).intersection(*not_linked_feeds[1:]))
        if len(not_linked_feeds)>0:
            warnings.warn(f'Some feeds were not linked to any product: {not_linked_feeds}')

        vprint("Calculating feed ration characteristics ...")
        self.calculate_ration_characteristics()

        vprint("Calculating enteric methane emissions ...")
        self.calculate_enteric_methane()

        vprint(type="end")

    def calculate_consumption_and_losses(self):
        """
        Calculate feeds lost during storage and feeding, as well consumption of feed
        products from feed demand, assigning it to the herd objects
        """

        for herd in self.herds:
            # Set species and breed filters for ParameterRetriever
            self.par.set(species=herd.species, breed=herd.breed)

            # Get the base demand, i.e. the net feed amount, before losses are made
            feed_demand = herd.data_attr.get("feed.demand")

            # Percentage points feeding-losses for each feed, e.g. 5 %
            pp_feeding_losses = (
                self.par.get_from_frame("feeding_losses", feed_demand) / 100
            )
            # Percentage points storage-losses for each feed, e.g. 10 %
            pp_storage_losses = (
                self.par.get_from_frame("storage_losses", feed_demand) / 100
            )

            # Compute the amounts of losses, as well as the total consumption
            storage_losses = feed_demand * pp_storage_losses
            feeding_losses = (feed_demand - storage_losses) * pp_feeding_losses

            # total consumption = demand * (1-pp_feeding) * (1-pp_storage)
            feed_consumption = feed_demand - feeding_losses - storage_losses

            # Add data attributes
            herd.data_attr.add(
                feed_consumption,
                name="feed.consumption",
                unit="kg DM/year",
                orig="FeedMgmt",
                desc="Demand for feed after accounting for storage and feeding losses",
            )
            herd.data_attr.add(
                storage_losses,
                name="feed.storage_losses",
                unit="kg DM/year",
                orig="FeedMgmt",
                desc="Losses of feed during storage",
            )
            herd.data_attr.add(
                feeding_losses,
                name="feed.feeding_losses",
                unit="kg DM/year",
                orig="FeedMgmt",
                desc="Losses of feed during feeding",
            )


