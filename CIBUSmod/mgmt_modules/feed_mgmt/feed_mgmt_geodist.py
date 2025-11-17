from itertools import product
import warnings
import pandas as pd
import numpy as np

from ...utils.verbose_print import verbose_init
from ...utils.misc import multiply_aligned

from .feed_mgmt import FeedMgmt

class GeoDistFeedMgmt(FeedMgmt):
    '''
    Class that that calculates amount of 'crop products' or 'by-products' needed for a
    certain demand of 'feed' accounting far all losses between harvest/prouction and
    final consumption by the animals.

    Parameters
    ----------
    herds : (pandas.Series of) AnimalHerd object(s)
    par : ParameterRetriever object
    '''
    def calculate(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='FeedMgmt')

        vprint('Calculating feed consumption ...')
        self.calculate_feed_consumption()

        vprint('Calculating feed losses ...')
        self.calculate_losses()

        not_linked_feeds = []
        vprint('Calculating demand for crop products ...')
        not_linked_feeds += [self.calculate_product_demand(prod_type='crop_prod')]
        self.calculate_max_crop_in_crop_prod()

        vprint('Calculating demand for by-products ...')
        not_linked_feeds += [self.calculate_product_demand(prod_type='by_prod')]

        vprint('Calculating demand for crop residues ...')
        not_linked_feeds += [self.calculate_product_demand(prod_type='crop_resid')]

        # Check for feeds not linked to any product
        not_linked_feeds = list(set(not_linked_feeds[0]).intersection(*not_linked_feeds[1:]))
        if len(not_linked_feeds)>0:
            warnings.warn(f'Some feeds were not linked to any product: {not_linked_feeds}')

        vprint(type='end')

    def calculate2(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='FeedMgmt')

        vprint('Adjusting feed rations (not implemented) ...')
        self.redistribute_feeds()

        vprint('Calculating feed ration characteristics ...')
        self.calculate_ration_characteristics()

        vprint('Calculating enteric methane emissions ...')
        self.calculate_enteric_methane()

        vprint(type='end')


    def calculate_feed_consumption(self):
        '''Calculates energy requirements per animal and from this + defined feed rations the total demand for feeds per animal.'''

        for herd in (h for h in self.herds if 'heads' in h.data_attr):

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )

            # Get ouput production systems
            pss = herd.data_attr.get('heads').columns.get_level_values('prod_system').unique()
            # Get animals
            anis = herd.animals
            # Get feeds in rations from feeds listed in the parameters 'f_feed' column
            fes = herd.par.get_unique('feed')

            # Create dataframe to store feed req.
            df_feeds = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    list(product(pss, anis, fes)),
                    names=['prod_system','animal','feed']
                ),
                dtype = float
            )

            # Get feed rations
            shares_per_feed = herd.par.get_from_frame('share_in_ration',df_feeds)/100

            # Check so that ration shares add up to 100%
            if not np.isclose(shares_per_feed.T.groupby(['prod_system','animal']).sum(),1).all():
                warnings.warn(f'\n\nAll feed ration shares did not add up to 100% for species: {herd.species}, breed: {herd.breed}. Feed rations were corrected.\n')
                shares_per_feed = (
                    shares_per_feed /
                    shares_per_feed.T.groupby(['prod_system','animal']).sum().T.align(shares_per_feed)[0]
                )

            # Get feed_param that decides feed requirements
            fp = herd.data_attr.get('feed_req_eq').columns.unique('feed_par')[0]

            # If herd has feed energy requirements calculate dry
            # matter requirements from energy requirements
            if fp != 'DM':

                # Get energy content of feeds [MJ/kg DM]
                E_per_feed = self.par.get_from_frame('feed_composition',df_feeds, feed_par = fp)

                # Calculate avg. energy in feed ration [MJ/kg DM]
                E_per_DM = (shares_per_feed * E_per_feed).T.groupby(['prod_system','animal']).sum().T

                # Calculate required DM
                feed_DM_req = herd.data_attr.get('feed_req_eq').xs(fp, level='feed_par', axis=1) / E_per_DM
            else:
                # Get DM requirements
                feed_DM_req = herd.data_attr.get('feed_req_eq').xs(fp, level='feed_par', axis=1)

            # Calculate and assign feed quantities [kg DM]
            df_feeds.loc[:,:] = multiply_aligned(shares_per_feed, feed_DM_req)

            # Add data attribute (drop zero cols)
            df_feeds = df_feeds.loc[:, df_feeds.sum() > 0]
            herd.data_attr.add(
                df_feeds,
                name = 'feed.consumption',
                unit = 'kg DM/year',
                orig = 'FeedMgmt',
                desc = 'Total feed consumption per feed'
            )

    def calculate_losses(self):
        '''Calculate feeds lost during storage and feeding and demand for feed products entering on-farm storage.
        '''
        for herd in (h for h in self.herds if 'heads' in h.data_attr):

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )

            feed_to_feeding = herd.data_attr.get('feed.consumption') * ( 1 / ( 1 - self.par.get_from_frame('feeding_losses', herd.data_attr.get('feed.consumption'))/100 ) )
            feeding_losses = feed_to_feeding - herd.data_attr.get('feed.consumption')

            feed_to_storage = feed_to_feeding * ( 1 / ( 1 - self.par.get_from_frame('storage_losses', feed_to_feeding)/100 ) )
            storage_losses = feed_to_storage - feed_to_feeding

            # Add data attributes
            herd.data_attr.add(
                feed_to_storage,
                name = 'feed.demand',
                unit = 'kg DM/year',
                orig = 'FeedMgmt',
                desc = 'Demand for feed after accounting for storage and feeding losses'
            )
            herd.data_attr.add(
                storage_losses,
                name = 'feed.storage_losses',
                unit = 'kg DM/year',
                orig = 'FeedMgmt',
                desc = 'Losses of feed during storage'
            )
            herd.data_attr.add(
                feeding_losses,
                name = 'feed.feeding_losses',
                unit = 'kg DM/year',
                orig = 'FeedMgmt',
                desc = 'Losses of feed during feeding'
            )


