import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr, multiply_aligned
from ..utils.misc import Container

class FeedMgmt():
    '''Class that that calculates ammount of 'crop products' or 'by-products' needed for a certain demand of 'feed'
    accounting far all losses between harvest/prouction and final cosnumption by the animals.
    
    Parameters
    ----------
    herds : (dict of) AnimalHerd object(s)
    par : ParameterRetriever object
    **kwargs: str or list
        Keyword arguments to be passed on as filters to the ParameterRetriever.
    '''

    def __init__(self,herds,par):
        
        self.par = par

        if isinstance(herds, pd.Series):
            self.herds = herds
        else:
             self.herds = pd.Series(
                data=herds,
                index=pd.MultiIndex.from_tuples(
                    [(herds.species,herds.breed,herds.prod_system,herds.sub_system)],
                    names=['species','breed','prod_system','sub_system']
                )
            )

        self.check_index()

        self.index = list(self.herds)[0].index

    def check_index(self):
        if len(self.herds)>0:
            for n in range(len(self.herds)-1):
                if (self.herds[n].index != self.herds[n+1].index).any():
                    raise Exception('Indexes does not match across herds!')

    

    def calculate(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='FeedMgmt')

        # Create feed objects in herds
        for herd in self.herds:
            if not hasattr(herd,'feed'):
                herd.feed = Feed()
        
        vprint('Calculating feed consumption ...')
        self.calculate_feed_consumption()
        vprint('Calculating demand for crop products ...')
        self.calculate_product_demand(of='crop_prod')
        vprint('Calculating demand for by-products ...')
        self.calculate_product_demand(of='by_prod')

        vprint(type='end')

    def calculate_feed_consumption(self):
        '''Calculates energy requirements per animal and from this + defined feed rations the total demand for feeds per animal.'''

        for herd in self.herds:
            
            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )

            # Get ouput production systems
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            # Get animals
            anis = herd.animals
            # Get feeds in rations from feeds listed in the parameters 'f_feed' column
            fes = herd.par.get_unique('feed')

            df_energy = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani) for ps in pss for ani in anis],
                    names=['prod_system','animal']
                    )
                )

            for ps in pss:
                for ani in anis:
                    herd.par.set(prod_system=ps, animal=ani)

                    # Calculate energy requirements per animal
                    E_req = np.atleast_1d(herd.calculate_feed_energy_req(ani))
                    if len(E_req) == 1:
                        E_req = E_req.repeat(len(herd.index))
                    
                    df_energy.loc[:,(ps,ani)] = E_req

            df_feeds = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,fe) for ps in pss for ani in anis for fe in fes],
                    names=['prod_system','animal','feed']
                    )
                )

            shares_per_feed = herd.par.get_from_frame('share_in_ration',df_feeds)/100
            E_per_feed = self.par.get_from_frame('feed_par_E',df_feeds)

            # Check so that ration shares add up to 100%
            if not np.isclose(shares_per_feed.groupby(['prod_system','animal'], axis=1).sum(),1).all():
                warnings.warn(f'\n\nAll feed ration shares did not add up to 100% for species: {herd.species}, breed: {herd.breed}. Feed rations were corrected.\n')
                shares_per_feed = (
                    shares_per_feed / 
                    shares_per_feed.groupby(['prod_system','animal'], axis=1).sum().align(shares_per_feed)[0]
                )
                
            # Calculate avg. energy in feed ration [MJ/kg DM]
            E_per_DM = (shares_per_feed * E_per_feed).groupby(['prod_system','animal'], axis=1).sum()

            # Calculate required DM 
            DM_req = df_energy / E_per_DM

            # Calculate and assign feed quantities [kg DM]
            df_feeds.loc[:,:] = shares_per_feed * DM_req.align(shares_per_feed)[0]

            df_energy = df_energy.multiply(herd.heads, axis=1)
            df_feeds = df_feeds.multiply(herd.heads, axis=1)

            herd.feed.energy_req = df_energy.reindex(columns=herd.heads.columns.get_level_values('prod_system').unique(), level='prod_system')
            herd.feed.consumption = df_feeds.reindex(columns=herd.animals, level='animal')
          
    def calculate_product_demand(self, of='crop_prod'):

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )

            # Get ouput production systems
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            # Get animals
            anis = herd.animals
            # Get feeds
            fes = herd.par.get_unique('feed')
            # Get crop/by products
            qry = 'f_feed.isin([' + ', '.join(f'"{fe}"' for fe in fes) + '])'
            prs = self.par.get_unique(of,qry=qry)

            result_df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ori,ps,ani,pr) for ori in ['domestic','regional','imported'] for ps in pss for ani in anis for pr in prs],
                    names=['origin','prod_system','animal',of]
                    )
                )
            
            if min(result_df.shape)>0:
                retrieve_df = pd.DataFrame(
                    index = herd.index,
                    columns = pd.MultiIndex.from_tuples(
                        [(ps,ani,pr,fe) for ps in pss for ani in anis for pr in prs for fe in fes],
                        names=['prod_system','animal',of,'feed']
                        )
                    )

                feed_to_prod = self.par.get_from_frame('feed_to_prod',retrieve_df)
                feed_to_imp_prod = feed_to_prod * self.par.get_from_frame('share_imported',retrieve_df)/100
                feed_to_dom_prod = feed_to_prod - feed_to_imp_prod
                feed_to_reg_prod = feed_to_dom_prod * self.par.get_from_frame('share_regional',retrieve_df)/100

                result_df.loc[:,('domestic')] = pd.concat(
                    {'domestic': (
                        multiply_aligned(feed_to_dom_prod,herd.feed.consumption)
                        .groupby(['prod_system','animal',of],sort=False,axis=1).sum()
                    )},
                    names=['origin'],
                    axis=1
                )

                result_df.loc[:,('regional')] = pd.concat(
                    {'regional': (
                        multiply_aligned(feed_to_reg_prod,herd.feed.consumption)
                        .groupby(['prod_system','animal',of],sort=False,axis=1).sum()
                    )},
                    names=['origin'],
                    axis=1
                )

                result_df.loc[:,('imported')] = pd.concat(
                    {'imported': (
                        multiply_aligned(feed_to_imp_prod,herd.feed.consumption)
                        .groupby(['prod_system','animal',of],sort=False,axis=1).sum()
                    )},
                    names=['origin'],
                    axis=1
                )

            # Set attribute feed.<crop/by>_product_demand
            rsetattr(herd,'feed.'+of+'uct_demand',result_df)


class Feed(Container):
    '''Class to store feed attributes in AnimalHerd obejcts'''
    