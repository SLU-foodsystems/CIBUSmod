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
    herds : (pandas.Series of) AnimalHerd object(s)
    par : ParameterRetriever object
    '''

    def __init__(self,herds,par):
        
        self.par = par

        if isinstance(herds, pd.Series):
            self.herds = herds
        else:
             if not isinstance(herds, list):
                 herds = [herds]
             self.herds = pd.Series(
                data=herds,
                index=pd.MultiIndex.from_tuples(
                    [(h.species,h.breed,h.prod_system,h.sub_system) for h in herds],
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
        vprint('Calculating feed losses ...')
        self.calculate_losses()
        vprint('Calculating demand for crop products ...')
        self.calculate_product_demand(of='crop_prod')
        self.calculate_max_crop_in_crop_prod()
        vprint('Calculating demand for by-products ...')
        self.calculate_product_demand(of='by_prod')

        vprint(type='end')

    def calculate2(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='FeedMgmt')

        vprint('Adjusting feed rations (not implemented) ...')
        self.redistribute_feeds()
        vprint('Calculating enteric methane emissions ...')
        self.calculate_enteric_methane()

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

            # Create dataframe to store feed req.
            df_feeds = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,fe) for ps in pss for ani in anis for fe in fes],
                    names=['prod_system','animal','feed']
                    )
                )

            # Get feed rations
            shares_per_feed = herd.par.get_from_frame('share_in_ration',df_feeds)/100

            # Check so that ration shares add up to 100%
            if not np.isclose(shares_per_feed.groupby(['prod_system','animal'], axis=1).sum(),1).all():
                warnings.warn(f'\n\nAll feed ration shares did not add up to 100% for species: {herd.species}, breed: {herd.breed}. Feed rations were corrected.\n')
                shares_per_feed = (
                    shares_per_feed / 
                    shares_per_feed.groupby(['prod_system','animal'], axis=1).sum().align(shares_per_feed)[0]
                )

            if hasattr(herd,'calculate_feed_energy_req'):
                # If herd has a method to calculate energy requirements of animals
                # energy requirements are calculated and feed demand is calculated
                # from this and supplied feed rations.
                df_energy = pd.DataFrame(
                    index = herd.index,
                    columns = pd.MultiIndex.from_tuples(
                        [(ps,ani) for ps in pss for ani in anis],
                        names=['prod_system','animal']
                        )
                    )

                for ps in pss:
                    for ani in anis:
                        herd.par.clear()
                        herd.par.set(
                            species = herd.species,
                            breed = herd.breed,
                            prod_system = ps,
                            animal = ani,
                            **herd.index.to_frame().to_dict('list')
                        )

                        # Calculate energy requirements per animal
                        E_req = np.atleast_1d(herd.calculate_feed_energy_req(ani))
                        if len(E_req) == 1:
                            E_req = E_req.repeat(len(herd.index))

                        df_energy.loc[:,(ps,ani)] = E_req

                # Get energy content of feeds [MJ/kg DM]
                E_per_feed = self.par.get_from_frame('feed_par_E',df_feeds)

                # Calculate avg. energy in feed ration [MJ/kg DM]
                E_per_DM = (shares_per_feed * E_per_feed).groupby(['prod_system','animal'], axis=1).sum()

                # Calculate required DM 
                DM_req = df_energy / E_per_DM

                df_energy = df_energy.multiply(herd.heads, axis=1)
                herd.feed.energy_req = df_energy.reindex(columns=herd.heads.columns.get_level_values('prod_system').unique(), level='prod_system')
                herd.data_attr.update(['feed.energy_req'])
            else:
                # If herd does not have a method to calculate energy requirements of animals
                # the dry matter feed requirements are calculated from feed conversion ratios
                # or a fixed feed intake per animal.

                df_feed_req = pd.DataFrame(
                    index = herd.index,
                    columns = pd.MultiIndex.from_tuples(
                        [(ps,ani) for ps in pss for ani in anis],
                        names=['prod_system','animal']
                        )
                    )
                
                for ps in pss:
                    for ani in anis:
                        herd.par.clear()
                        herd.par.set(
                            species = herd.species,
                            breed = herd.breed,
                            prod_system = ps,
                            animal = ani,
                            **herd.index.to_frame().to_dict('list')
                        )

                        # Calculate feed requirements per animal
                        feed_req = np.atleast_1d(herd.calculate_feed_req(ani))
                        if len(feed_req) == 1:
                            feed_req = feed_req.repeat(len(herd.index))

                        df_feed_req.loc[:,(ps,ani)] = feed_req
                
                # Get DM content of feeds [kg DM/kg]
                DM_per_feed = self.par.get_from_frame('feed_par_DM',df_feeds)/100

                # Calculate avg. DM in feed ration [MJ/kg DM]
                ration_DM = (shares_per_feed * DM_per_feed).groupby(['prod_system','animal'], axis=1).sum()

                # Calculate required DM 
                DM_req = df_feed_req * ration_DM

            # Calculate and assign feed quantities [kg DM]
            df_feeds.loc[:,:] = shares_per_feed * DM_req.align(shares_per_feed)[0]

            df_feeds = df_feeds.multiply(herd.heads, axis=1)

            herd.feed.consumption = df_feeds.reindex(columns=herd.animals, level='animal')
            herd.data_attr.update(['feed.consumption'])

    def calculate_losses(self):
        '''Calculate feeds lost during storage and feeding and demand for feed products entering on-farm storage.
        '''
        for herd in self.herds:
            
            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )
            
            feed_to_feeding = herd.feed.consumption * ( 1 / ( 1 - self.par.get_from_frame('feeding_losses', herd.feed.consumption)/100 ) )
            feeding_losses = feed_to_feeding - herd.feed.consumption

            feed_to_storage = feed_to_feeding * ( 1 / ( 1 - self.par.get_from_frame('storage_losses', feed_to_feeding)/100 ) )
            storage_losses = feed_to_storage - feed_to_feeding

            herd.feed.demand = feed_to_storage
            herd.feed.storage_losses = storage_losses
            herd.feed.feeding_losses = feeding_losses
            herd.data_attr.update(['feed.demand','feed.storage_losses','feed.feeding_losses'])
          
    def calculate_product_demand(self, of='crop_prod'):

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Get ouput production systems
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            # Get animals
            anis = herd.animals
            # Get feeds
            fes = herd.par.get_unique('feed')
            # Get crop/by products
            qry = 'f_feed.isin([' + ', '.join(f'"{fe}"' for fe in fes) + '])'
            prs = self.par.get_unique(['feed',of],qry=qry).set_index('feed')[of]
            # Remove feeds not supplied by crop/by products
            fes = fes[np.isin(fes,prs.index)]

            result_df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ori,ps,ani,pr) for ori in ['domestic','regional','imported'] for ps in pss for ani in anis for pr in prs.unique()],
                    names=['origin','prod_system','animal',of]
                    )
                )
            
            if min(result_df.shape)>0:
                retrieve_df = pd.DataFrame(
                    index = herd.index,
                    columns = pd.MultiIndex.from_tuples(
                        [(ps,ani,prs[fe],fe) for ps in pss for ani in anis for fe in fes],
                        names=['prod_system','animal',of,'feed']
                        )
                    )

                feed_to_prod = self.par.get_from_frame('feed_to_prod',retrieve_df)
                feed_to_imp_prod = feed_to_prod * self.par.get_from_frame('share_imported',retrieve_df)/100
                feed_to_dom_prod = feed_to_prod - feed_to_imp_prod
                feed_to_reg_prod = feed_to_dom_prod * self.par.get_from_frame('share_regional',retrieve_df)/100

                result_df.loc[:,('domestic')] = pd.concat(
                    {'domestic': (
                        multiply_aligned(feed_to_dom_prod,herd.feed.demand)
                        .groupby(['prod_system','animal',of],sort=False,axis=1).sum()
                    )},
                    names=['origin'],
                    axis=1
                )

                result_df.loc[:,('regional')] = pd.concat(
                    {'regional': (
                        multiply_aligned(feed_to_reg_prod,herd.feed.demand)
                        .groupby(['prod_system','animal',of],sort=False,axis=1).sum()
                    )},
                    names=['origin'],
                    axis=1
                )

                result_df.loc[:,('imported')] = pd.concat(
                    {'imported': (
                        multiply_aligned(feed_to_imp_prod,herd.feed.demand)
                        .groupby(['prod_system','animal',of],sort=False,axis=1).sum()
                    )},
                    names=['origin'],
                    axis=1
                )

            # Set attribute feed.<crop/by>_product_demand
            rsetattr(herd,'feed.'+of+'uct_demand',result_df)
            herd.data_attr.update(['feed.'+of+'uct_demand'])

    def calculate_max_crop_in_crop_prod(self):
        idx = pd.IndexSlice

        # Get crops to handle
        crs = (
            self.par.get_unique(['crop','crop_prod'], qry='parameter == "max_crop_in_crop_prod"')
            .set_index('crop_prod')['crop']
        )

        for herd in self.herds:
            if herd.feed.crop_product_demand.columns.get_level_values('crop_prod').isin(crs.index).any():

                # Set species and breed filters for ParameterRetriever
                self.par.clear()
                self.par.set(
                    species = herd.species,
                    breed = herd.breed,
                    sub_system = herd.sub_system
                )

                # Get crop product demand supplied by crops in crs
                df = herd.feed.crop_product_demand.loc[:,idx['domestic',:,:,crs.index]]

                # Append crops to column index
                df.columns = pd.MultiIndex.from_tuples(map(lambda x: (x + tuple([crs[x[-1]]])), df.columns), names = df.columns.names + ['crop'])

                # Calculate maximum supply of crop_prod from crop
                res = (df * self.par.get_from_frame(
                    'max_crop_in_crop_prod',
                    df,
                    species=herd.species,
                    breed=herd.breed,
                    sub_system=herd.sub_system
                )).groupby(['prod_system','crop_prod','crop'], axis=1).sum()/100

                herd.feed.max_supply_from_crop = res
                herd.data_attr.update(['feed.max_supply_from_crop'])

            else:
                herd.feed.max_supply_from_crop = None
                herd.data_attr.update(['feed.max_supply_from_crop'])

    def redistribute_feeds(self):
        # IMPLEMENT METHOD TO REDISTRIBUTE FEEDS
        # IN ORDER TO ALIGN WITH GENERATED/IMPORTED
        # BY-PRODUCTS
        pass 
    
    def calculate_enteric_methane(self):

        idx = pd.IndexSlice

        for herd in self.herds:

            self.par.set(
                species=herd.species,
                breed=herd.breed,
                sub_system=herd.sub_system
            )

            if herd.species in ['cattle','sheep']:

                CH4_specific_energy = 55.6 # [MJ/kg]

                # Get gross energy intake [MJ]
                GE_intake = (
                    (
                        self.par.get_from_frame('feed_par_GE',herd.feed.consumption)
                        * herd.feed.consumption
                    )
                    .groupby(['prod_system','animal'], axis=1).sum()
                )

                if herd.species == 'cattle':
                    # Calculate Ym (i.e. % of gross energy intake resulting in mtehane
                    # emissions) based on method presented in <<Bertilsson (2016) Updating
                    # Swedish emission factors for cattle to be used for calculations of
                    # greenhouse gases>> which is used in the Swedish NIR.

                    # Get dry matter intake [kg DM]
                    dry_matter_intake = (
                        herd.feed.consumption
                        .groupby(['prod_system','animal'], axis=1)
                        .sum()
                    )

                    # Get fat in ration [g/kg DM]
                    fat_in_ration = (
                        (
                            self.par.get_from_frame('feed_par_fat',herd.feed.consumption)
                            * herd.feed.consumption
                        )
                        .groupby(['prod_system','animal'], axis=1).sum()
                        / dry_matter_intake
                    )

                    sel_rough = ['ley silage, 1st cut','ley silage, regrowth','other silage','maize silage','grazing']

                    # Calculate concentrate share [% of DM]
                    concentrate_share = 100 - (
                        (
                            herd.feed.consumption
                            .loc[:,idx[:,:,sel_rough]]
                            .groupby(['prod_system','animal'], axis=1)
                            .sum()
                        ) / dry_matter_intake
                    ) * 100
                    

                    # Calculate Ym (i.e. % of gross energy intake resulting in mtehane emissions)
                    Ym = (-0.046 * concentrate_share + 7.1379)

                    # Update values using specific method for cows
                    Ym.update(
                        (
                            (1.39 * dry_matter_intake - 0.091 * fat_in_ration * herd.heads * 365.25)
                            / GE_intake * 100
                        ).xs('cows', level='animal', axis=1, drop_level=False)
                    )
                else:
                    # Get specified Ym factors per animal
                    Ym = self.par.get_from_frame('Ym_enteric',herd.heads)

                # Calculate enteric methane emissions from Ym and GE intake [kg CH4]
                enteric_methane = GE_intake * Ym/100 / CH4_specific_energy

            else:
                # Calculate enteric fermentation based on EFs [kg CH4/animal/year] per animal
                
                # Calculate enteric methane emissions [kg CH4]
                enteric_methane = herd.heads * self.par.get_from_frame('EF_enteric', herd.heads)

            # Store enteric methane emissions [kg CH4]
            herd.enteric_methane = enteric_methane.fillna(0)
            herd.data_attr.update(['enteric_methane'])

            self.par.clear()



class Feed(Container):
    '''Class to store feed attributes in AnimalHerd obejcts'''
    