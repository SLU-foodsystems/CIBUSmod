from itertools import product
from typing import Literal
import warnings
import pandas as pd
import numpy as np

from CIBUSmod.main_modules.animal_herd import AnimalHerd
from CIBUSmod.utils.retriever import ParameterRetriever

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned, fix_herds

class FeedMgmt():
    '''
    Class that that calculates amount of 'crop products' or 'by-products' needed for a
    certain demand of 'feed' accounting far all losses between harvest/prouction and
    final consumption by the animals.

    Parameters
    ----------
    herds: (pandas.Series of) AnimalHerd object(s)
    par: ParameterRetriever object
    '''

    def __init__(self, herds: AnimalHerd | list | pd.Series, par: ParameterRetriever):
        self.par = par
        self.herds: pd.Series = fix_herds(herds)
        self.index = list(self.herds)[0].index

    def check_index(self):
        if len(self.herds) == 0:
            return
        for n in range(len(self.herds)-1):
            if (self.herds[n].index != self.herds[n+1].index).any():
                raise Exception('Indexes does not match across herds!')

    def calculate(self, verbose=False):
        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='FeedMgmt')

        vprint('Calculating feed consumption ...')
        self.calculate_feed_consumption()

        vprint('Calculating feed losses ...')
        self.calculate_losses()

        vprint('Calculating demand for crop products ...')
        self.calculate_product_demand(of='crop_prod')
        self.calculate_max_crop_in_crop_prod()

        vprint('Calculating demand for by-products ...')
        self.calculate_product_demand(of='by_prod')

        vprint('Calculating demand for crop residues ...')
        self.calculate_product_demand(of='crop_resid')

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
        '''Calculates energy requirements per animal and from this + defined feed
        rations the total demand for feeds per animal.'''

        for herd in self.herds:
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
            shares_per_feed = herd.par.get_from_frame('share_in_ration', df_feeds).fillna(0) / 100

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
                E_per_feed = self.par.get_from_frame(
                    "feed_composition", df_feeds, feed_par=fp
                )

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

    def calculate_ration_characteristics(self):
        '''Calculates ration characteristics'''
        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )

            feed_DM = herd.data_attr.get('feed.consumption')
            # Get dry matter intake [kg DM]
            ration_DM = (
                feed_DM
                .T.groupby(['prod_system','animal']).sum().T
            )

            if herd.species == 'cattle':
                pars = ['N','P','K','ASH','GE','DE','fat']
            elif herd.species == 'sheep':
                pars = ['N','P','K','ASH','GE','DE']
            elif herd.species == 'horses':
                pars = ['N','P','K','ASH','GE','DE']
            elif herd.species == 'pigs':
                pars = ['N','P','K','ASH','GE','DE']
            elif herd.species == 'poultry':
                pars = ['N','P','K','ASH','GE','AME']
            else:
                raise ValueError(f"Received unexpected species of herd: {herd.species}")

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
                    name = 'feed.ration_'+par,
                    unit = 'MJ' if par in ['GE','DE'] else '%',
                    orig = 'FeedMgmt',
                    desc = 'Ration '+par+' content',
                    scalable = False
                )


    def calculate_losses(self):
        '''Calculate feeds lost during storage and feeding and demand for feed products entering on-farm storage.
        '''
        for herd in self.herds:

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


    def calculate_product_demand(
        self, of: Literal["crop_prod", "crop_resid", "by_prod"]
    ):
        # Get Series of crop/by- products or crop residues with feed as index
        prs = self.par.get_unique(['feed',of]).set_index('feed')[of]

        if (of != 'crop_prod') and (len(self.par.get_unique(of, 'parameter == "share_imported"')) > 0):
            warnings.warn(f"FeedMgmt: Spicified import shares for '{of}' will have no effect! For by-products, imports are handled in the ByProductMgmt module and for crop residues, imports are not allowed.")

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Construct retrieve dataframe columns
            df_cols = []
            for ps, ani, fe in herd.data_attr.get('feed.demand').columns:
                if fe in prs.index:
                    for pr in prs[fe] if not isinstance(prs[fe], str) else [prs[fe]]:
                        df_cols += [(ps, ani, pr, fe)]

            retrieve_df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    df_cols,
                    names=['prod_system', 'animal', of, 'feed']
                    ),
                dtype = float
            )

            if retrieve_df.shape[1] > 0:

                # Get factors feed --> product
                feed_to_prod = self.par.get_from_frame('feed_to_prod', retrieve_df)

                if of == 'crop_prod':

                    # Get import shares
                    feed_to_imp_prod = feed_to_prod * self.par.get_from_frame('share_imported', retrieve_df)/100
                    feed_to_dom_prod = feed_to_prod - feed_to_imp_prod

                    # Calculate domestic and imported crop product demand
                    dom_prod = (
                        multiply_aligned(feed_to_dom_prod,herd.data_attr.get('feed.demand'))
                        .T.groupby(['prod_system','animal',of], sort=False).sum().T
                    )
                    imp_prod = (
                        multiply_aligned(feed_to_imp_prod,herd.data_attr.get('feed.demand'))
                        .T.groupby(['prod_system','animal',of],sort=False).sum().T
                    )

                    # Combine
                    result_df = pd.concat([
                        pd.concat(
                            {'domestic': (dom_prod)},
                            names=['origin'],
                            axis=1
                        ),

                        pd.concat(
                            {'imported': (imp_prod)},
                            names=['origin'],
                            axis=1
                        )
                    ], axis=1)

                    # Calculate regional demand for crop products
                    feed_to_reg_prod = feed_to_dom_prod * self.par.get_from_frame('share_regional', retrieve_df)/100
                    result_df_reg = (
                        multiply_aligned(feed_to_reg_prod,herd.data_attr.get('feed.demand'))
                        .T.groupby(['prod_system','animal',of], sort=False).sum().T
                    )
                else:
                    # Calculate demand
                    result_df = (
                        multiply_aligned(feed_to_prod,herd.data_attr.get('feed.demand'))
                        .T.groupby(['prod_system','animal',of],sort=False).sum().T
                    )
            else:
                # Empty result dataframes
                result_df = pd.DataFrame(
                    index = herd.index,
                    columns = pd.MultiIndex.from_tuples(
                        [],
                        names = ['origin', 'prod_system', 'animal', of] if of == 'crop_prod'
                        else ['prod_system', 'animal', of]
                    ),
                    dtype = float
                )
                result_df_reg = pd.DataFrame(
                    index = herd.index,
                    columns = pd.MultiIndex.from_tuples(
                        [],
                        names = ['prod_system', 'animal', of]
                    ),
                    dtype = float
                )

            # Add data attributes (drop zero cols)
            result_df = result_df.loc[:, result_df.sum() > 0]
            herd.data_attr.add(
                result_df,
                name = 'feed.' + of + ('ue_demand' if of == 'crop_resid' else 'uct_demand'),
                unit = 'kg DM/year' if of == 'crop_resid' else 'kg/year',
                orig = 'FeedMgmt',
                desc = f'Demand for {"crop products" if of == "crop_prod" else "by-products" if of=="by_prod" else "crop residues"} for feed'
            )

            if of == 'crop_prod':
                result_df_reg = result_df_reg.loc[:, result_df_reg.sum() > 0]
                herd.data_attr.add(
                    result_df_reg,
                    name = 'feed.regional_crop_product_demand',
                    unit = 'kg/year',
                    orig = 'FeedMgmt',
                    desc = 'Demand for crop products for feed that must be supplied regionally'
                )

    def calculate_max_crop_in_crop_prod(self):
        idx = pd.IndexSlice

        # Get crop_groups to handle
        cgs = (
            self.par.get_unique(['crop_group','crop_prod'], qry='parameter == "max_crop_in_crop_prod"')
            .set_index('crop_prod')['crop_group']
        )

        for herd in self.herds:
            if herd.data_attr.get('feed.crop_product_demand').columns.get_level_values('crop_prod').isin(cgs.index).any():

                # Set species and breed filters for ParameterRetriever
                self.par.clear()
                self.par.set(
                    species = herd.species,
                    breed = herd.breed,
                    sub_system = herd.sub_system
                )

                # Get crop product demand supplied by crops in crs
                df = herd.data_attr.get('feed.crop_product_demand').loc[:,idx['domestic',:,:,cgs.index]]

                # Append crops to column index
                df.columns = pd.MultiIndex.from_tuples(map(lambda x: (x + tuple([cgs[x[-1]]])), df.columns), names = df.columns.names + ['crop_group'])

                # Calculate maximum supply of crop_prod from crop
                res = (df * self.par.get_from_frame(
                    'max_crop_in_crop_prod',
                    df,
                    species=herd.species,
                    breed=herd.breed,
                    sub_system=herd.sub_system
                )).T.groupby(['prod_system','crop_prod','crop_group']).sum().T/100

                # Add data attribute
                herd.data_attr.add(
                    res,
                    name = 'feed.max_supply_from_crop_group',
                    unit = 'kg/year',
                    orig = 'FeedMgmt',
                    desc = 'Maximum supply of feed from crop_group. Used in GeoDistributor constraint'
                )

            else:
                # Add data attribute
                herd.data_attr.add(
                    None,
                    name = 'feed.max_supply_from_crop_group',
                    unit = 'kg/year',
                    orig = 'FeedMgmt',
                    desc = 'Maximum supply of feed from crop_group. Used in GeoDistributor constraint'
                )

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

                # Get dry matter intake [kg DM]
                dry_matter_intake = (
                    herd.data_attr.get('feed.consumption')
                    .T.groupby(['prod_system','animal']).sum().T
                )


                # Get gross energy intake [MJ]
                GE_intake = herd.data_attr.get('feed.ration_GE') *  herd.data_attr.get('feed.consumption').T.groupby(['prod_system','animal']).sum().T

                if herd.species == 'cattle':
                    # Calculate Ym (i.e. % of gross energy intake resulting in mtehane
                    # emissions) based on method presented in <<Bertilsson (2016) Updating
                    # Swedish emission factors for cattle to be used for calculations of
                    # greenhouse gases>> which is used in the Swedish NIR.

                    # Get fat in ration [g/kg DM]
                    fat_in_ration = herd.data_attr.get('feed.ration_fat')

                    # FUTURE FIX: Not ideal that roughage feeds are hardcoded here... Use relation_tables instead??
                    rough_feeds = {'ley silage, 1st cut','ley silage, regrowth','other silage','maize silage','grazing'}
                    sel_rough = list(set(herd.data_attr.get('feed.consumption').columns.get_level_values('feed')).intersection(rough_feeds))

                    # Calculate concentrate share [% of DM]
                    concentrate_share = 100 - (
                        (
                            herd.data_attr.get('feed.consumption')
                            .loc[:,idx[:,:,sel_rough]]
                            .T.groupby(['prod_system','animal']).sum().T
                        ) / dry_matter_intake.replace({0:np.nan}) # To avoid div by 0 error
                    ) * 100


                    # Calculate Ym (i.e. % of gross energy intake resulting in mtehane emissions)
                    Ym = (-0.046 * concentrate_share + 7.1379)

                    # Update values using specific method for cows
                    Ym.update(
                        (
                            (1.39 * dry_matter_intake - 0.091 * fat_in_ration * herd.data_attr.get('heads') * 365.25)
                            / GE_intake * 100
                        ).xs('cows', level='animal', axis=1, drop_level=False)
                    )
                else:
                    # Get specified Ym factors per animal
                    Ym = self.par.get_from_frame('Ym_enteric',herd.data_attr.get('heads'))

                # Calculate enteric methane emissions from Ym and GE intake [kg CH4]
                enteric_methane = GE_intake * Ym/100 / CH4_specific_energy

            else:
                # Calculate enteric fermentation based on EFs [kg CH4/animal/year] per animal

                # Calculate enteric methane emissions [kg CH4]
                enteric_methane = herd.data_attr.get('heads') * self.par.get_from_frame('EF_enteric', herd.data_attr.get('heads'))

            # Adjust based on enteric methane adjustment factor
            enteric_methane = enteric_methane * self.par.get_from_frame('enteric_adj', enteric_methane)

            # Add compound to column index
            enteric_methane.columns = pd.MultiIndex.from_tuples(
                [(ps,an,'CH4bio') for ps,an in enteric_methane.columns],
                names = ['prod_system', 'animal', 'compound']
            )

            # Store enteric methane emissions [kg CH4]
            herd.data_attr.add(
                enteric_methane.fillna(0),
                name = 'enteric_methane',
                unit = 'kg/year',
                orig = 'FeedMgmt',
                desc = 'Emissions of methane (CH4) from enteric fermentation'
            )

            self.par.clear()
