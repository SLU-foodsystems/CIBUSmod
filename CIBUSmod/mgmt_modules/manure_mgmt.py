import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned, fix_herds

if TYPE_CHECKING:
    from ..utils.retriever import ParameterRetriever
    from .feed_mgmt import FeedMgmt

class ManureMgmt():
    '''Management module that calculates animal manure excretion and losses.
    
    Parameters
    ----------
    herds : pd.Series of AnimalHerd objects
    feed_mgmt : FeedMgmt object
    par : ParameterRetriever object
    settings : dict
        Dict with <setting name> : <value>
        Allowed settings are:
            'MMS_grazing_from_feed' : bool, default False
                If True, use share of dry matter in ration
                from grazing to estimate the share of manure
                deposited on pasture
            'NPK_excretion_from_balance' : bool, default False
                If True, calculate N, P and K excretion from nutrients
                in feed minus nutrients incorporated in live weight gain
                and animal products. Only applies to AnimalHerds where
                the data attribute 'lwg' is calculated and relise on
                parameters '<N/P/K>_in_WL' and '<N/P/K>_in_prod'.
                If False, and for AnimalHerds where 'lwg' is not calculated
                N, P and K excretion is calculated from the parameter
                'manure_excr_<N/P/K>' 
    '''

    def __init__(
            self,
            herds: pd.Series,
            feed_mgmt: "FeedMgmt",
            par: "ParameterRetriever",
            settings = {}
        ):
        
        self.par = par

        # Default settings
        self.settings = {
            'MMS_grazing_from_feed' : False,
            'NPK_excretion_from_balance' : False
        }
        # Update settings if valid input
        for k,v in settings.items():
            if k in self.settings:
                if type(v) is type(self.settings[k]):
                    self.settings.update({k:v})
                else:
                    raise TypeError(f'Expected {type(self.settings[k])} for setting "{k}"')
            else:
                raise ValueError(f'"{k}" is not a valid setting')

        self.herds = fix_herds(herds)
        self.feed_mgmt = feed_mgmt

        self.index = list(self.herds)[0].index
    
    def calculate(
            self,
            verbose=False
            ):
        '''Calculates all manure excretion and losses.

        Parameters
        ----------
        verbose : bool, default True
            Print progress messages

        Returns
        -------
        Nothing. Stores output in pandas.DataFrames in the attrubutes: '<element>_excr', '<element>_loss' and '<element>_to_spread'.
        <element> is VS (volatile solids), N (nitrogen), P (phosphorous) and K (potassium).
        '''

        # Define function to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='ManureMgmt')

        # Calculate manure management system (MMS) shares
        vprint('Calculating MMS shares ...')
        self.calculate_MMS_shares()

        # Calculate bedding material use
        vprint('Calculating bedding material use ...')
        self.calculate_bedding_material_use()
      
        # Volatile solids (VS)
        vprint('Calculating VS excretion ...')
        self.calculate_VS_excretion()
        vprint('Calculating VS losses ...')
        self.calculate_VS_losses()

        # Nitrogen (N), Phosphorous (P) and Potassioum (K)
        vprint('Calculating manure excretion ...')
        self.calculate_NPK_excretion(element = 'N')
        self.calculate_NPK_excretion(element = 'P')
        self.calculate_NPK_excretion(element = 'K')

        vprint('Calculating manure losses ...')
        self.calculate_NPK_losses(element = 'N')
        self.calculate_NPK_losses(element = 'P')
        self.calculate_NPK_losses(element = 'K')

        vprint(type='end')

    def calculate_MMS_shares(self):
        '''Calculate share of manure per manure management system (MMS). MMS = 'grazing' is calculated from the share of
        dry matter feed intake from grazing and the 'share_feed_on_pasture' parameter. Other MMS shares are defined with the
        MMS_share parameter'''

        # Clear ParameterRetriever filters 
        self.par.clear()

        # Get defined manure management systems
        mmss = list(self.par.get_unique('MMS', qry="parameter == 'mms_share'"))

        for herd in self.herds:
            
            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )
            
            heads = herd.data_attr.get('heads')
            
            # Create dataframe
            mms_shares = pd.DataFrame(
                0.0,
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [tuple(list(i) + [mms]) for i in heads.columns for mms in mmss + ['grazing']],
                    names = heads.columns.names + ['MMS']
                )
            )
            
            # Get MMS shares (excl. grazing) according to parameters
            mms_shares.update(
                self.par.get_from_frame(
                    'mms_share',
                    mms_shares.loc[:,(slice(None), slice(None), mmss)]
                )/100
            )
            
            # Check that MMS shares add up to 100%, if not warn and correct
            if not np.isclose(mms_shares.T.groupby(['prod_system','animal']).sum().T, 1).all():
                warnings.warn(f'MMS shares did not add up to 100% for species: {herd.species}, breed: {herd.breed}. MMS shares were corrected.')
                mms_shares = mms_shares.T.groupby(['prod_system','animal']).transform(lambda x: x/x.sum()).T
            
            if self.settings['MMS_grazing_from_feed']:
                # Calculate share in MMS='grazing' from share of dry matter feed intake from grazing
                
                # Get feed intake
                feed_cons = herd.data_attr.get('feed.consumption')
                if 'grazing' in feed_cons.columns.get_level_values('feed'):

                    # Calculate share of feed intake from grazing
                    grazing_share = (
                        feed_cons.xs('grazing', level='feed', axis=1) /
                        feed_cons.T.groupby(['prod_system','animal']).sum().T.replace({0:np.nan})
                    )

                    # Add contribution of other feeds fed while on pasture
                    grazing_share += (
                        (100 - grazing_share) *
                        self.par.get_from_frame('share_feed_on_pasture', grazing_share)
                    )

                    # Add MMS column level
                    grazing_share = (
                        pd.concat({'grazing': grazing_share}, names=['MMS'], axis=1)
                        .reorder_levels(['prod_system','animal','MMS'], axis=1)
                    )

                    # Update DataFrame
                    mms_shares.update(grazing_share)
                    
            else:
                # Calculate share in MMS='grazing' from 'grazing_period' parameter

                if 'grazing_period' in herd.par.data.index.get_level_values('parameter'):

                    # Clear and set filters
                    herd.par.clear()
                    herd.par.set(
                        species = herd.species,
                        breed = herd.breed,
                        sub_system = herd.sub_system
                    )

                    # Get grazing share
                    grazing_share = (
                        (herd.par.get_from_frame('grazing_period', herd.data_attr.get('heads')) / 12) *
                        (1 - herd.par.get_from_frame('indoors_during_grazing', herd.data_attr.get('heads'))/100)
                    )

                    # Add MMS column level
                    grazing_share = (
                        pd.concat({'grazing': grazing_share}, names=['MMS'], axis=1)
                        .reorder_levels(['prod_system','animal','MMS'], axis=1)
                    )

                    # Update DataFrame
                    mms_shares.update(grazing_share)
            
            if not np.isclose(mms_shares.T.groupby(['prod_system','animal']).sum().T, 1).all():
                # Adjust non-grazing MMS shares
                adjusted = (
                    mms_shares.loc[:, (slice(None), slice(None), mmss)]
                    .T.groupby(['prod_system','animal']).transform(lambda x: x/x.sum()).T
                )
                adjusted = multiply_aligned(
                    adjusted,
                    1 - mms_shares.xs('grazing', level='MMS', axis=1)
                )
                mms_shares.update(adjusted)

            herd.data_attr.add(
                mms_shares * 100,
                name = 'manure.mms_shares',
                unit = '%',
                orig = 'ManureMgmt',
                desc = 'Manure management systems shares',
                scalable = False
            )

        return None

    def calculate_bedding_material_use(self):
        '''Calculate use of bedding materials'''
        
        # Clear ParameterRetriever filters 
        self.par.clear()
        self.feed_mgmt.par.clear()

        # Get types of bedding materials used
        fes = self.par.get_unique('feed', qry='parameter == "bedding_material_use"')

        for herd in self.herds:

            # Set species and breed filters
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Get manure management systems
            mmss = herd.data_attr.get('manure.mms_shares').columns.get_level_values('MMS').unique()
            
            # Create dataframe
            df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [tuple(list(i) + [mms,fe]) for i in herd.data_attr.get('heads').columns for mms in mmss for fe in fes],
                    names = herd.data_attr.get('heads').columns.names + ['MMS','feed']
                )
            )
            
            # Calculate bedding material use [kg DM/year]
            DM = multiply_aligned(
                multiply_aligned(
                    self.par.get_from_frame('bedding_material_use', df),
                    herd.data_attr.get('manure.mms_shares')/100
                ) * 365.25,
                herd.data_attr.get('heads')
            )
            # Set bedding material use during grazing to zero
            DM.loc[:,(slice(None),slice(None),['grazing'],slice(None))] *= 0
            
            # Calculate nitrogen,phosphorous and potassium in bedding materials
            N = ( # [kg N/year]
                DM
                .mul(self.feed_mgmt.par.get_from_frame('feed_par_N', df)/100)
                .T.groupby(['prod_system','animal','MMS']).sum().T
            )
            P = ( # [kg P/year]
                DM
                .mul(self.feed_mgmt.par.get_from_frame('feed_par_P', df)/100)
                .T.groupby(['prod_system','animal','MMS']).sum().T
            )
            K = ( # [kg K/year]
                DM
                .mul(self.feed_mgmt.par.get_from_frame('feed_par_K', df)/100)
                .T.groupby(['prod_system','animal','MMS']).sum().T
            )

            # Add data attributes
            for df, element in zip([DM,N,P,K], ['DM','N','P','K']):
                herd.data_attr.add(
                    df,
                    name = f'bedding_material{"_"+element if not element=="DM" else ""}',
                    unit = f'kg {element}/year',
                    orig = 'ManureMgmt',
                    desc = 'Bedding material use' + ((' in terms of ' + element) if not element=='DM' else '')
                )

    def calculate_VS_excretion(self):
        '''Calculate VS excretion'''

        self.par.clear()
        pf = self.par.get_from_frame

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Get production systems, animals in herd
            pss = herd.data_attr.get('heads').columns.get_level_values('prod_system').unique()
            anis = herd.animals

            DM_intake = herd.data_attr.get('feed.consumption').T.groupby(['prod_system','animal']).sum().T
            if herd.species != 'poultry':
                energy_manure = (
                    (
                        herd.data_attr.get('feed.ration_GE') -
                        herd.data_attr.get('feed.ration_DE') + 
                        herd.data_attr.get('feed.ration_GE') * pf('UE_of_GE',herd.data_attr.get('feed.ration_GE'))
                    ) * DM_intake
                )
            else:
                energy_manure = (
                    (herd.data_attr.get('feed.ration_GE') - herd.data_attr.get('feed.ration_AME')) * DM_intake
                )

            VS_excr = (
                energy_manure * 
                ((1 - herd.data_attr.get('feed.ration_ASH')/100) / herd.data_attr.get('feed.ration_GE'))
            ).fillna(0)

            # Distribute across MMS
            VS_excr = multiply_aligned(
                herd.data_attr.get('manure.mms_shares')/100,
                VS_excr
            )

            # ADD BEDDING TO VS EXCRETION!!!

            # Add data attribute
            herd.data_attr.add(
                VS_excr,
                name = 'manure.VS_excr',
                unit = 'kg VS/year',
                orig = 'ManureMgmt',
                desc = 'Manure volatile solids (VS) excretion'
            )
        
        return None
    
    def calculate_VS_losses(self):

        idx = pd.IndexSlice

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            # Get manure management systems
            mmss = herd.data_attr.get('manure.mms_shares').columns.get_level_values('MMS').unique()

            # Get production systems, animals in herd
            pss = herd.data_attr.get('heads').columns.get_level_values('prod_system').unique()
            anis = herd.animals
            css = ['CH4bio','CO2bio']

            # Create dataframe
            df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms,cs) for ps in pss for ani in anis for mms in mmss for cs in css],
                    names=['prod_system','animal','MMS','compound']
                ),
                dtype = float
            )

            # Calculate CH4 emissions using the IPCC Tier 2 method
            # from maximum methane production (B0) and MCF
            CH4_loss = (
                herd.data_attr.get('manure.VS_excr') *
                (self.par.get_from_frame('methane_B0', herd.data_attr.get('manure.VS_excr'))*0.67) *
                (self.par.get_from_frame('methane_MCF', herd.data_attr.get('manure.VS_excr'))/100)
            )

            if False:
                # Calculate CH4 emissins using the IPCC Tier 1 method
                # from manure emissions per head and year
                CH4_Tier1 = multiply_aligned(
                    self.par.get_from_frame('methane_per_head',df),
                    herd.data_attr.get('heads')
                ).replace({0:np.nan})

                # Take Tier 2 if possible otherwise Tier 1
                VS_loss = CH4_Tier2.copy()
                VS_loss.update(CH4_Tier1, overwrite=False)
                VS_loss = VS_loss.fillna(0)

            # Calculate C and CO2 losses
            C_excr = herd.data_attr.get('manure.VS_excr') * (self.par.get_from_frame('manure_VS_C', herd.data_attr.get('manure.VS_excr'))/100)
            C_loss_tot = C_excr * (self.par.get_from_frame('C_loss',C_excr)/100)
            C_to_spread = C_excr - C_loss_tot

            C_loss_CH4 = CH4_loss * (12/(12+1*4))
            C_loss_CO2 = C_loss_tot - C_loss_CH4
            # Set negative CO2 loss to zero. Occurs if parameter 'C_loss' yields lower
            # C losses than calculated methane emissions (e.g. for MMS=grazing)
            C_loss_CO2 = C_loss_CO2.where(C_loss_CO2 > 0, 0.0)
            CO2_loss = C_loss_CO2 * ((12+16*2)/12)

            # Put results in dataframe
            df.loc[:,idx[:,:,:,'CH4bio']] = \
            CH4_loss.reindex(df.xs('CH4bio', level='compound', axis=1, drop_level=False).columns, axis=1)
            df.loc[:,idx[:,:,:,'CO2bio']] = \
            CO2_loss.reindex(df.xs('CO2bio', level='compound', axis=1, drop_level=False).columns, axis=1)

            VS_loss = df

            # Add data attributes
            herd.data_attr.add(
                VS_loss,
                name = 'manure.VS_loss',
                unit = 'kg/year',
                orig = 'ManureMgmt',
                desc = 'Manure losses of C-containing compounds in stables and during manure storage'
            )
            herd.data_attr.add(
                C_to_spread,
                name = 'manure.C_to_spread',
                unit = 'kg C/year',
                orig = 'ManureMgmt',
                desc = 'Carbon (C) in manure available to spread'
            )
        
    def calculate_NPK_excretion(self, element):

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system
            )

            if self.settings['NPK_excretion_from_balance'] and 'lwg' in herd.data_attr:
                # Calculate N excretion based on mass balance

                # Nutrient in feed input (excl. storage and feeding losses)
                # SOME LOSSES SHOULD BE INCLUDED!! 
                # !!! NEED TO THINK ABOUT SILAGE LOSSES !!!
                feed = (
                    herd.data_attr.get('feed.consumption')
                    .T.groupby(['prod_system','animal']).sum().T *
                    (herd.data_attr.get('feed.ration_' + element) / 100)
                )

                # Nutrients in bedding materials
                bedding = herd.data_attr.get('bedding_material_' + element)

                # Nutrients in live weight gain
                lwg = (
                    herd.data_attr.get('lwg') *
                    self.par.get_from_frame(element + '_in_LW', herd.data_attr.get('lwg'))/1000
                )

                # Nutrients in products (excl. meat)
                prod = (
                    (
                        herd.data_attr.get('production') *
                        self.par.get_from_frame(
                            element + '_in_prod',
                            herd.data_attr.get('production')
                        )/1000
                    )
                    .T.groupby(['prod_system','animal']).sum().T
                )

                # Calculate excretion from animals
                excr_df = multiply_aligned(
                    herd.data_attr.get('manure.mms_shares')/100,
                    (feed - lwg - prod)
                )

                # Add nutrients in bedding materials
                excr_df = excr_df + bedding

            else:
                # Calculate N excretion from fixed factor per head
                excr_df = multiply_aligned(
                    (
                        herd.data_attr.get('manure.mms_shares')/100 *
                        self.par.get_from_frame('manure_excr_'+element, herd.data_attr.get('manure.mms_shares'))
                    ),
                    herd.data_attr.get('heads')
                )

            # Add data attribute
            herd.data_attr.add(
                excr_df,
                name = f'manure.{element}_excr',
                unit = f'kg {element}/year',
                orig = 'ManureMgmt',
                desc = f'Manure {_elem_to_name[element]} excretion'
            )

        return None
    
    def calculate_NPK_losses(self, element):

        # Get compounds
        cmps = self.par.get_unique('compound', qry=f'f_element == "{element}"')
        
        for herd in self.herds:
        
            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed,
                sub_system = herd.sub_system,
                element = element
            )
        
            # Get manure management systems
            mmss = herd.data_attr.get('manure.mms_shares').columns.get_level_values('MMS').unique()
        
            # Get production systems, animals in herd
            pss = herd.data_attr.get('heads').columns.get_level_values('prod_system').unique()
            anis = herd.animals
        
            # Create dataframe
            df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms,cmp) for ps in pss for ani in anis for mms in mmss for cmp in cmps],
                    names=['prod_system','animal','MMS','compound']
                )
            )
            
            # Get excretion and propagate values to all compound columns
            excr = herd.data_attr.get(f'manure.{element}_excr').align(df)[0].reindex(index=df.index, columns=df.columns)
           
            loss_factors_stable = self.par.get_from_frame('loss_stable', df)/100
            loss_stable = loss_factors_stable * excr
        
            loss_factors_storage = self.par.get_from_frame('loss_storage', df)/100
            loss_storage = loss_factors_storage * (excr - loss_stable)
        
            if element == 'N':
                # NOx-N and N2 storage losses are calculated from plant available nitrogen
                loss_storage.update(
                    (loss_factors_storage * (excr - loss_stable) * (self.par.get_from_frame('TAN_share',df)/100))
                    .loc[:,(slice(None), slice(None), slice(None), ['NOx-N', 'N2'])]
                )
        
            available_to_spread = (
                herd.data_attr.get(f'manure.{element}_excr') - 
                (loss_stable + loss_storage).T.groupby(['prod_system','animal','MMS']).sum().T
            )
        
            # Add data attribute
            herd.data_attr.add(
                loss_stable + loss_storage,
                name = f'manure.{element}_loss',
                unit = f'kg {element}/year',
                orig = 'ManureMgmt',
                desc = f'Manure losses of {element}-containing compounds in stables and during manure storage'
            )
            herd.data_attr.add(
                    available_to_spread,
                    name = f'manure.{element}_to_spread',
                    unit = f'kg {element}/year',
                    orig = 'ManureMgmt',
                    desc = f'Total {_elem_to_name[element]} in manure available to spread after stable and storage losses'
            )
            
            if element=='N':
                # For nitrogen, also calculate and store plant available nitrogen available to spread
                TAN_to_spread = (
                    available_to_spread * 
                    (self.par.get_from_frame('TAN_share', available_to_spread)/100)
                ).T.groupby(['prod_system','animal','MMS']).sum().T
                
                herd.data_attr.add(
                    TAN_to_spread,
                    name = 'manure.TAN_to_spread',
                    unit = 'kg TAN/year',
                    orig = 'ManureMgmt',
                    desc = 'Total plant available nitrogen (TAN) in manure available to spread after stable and storage losses'
                )
    
        return None
    
_elem_to_name = {
    'N' : 'nitrogen (N)',
    'P' : 'phosphorous (P)',
    'K' : 'potassium (P)'
}