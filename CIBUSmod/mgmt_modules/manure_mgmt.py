import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import Container, multiply_aligned, rgetattr, rsetattr

class ManureMgmt():
    '''Class that takes a (list of) AnimalHerd object(s) and calculates manure excretion and losses.'''

    def __init__(self,herds,par,**kwargs):
        
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
    
    def make_df(self,df_dict):
        col_levels = ['species','breed','prod_system','animal','mms','compound']
        df = pd.DataFrame.from_dict(df_dict)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=col_levels[:len(df.columns[0])])
        
        return df
    
    def calculate(self, verbose=False):
        '''Calculates all manure excretion and losses.

        Parameters
        ----------
        filters_from_index : bool or list (default True)
            If True indexes of AnimalHerd objects are use as filters for the ParameterRetriever. If a list is supplied
            index levels in that list are used.
        **kwargs : str or list
            Keyword arguments to be passed on as filters to the ParameterRetriever.

        Returns
        -------
        Nothing. Stores output in pandas.DataFrames in the attrubutes: '<element>_excr', '<element>_loss' and '<element>_to_spread'.
        <element> is VS (volatile solids), N (nitrogen), P (phosphorous) and K (potassium).
        '''

        # Define function to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='ManureMgmt')

        # Create manure objects in herds
        for herd in self.herds:
            if not hasattr(herd,'manure'):
                herd.manure = Manure()
      
        # Volatile solids (VS)
        vprint('Calculating VS excretion ...')
        self.calculate_VS_excretion()
        vprint('Calculating VS losses ...')
        self.calculate_VS_losses()

        # Nitrogen (N) [kg N or TAN per year]
        vprint('Calculating N excretion ...')
        self.calculate_NPK_excretion(compound = 'N')
        vprint('Calculating N losses ...')
        self.calculate_N_losses()

        # Phosphorous (P)
        vprint('Calculating P excretion ...')
        self.calculate_NPK_excretion(compound = 'P')

        # Potassioum (K)
        vprint('Calculating K excretion ...')
        self.calculate_NPK_excretion(compound = 'K')

        vprint('Calculating N available to spread ...')
        for herd in self.herds:
            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
                )
            
            # Calculate total nitrogen available to spread
            N_to_spread = herd.manure.N_excr - herd.manure.N_loss.groupby(['prod_system','animal','MMS'], axis=1).sum()

            # Calculate plant available nitrogen available to spread
            TAN_to_spread = N_to_spread * self.par.get_from_frame('TAN_share',N_to_spread)/100

            herd.manure.N_to_spread = N_to_spread
            herd.manure.TAN_to_spread = TAN_to_spread
            herd.data_attr.update(['manure.N_to_spread','manure.TAN_to_spread'])

        vprint(type='end')

    def calculate_VS_excretion(self):
        '''Calculate VS excretion'''

        self.par.clear()
        pf = self.par.get_from_frame

        # Get manure management systems
        mmss = self.par.get_unique('MMS')

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
            )

            # Get production systems, animals in herd
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            anis = herd.animals



            DM_intake = herd.feed.consumption.groupby(['prod_system','animal'], axis=1).sum()
            if herd.species != 'poultry':
                energy_manure = (
                    (
                        herd.feed.ration_GE -
                        herd.feed.ration_DE + 
                        herd.feed.ration_GE * pf('UE_of_GE',herd.feed.ration_GE)
                    ) * DM_intake
                )
            else:
                energy_manure = (
                    (herd.feed.ration_GE - herd.feed.ration_AME) * DM_intake
                )

            VS_excr = (
                energy_manure * 
                ((1 - herd.feed.ration_ASH/100) / herd.feed.ration_GE)
            ).fillna(0)

            # Distribute across MMS
            VS_excr = VS_excr.reindex(
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms) for ps in pss for ani in anis for mms in mmss],
                    names=['prod_system','animal','MMS']
                )
            )
            VS_excr = VS_excr * self.par.get_from_frame('mms_share',VS_excr)/100

            herd.manure.VS_excr = VS_excr
            herd.data_attr.update(['manure.VS_excr'])
        
        return None
    
    def calculate_VS_losses(self):

        idx = pd.IndexSlice

        # Get manure management systems and compounds
        mmss = self.par.get_unique('MMS')

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
            )

            # Get production systems, animals in herd
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            anis = herd.animals
            css = ['CH4bio','CO2bio']

            # Create dataframe
            df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms,cs) for ps in pss for ani in anis for mms in mmss for cs in css],
                    names=['prod_system','animal','MMS','compound']
                )
            )

            # Calculate CH4 emissions using the IPCC Tier 2 method
            # from maximum methane production (B0) and MCF
            CH4_loss = (
                herd.manure.VS_excr *
                (self.par.get_from_frame('methane_B0',herd.manure.VS_excr)*0.67) *
                (self.par.get_from_frame('methane_MCF',herd.manure.VS_excr)/100)
            )

            if False:
                # Calculate CH4 emissins using the IPCC Tier 1 method
                # from manure emissions per head and year
                CH4_Tier1 = multiply_aligned(
                    self.par.get_from_frame('methane_per_head',df),
                    herd.heads
                ).replace({0:np.nan})

                # Take Tier 2 if possible otherwise Tier 1
                VS_loss = CH4_Tier2.copy()
                VS_loss.update(CH4_Tier1, overwrite=False)
                VS_loss = VS_loss.fillna(0)

            # Calculate C and CO2 losses
            C_excr = herd.manure.VS_excr * (self.par.get_from_frame('manure_VS_C',herd.manure.VS_excr)/100)
            C_loss_tot = C_excr * (self.par.get_from_frame('C_loss',C_excr)/100)
            C_to_spread = C_excr - C_loss_tot

            C_loss_CH4 = CH4_loss * (12/(12+1*4))
            C_loss_CO2 = C_loss_tot - C_loss_CH4
            CO2_loss = C_loss_CO2 * ((12+16*2)/12)

            # Put results in dataframe
            df.loc[:,idx[:,:,:,'CH4bio']] = \
            CH4_loss.reindex(df.xs('CH4bio', level='compound', axis=1, drop_level=False).columns, axis=1)
            df.loc[:,idx[:,:,:,'CO2bio']] = \
            CO2_loss.reindex(df.xs('CO2bio', level='compound', axis=1, drop_level=False).columns, axis=1)

            VS_loss = df

            herd.manure.VS_loss = VS_loss
            herd.manure.C_to_spread = C_to_spread
            herd.data_attr.update(['manure.VS_loss','manure.C_to_spread'])
        
    def calculate_NPK_excretion(self, compound):

        # Get manure management systems
        mmss = self.par.get_unique('MMS')

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.clear()
            self.par.set(
                species = herd.species,
                breed = herd.breed
            )

            # Get production systems, animals in herd
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            anis = herd.animals

            # Create dataframe
            excr_df = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms) for ps in pss for ani in anis for mms in mmss],
                    names=['prod_system','animal','MMS']
                    )
                )

            if hasattr(herd,'lwg'):
                # Calculate N excretion based on mass balance

                # Nutrient in feed input (excl. storage and feeding losses)
                # SOME LOSSES SHOULD BE INCLUDED!! 
                # !!! NEED TO THINK ABOUT SILAGE LOSSES !!!
                feed = (
                    herd.feed.demand
                    .groupby(['prod_system','animal'], axis=1)
                    .sum() *
                    (rgetattr(herd, 'feed.ration_' + compound) / 100)
                )

                # Nutrients in bedding materials
                # !!! TO BE ADDED !!!
                bedding = 0

                # Nutrients in live weight gain
                lwg = (
                    herd.lwg *
                    self.par.get_from_frame(compound + '_in_LW', herd.lwg)/1000
                )

                # Nutrients in products (excl. meat)
                prod = (
                    (
                        herd.production *
                        self.par.get_from_frame(
                            compound + '_in_prod',
                            herd.production
                        )/1000
                    )
                    .groupby(['prod_system','animal'], axis=1)
                    .sum()
                )

                excr_df.loc[:,:] = multiply_aligned(
                    self.par.get_from_frame('mms_share',excr_df)/100,
                    (feed + bedding - lwg - prod)
                )

            else:
                # Calculate N excretion from fixed factor per head
                excr_df.loc[:,:] = multiply_aligned(
                    (
                        self.par.get_from_frame('manure_excr_'+compound,excr_df)
                        * self.par.get_from_frame('mms_share',excr_df)/100
                    ),
                    herd.heads
                )

            rsetattr(
                herd,
                'manure.' + compound + '_excr',
                excr_df
            )
            herd.data_attr.update(['manure.' + compound + '_excr'])

        return None
    
    def calculate_N_losses(self):
        
        # Get manure management systems and compounds
        mmss = self.par.get_unique('MMS')
        cmps = self.par.get_unique('compound', qry='f_compound.str.contains("N", na=False)')
        
        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
            )

            # Get production systems, animals in herd
            pss = herd.heads.columns.get_level_values('prod_system').unique()
            anis = herd.animals

            # Create dataframe
            herd.manure.N_loss = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms,cmp) for ps in pss for ani in anis for mms in mmss for cmp in cmps],
                    names=['prod_system','animal','MMS','compound']
                )
            )
            
            # Calculate N losses

            # Get share of total N that is available (this only applies to NOx-N and N2 losses
            # TAN_share=1 for other compounds
            TAN_share = self.par.get_from_frame('TAN_share',herd.manure.N_loss)/100
            TAN_share.loc[:,~TAN_share.columns.get_level_values('compound').isin(['NOx-N','N2'])] = 1

            # Get N excr and propagate values to all compound columns
            N_excr = herd.manure.N_excr.align(herd.manure.N_loss)[0].reindex(index=herd.manure.N_loss.index, columns=herd.manure.N_loss.columns)


            loss_stable = (
                self.par.get_from_frame('loss_stable',herd.manure.N_loss)/100
                * N_excr
            )

            loss_storage = (
                self.par.get_from_frame('loss_storage',herd.manure.N_loss)/100
                * TAN_share
                * (N_excr - loss_stable)
            )

            herd.manure.N_loss.loc[:,:] = loss_stable + loss_storage
            herd.data_attr.update(['manure.N_loss'])

        return None

class Manure(Container):
    '''Class to store manure attributes in AnimalHerd obejcts'''