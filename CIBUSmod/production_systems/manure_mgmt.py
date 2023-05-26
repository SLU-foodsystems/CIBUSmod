import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import Container, multiply_aligned

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
        # To be included ...

        # Nitrogen (N) [kg N per year]
        vprint('Calculating N excretion ...')
        self.calculate_N_excretion()
        vprint('Calculating N losses ...')
        self.calculate_N_losses()
        vprint('Calculating N available to spread ...')
        for herd in self.herds:
            herd.manure.N_to_spread = herd.manure.N_excr - herd.manure.N_loss.groupby(['prod_system','animal','MMS'], axis=1).sum()
            herd.data_attr.update(['manure.N_to_spread'])

        # Phosphorous (P)
        # To be included ...

        # Potassioum (K)
        # To be included ...

        vprint(type='end')
        
    def calculate_N_excretion(self):

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

            # Create dataframe
            N_excr = pd.DataFrame(
                index = herd.index,
                columns = pd.MultiIndex.from_tuples(
                    [(ps,ani,mms) for ps in pss for ani in anis for mms in mmss],
                    names=['prod_system','animal','MMS']
                    )
                )
            
            # Calculate N excretion
            N_excr.loc[:,:] = multiply_aligned(
                (
                    self.par.get_from_frame('manure_excr_N',N_excr)
                    * self.par.get_from_frame('mms_share',N_excr)/100
                ),
                herd.heads
            )

            herd.manure.N_excr = N_excr
            herd.data_attr.update(['manure.N_excr'])
        
        return None
        
    def calculate_N_losses(self):

        p = self.par.get
        
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