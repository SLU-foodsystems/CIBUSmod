import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr

if TYPE_CHECKING:
    from .demand_and_conversions import DemandAndConversions
    from .crop_prod import CropProduction
    from ..utils.retriever import ParameterRetriever

class WasteAndCircularity(object):
    '''
    '''

    module_name = 'WasteAndCircularity'

    def __init__(
            self,
            demand: "DemandAndConversions",
            crops: "CropProduction",
            herds: pd.Series,
            par: "ParameterRetriever"
        ):
        
        # Create object for storing data attributes
        self.data_attr = DataAttr(self)
        self.par = par

        self.demand = demand
        self.crops = crops
        self.herds = herds

    def calculate(self, verbose=False):
        '''
        '''

        # Define function to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='ManureMgmt')

        vprint('Collecting biomass and calculating composition ...')
        self.collect_waste()

        for treatment in self.data_attr.get('feedstock_VS').columns.unique('waste_treatment'):
            vprint(f'Calculating {treatment} ...')

            try:
                calc_fun = getattr(self, 'calculate_' + treatment.replace(' ', '_'))
            except AttributeError:
                warnings.warn(f"No method to handle waste treatment '{treatment}'")
            else:
                calc_fun()

        vprint('Calculating regional distribution of bio-fertilisers ...')
        self.distribute_biofert()

        vprint(type='end')


    def collect_waste(self):
        '''
        '''

        self.par.clear()

        # COLLECT FOOD WASTE --------------------------------------------------------|
        # Get waste
        food_waste = (
            self.demand.data_attr.get('waste')
            .groupby(['food', 'food_group'])
            .sum()
            .stack()
            .rename_axis(index={'food':'waste', 'food_group':'waste_group'})
        )
        
        # Get human population distribution
        pop_dist = self.demand.data_attr.get('population_per_region').transform(lambda x: x/x.sum())
        
        # Distribute over regions based on population
        food_waste_reg = pd.DataFrame(
            1.0,
            index = pop_dist.index,
            columns = food_waste.index
        )
        food_waste_reg = (
            food_waste_reg
            .mul(pop_dist, axis=0)
            .mul(food_waste, axis=1)
        )
        
        # COLLECT CARCASSES -----------------------------------------------------------|
        # TO BE DONE...

        # COLLECT WASTE FROM SLAUGHTERHOUSES ------------------------------------------|
        # TO BE DONE...

        # COLLECT SURPLUSS BY-PRODUCTS ------------------------------------------|
        # TO BE DONE... ???
        
        # COLLECT CROP FEEDSTOCKS ----------------------------------------------------|
        waste_level = 'crop feedstock'
        sel = self.par.get_unique('waste', qry=f'f_waste_level == "{waste_level}"')
        non_waste = (
            demand.data_attr.get('non_food_demand').loc[sel]
            .groupby(['food', 'food_group'])
            .sum()
            .rename_axis(index={'food':'waste', 'food_group':'waste_group'})
            .to_frame()
            .rename_axis(columns='waste_level')
            .rename(columns={0:waste_level})
            .stack()
        )
        # Distribute over regions based on production for non-food use
        # of crops supplying the non-waste feedstocks
        non_waste_reg = pd.DataFrame(
            1.0,
            index = pop_dist.index,
            columns = non_waste.index
        )
        non_waste_reg = non_waste_reg.mul(non_waste, axis=1)
        for item in non_waste.index:
            food = item[0]
            # Get crop products supplying item
            cps = demand.par.get_unique(
                'crop_prod',
                qry=f'f_food == "{food}" & parameter == "conv_factor_main"'
            )
            # Get crops supplying crop product(s)
            crs = crops.par.get_unique(
                'crop',
                qry=f'f_crop_prod.isin({list(cps)}) & parameter == "crop_to_prod"'
            )
            # Calculate regional distribution key
            reg_dist = (
                crops.data_attr.get('production_per_use')
                .loc[crs, 'non-food']
                .groupby('region')
                .sum()
                .transform(lambda x: x/x.sum())
            )
            non_waste_reg.loc[:,item] *= reg_dist
        
        # COMBINE --------------------------------------------------------------------------|
        
        waste = pd.concat([food_waste_reg, non_waste_reg], axis=1).fillna(0)

        # CALCULATE FEEDSTOCK COMPOSITION AND GET MANURE FOR CENTRALISED TREATMENT ---------|
        
        # Set ParameterRetriver filters to get feedstock composition
        self.par.set(**waste.columns.to_frame().to_dict('list'))
        
        # Calculate dry matter
        waste_DM = waste.mul(self.par.get('waste_DM'), axis=1)
        
        # Calculate volatile solids (VS), methane production potential (B0), nitrogen (N), phosphorous (P) and potassium (K)
        items = {
            'VS' : 'volatile solids (VS)',
            'B0' : 'methane production potential (B0)',
            'C' : 'carbon (C)',
            'N' : 'nitrogen (N)',
            'P' : 'phosphorous (P)',
            'K' : 'potassium (K)'
        }
        
        waste_dfs = dict()
        manure_dfs = dict()
        
        for i in items:
            if i in ['B0','C']:
                df = waste_dfs['VS']
            else:
                df = waste_DM
        
            # Calculate item for waste
            waste_dfs.update({i : df.mul(self.par.get(f'waste_{i}'), axis=1)})
        
            # Get item from manure
            m_list = []
            for herd in self.herds:
                m = herd.data_attr.get(f'manure.{i}_to_treatment').T.groupby('MMS').sum().T
            
                # Create column index
                m.columns = pd.MultiIndex.from_tuples(
                    [(f'Manure, {herd.species}, {herd.breed}', mms, 'manure') for mms in m.columns],
                    names = ['waste', 'waste_group', 'waste_level']
                )
            
                m_list += [m]
                    
            manure_dfs.update({
                i : 
                pd.concat(m_list, axis=1)
                # Sum duplicates
                .T.groupby(['waste', 'waste_group', 'waste_level']).sum().T
                # Drop columns with all zeros
                .replace({0:np.nan})
                .dropna(axis=1, how='all')
                .fillna(0)
            })
        
        dfs = {i : pd.concat([waste_dfs[i], manure_dfs[i]], axis=1) for i in items}

        # DISTRIBUTE ACROSS WASTE TREATMENTS ----------------------------------------------------|
        
        # Create df to retrieve treatment shares
        retrieve_df = pd.DataFrame(
            index = list(dfs.values())[0].index,
            columns = pd.MultiIndex.from_tuples(
                [(w, wg, wl, tr) for w,wg,wl in list(dfs.values())[0].columns
                 for tr in self.par.get_unique('waste_treatment')],
                names = ['waste', 'waste_group', 'waste_level', 'waste_treatment']
            )
        )
        
        # Get treatment shares
        treatment_shares = self.par.get_from_frame('treatment_share', retrieve_df)
        
        # Check that treatment shares all add up to 100%
        shares_sum = treatment_shares.T.groupby(['waste','waste_group','waste_level']).sum().T
        check_shares = shares_sum != 100
        if check_shares.any().any():
            warnings.warn(f''''treatment_share' did not add up to 100% for:
            {shares_sum.loc[check_shares.any(axis=1),check_shares.any()]}
            ''')
        
        # Distribute across treatments and store data attributes
        for i in items:
            self.data_attr.add(
                multiply_aligned(treatment_shares/100, dfs[i]),
                name = f'feedstock_{i}',
                unit = f'kg {i}' if i != 'B0' else 'Nm3 CH4',
                orig = 'WasteAndCircularity',
                desc = f'Total {items[i]} in biomass for treatment'
            )

        return None
        
    def calculate_anaerobic_digestion(self):
        print('NOT IMPLEMENTED!', end=' ')

    def calculate_composting(self):
        print('NOT IMPLEMENTED!', end=' ')

    def calculate_incineration(self):
        print('NOT IMPLEMENTED!', end=' ')

    def calculate_landfill(self):
        print('NOT IMPLEMENTED!', end=' ')

    def distribute_biofert(self):
        print('NOT IMPLEMENTED!', end=' ')
        