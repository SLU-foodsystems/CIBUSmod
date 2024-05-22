import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr
from .. utils.misc import multiply_aligned

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
        vprint = verbose_init(verbose, id_str='WasteAndCircularity')

        vprint('Collecting feedstocks and calculating composition ...')
        self.collect_feedstock()

        vprint('Creating data attribute tables ...')
        self.create_data_attribute_tables()

        for treatment in self.data_attr.get('feedstock_VS').columns.unique('treatment'):
            vprint(f'Calculating {treatment} ...')

            try:
                calc_fun = getattr(self, 'calculate_' + treatment.replace(' ', '_'))
            except AttributeError:
                warnings.warn(f"WasteAndCircularity: No method to handle waste treatment '{treatment}'")
            else:
                calc_fun()

        vprint('Calculating regional distribution of bio-fertilisers ...')
        self.distribute_biofert()

        vprint(type='end')

    def create_data_attribute_tables(self):
        # Create empty dataframes and store data attributes to write to in the
        # treatment-specific methods

        dfs = {
            'losses_VS' : ('kg', 'Losses of volatile solids (VS) during waste treatment'),
            'losses_N' : ('kg N', 'Losses of nitrogen (N) during waste treatment'),
            'losses_P' : ('kg P', 'Losses of phosphorous (P) during waste treatment'),
            'losses_K' : ('kg K', 'Losses of potassium (K) during waste treatment'),
            'organic_fertiliser_C' : ('kg C', 'Carbon (C) in organic fertilisers available to spread'),
            'organic_fertiliser_N' : ('kg N', 'Nitrogen (N) in organic fertilisers available to spread'),
            'organic_fertiliser_P' : ('kg P', 'Phosphorous (P) in organic fertilisers available to spread'),
            'organic_fertiliser_K' : ('kg K', 'Potassium (K) in organic fertilisers available to spread'),
            'energy_prod' : ('kWh', 'Total energy production'),
            'energy_use' : ('kWh', 'Total energy use')
        }
        empty_df = pd.DataFrame(
            index=self.data_attr.get('feedstock_VS').index,
            columns=self.data_attr.get('feedstock_VS').columns
        )

        for n in dfs:
            if 'losses' in n:
                element = n.split('_')[1]
                # Get compounds emitted
                if element == 'VS':
                    cmps = ['CH4bio', 'CO2bio']
                else:
                    cmps = self.par.get_unique('compound', qry=f'f_element == "{element}"')
                df = pd.DataFrame(
                    index = empty_df.index,
                    columns = pd.MultiIndex.from_tuples(
                        [i + (cmp,) for i in empty_df.columns for cmp in cmps],
                        names = empty_df.columns.names + ['compound']
                    )
                )
            elif 'energy' in n:
                level = 'energy_source' if 'use' in n else 'energy_prod'
                ens = self.par.get_unique(level)
                df = pd.DataFrame(
                    index = empty_df.index,
                    columns = pd.MultiIndex.from_tuples(
                        [i + (en,) for i in empty_df.columns for en in ens],
                        names = empty_df.columns.names + [level]
                    )
                )
            else:
                df = empty_df.copy()

            self.data_attr.add(
                df,
                name = n,
                unit = dfs[n][0],
                orig = 'WasteAndCircularity',
                desc = dfs[n][1]
            )


    def collect_feedstock(self):
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
            .rename_axis(index={'food':'feedstock', 'food_group':'feedstock_group', 'waste_level':'feedstock_type'})
            .rename(index = lambda x: 'food waste, ' + x, level='feedstock_type')
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
        feedstock_type = 'crop feedstock'
        sel = self.par.get_unique('feedstock', qry=f'f_feedstock_type == "{feedstock_type}"')
        crops = (
            self.demand.data_attr.get('non_food_demand').loc[sel]
            .groupby(['food', 'food_group'])
            .sum()
            .rename_axis(index={'food':'feedstock', 'food_group':'feedstock_group'})
            .to_frame()
            .rename_axis(columns='feedstock_type')
            .rename(columns={0:feedstock_type})
            .stack()
        )
        # Distribute over regions based on production for non-food use
        # of crops supplying the feedstocks
        crops_reg = pd.DataFrame(
            1.0,
            index = pop_dist.index,
            columns = crops.index
        )
        crops_reg = crops_reg.mul(crops, axis=1)
        for item in crops.index:
            food = item[0]
            # Get crop products supplying item
            cps = self.demand.par.get_unique(
                'crop_prod',
                qry=f'f_food == "{food}" & parameter == "conv_factor_main"'
            )
            # Get crops supplying crop product(s)
            crs = self.crops.par.get_unique(
                'crop',
                qry=f'f_crop_prod.isin({list(cps)}) & parameter == "crop_to_prod"'
            )
            # Calculate regional distribution key
            reg_dist = (
                self.crops.data_attr.get('production_per_use')
                .loc[crs, 'non-food']
                .groupby('region')
                .sum()
                .transform(lambda x: x/x.sum())
            )
            crops_reg.loc[:,item] *= reg_dist
        
        # COMBINE --------------------------------------------------------------------------|
        
        feedstock = pd.concat([food_waste_reg, crops_reg], axis=1).fillna(0)

        # CALCULATE FEEDSTOCK COMPOSITION AND GET MANURE FOR CENTRALISED TREATMENT ---------|
        
        # Set ParameterRetriver filters to get feedstock composition
        self.par.set(**feedstock.columns.to_frame().to_dict('list'))
        
        # Calculate dry matter
        feedstock_DM = feedstock.mul(self.par.get('feedstock_DM'), axis=1)
        
        # Calculate volatile solids (VS), methane production potential (B0), nitrogen (N), phosphorous (P) and potassium (K)
        items = {
            'VS' : 'volatile solids (VS)',
            'B0' : 'methane production potential (B0)',
            'C' : 'carbon (C)',
            'N' : 'nitrogen (N)',
            'P' : 'phosphorous (P)',
            'K' : 'potassium (K)'
        }
        
        feedstock_dfs = dict()
        manure_dfs = dict()
        
        for i in items:
            if i in ['B0','C']:
                df = feedstock_dfs['VS']
            else:
                df = feedstock_DM
        
            # Calculate item for waste
            feedstock_dfs.update({i : df.mul(self.par.get(f'feedstock_{i}'), axis=1)})
        
            # Get item from manure
            m_list = []
            for herd in self.herds:
                m = herd.data_attr.get(f'manure.{i}_to_treatment').T.groupby('MMS').sum().T
            
                # Create column index
                m.columns = pd.MultiIndex.from_tuples(
                    [(f'Manure, {herd.species}, {herd.breed}', mms, 'manure') for mms in m.columns],
                    names = ['feedstock', 'feedstock_group', 'feedstock_type']
                )
            
                m_list += [m]
                    
            manure_dfs.update({
                i : 
                pd.concat(m_list, axis=1)
                # Sum duplicates
                .T.groupby(['feedstock', 'feedstock_group', 'feedstock_type']).sum().T
                # Drop columns with all zeros
                .replace({0:np.nan})
                .dropna(axis=1, how='all')
                .fillna(0)
            })
        
        dfs = {i : pd.concat([feedstock_dfs[i], manure_dfs[i]], axis=1) for i in items}

        # DISTRIBUTE ACROSS WASTE TREATMENTS ----------------------------------------------------|
        
        # Create df to retrieve treatment shares
        retrieve_df = pd.DataFrame(
            index = list(dfs.values())[0].index,
            columns = pd.MultiIndex.from_tuples(
                [(w, wg, wl, tr) for w,wg,wl in list(dfs.values())[0].columns
                 for tr in self.par.get_unique('treatment')],
                names = ['feedstock', 'feedstock_group', 'feedstock_type', 'treatment']
            )
        )
        
        # Get treatment shares
        treatment_shares = self.par.get_from_frame('treatment_share', retrieve_df)
        
        # Check that treatment shares all add up to 100%
        shares_sum = treatment_shares.T.groupby(['feedstock','feedstock_group','feedstock_type']).sum().T
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

        # Shorthands to retriever
        g = self.par.get
        gf = self.par.get_from_frame

        # Get CH4 and CO2 density and CH4 specific energy
        self.par.clear()
        CH4d = g('CH4_density')[0] # kg/Nm3
        CO2d = g('CO2_density')[0] # kg/Nm3
        CH4se = g('CH4_specific_energy')[0] # kWh/kg

        # Get feedstock VS, C and B0
        feedstock_VS = self.data_attr.get('feedstock_VS').xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)
        feedstock_C = self.data_attr.get('feedstock_C').xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)
        feedstock_B0 = self.data_attr.get('feedstock_B0').xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)

        # Calculate volume [Nm3] of generated biogas CH4 and CO2 
        CH4_prod_vol = feedstock_B0 * gf('biogas_CH4_yield', feedstock_B0)/100
        CO2_prod_vol = CH4_prod_vol * (1/(gf('biogas_CH4_frac', CH4_prod_vol)/100) - 1)

        # Calculate mass [kg] of generated biogas CH4 and CO2
        CH4_prod = CH4_prod_vol * CH4d
        CO2_prod = CO2_prod_vol * CO2d

        # Calculate CH4 and CO2 losses (slip and flare) during biogas production [kg]
        slip = gf('biogas_CH4_slip', CH4_prod)/100
        flare = gf('biogas_flare_frac', CH4_prod)/100
        flare_slip = gf('biogas_flare_slip', CH4_prod)/100

        CH4_loss_slip = CH4_prod * slip
        CO2_loss_slip = CO2_prod * slip

        CH4_loss_flare = CH4_prod * flare
        CO2_loss_flare = CO2_prod * flare

        # Sum production losses and recalculate flared CH4 to CO2
        CH4_loss_prod = CH4_loss_slip + CH4_loss_flare * flare_slip
        CO2_loss_prod = CO2_loss_slip + CO2_loss_flare + \
            CH4_loss_flare *  (1-flare_slip) * ((12+2*16)/(12+4*1))

        # Get uses of biogas
        uses = self.par.get_unique('energy_prod', qry="parameter == 'biogas_use_share'")

        # Construct dataframe and get use shares
        use_shares = gf(
            'biogas_use_share',
            pd.DataFrame(
                index = CH4_prod.index,
                columns = pd.MultiIndex.from_tuples(
                    [i+(u,) for i in CH4_prod.columns for u in uses],
                    names = CH4_prod.columns.names + ['energy_prod']
                )
            )
        )/100

        # Check that shares add up to 100%
        if not (use_shares.T.groupby(use_shares.columns.names[:-1]).sum().T == 1).all().all():
            raise ValueError("WasteAndCircularity: 'biogas_use_share' did not add up to 100%")

        # Calculate CH4 per use [kg]
        CH4_per_use = multiply_aligned(
            use_shares,
            ( CH4_prod - CH4_loss_slip - CH4_loss_flare )
        )

        # Calculate CH4 slip per use and aggregate
        CH4_loss_use = CH4_per_use * (gf('biogas_use_CH4_slip', CH4_per_use)/100)
        CH4_loss_use_agg = CH4_loss_use.T.groupby(CH4_loss_use.columns.names[:-1]).sum().T

        CH4_per_use_after_losses = CH4_per_use - CH4_loss_use

        # Calculate energy in produced biogas per use [kWh]
        energy_prod = CH4_per_use_after_losses * CH4se

        # Update energy_prod data attribute
        self.data_attr.get('energy_prod').loc[:, energy_prod.columns] = energy_prod

        # Calculate energy use in biogas production and use [kWh]
        energy_sources = self.par.get_unique('energy_source')

        energy_use_prod = gf(
            'biogas_energy_use',
            pd.DataFrame(
                index = CH4_prod_vol.index,
                columns = pd.MultiIndex.from_tuples(
                    [i+(es,) for i in CH4_prod_vol.columns for es in energy_sources],
                    names = CH4_prod_vol.columns.names + ['energy_source']
                )
            )
        )
        energy_use_prod = multiply_aligned(energy_use_prod, CH4_prod_vol)

        energy_use_use = gf(
            'biogas_use_energy_use',
            pd.DataFrame(
                index = CH4_per_use.index,
                columns = pd.MultiIndex.from_tuples(
                    [i+(es,) for i in CH4_per_use.columns for es in energy_sources],
                    names = CH4_per_use.columns.names + ['energy_source']
                )
            )
        )
        energy_use_use = multiply_aligned(energy_use_use, CH4_per_use).T.groupby(energy_use_prod.columns.names).sum().T

        energy_use = energy_use_prod + energy_use_use

        # Update energy_use data attribute
        self.data_attr.get('energy_use').loc[:, energy_use.columns] = energy_use

        # Calculate VS remaining in digestate assuming that C/VS
        # remains constant [kg]
        digestate_C = (
            feedstock_C -
            CH4_prod * (12/(12+4*1)) -
            CO2_prod * (12/(12+2*16))
        )
        digestate_VS = digestate_C * (feedstock_VS/feedstock_C).fillna(0)

        # Calculate storage losses of CH4 and CO2 assuming same CH4 frac as
        # in produced biogas [Nm3]
        digestate_B0 = digestate_VS * gf('digestate_B0', digestate_VS)
        CH4_loss_store_vol = digestate_B0 * gf('digestate_MCF', digestate_B0)/100
        CO2_loss_sore_vol = CH4_loss_store_vol * (1/(gf('biogas_CH4_frac', CH4_loss_store_vol)/100) - 1)

        # Calculate mass [kg] of storage losses
        CH4_loss_store = CH4_loss_store_vol * CH4d
        CO2_loss_store = CO2_loss_sore_vol * CO2d

        # Calculate carbon (C) in digestate after storage [kg]
        digestate_C_to_spread = (
            digestate_C -
            CH4_loss_store * (12/(12+4*1)) -
            CO2_loss_store * (12/(12+2*16))
        )

        # Aggregate VS losses and update data attribute
        VS_loss = pd.concat([
            pd.concat({'CH4bio': (CH4_loss_prod + CH4_loss_use_agg + CH4_loss_store)}, names=['compound'], axis=1).reorder_levels([1,2,3,4,0], axis=1),
            pd.concat({'CO2bio': (CO2_loss_prod + CO2_loss_store)}, names=['compound'], axis=1).reorder_levels([1,2,3,4,0], axis=1)
        ], axis=1)
        # self.data_attr.get('losses_VS').update(VS_loss)
        self.data_attr.get('losses_VS').loc[:, VS_loss.columns] = VS_loss

        # Update data attribute
        self.data_attr.get('organic_fertiliser_C').loc[:, digestate_C_to_spread.columns] = digestate_C_to_spread

        # Calculate losses of NPK during digestate storage and store data attributes
        for element in ['N','P','K']:

            self.par.clear()
            self.par.set(element = element)

            # Get ammount of N, P or K in digestate assuming
            # no losses of NPK during biogas production
            digestate = self.data_attr.get('feedstock_'+element).xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)
            
            # Get compounds emitted
            cmps = self.par.get_unique('compound', qry=f'f_element == "{element}"')

            # Get losses dataframe slice
            df = self.data_attr.get('losses_'+element).xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)

            if (self.par.data.xs((element,'digestate_loss_storage'),level=('f_element','parameter'))>0).any():

                loss_factors_storage = self.par.get_from_frame('digestate_loss_storage', df)/100
                loss_storage = multiply_aligned(loss_factors_storage, digestate)
            
                if element == 'N':
                    # NOx-N and N2 storage losses are calculated from plant available nitrogen
                    loss_storage.update(
                        (
                            multiply_aligned(loss_factors_storage, digestate) *
                            (self.par.get_from_frame('digestate_TAN_share', df)/100)
                        )
                        .loc[:,(slice(None), slice(None), slice(None), slice(None), ['NOx-N', 'N2'])]
                    )
            else:
                loss_storage = df * 0.0

            digestate_to_spread = \
                digestate - loss_storage.T.groupby(['feedstock','feedstock_group','feedstock_type','treatment']).sum().T

            self.data_attr.get('organic_fertiliser_'+element).loc[:, digestate_to_spread.columns] = digestate_to_spread
            self.data_attr.get('losses_'+element).loc[:, loss_storage.columns] = loss_storage

    def calculate_composting(self):
        print('NOT IMPLEMENTED!', end=' ')

    def calculate_incineration(self):
        print('NOT IMPLEMENTED!', end=' ')

    def calculate_landfill(self):
        print('NOT IMPLEMENTED!', end=' ')

    def distribute_biofert(self):
        print('NOT IMPLEMENTED!', end=' ')
        