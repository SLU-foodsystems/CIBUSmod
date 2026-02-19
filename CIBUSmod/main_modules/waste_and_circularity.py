import warnings
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr
from ..utils.misc import multiply_aligned, fix_herds
from ..main_modules.animal_herd import concat_herds

if TYPE_CHECKING:
    from .demand_and_conversions import DemandAndConversions
    from .crop_prod import CropProduction
    from ..utils.retriever import ParameterRetriever

class WasteAndCircularity(object):
    '''Main module that handles waste management and other circularity strategies.

    Parameters
    ----------
    demand : DemandAndConversions object
    crops : CropProduction object
    herds : pandas.Series of AnimalHerd objects
    par : ParameterRetriever object
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

        # Create dict with treatment-specific functions
        self.tratment_funs = {
            'anaerobic digestion' : anaerobic_digestion,
            'composting' : composting,
            'incineration' : incineration,
            'landfill' : landfill
        }

        self.demand = demand
        self.crops = crops
        self.herds = fix_herds(herds)

    def calculate(self, verbose=False):
        '''
        '''

        # Define function to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='WasteAndCircularity')

        if 'wastewater_share_connected' in self.par.data.index.get_level_values('parameter'):
            vprint('Calculating wastewater treatment ...')
            self.calculate_wastewater()

        vprint('Collecting feedstocks and calculating composition ...')
        self.collect_feedstock()

        vprint('Creating data attribute tables ...')
        self.create_data_attribute_tables()

        for treatment in self.data_attr.get('feedstock_VS').columns.unique('treatment'):
            vprint(f'Calculating {treatment} ...')

            self.calculate_treatment(treatment)

        vprint(type='end')

    def make_data_attribute_columns(self, table):
        # Create columns for the diferent data attributes

        feedstock_cols = self.data_attr.get('feedstock_VS').columns

        if 'losses' in table:
            element = table.split('_')[1]
            # Get compounds emitted
            if element == 'VS':
                cmps = ['CH4bio', 'CO2bio']
            else:
                cmps = self.par.get_unique('compound', qry=f'f_element == "{element}"')
            cols = pd.MultiIndex.from_tuples(
                [i + (cmp,) for i in feedstock_cols for cmp in cmps],
                names = feedstock_cols.names + ['compound']
            )
        elif 'energy' in table:
            level = 'energy_source' if 'use' in table else 'energy_prod'
            ens = self.par.get_unique(level)
            cols = pd.MultiIndex.from_tuples(
                [i + (en,) for i in feedstock_cols for en in ens],
                names = feedstock_cols.names + [level]
            )
        else:
            cols = feedstock_cols.copy()

        return cols

    def create_data_attribute_tables(self, reindex=False):
        '''Creates empty dataframes and stores data attributes to write to in the
        treatment-specific methods
        '''

        dfs = {
            'losses_VS' : ('kg', 'Losses of volatile solids (VS) during waste treatment'),
            'losses_N' : ('kg N', 'Losses of nitrogen (N) during waste treatment'),
            'losses_P' : ('kg P', 'Losses of phosphorus (P) during waste treatment'),
            'losses_K' : ('kg K', 'Losses of potassium (K) during waste treatment'),
            'organic_fertiliser_C' : ('kg C', 'Carbon (C) in organic fertilisers available to spread'),
            'organic_fertiliser_N' : ('kg N', 'Nitrogen (N) in organic fertilisers available to spread'),
            'organic_fertiliser_P' : ('kg P', 'Phosphorus (P) in organic fertilisers available to spread'),
            'organic_fertiliser_K' : ('kg K', 'Potassium (K) in organic fertilisers available to spread'),
            'organic_fertiliser_TAN' : ('kg TAN', 'Total plant available nitrogen (TAN) in organic fertilisers available to spread'),
            'energy_prod' : ('kWh', 'Total energy production'),
            'energy_use' : ('kWh', 'Total energy use')
        }

        # Create the diferent data attribute tables
        for table in dfs:
            df = pd.DataFrame(
                index = self.data_attr.get('feedstock_VS').index,
                columns = self.make_data_attribute_columns(table),
                dtype = float
            )

            if reindex:
                self.data_attr.update(
                    name = table,
                    data = self.data_attr.get(table).reindex_like(df)
                )
            else:
                self.data_attr.add(
                    df,
                    name = table,
                    unit = dfs[table][0],
                    orig = 'WasteAndCircularity',
                    desc = dfs[table][1]
                )

    def calculate_wastewater(self):

        self.par.clear()
        p = self.par.get
        pf = self.par.get_from_frame

        #######################
        # INCOMING WASTEWATER #
        #######################

        # Define "elements" to consider in incoming waste water
        els = ['VS','COD','N','P','K']
        # Get population connected to WWT
        pop = self.demand.data_attr.get('population_per_region')
        pop_wwt = self.demand.data_attr.get('population_per_region') * (p('wastewater_share_connected', **pop.index.to_frame().to_dict('list'))/100)

        # Create dataframe for incomming VS, COD, N, P, and K
        in_toilet = pd.DataFrame(
            index = pop_wwt.index,
            columns = pd.Index(els, name='element')
        )
        in_grey = in_toilet.copy()
        # Calculate incoming VS, COD, N, P, and K based on parameters and population connected
        in_toilet = pf('wastewater_incoming_toilet', in_toilet, warn_if_nan=False).mul(pop_wwt, axis=0)
        in_grey = pf('wastewater_incoming_grey', in_toilet, warn_if_nan=False).mul(pop_wwt, axis=0)
        # Get incoming N, P, and K based on food consumption if available
        in_food = self.demand.data_attr.get('nutrient_supply').sum()
        in_food = in_food.loc[list(set(in_food.index) & {'N','P','K'})].rename_axis('element')
        in_food = pd.DataFrame(
            np.outer(pop.transform(lambda x: x/x.sum()), in_food),
            index = pop.index,
            columns = in_food.index
        )
        # Update incoming toilet water to N, P, and K based on food consumption
        in_toilet.update(in_food * (pop_wwt.sum()/pop.sum()))

        # Make sure there are no NaNs remaining
        assert ~in_toilet.isna().any().any(), "WasteAndCircularity: NaNs for VS, COD, N, P, or K in incoming toilet water to WWT"
        assert ~in_grey.isna().any().any(), "WasteAndCircularity: NaNs for VS, COD, N, P, or K in incoming grey water to WWT"

        # Combine toilet and grey water
        in_total = in_toilet + in_grey

        ##############
        # ENERGY USE #
        ##############

        energy_use = pd.DataFrame(index=in_total.index, columns=in_total.columns)

        #############
        # INPUT USE #
        #############

        input_use = pd.DataFrame(index=in_total.index, columns=in_total.columns)

        ##############
        # EMISSIONS #
        #############

        # Get emitted compounds
        cmps = self.par.get_unique('compound', qry="parameter=='wastewater_emissions'")
        # Create dataframe for emissions
        emissions = in_total.reindex(
            pd.MultiIndex.from_product([in_total.columns, cmps], names=['element','compound']),
            level = 'element',
            axis=1
        )
        # Calculate emissions and sum by compound
        emissions = (
            (emissions * pf('wastewater_emissions', emissions))
            .T.groupby('compound').sum().T
        )

        #####################
        # WASTEWATER SLUDGE #
        #####################

        # Calculate VS,N,P,K in sludge
        sludge = in_total.drop('COD', axis=1) * (pf('wastewater_recovery_sludge', in_total.drop('COD', axis=1))/100)

        #########################
        # STORE DATA ATTRIBUTES #
        #########################

        self.data_attr.add(
            in_total,
            name = 'wastewater.incoming',
            unit = 'kg',
            orig = 'WasteAndCircularity',
            desc = 'VS, COD, N, P, and K in incoming wastewater'
        )
        self.data_attr.add(
            energy_use,
            name = 'wastewater.energy_use',
            unit = 'kg',
            orig = 'WasteAndCircularity',
            desc = 'Energy use in wastewater treatment'
        )
        self.data_attr.add(
            input_use,
            name = 'wastewater.input_use',
            unit = 'kg',
            orig = 'WasteAndCircularity',
            desc = 'Inputs used in wastewater treatment'
        )
        self.data_attr.add(
            emissions,
            name = 'wastewater.emissions',
            unit = 'kg',
            orig = 'WasteAndCircularity',
            desc = 'Direct emissions from wastewater treatment'
        )
        self.data_attr.add(
            sludge,
            name = 'wastewater.sludge',
            unit = 'kg',
            orig = 'WasteAndCircularity',
            desc = 'VS, N, P, and K in outgoing wastewater sludge'
        )
        return None

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

        # COLLECT ANIMAL CARCASSES -----------------------------------------------------------|
        animals_reg = (
            concat_herds(self.herds, ['lost_lw']).data_attr.get('lost_lw')
            .T.groupby(['species','animal']).sum().T
            .rename(lambda x: 'carcasses, ' + x, level='species', axis=1)
            .rename_axis(columns=['feedstock_group','feedstock'])
        )
        animals_reg = pd.concat({'animal carcasses': animals_reg}, names=['feedstock_type'], axis=1)
        animals_reg = animals_reg.reorder_levels(['feedstock', 'feedstock_group', 'feedstock_type'], axis=1)

        # COLLECT SURPLUSS BY-PRODUCTS ------------------------------------------|
        # Get surpluss by-product to waste treatment and add 'feedstock_group' and 'feedstock_type' to index
        byprod = self.demand.data_attr.get('by_prod_to_waste').groupby('by_prod').sum()
        byprod = byprod.to_frame()
        byprod['feedstock_group'] = byprod.index.map(self.par.get_rel('by_prod', 'by_prod_group'))
        byprod['feedstock_type'] = 'by-product waste'
        byprod = byprod.set_index(['feedstock_group', 'feedstock_type'], append = True).rename_axis(index={'by_prod':'feedstock'})
        byprod = byprod[0]

        # Create distribution key
        # Surpluss by-products are distributed based on cropland area
        # assuming (roughly) that locations of processing facilities
        # follows cropland area distribution.
        dist_key = (
            self.crops.data_attr.get('area')
            # Select cropland
            .rename(self.par.get_rel('crop', 'land_use')).xs('cropland')
            # Sum area per region
            .groupby('region').sum()
            # Calculate share of total cropland per region
            .transform(lambda x: x/x.sum())
        )

        # Create dataframe with by-products as columns and regions as index
        # and calculate regional distribution of surpluss by-products
        byprod_reg = pd.DataFrame(
            1.0,
            index = dist_key.index,
            columns = byprod.index
        )
        byprod_reg = byprod_reg.mul(dist_key, axis=0).mul(byprod, axis=1)

        # COLLECT CROP FEEDSTOCKS ----------------------------------------------------|
        feedstock_type = 'crop feedstock'
        sel = self.par.get_unique('feedstock', qry=f'f_feedstock_type == "{feedstock_type}"')
        crops = (
            self.demand.data_attr.get('non_food_demand').loc[sel,'domestic']
            .groupby(['food', 'food_group'])
            .sum()
            .rename_axis(index={'food':'feedstock', 'food_group':'feedstock_group'})
            .to_frame()
            .rename_axis(columns='feedstock_type')
            .rename(columns={'domestic':feedstock_type})
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

        feedstock = pd.concat([food_waste_reg, byprod_reg, crops_reg, animals_reg], axis=1).fillna(0)

        # CALCULATE FEEDSTOCK COMPOSITION AND GET MANURE FOR CENTRALISED TREATMENT ---------|

        # Set ParameterRetriver filters to get feedstock composition
        self.par.set(**feedstock.columns.to_frame().to_dict('list'))

        # Calculate dry matter
        feedstock_DM = feedstock.mul(self.par.get('feedstock_DM'), axis=1)

        # Calculate volatile solids (VS), methane production potential (B0), nitrogen (N), phosphorus (P) and potassium (K)
        items = {
            'VS' : 'volatile solids (VS)',
            'B0' : 'methane production potential (B0)',
            'C' : 'carbon (C)',
            'N' : 'nitrogen (N)',
            'P' : 'phosphorus (P)',
            'K' : 'potassium (K)'
        }

        df_dicts = []

        feedstock_dfs = dict()
        for i in items:
            if i in ['B0','C']:
                df = feedstock_dfs['VS']
            else:
                df = feedstock_DM

            # Calculate item for waste
            feedstock_dfs.update({i : df.mul(self.par.get(f'feedstock_{i}'), axis=1)})

        df_dicts += [feedstock_dfs]

        if 'wastewater.sludge' in self.data_attr:
            self.par.clear()
            sludge_dfs = dict()
            for i in items:
                # Get item for wastewater sludge
                if i in ['B0','C']:
                    VS_df = sludge_dfs['VS']
                    self.par.set(**VS_df.columns.to_frame().to_dict('list'))
                    sludge_dfs.update({i : VS_df.mul(self.par.get(f'feedstock_{i}'), axis=1)})
                else:
                    s_df = self.data_attr.get('wastewater.sludge').loc[:,i].to_frame()
                    s_df.columns = pd.MultiIndex.from_tuples(
                        [('wastewater sludge',)*3],
                        names=['feedstock','feedstock_group','feedstock_type']
                        )
                    sludge_dfs.update({i : s_df})

            df_dicts += [sludge_dfs]
        
        manure_dfs = dict()
        for i in items:
            # Get item from manure
            m_list = []
            for herd in (h for h in self.herds if 'manure.VS_excr' in h.data_attr):
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

        df_dicts += [manure_dfs]

        dfs = {i : pd.concat([df_dict[i] for df_dict in df_dicts], axis=1) for i in items}

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

    def calculate_treatment(self, treatment):
        try:
            treatment_fun = self.tratment_funs[treatment]
        except KeyError:
            warnings.warn(f"WasteAndCircularity: No function to handle waste treatment '{treatment}' found in .treatment_funs")
        else:
            treatment_fun(self)


# BELOW TREATMENT-SPECIFIC FUNCTIONS ADDED TO .treatment_funs ARE DEFINED
# THESE SHOULD TAKE A WasteAndCircularity OBJECT AS ONLY INPUT

def anaerobic_digestion(waste:WasteAndCircularity):

    # Shorthands to retriever
    g = waste.par.get
    gf = waste.par.get_from_frame

    # Get CH4 and CO2 density and CH4 specific energy
    waste.par.clear()
    CH4d = g('CH4_density')[0] # kg/Nm3
    CO2d = g('CO2_density')[0] # kg/Nm3
    CH4se = g('CH4_specific_energy')[0] # kWh/kg

    # Get feedstock VS, C and B0
    feedstock_VS = waste.data_attr.get('feedstock_VS').xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)
    feedstock_C = waste.data_attr.get('feedstock_C').xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)
    feedstock_B0 = waste.data_attr.get('feedstock_B0').xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)

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
    uses = waste.par.get_unique('energy_prod', qry="parameter == 'biogas_use_share'")

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
    waste.data_attr.get('energy_prod').loc[:, energy_prod.columns] = energy_prod

    # Calculate energy use in biogas production and use [kWh]
    energy_sources = waste.par.get_unique('energy_source')

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
    waste.data_attr.get('energy_use').loc[:, energy_use.columns] = energy_use

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

    # Create dict for storing DFs of digestate composition after storage
    digestate_after_storage = dict()

    # Calculate C and VS in digestate after storage assuming that C/VS
    # remains constant [kg]
    digestate_after_storage['C'] = (
        digestate_C -
        CH4_loss_store * (12/(12+4*1)) -
        CO2_loss_store * (12/(12+2*16))
    )
    digestate_after_storage['VS'] = digestate_after_storage['C'] * (feedstock_VS/feedstock_C).fillna(0)

    # Aggregate VS losses and update data attribute
    VS_loss = pd.concat([
        pd.concat({'CH4bio': (CH4_loss_prod + CH4_loss_use_agg + CH4_loss_store)}, names=['compound'], axis=1).reorder_levels([1,2,3,4,0], axis=1),
        pd.concat({'CO2bio': (CO2_loss_prod + CO2_loss_store)}, names=['compound'], axis=1).reorder_levels([1,2,3,4,0], axis=1)
    ], axis=1)
    waste.data_attr.get('losses_VS').loc[:, VS_loss.columns] = VS_loss

    # Calculate losses of NPK during digestate storage and store data attributes
    for element in ['N','P','K']:

        waste.par.clear()
        waste.par.set(element = element)

        # Get ammount of N, P or K in digestate assuming
        # no losses of NPK during biogas production
        digestate = waste.data_attr.get('feedstock_'+element).xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)

        # Get losses dataframe slice
        df = waste.data_attr.get('losses_'+element).xs('anaerobic digestion', level='treatment', axis=1, drop_level=False)

        if (waste.par.data.xs((element,'digestate_loss_storage'),level=('f_element','parameter'))>0).any():

            loss_factors_storage = waste.par.get_from_frame('digestate_loss_storage', df)/100
            loss_storage = multiply_aligned(loss_factors_storage, digestate)

            if element == 'N':
                # NOx-N and N2 storage losses are calculated from plant available nitrogen
                loss_storage.update(
                    (
                        multiply_aligned(loss_factors_storage, digestate) *
                        (waste.par.get_from_frame('digestate_TAN_share', df)/100)
                    )
                    .loc[:,(slice(None), slice(None), slice(None), slice(None), ['NOx-N', 'N2'])]
                )
        else:
            loss_storage = df * 0.0

        digestate_after_storage[element] = \
            digestate - loss_storage.T.groupby(['feedstock','feedstock_group','feedstock_type','treatment']).sum().T

        # Update data attribute for storage losses 
        waste.data_attr.get('losses_'+element).loc[:, loss_storage.columns] = loss_storage

    # Create dicts for storing DFs of digestate to spread as fertiliser
    # and digestate going to landfill
    digestate_to_spread = dict()
    digestate_to_landfill = dict()

    # Get fraction of digestate assumed to be spread on
    # cropland as organic fertiliser
    try:
        waste.par.clear()
        frac_to_spread = pd.Series(
            waste.par.get(
                'digestate_share_as_fertiliser',
                **digestate_after_storage['C'].columns
                .to_frame().to_dict('list')
            )/100,
            index = digestate_after_storage['C'].columns
        )
    except KeyError:
        # Set to 1 if parameter not defined (for compatibility)
        frac_to_spread = 1
    
    # Separate digestate going to landfill and returned to cropland
    for k,df in digestate_after_storage.items():
        digestate_to_landfill[k] = (
            df.mul(1-frac_to_spread, axis=1)
            .rename(lambda x: 'digestate', axis=1)
            .rename(lambda x: 'landfill', level='treatment', axis=1)
            .T.groupby(df.columns.names).sum().T
        )
        digestate_to_spread[k] = df.mul(frac_to_spread, axis=1)
    # Calculate B0 for digestate going to landfill
    waste.par.clear()
    digestate_to_landfill['B0'] = digestate_to_landfill['VS'] * gf('digestate_B0', digestate_to_landfill['VS'])

    # Uppdate data attributes for digestate used as organic fertiliser
    for element in ['C','N','P','K']:
        waste.data_attr.get('organic_fertiliser_'+element).loc[:, digestate_to_spread[element].columns] = digestate_to_spread[element]
        if element == 'N':
            # Calculate and store plant available nitrogen in digestate
            waste.data_attr.get('organic_fertiliser_TAN').loc[:, digestate_to_spread[element].columns] = \
                digestate_to_spread[element] * (waste.par.get_from_frame('digestate_TAN_share', digestate_to_spread[element])/100)

    # Add digestate going to landfill to the feedstock dataframes
    # so it is handled in that treatment function
    for element in ['C','VS','B0','N','P','K']:
        waste.data_attr.get('feedstock_'+element)[digestate_to_landfill[element].columns] = digestate_to_landfill[element]
        # Reindex data attribute tables with the added feedstock
        waste.create_data_attribute_tables(reindex=True)

    return None


def composting(waste:WasteAndCircularity):
    print('NOT IMPLEMENTED!', end=' ')
    return None

def incineration(waste:WasteAndCircularity):

    # Get feedstock VS
    feedstock_VS = (
        waste.data_attr.get('feedstock_VS')
        .xs('incineration', level='treatment', axis=1, drop_level=False)
    )

    # Calculate heating value [kWh]
    feedstock_heating_value = (
        feedstock_VS * 
        waste.par.get_from_frame('incineration_heating_value', feedstock_VS)
    )

    # Get energy production efficiency
    energy_prod_efficiency = waste.par.get_from_frame(
        'incineration_efficiency',
        waste.data_attr.get('energy_prod')
        .xs('incineration', level='treatment',
            axis=1, drop_level=False)
    )

    # Calculate energy production
    waste.data_attr.get('energy_prod').update(
        multiply_aligned(energy_prod_efficiency, feedstock_heating_value)
    )
        
    # Calculate combustion emissions
    for element in ['VS','N','P','K']:

        # Get emission factors
        waste.par.clear()
        EFs = waste.par.get_from_frame(
            'incineration_emissions',
            waste.data_attr.get(f'losses_{element}')
            .xs('incineration', level='treatment', axis=1, drop_level=False),
            element = element
        )

        # Calculate emissions
        waste.data_attr.get(f'losses_{element}').update(
            multiply_aligned(EFs, feedstock_heating_value)
        )

    # No organic fertilisers generated, set to zero
    idx = pd.IndexSlice
    for element in ['C','N','P','K','TAN']:
        waste.data_attr.get(f'organic_fertiliser_{element}').loc[:,idx[:,:,:,'incineration']] = 0.0

    return None

def landfill(waste:WasteAndCircularity):

    # Get feedstock VS
    feedstock_VS = (
        waste.data_attr.get('feedstock_VS')
        .xs('landfill', level='treatment', axis=1, drop_level=False)
    )

    for element in ['VS','N','P','K']:
        # Get emission factors
        waste.par.clear()
        EFs = waste.par.get_from_frame(
            'landfill_emissions',
            waste.data_attr.get(f'losses_{element}')
            .xs('landfill', level='treatment', axis=1, drop_level=False),
            element = element
        )

        # Calculate emissions
        waste.data_attr.get(f'losses_{element}').update(
            multiply_aligned(EFs, feedstock_VS)
        )

    # No organic fertilisers generated, set to zero
    idx = pd.IndexSlice
    for element in ['C','N','P','K','TAN']:
        waste.data_attr.get(f'organic_fertiliser_{element}').loc[:,idx[:,:,:,'landfill']] = 0.0

    return None

    