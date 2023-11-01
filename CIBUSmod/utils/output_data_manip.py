import warnings
import pandas as pd

from ..utils.retriever import ParameterRetriever

from ..main_modules.animal_herd import AnimalHerd, StaticAnimalHerd
from ..mgmt_modules.feed_mgmt import Feed
from ..mgmt_modules.manure_mgmt import Manure

from ..utils.misc import rgetattr,rsetattr,inv_dict

def concat_herds(herds):
    '''Combines multiple AnimalHerd objects
    
    Parameters
    ----------
    herds : itterable of AnimalHerd objects
    
    Returns
    -------
    StaticAnimalHerd object'''
    res_herd = StaticAnimalHerd()

    res_herd.id_attr = AnimalHerd.id_attr
    for attr in AnimalHerd.id_attr:
        setattr(res_herd,attr,'aggregated')

    res_herd.feed = Feed()
    res_herd.manure = Manure()

    # Check presence of data attributes in AnimalHerd objects
    # Only attributes present in all AnimalHerd objects are 
    # retained in the combined StaticAnimalHerd object
    data_attr_union = set.union(*[set(h.data_attr) for h in herds])
    data_attr_in_all = set.intersection(*[set(h.data_attr) for h in herds])
    data_attr_in_some = data_attr_union - data_attr_in_all
    if len(data_attr_in_some) > 0:
        pass
        # Should a warning be printed here?
        # warnings.warn(f'Data attributes {data_attr_in_some} not pressent in all AnimalHerds and therfore not retained.')

    # Go through data attributes and concatenate
    for attr in data_attr_in_all:

        df = pd.concat(
            [
                pd.concat({herd.species : 
                    pd.concat({herd.breed :
                        pd.concat({herd.sub_system : rgetattr(herd,attr)},
                            names=['sub_system'],axis=1)},
                        names=['breed'],axis=1)},
                    names=['species'],axis=1)
                if rgetattr(herd,attr) is not None else None for herd in herds
            ],
            axis=1
        )

        # Group and sum columns to avoid duplicates
        df = df.groupby(df.columns.names, axis=1).sum()

        rsetattr(res_herd,attr,df)
    
    res_herd.data_attr = data_attr_in_all

    return res_herd

def get_attr(
    output,
    module,
    attr,
    groupby='all'
):
    '''Get specified data attribute from output.
    
    Parameters
    ----------
    output : Output or pandas.DataFrame
        CIBUSmod outputs
    module : str
        Module to get output from: 'DemandAndConversions', 'Regions', 'CropProduction' or 'AnimalHerd'
    attr : str
        data attribute to get
    groupby : str, list or dict
        If str or list data is grouped and aggregated by these index/column levels.
        If 'all' data is not aggregated
        If 'none'  data is summed over all index/columns
        If a dict is supplied relation tables are used
        
    Returns
    -------
    pandas.DataFrame or Series with scenario (scn) and year as index and <groupby>
    as columns.
    '''

    short_hands = {
        'D':'DemandAndConversions', 'R':'Regions',
        'C':'CropProduction', 'A':'AnimalHerd'
    }
    if module not in short_hands.values():
        try:
            module = short_hands[module.upper()]
        except KeyError:
            raise ValueError('module not found')
    if groupby == 'all':
        groupby = []
        dont_agg = True
    else:
        dont_agg = False
    if groupby == 'none':
        groupby = []
    
    if isinstance(groupby,str):
        groupby = [groupby]
    if isinstance(groupby,dict):
        rel = {k:v for k,v in groupby.items() if v is not None and k != v}
        groupby = list(groupby)
    else:
        rel = {}
        
    d = []
    for idx in output.index:
        # Get attribute
        x = rgetattr(output.loc[idx,module],attr)
        
        # Get index levels to group by
        ig = [g for g in groupby if g in x.index.names]
        if isinstance(x,pd.DataFrame):
            # Get column levels to group by
            cg = [g for g in groupby if g in x.columns.names]
        else:
            cg = None
        
        for lvl in [g for g in ig if g in rel]:
            # Rename index based on relation table
            x = x.rename(ParameterRetriever.get_rel(lvl,rel[lvl]), level=lvl)
        if cg is not None:
            for lvl in [g for g in cg if g in rel]:
                # Rename columns based on relation table
                x = x.rename(ParameterRetriever.get_rel(lvl,rel[lvl]), axis=1, level=lvl)
        
        if len(ig)>0:
            # Group by index levels and aggregate
            x = x.groupby(ig).sum()
        elif not dont_agg:
            # Aggregate across all index levels
            x = x.sum()

        if isinstance(x,pd.DataFrame) and cg is not None:
            if len(cg)>0:
                # Group by column levels and aggregate
                x = x.groupby(cg, axis=1).sum()
            elif not dont_agg:
                # Aggregate across all column levels
                x = x.sum(axis=1)
        elif isinstance(x,pd.Series):
            if cg is not None:
                if len(cg)>0:
                    # Group by column (now index) levels and aggregate
                    x = x.groupby(cg).sum()
                elif not dont_agg:
                    # Aggregate across all column (now index) levels
                    x = x.sum()

        if isinstance(x,pd.DataFrame):
            nlevels = x.columns.nlevels
            if nlevels == 1 and isinstance(x.columns,pd.MultiIndex):
                # Fix problem with single-level MultiIndex stacking by
                # converting to Index
                x.columns = x.columns.get_level_values(0)
            # Stack dataframe to sries
            x = x.stack(list(range(nlevels)))
        if not isinstance(x,pd.Series):
            # If float returned create series
            x = pd.Series(x)

        d.append(x)

    # Combine and transpose
    data = pd.concat(d, axis=1).T
    data.index = output.index

    if len(data.columns) == 1:
        # If only one column. Make series with attr as name
        data = data.iloc[:,0]
        data.name = attr
        
    if len(groupby)>1:
        # Reorder column levels as specified in groupby
        data = data.reorder_levels(groupby, axis=1)

    return data

def get_GHG(output, CO2eq=True):
    
    # Conversion factors --------->
    to_GHG = {
        'CO2' : 1,
        'CH4bio' : 1,
        'CH4fos' : 1,
        'N2O' : 1,
        'N2O-N' : (44/28),
        'NH3-N' : 0.01 * (44/28),

    }
    # -----
    to_GHG_names = {
        'CO2' : 'CO2',
        'CH4bio' : 'CH4bio',
        'CH4fos' : 'CH4fos',
        'N2O-N' : 'N2O',
        'NH3-N' : 'N2Oind'
    }
    # -----
    to_CO2eq = {
        'CO2' : 1,
        'CH4bio' : 25,
        'CH4fos' : 26,
        'N2O' : 298,
        'N2Oind' : 298
    }
    # <--------------------------
    
    for scn, year in output.index:
        # ENTERIC METHANE
        enteric = (
            output.loc[(scn,year),'AnimalHerd'].enteric_methane
            .groupby(['prod_system','species','breed','compound'], axis=1)
            .sum()
        )
        enteric.columns = pd.MultiIndex.from_tuples(
            [(ps,f'{sp}, {br}',cp) for ps,sp,br,cp in enteric.columns],
            names = ['prod_system', 'item', 'compound']
        )
        enteric = pd.concat([enteric], keys=['enteric fermentation'], names=['process'], axis=1)

        # MANURE MANAGEMENT
        manure = (
            pd.concat([
                # N losses
                output.loc[(scn,year),'AnimalHerd']
                .manure.N_loss
                .groupby(['prod_system','species','breed','compound'], axis=1)
                .sum(),
                # VS losses
                output.loc[(scn,year),'AnimalHerd']
                .manure.VS_loss
                .groupby(['prod_system','species','breed','compound'], axis=1)
                .sum()
            ], axis=1)
        )
        manure.columns = pd.MultiIndex.from_tuples(
            [(ps,f'{sp}, {br}',cp) for ps,sp,br,cp in manure.columns],
            names = ['prod_system', 'item', 'compound']
        )
        manure = pd.concat([manure], keys=['manure management'], names=['process'], axis=1)

        # AGRICULTURAL SOILS
        rel = ParameterRetriever.get_rel('crop','crop_group2')
        soils = (
            pd.concat([
                getattr(output.loc[(scn,year),'CropProduction'].fertiliser, attr)
                .groupby('compound', axis=1).sum()
                .rename(rel, level='crop', axis=0)
                .rename_axis(index={'crop':'item'})
                .groupby(['region','prod_system','item']).sum()
                .unstack(['prod_system','item'])
                for attr in
                ['manure_N_application_loss','manure_N_soil_loss',
                'mineral_N_application_loss','mineral_N_soil_loss',
                'crop_residues_N_soil_loss','organic_soil_N_loss']
            ], axis=1)
            .reorder_levels(['prod_system','item','compound'], axis=1)
        )
        soils = pd.concat([soils], keys=['agricultural soils'], names=['process'], axis=1)

        # ENERGY USE EMISSIONS
        # crops
        energy_crops = (
            output.loc[(scn,year),'CropProduction'].energy_use_emissions
            .groupby('compound', axis=1).sum()
            .rename(rel, level='crop', axis=0)
            .rename_axis(index={'crop':'item'})
            .groupby(['region','prod_system','item']).sum()
            .unstack(['prod_system','item'])
            .reorder_levels(['prod_system','item','compound'], axis=1)
        )

        # livestock
        energy_livestock = (
            output.loc[(scn,year),'AnimalHerd'].energy_use_emissions
            .groupby(['prod_system','species','breed','compound'], axis=1)
            .sum()
        )
        energy_livestock.columns = pd.MultiIndex.from_tuples(
            [(ps,f'{sp}, {br}',cp) for ps,sp,br,cp in energy_livestock.columns],
            names = ['prod_system', 'item', 'compound']
        )

        # combine
        energy = pd.concat([energy_crops,energy_livestock], axis=1)
        energy = pd.concat([energy], keys=['energy use'], names=['process'], axis=1)

        # INPUTS SUPPLY CHAIN EMISSIONS
        # crops
        inputs_crops = (
            output.loc[(scn,year),'CropProduction'].input_supply_chain_emissions
            .groupby('compound', axis=1).sum()
            .rename(rel, level='crop', axis=0)
            .rename_axis(index={'crop':'item'})
            .groupby(['region','prod_system','item']).sum()
            .unstack(['prod_system','item'])
            .reorder_levels(['prod_system','item','compound'], axis=1)
            .sort_index(axis=1)
        )
        inputs_crops = pd.concat([inputs_crops], keys=['inputs'], names=['process'], axis=1)
        
        # livestock

        # combine
        inputs = inputs_crops

        # COMBINE ALL PROCESSES
        res = (
            pd.concat([enteric,manure,soils,energy,inputs], axis=1)
            .groupby(['process','prod_system','item','compound'], axis=1)
            .sum()
        )

        # Get only GHGs and compunds with indirect GHG emissions
        res = res.loc[:,res.columns.isin(to_GHG, level='compound')]

        # Convert to GHG emissions
        res = (
            res
            .mul([to_GHG[cp] for cp in res.columns.get_level_values('compound')], axis=1)
            .rename(to_GHG_names, axis=1)
        )

        res = pd.concat([res], keys=[year], names=['year'], axis=1)
        res = pd.concat([res], keys=[scn], names=['scn'], axis=1)

        if CO2eq:
            # Calculate CO2 equivalents
            res = (
                res
                .mul([to_CO2eq[cp] for cp in res.columns.get_level_values('compound')], axis=1)
            )
        
        try:
            result = pd.concat([result,res], axis=1)
        except NameError:
            result = res

    return result

def to_ICBM(output):
    ''''''
    output = output.copy()
    cropland = inv_dict(ParameterRetriever.get_rel('crop','land_use'))['cropland']
    first_par = True
    for par in ['area','harvest','manure']:
        first_scnyear = True
        for scn, year in output.index:
            if par == 'area':
                res = (
                    output.loc[(scn,year),'CropProduction']
                    .area.loc[(cropland,slice(None),slice(None))]
                )
                res = pd.concat([res], keys=['area_ha'], names=['par'])
            elif par == 'harvest':
                res = (
                    output.loc[(scn,year),'CropProduction']
                    .harvest_dm.loc[(cropland,slice(None),slice(None))]
                )
                res = pd.concat([res], keys=['harvest_kgdm'], names=['par'])
            elif par == 'manure':
                res = (
                    output.loc[(scn,year),'CropProduction']
                    .fertiliser.manure_C.loc[(cropland,slice(None),slice(None)),:]
                    .groupby('species', axis=1).sum().stack()
                )
                res.index.names = res.index.names[0:3]+['par']
                res = res.rename({sp:'manure_'+sp+'_kgC' for sp in res.index.get_level_values('par').unique()})

            res = pd.concat([res], keys=[year], names=['year'])
            res = pd.concat([res], keys=[scn], names=['scn'])

            if first_scnyear:
                result = res
                first_scnyear = False
            else:
                result = pd.concat([result,res], axis=0)

        y0 = result.index.get_level_values('year').astype(int).min()
        yend = result.index.get_level_values('year').astype(int).max()

        result = result.unstack('year').reindex(columns=pd.Index([str(y) for y in range(y0,yend+1)], name='year'))
        result.columns = result.columns.astype(int)
        result = result.interpolate(axis=1)

        result = result.stack().unstack('par')
        
        if first_par:
            comb_result = result
            first_par = False
        else:
            comb_result = pd.concat([comb_result,result], axis=1)
    
    return comb_result