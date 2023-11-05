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
    groupby = 'all',
    interpolate = False
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
            x = x.groupby(ig if len(ig)>1 else ig[0]).sum()
        elif not dont_agg:
            # Aggregate across all index levels
            x = x.sum()

        if isinstance(x,pd.DataFrame) and cg is not None:
            if len(cg)>0:
                # Group by column levels and aggregate
                x = x.groupby(cg if len(cg)>1 else cg[0], axis=1).sum()
            elif not dont_agg:
                # Aggregate across all column levels
                x = x.sum(axis=1)
        elif isinstance(x,pd.Series):
            if cg is not None:
                if len(cg)>0:
                    # Group by column (now index) levels and aggregate
                    x = x.groupby(cg if len(cg)>1 else cg[0]).sum()
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

    if interpolate:
        # Interpolate to yearly data
        
        # Create new index with all years represented
        new_idx = pd.MultiIndex.from_tuples(
            [
                (scn,str(year))
                for scn in data.index.get_level_values('scn').unique()
                for year in range(
                    min(data.loc[scn].index.astype(int)),
                    max(data.loc[scn].index.astype(int))+1
                )
            ],
            names = ['scn','year']
        )
        # Reindex and interpolate
        data = data.reindex(new_idx).interpolate()

    return data

def get_emissions(output, interpolate=False):
    # Define emissions processes and corresponding modules
    # and data attributes
    prs = {
        'enteric fermentation' : {
            'module' : ['AnimalHerd'],
            'attr' : ['enteric_methane']
        },
        'manure management' : {
            'module' : ['AnimalHerd'],
            'attr' : ['manure.N_loss',
                      'manure.VS_loss']
        },
        'energy use' : {
            'module' : ['AnimalHerd',
                        'CropProduction'],
            'attr' : ['energy_use_emissions',
                      'energy_use_supply_chain_emissions']
        },
        'fertiliser production' : {
            'module' : ['CropProduction'],
            'attr' : ['fertiliser.mineral_N_supply_chain_emissions']
        },
        'agricultural soils' : {
            'module' : ['CropProduction'],
            'attr' : ['fertiliser.manure_N_application_loss',
                      'fertiliser.manure_N_soil_loss',
                      'fertiliser.mineral_N_application_loss',
                      'fertiliser.mineral_N_soil_loss',
                      'fertiliser.crop_residues_N_soil_loss',
                      'fertiliser.organic_soil_N_loss']
        }
    }

    d = []

    for pr in prs:
        mds = prs[pr]['module']
        ats = prs[pr]['attr']
        for md in mds:
            for at in ats:
                if md == 'CropProduction':
                    df = output.get_attr(
                        module = 'CropProduction',
                        attr = at,
                        groupby = {'prod_system':None, 'crop':'crop_group2',
                                   'region':None, 'compound':None},
                        interpolate = interpolate
                    )
                    df = df.rename_axis(columns = {'crop' : 'item'})
                elif md == 'AnimalHerd':
                    df = output.get_attr(
                        module = 'AnimalHerd',
                        attr = at,
                        groupby = ['prod_system','species',
                                   'breed','region','compound'],
                        interpolate = interpolate
                    )
                    df.columns = pd.MultiIndex.from_tuples(
                        [(ps,f'{sp}, {br}',re,cp) for ps,sp,br,re,cp in df.columns],
                        names = ['prod_system', 'item',
                                 'region', 'compound']
                    )

                df = pd.concat([df], keys=[pr], names=['process'], axis=1)
                d += [df]

    res = pd.concat(d, axis=1)
    res = (
        res.groupby(
            ['process','prod_system', 'item',
             'region', 'compound'],
            axis=1
        ).sum()
    )
    
    return res

def get_GHG(output, CO2eq=True, interpolate=False):
    
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
    
    # Get emissions
    res = get_emissions(output, interpolate = interpolate)

    # Get only GHGs and compunds with indirect GHG emissions
    res = res.loc[:,res.columns.isin(to_GHG, level='compound')]

    # Convert to GHG emissions
    res = (
        res
        .mul([to_GHG[cp] for cp in res.columns.get_level_values('compound')], axis=1)
        .rename(to_GHG_names, axis=1)
        .groupby(res.columns.names, axis=1)
        .sum()
    )

    if CO2eq:
        # Calculate CO2 equivalents
        res = (
            res
            .mul([to_CO2eq[cp] for cp in res.columns.get_level_values('compound')], axis=1)
        )

    return res

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