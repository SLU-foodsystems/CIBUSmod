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
    interpolate = False,
    keep_duplicate_levels = 'index',
    suffixes = ('_idx','_col')
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
    groupby : str, list or dict, default 'all'
        If str or list data is grouped and aggregated by these index/column levels.
        If 'all' data is not aggregated
        If 'none'  data is summed over all index/columns
        If a dict is supplied relation tables are used
    interpolate : Bool, default True
        If True interpolate between defined years
    keep_duplicate_levels: {'index','columns','both'}, default 'index'
        If the same groupby level is in both index and columns of data attribute
        then keep level on the specified axis. If 'both', both levels are
        retained and renamed with 'suffixes'
    suffixes : itterable of len 2, default ('_idx','_col')
        Suffixes to use for index and column levels if 'keep_duplicate_levels' is 'both'
        
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
    
    # Get first scn and year
    x = rgetattr(output.iloc[0].loc[module],attr)

    if groupby == 'all':
        groupby = list(x.index.names)
        if isinstance(x,pd.DataFrame):
            groupby += [lvl for lvl in x.columns.names if lvl not in groupby]
    if groupby == 'none':
        groupby = []
    
    if isinstance(groupby,str):
        groupby = [groupby]
    if isinstance(groupby,dict):
        rel = {k:v for k,v in groupby.items() if v is not None and k != v}
        groupby = list(groupby)
    else:
        rel = {}
        
    # Check for duplicate groupby levels in both index and
    # columns and if 'keep_duplicate_levels' is 'both', add
    # suffixes in data and groupby list
    if isinstance(x, pd.DataFrame):
        idx_col_same = \
        [lvl for lvl in groupby if lvl in x.index.names and lvl in x.columns.names]
    else:
        idx_col_same = []
    if len(idx_col_same)>0:
        if keep_duplicate_levels == 'both':
            new_groupby = []
            idx_rename = {}
            col_rename = {}
            idx_drop = None
            col_drop = None
            for lvl in groupby:
                if lvl in idx_col_same:
                    idx_rename.update({lvl:lvl+suffixes[0]})
                    col_rename.update({lvl:lvl+suffixes[1]})
                    new_groupby += [lvl+suffixes[0]]
                    new_groupby += [lvl+suffixes[1]]
                else:
                    new_groupby += [lvl]
            groupby = new_groupby
        elif keep_duplicate_levels == 'index':
            idx_rename = None
            col_rename = None
            idx_drop = None
            col_drop = idx_col_same
        elif keep_duplicate_levels == 'columns':
            idx_rename = None
            col_rename = None
            idx_drop = idx_col_same
            col_drop = None
        else:
            raise ValueError("'keep_duplicate_levels' must be one of {'index','columns','both'}")
    else:
        idx_rename = None
        col_rename = None
        idx_drop = None
        col_drop = None
        
    d = []
    for idx in output.index:
        # Get attribute
        x = rgetattr(output.loc[idx,module],attr)
        
        # Drop or add suffixes to handle duplicate levels in index and columns
        if idx_rename is not None:
            x = x.rename_axis(index=idx_rename, columns=col_rename)
        if idx_drop is not None:
            x = x.droplevel(idx_drop)
        if col_drop is not None:
            x = x.droplevel(col_drop, axis=1)
        
        # Get index levels to group by
        ig = [g for g in groupby if g in x.index.names]
        if isinstance(x, pd.DataFrame):
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
        else:
            # Aggregate across all index levels
            x = x.sum()

        if isinstance(x,pd.DataFrame) and cg is not None:
            if len(cg)>0:
                # Group by column levels and aggregate
                x = x.groupby(cg if len(cg)>1 else cg[0], axis=1).sum()
            else:
                # Aggregate across all column levels
                x = x.sum(axis=1)
        elif isinstance(x,pd.Series):
            if cg is not None:
                if len(cg)>0:
                    # Group by column (now index) levels and aggregate
                    x = x.groupby(cg if len(cg)>1 else cg[0]).sum()
                else:
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
        
    if isinstance(data, pd.DataFrame) and data.columns.nlevels>1:
        # Reorder column levels as specified in groupby
        data = data.reorder_levels([g for g in groupby if g in data.columns.names], axis=1)

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
    ats = ['area','harvest_dm','fertiliser.manure_C']
    d = []
    for at in ats:
        df = (
            output
            .get_attr(
                module = 'CropProduction',
                attr = at,
                groupby = ['crop','prod_system','region','species'],
                interpolate=True
            )
            .stack(['crop','prod_system','region'])
            .reorder_levels(['scn','crop','prod_system','region','year'])
            .sort_index()
        )
        if at == 'area':
            df = df.rename('area_ha')
        if at == 'harvest_dm':
            df = df.rename('harvest_kgdm')
        if at == 'fertiliser.manure_C':
            df = df.rename({sp:'manure_'+sp+'_kgC' for sp in df.columns}, axis=1)
            
        d += [df]
    
    res = pd.concat(d, axis=1)
    
    return res