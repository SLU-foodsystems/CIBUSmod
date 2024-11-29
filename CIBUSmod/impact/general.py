import pandas as pd

from ..utils.retriever import ParameterRetriever

from ..utils.misc import inv_dict

def get_emissions(session, scn='all', years='all', interpolate=False):
    # Define emissions processes and corresponding modules
    # and data attributes
    '''Collect emissions to the environment stored in various data attributes
    and modules. 

    Parameters
    ----------
    session : Session object
    scn : (list of) str, default 'all'
    years : (list of) str, default 'all'
    interpolate : Bool, default False
        Interpolate between defined years
    
    Returns
    -------
    pandas.DataFrame'''

    prs = {
        'enteric fermentation' : {
            'module' : ['AnimalHerd'],
            'attr' : ['enteric_methane']
        },
        'manure management' : {
            'module' : ['AnimalHerd'],
            'attr' : ['manure.N_loss',
                      'manure.P_loss',
                      'manure.K_loss',
                      'manure.VS_loss']
        },
        'energy use' : {
            'module' : ['AnimalHerd',
                        'CropProduction',
                        'WasteAndCircularity'],
            'attr' : ['energy_use_emissions',
                      'energy_use_supply_chain_emissions']
        },
        'fertiliser production' : {
            'module' : ['CropProduction'],
            'attr' : ['fertiliser.mineral_N_supply_chain_emissions',
                      'fertiliser.mineral_P_supply_chain_emissions',
                      'fertiliser.mineral_K_supply_chain_emissions',
                      'fertiliser.liming_supply_chain_emissions']
        },
        'agricultural soils' : {
            'module' : ['CropProduction'],
            'attr' : ['fertiliser.manure_N_application_loss',
                      'fertiliser.manure_N_soil_loss',
                      'fertiliser.organic_N_application_loss',
                      'fertiliser.organic_N_soil_loss',
                      'fertiliser.mineral_N_application_loss',
                      'fertiliser.mineral_N_soil_loss',
                      'fertiliser.crop_residues_N_soil_loss',
                      'fertiliser.organic_soil_N_loss',
                      'fertiliser.leaching_N']
        },
        'liming' : {
            'module' : ['CropProduction'],
            'attr' : ['fertiliser.liming_emissions']
        },
        'waste and circularity' : {
            'module' : ['WasteAndCircularity'],
            'attr' : ['losses_N',
                      'losses_P',
                      'losses_K',
                      'losses_VS']
        }
    }

    d = []

    for pr in prs:
        mds = prs[pr]['module']
        ats = prs[pr]['attr']
        for md in mds:
            for at in ats:
                if md == 'CropProduction':
                    df = session.get_attr(
                        module = 'CropProduction',
                        attr = at,
                        groupby = {'prod_system':None, 'crop':'crop_group2',
                                   'region':None, 'compound':None},
                        scn = scn,
                        years = years,
                        interpolate = interpolate
                    )
                    df = df.rename_axis(columns = {'crop_group2' : 'item'})
                elif md == 'WasteAndCircularity':
                    df = session.get_attr(
                        module = 'WasteAndCircularity',
                        attr = at,
                        groupby = {'treatment':None,
                                   'region':None, 'compound':None},
                        scn = scn,
                        years = years,
                        interpolate = interpolate
                    )
                    df = df.rename_axis(columns = {'treatment' : 'item'})
                    # Add 'prod_system' to column level as not applicable (n/a)
                    df = pd.concat({'n/a': df}, names=['prod_system'], axis=1)
                elif md == 'AnimalHerd':
                    df = session.get_attr(
                        module = 'AnimalHerd',
                        attr = at,
                        groupby = ['prod_system','species',
                                   'breed','region','compound'],
                        scn = scn,
                        years = years,
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
        res
        .T.groupby(
            ['process','prod_system', 'item',
             'region', 'compound']
        ).sum().T
    )
    
    return res

def to_ICBM(session):
    '''Create data for soil carbon modelling.

    Parameters
    ----------
    session : Session object
    
    Returns
    -------
    pandas.DataFrame'''

    ats = ['area','harvest_dm','crop_residues_harvest','fertiliser.manure_C', 'fertiliser.organic_C']
    d = []
    for at in ats:
        df = (
            session.get_attr(
                module = 'CropProduction',
                attr = at,
                groupby = ['crop','prod_system','region']
                + (['species'] if 'manure' in at else [])
                + (['treatment'] if 'organic' in at else []),
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
        if at == 'crop_residues_harvest':
            df = df.rename('crop_residues_harvest_kgdm')
        if at == 'fertiliser.manure_C':
            df = df.rename({sp:'manure_'+sp+'_kgC' for sp in df.columns}, axis=1)
        if at == 'fertiliser.organic_C':
            df = df.rename({tr:'organic_'+tr+'_kgC' for tr in df.columns}, axis=1)
            
        d += [df]
    
    res = pd.concat(d, axis=1)

    # Only select cropland
    sel_crops = inv_dict(ParameterRetriever.get_rel('crop','land_use'))['cropland']
    res = res.loc[(slice(None), sel_crops),:]
    
    return res