import os
import pandas as pd

from . import IMPACT_DATA_PATH

from ..utils.retriever import ParameterRetriever

from ..utils.misc import inv_dict

def get_emissions(session, scn='all', years='all', interpolate=False):
    '''Collect emissions to the environment stored in various data attributes
    and modules. The data attributes from where emissions are collected are defined
    in the 'CIBUSmod/impact/data/emissions_attrs.csv' file. This file also group
    emissions into a 'process' and a 'sub-process', which can be adjusted by changing
    this file.

    Output are grouped by process, sub-process, prod_system, item, region, and compound.
    The 'item' level uses the 'crop_group2' aggregation for CropProduction emissions, 'species, breed' for
    AnimalHerd emissions and 'treatment' for WasteAndCircularity emissions.

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

    # Get all modules and data attributes with emissions and
    # their corresponding process and sub-process from csv
    emissions_attrs = pd.read_csv(os.path.join(IMPACT_DATA_PATH,'emissions_attrs.csv'))

    d = []

    for md,at,pr,spr in emissions_attrs.values:
        try:
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
        except ValueError as e:
            if 'No match for module' in str(e):
                # If Session.get_attr() raises ValueError due to module and data attribute not found
                # silently continue. This occurs if e.g. not all Mgmt modules have been included in
                # the model run.
                continue
            else:
                # If some other ValueError is raised re-raise it.
                raise

        df = pd.concat([df], keys=[spr], names=['sub-process'], axis=1)
        df = pd.concat([df], keys=[pr], names=['process'], axis=1)
        d += [df]

    res = pd.concat(d, axis=1)
    res = (
        res
        .T.groupby(
            ['process','sub-process','prod_system', 'item',
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