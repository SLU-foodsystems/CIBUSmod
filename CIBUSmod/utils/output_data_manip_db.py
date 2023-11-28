import warnings
import pandas as pd

from ..utils.retriever import ParameterRetriever

from ..main_modules.animal_herd import AnimalHerd, StaticAnimalHerd
from ..mgmt_modules.feed_mgmt import Feed
from ..mgmt_modules.manure_mgmt import Manure

from ..utils.misc import rgetattr, rsetattr, inv_dict

def get_emissions(session, interpolate=False):
    # Define emissions processes and corresponding modules
    # and data attributes
    '''Collect emissions to the environment stored in various data attributes
    and modules. 

    Parameters
    ----------
    session : Session object
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
            'attr' : ['fertiliser.mineral_N_supply_chain_emissions',
                      'fertiliser.mineral_P_supply_chain_emissions',
                      'fertiliser.mineral_K_supply_chain_emissions']
        },
        'agricultural soils' : {
            'module' : ['CropProduction'],
            'attr' : ['fertiliser.manure_N_application_loss',
                      'fertiliser.manure_N_soil_loss',
                      'fertiliser.mineral_N_application_loss',
                      'fertiliser.mineral_N_soil_loss',
                      'fertiliser.crop_residues_N_soil_loss',
                      'fertiliser.organic_soil_N_loss',
                      'fertiliser.leaching_N']
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
                        interpolate = interpolate
                    )
                    df = df.rename_axis(columns = {'crop' : 'item'})
                elif md == 'AnimalHerd':
                    df = session.get_attr(
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

def get_GHG(session, CO2eq=True, interpolate=False):
    '''
    Parameters
    ----------
    session : Session object
    CO2eq : Bool, default True
        Translate GHGs to CO2-eq
    interpolate : Bool, default False
        Interpolate between defined years'''
    
    # Conversion factors --------->
    to_GHG = {
        'CO2' : 1,
        'CH4bio' : 1,
        'CH4fos' : 1,
        'N2O' : 1,
        'N2O-N' : (44/28),
        'NH3-N' : 0.010 * (44/28), # IPCC 2019 Guidelines Table 11.3
        'NOx-N' : 0.010 * (44/28), # IPCC 2019 Guidelines Table 11.3
        'NO3-N' : 0.011 * (44/28), # IPCC 2019 Guidelines Table 11.3

    }
    # -----
    to_GHG_names = {
        'CO2' : 'CO2',
        'CH4bio' : 'CH4bio',
        'CH4fos' : 'CH4fos',
        'N2O-N' : 'N2O',
        'NH3-N' : 'N2Oind',
        'NOx-N' : 'N2Oind',
        'NO3-N' : 'N2Oind'
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
    res = get_emissions(session, interpolate = interpolate)

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

def to_ICBM(session):
    '''Create data for soil carbon modelling.

    Parameters
    ----------
    session : Session object
    
    Returns
    -------
    pandas.DataFrame'''

    ats = ['area','harvest_dm','crop_residues_harvest','fertiliser.manure_C']
    d = []
    for at in ats:
        df = (
            session.get_attr(
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
        if at == 'crop_residues_harvest':
            df = df.rename('crop_residues_harvest_kgdm')
        if at == 'fertiliser.manure_C':
            df = df.rename({sp:'manure_'+sp+'_kgC' for sp in df.columns}, axis=1)
            
        d += [df]
    
    res = pd.concat(d, axis=1)

    # Only select cropland
    sel_crops = inv_dict(ParameterRetriever.get_rel('crop','land_use'))['cropland']
    res = res.loc[(slice(None), sel_crops),:]
    
    return res