import os
import pandas as pd
import numpy as np

from . import  IMPACT_DATA_PATH
from .general import get_emissions
from ..temp_calc.temp_utils import ghg_to_temp
from ..utils.session_db import Session

def get_GHG(session, scn='all', years = 'all', CO2eq='GWP100 AR4', interpolate=False):
    '''
    Parameters
    ----------
    session : Session object
    scn : (list of) str, default 'all'
    years : (list of) str, default 'all'
    CO2eq : Bool, default True
        Translate GHGs to CO2-eq
    interpolate : Bool, default False
        Interpolate between defined years'''

    # Get conversion factors, compound --> GHG
    emi_to_ghg = pd.read_csv(os.path.join(IMPACT_DATA_PATH, 'emi_to_ghg.csv'), index_col='compound')
    to_GHG = emi_to_ghg['factor'].to_dict()
    to_GHG_names = emi_to_ghg['ghg'].to_dict()

    # Get emissions
    res = get_emissions(session, scn=scn, years=years, interpolate = interpolate)

    # Get only GHGs and compunds with indirect GHG emissions
    res = res.loc[:,res.columns.isin(to_GHG, level='compound')]

    # Convert to GHG emissions
    res = (
        res
        .mul([to_GHG[cp] for cp in res.columns.get_level_values('compound')], axis=1)
        .rename(to_GHG_names, axis=1)
        .T.groupby(res.columns.names).sum().T
    )

    if CO2eq:
        # Get conversion factors
        to_CO2eq = pd.read_csv(os.path.join(IMPACT_DATA_PATH, 'ghg_to_CO2eq.csv'), index_col=['ghg','method'])['factor']

        # Select method
        to_CO2eq = to_CO2eq.xs(CO2eq, level='method').to_dict()

        # Calculate CO2 equivalents
        res = (
            res
            .mul([to_CO2eq[cp] for cp in res.columns.get_level_values('compound')], axis=1)
        )

    return res

def get_deltaT(
    session : Session,
    groupby : list[str]|str = 'all',
    scn : str = 'all',
    years : str = 'all',
    extend : int = 0
):

    # Get greenhouse gas emissions
    print('Getting GHG emissions ...')
    ghg = get_GHG(session, scn, years, CO2eq=None, interpolate=True)

    if groupby == 'all':
        groupby = ['process', 'sub-process', 'prod_system', 'item', 'region', 'compound']
    elif groupby == 'none':
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]

    groupby_orig = groupby.copy()

    # Make sure 'compound' is first in groupby
    try:
        groupby.insert(0,groupby.pop(groupby.index('compound')))
    except IndexError:
        groupby.insert(0,'compound')

    rename_ghg = {
        'CO2' : 'co2',
        'CH4bio' : 'ch4',
        'CH4fos' : 'ch4',
        'N2O' : 'n2o',
        'N2Oind' : 'n2o'
    }

    # Group and sum
    ghg = ghg.T.groupby(groupby).sum().T

    start_year = ghg.index.get_level_values('year').astype(int).min()
    end_year = ghg.index.get_level_values('year').astype(int).max()

    print('Calculating temperature response ...')
    # Costruct output dataframe with extended year index
    scns = ghg.index.unique('scn')
    deltaT = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_product(
            [scns, map(str, range(start_year, end_year + extend + 1))],
            names=['scn', 'year']
        ),
        columns=ghg.columns
    )

    for scn in deltaT.index.unique('scn'):
        for cmp in deltaT.loc[scn].columns.unique('compound'):
            # Get GHG time-series for col
            ghg_data = ghg.loc[scn,cmp]
            # Pre-compute temp response curve
            temp_curve = np.atleast_2d(
                ghg_to_temp(
                    ghg = rename_ghg[cmp],
                    time_horizon=end_year+extend-start_year
                )
            ).T
            # Calculate temperature response
            temp_resp = sum([
                np.pad(temp_curve[0:end_year+extend-y+1],[(y-start_year,0),(0,0)]) @ np.atleast_2d(ghg_data.loc[str(y)])
                for y in range(start_year, end_year + 1)
            ])
            # Store results
            deltaT.loc[scn,cmp] = temp_resp

    if groupby != groupby_orig:
        if len(groupby_orig)>0:
            deltaT = deltaT.T.groupby(groupby_orig).sum().T
        else:
            deltaT = deltaT.sum(axis=1)

    return deltaT