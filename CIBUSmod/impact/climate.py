import os
import pandas as pd
import numpy as np

from . import  IMPACT_DATA_PATH
from .general import get_emissions
from ..temp_calc.temp_utils import ghg_to_temp
from ..utils.session_db import Session

def get_GHG(
        session : Session,
        scn : list[str]|str = 'all',
        years : list[str]|str = 'all',
        CO2eq : str|None = 'GWP100 AR4',
        interpolate : bool = False
        ) -> pd.DataFrame:
    '''Function to get greenhouse gas emissions from Session
    
    Parameters
    ----------
    session : Session object
    scn : (list of) str, default 'all'
        Scenarios to include
    years : (list of) str, default 'all'
        Years to include
    CO2eq : str or None, default 'GWP100 AR4'
        Method for translating GHGs to CO2-eq, if None emissions are not translated to CO2-eq
    interpolate : Bool, default False
        Interpolate between defined years
        
    Returns
    -------
    pandas.DataFrame'''

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
    extend : int = 0,
    extend_emissions : bool = False
) -> pd.DataFrame:
    '''Function to get the temperature response from greenhouse gas emissions
    
    Paramters
    ---------
    session : Session object
    groupby : (list of) str, default 'all'
        Group results by levels
    scn : (list of) str, default 'all'
        Scenarios to include
    years : (list of) str, default 'all'
        Years to include
    extend : int, default 0
        Years by which to extend analysis of temperature
        response from last year in scenario
    extend_emissions : Bool, default False
        If True emissions are assumed to remain constant
        after last year in scenario, if False emissions
        are assumed to be zero after last year

    Returns
    -------
    pandas.DataFrame'''

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
    
    print('Calculating temperature response ...')
    deltaTs = []

    for scn in ghg.index.unique('scn'):
        ghg_scn = ghg.loc[[scn]]
        # Get start/end year
        start_year = ghg.index.get_level_values('year').astype(int).min()
        end_year = ghg.index.get_level_values('year').astype(int).max()
        # Costruct output dataframe with extended year index
        deltaT = pd.DataFrame(
            0.0,
            index=pd.MultiIndex.from_tuples(
                [(scn,y) for y in map(str, range(start_year, end_year + extend + 1))],
                names = ['scn', 'year']
            ),
            columns=ghg_scn.columns
        )
        if extend_emissions:
            # Shift end year, extend index and fill with emissions from last year
            end_year += extend
            ext = 0
            ghg_scn = ghg_scn.reindex_like(deltaT).ffill(axis=0)
        else:
            ext = extend

        for cmp in deltaT.loc[scn].columns.unique('compound'):
            # Get GHG time-series for col
            ghg_data = ghg_scn.loc[scn,cmp]
            # Pre-compute temp response curve
            temp_curve = np.atleast_2d(
                ghg_to_temp(
                    ghg = rename_ghg[cmp],
                    time_horizon=end_year+ext-start_year
                )
            ).T
            # Calculate temperature response
            temp_resp = sum([
                np.pad(temp_curve[0:end_year+ext-y+1],[(y-start_year,0),(0,0)]) @ np.atleast_2d(ghg_data.loc[str(y)])
                for y in range(start_year, end_year + 1)
            ])
            # Store results
            deltaT.loc[scn,cmp] = temp_resp

        deltaTs += [deltaT]

    # Combine all scenarios
    deltaT_combined = pd.concat(deltaTs)

    if groupby != groupby_orig:
        if len(groupby_orig)>0:
            deltaT_combined = deltaT_combined.T.groupby(groupby_orig).sum().T
        else:
            deltaT_combined = deltaT_combined.sum(axis=1)

    return deltaT_combined