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

        to_CO2eq = _get_CO2eq_dict(CO2eq)

        # Calculate CO2 equivalents
        res = (
            res
            .mul([to_CO2eq[cp] for cp in res.columns.get_level_values('compound')], axis=1)
        )

    return res

def _get_CO2eq_dict(method):
    # Get conversion factors
    to_CO2eq = pd.read_csv(os.path.join(IMPACT_DATA_PATH, 'ghg_to_CO2eq.csv'), index_col=['ghg','method'])['factor']

    # Select method
    to_CO2eq = to_CO2eq.xs(method, level='method').to_dict()

    return to_CO2eq

def get_deltaT(
    session : None|Session = None,
    ghg_data : None|pd.DataFrame = None,
    groupby : list[str]|str = 'all',
    scn : str = 'all',
    years : str = 'all',
    extend : int = 0,
    extend_emissions : bool = True
) -> pd.DataFrame:
    '''Function to get the temperature response from greenhouse gas emissions
    
    Paramters
    ---------
    session : Session object
    ghg_data : pandas.DataFrame, default None
        Alternative to supplying a session object, should be a
        DataFrame of greenhouse gas emissions as returned by
        impact.get_GHG(CO2eq=None, interpolate=True)
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
    if ghg_data is not None:
        ghg = ghg_data
    else:
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

def get_rewetting_emissions(
        session : Session,
        year0 : str = '2020',
        CO2eq : str|None = 'GWP100 AR4',
        interpolate : bool = False,
        return_area : bool = False,
        EF_CO2 : float = 0.5*(44/12)*1000, # kg CO2/ha
        EF_CH4 : float = 123 # kg CH4/ha
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    '''Function to calculate emissions of CO2 and CH4 from rewetted organic soils. Any
    reduction in the area of organic soils in year0 is assumed to result in an equivalent
    area of rewetted wetlands.

    This will likely be replaced by a more comprehensive framework for handling land use
    change and associated emissions in the future.
    
    Parameters
    ----------
    session : Session object
    year0 : str
    CO2eq : str or None, default 'GWP100 AR4'
        Method for translating GHGs to CO2-eq, if None emissions are not translated to CO2-eq
    interpolate : Bool, default False
        Interpolate between defined years
    return_area : Bool, default False
        If True, returns area tuple of (rewetted area, emissions)
    EF_CO2 : float, default from Lindgren & Lundblad (2014)
        Emission factor for CO2 emissions in kg CO2/ha
    EF_CH4 : float, default from Lindgren & Lundblad (2014)
        Emission factor for CH4 emissions in kg CH4/ha

    Returns
    -------
    pandas.DataFrame
    of the same structure as returned by impact.get_GHG()
    '''

    from CIBUSmod.impact import IMPACT_DATA_PATH

    # Get area of organic soils
    area_org_soil = session.get_attr('c', 'organic_soil_area', 'region', interpolate=interpolate)

    # Get scenarios in data 
    scns = area_org_soil.index.unique('scn')

    # Calculate area of rewetted organic soils per year
    part_dfs = []
    for scn in scns:
        df_scn = area_org_soil.loc[[scn],:]
        part_dfs.append(
            -area_org_soil.loc[[scn],:].sub(
                area_org_soil.loc[(scn,year0),:],
                axis=1
            # If org_soils @ year <= org_soils @ year0 --> No wetlands
            # There are many other ways to think here...
            ).clip(upper=0)
        )
    # Combine areas for all scenarios
    area_rewetted = pd.concat(part_dfs)

    # Calculate CO2 and CH4 emissions
    CO2_rewetted = area_rewetted * EF_CO2
    CH4_rewetted = area_rewetted * EF_CH4
    # Add compound to column index
    CO2_rewetted = pd.concat({'CO2': CO2_rewetted}, names=['compound'], axis=1)
    CH4_rewetted = pd.concat({'CH4bio': CH4_rewetted}, names=['compound'], axis=1)

    if CO2eq:
        # Convert to CO2eq
        CF_CH4 = pd.read_csv(os.path.join(IMPACT_DATA_PATH, 'ghg_to_CO2eq.csv'), index_col=['ghg','method'])['factor'].loc[('CH4bio',CO2eq)]
        CH4_rewetted *= CF_CH4

    # Combine CO2 and CH4 emissions
    rewetting_emissions = pd.concat([CO2_rewetted, CH4_rewetted], axis=1)

    # Fix column index to match df returned by impact.get_GHG()
    rewetting_emissions = pd.concat({'rewetting': rewetting_emissions}, names=['process'], axis=1)
    rewetting_emissions = pd.concat({'rewetting': rewetting_emissions}, names=['sub-process'], axis=1)
    rewetting_emissions = pd.concat({'n/a': rewetting_emissions}, names=['prod_system'], axis=1)
    rewetting_emissions = pd.concat({'wetlands': rewetting_emissions}, names=['item'], axis=1)
    rewetting_emissions = rewetting_emissions.reorder_levels(['process', 'sub-process', 'prod_system', 'item', 'region', 'compound'], axis=1)

    if return_area:
        return (area_rewetted, rewetting_emissions)
    else:
        return rewetting_emissions