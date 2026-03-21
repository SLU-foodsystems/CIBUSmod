import os
import pandas as pd
import numpy as np

from . import  IMPACT_DATA_PATH
from .general import get_emissions
from .temp_funcs import generate_temp_responses
from ..utils.session_db import Session
from ..utils.verbose_print import verbose_init

def get_GHG(
        session : Session,
        scn : list[str]|str = 'all',
        years : list[str]|str = 'all',
        CO2eq : str|None = 'GWP100 AR4',
        interpolate : bool = False
        ) -> pd.DataFrame:
    '''Function to get greenhouse gas emissions from Session. Emissions are expressed in kg or kg CO2-eq
    depending on 'CO2eq' setting.
    
    This function relies on impact.get_emissions() to collect all emissions and then uses the file
    'CIBUSmod/impact/data/emi_to_ghg.csv' to 1) select only greenhouse gas emissions, 2) convert units if
    necessary (e.g. N2O-N --> N2O), and 3) calculate indirect N2O emissions from other nitrogen emissions
    and leaching.
    
    Characterization factors are defined in 'CIBUSmod/impact/data/ghg_to_CO2eq.csv' where the method column
    corresponds to the allowable settings for 'CO2eq'. It is possible to add additional characterization methods
    by adding rows in this file.
    
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
    pandas.DataFrame of greenhouse gas emissions in kg or kg CO2-eq'''

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
    session : Session|pd.DataFrame,
    groupby : list[str]|str = 'all',
    scn : str = 'all',
    years : str = 'all',
    extend : int = 0,
    extend_emissions : bool = True,
    temp_resp_model : str = 'C2012',
    temp_resp_version : str = 'AR5'
) -> pd.DataFrame:
    '''Function to calculate the temperature response measured in Kelvin (K) from time-series of greenhouse gas emissions.
    
    Paramters
    ---------
    session : Session object
        Alternatively a pandas.DataFrame can be supplyed,
        should be a DataFrame of greenhouse gas emissions
        as returned by impact.get_GHG(CO2eq=None, interpolate=True)
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
    temp_resp_model : str, default 'C2012'
        Temperature response model to use. One of:
            - 'C2012' (Collins et al. 2012, AGTP formulation)
            - 'E2013' (Ericsson et al. 2013, convolution method)
            - 'E2014' (Ericsson et al. 2014, empirical constants)
    temp_resp_version : str, default 'AR5
        IPCC version for constants. One of:
            - 'AR5'
            - 'AR4'
    
    Returns
    -------
    pandas.DataFrame of temperature response in Kelvin (K)'''

    vprint = verbose_init(True, id_str='get_deltaT')

    # Get greenhouse gas emissions
    vprint('Getting GHG emissions ...')
    if isinstance(session, Session):
        ghg = get_GHG(session, scn, years, CO2eq=None, interpolate=True)
    elif isinstance(session, pd.DataFrame):
        ghg = session
    else:
        raise ValueError('')

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
    except (IndexError, ValueError):
        groupby.insert(0,'compound')

    rename_ghg = {
        'CO2' : 'co2',
        'CH4bio' : 'ch4bio',
        'CH4fos' : 'ch4fos',
        'N2O' : 'n2o',
        'N2Oind' : 'n2o'
    }

    # Group and sum
    ghg = ghg.T.groupby(groupby).sum().T
    
    vprint('Calculating temperature response ...')
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

        # Pre-compute temp response curves
        temp_curves = generate_temp_responses(n=end_year+ext-start_year, model=temp_resp_model, vers=temp_resp_version)

        # Separate fossil and biogenic CH4 by assuming that one CO2-C is emitted per fossil CH4-C emitted.
        # This is a pragmatic approach to account for the oxidation of fossil CH4 into CO2 but does not
        # capture the true time dynamics.
        temp_curves['ch4bio'] = temp_curves['ch4']
        temp_curves['ch4fos'] = temp_curves.pop('ch4') + ((12 + 2*16) / (12 + 4*1)) * temp_curves['co2']

        for cmp in deltaT.loc[scn].columns.unique('compound'):
            # Get GHG time-series for col
            ghg_data = ghg_scn.loc[scn,cmp]

            # Get temp curve
            temp_curve = np.atleast_2d(
                temp_curves[rename_ghg[cmp]],
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

    vprint(type='end')

    return deltaT_combined

def get_GWPstar(
    session : Session|pd.DataFrame,
    scn : list[str]|str = 'all',
    years : list[str]|str = 'all',
    CO2eq : str = 'GWP100 AR4',
    r : float = 0.75,
    s : float = 0.25,
    dt : int = 20,
    interpolate : bool = False
) -> pd.DataFrame:
    '''Function to get greenhouse gas emissions from Session expressed as kg CO2-w.e.
    according to GWP* (Lynch et.al. 2020), an alternative application of GWPs where
    the CO2-equivalence of short-lived climate pollutant (SLCP) emissions is predominantly
    determined by changes in their emission rate.
    
    Parameters
    ----------
    session : Session object
    scn : (list of) str, default 'all'
        Scenarios to include
    years : (list of) str, default 'all'
        Years to include
    CO2eq : str or None, default 'GWP100 AR4'
        Method for translating GHGs to CO2-eq Should be one of the GWP methods
        available in impact.get_GHG()
    r : float, default 0.75 (as in Lynch et.al. 2020)
        Weight given changes in rate of SLCP emissions
    s : float, default 0.25 (as in Lynch et.al. 2020)
        Weight given to SLCP emissions
    dt : int, default 20 (as in Lynch et.al. 2020)
        Number of years to average changes in SLCP emissions rate over (Δt)
    interpolate : Bool, default False
        Interpolate between defined years
        
    Returns
    -------
    pandas.DataFrame of greenhouse gas emissions in kg CO2-w.e.

        
    
    The GWP* calculations are implemented as described in Lynch et.al. (2020). Emissions before the start
    of the scenarios are assumed equal to emissions in the first year in calculating GWP* for the years up
    unitil 'dt' years after the start year. Only methane is considered a SLCP in the calculations implemented
    in this function.
    
    Lynch, J., Cain, M., Pierrehumbert, R. & Allen, M. (2020).
    Demonstrating GWP*: a means of reporting warming-equivalent emissions that
    captures the contrasting impacts of short- and long-lived climate pollutants.
    Environmental Research Letters, 15(4), 044023. 10.1088/1748-9326/ab6d7e
    '''

    import re
    if match := re.search(r'GWP(\d+)', CO2eq):
        H = int(match.group(1))
    else:
        raise ValueError("'CO2eq' must be one of the GWP methods")
    
    SLCPs = ['CH4bio', 'CH4fos']
    if isinstance(session, Session):
        ghg = get_GHG(session, scn, years, CO2eq, interpolate)
    elif isinstance(session, pd.DataFrame):
        ghg = session
    else:
        raise ValueError('')
    
    SLCP_E = ghg.loc[:,(slice(None),slice(None),slice(None),slice(None),slice(None),SLCPs)]

    # Create df of SLCP emissions at t - Δt
    # Emissions before t0 area assumed equalt to emissions at t0
    SLCP_E_sub = SLCP_E.reindex(
        pd.MultiIndex.from_tuples(
            [(s,str(int(y)-dt)) for s,y in SLCP_E.index],
            names = SLCP_E.index.names
        )
    ).interpolate(limit_direction='both')
    SLCP_E_sub.index = SLCP_E.index

    # Calculate GWP*
    GWPstar = (r * ((SLCP_E-SLCP_E_sub)/dt) * H) + (s * SLCP_E)

    # Re-insert GWP* to main df
    ghg.loc[:,(slice(None),slice(None),slice(None),slice(None),slice(None),SLCPs)] = GWPstar
    
    return ghg