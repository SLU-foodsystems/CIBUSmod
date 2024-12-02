import os
import pandas as pd

from . import  IMPACT_DATA_PATH
from .general import get_emissions

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