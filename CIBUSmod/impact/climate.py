from .general import get_emissions

def get_GHG(session, scn='all', years = 'all', CO2eq=True, interpolate=False):
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
    
    # Conversion factors --------->
    to_GHG = {
        'CO2' : 1,
        'CH4bio' : 1,
        'CH4fos' : 1,
        'N2O' : 1,
        'N2O-N' : (44/28),
        'NH3' : (14/17) * 0.10 * (44/28), # NH3 -> NH3-N -> N2O-N -> N2O
        'NH3-N' : 0.010 * (44/28), # NH3-N -> N2O-N -> N2O (IPCC 2019 Guidelines Table 11.3)
        'NOx' : (14/30) * 0.10 * (44/28), # # NOx -> NOx-N -> N2O-N -> N2O (Assumes NOx = NO)
        'NOx-N' : 0.010 * (44/28), # NOx-N -> N2O-N -> N2O (IPCC 2019 Guidelines Table 11.3)
        'NO3-N' : 0.011 * (44/28), # NO3-N -> N2O-N -> N2O (IPCC 2019 Guidelines Table 11.3)

    }
    # -----
    to_GHG_names = {
        'CO2' : 'CO2',
        'CH4bio' : 'CH4bio',
        'CH4fos' : 'CH4fos',
        'N2O-N' : 'N2O',
        'NH3' : 'N2Oind',
        'NH3-N' : 'N2Oind',
        'NOx' : 'N2Oind',
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
        # Calculate CO2 equivalents
        res = (
            res
            .mul([to_CO2eq[cp] for cp in res.columns.get_level_values('compound')], axis=1)
        )

    return res