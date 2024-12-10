import numpy as np

def get_LSU(session, groupby = 'all', scn='all', years = 'all', method='factors', interpolate=False):
    '''Gives animal numbers in terms of livestock units (LSU)

    Parameters
    ----------
    session : Session object
    groupby : str, list or dict, default 'all'
        If str or list data is grouped and aggregated by these levels.
        If 'all' all available levels are returned
        If 'none'  data is summed over all levels
        If a dict is supplied relation tables are used
    scn : (list of) str
    years : (list of) str
    method : str, default 'factors'
        Method to calculate LSUs
            'factors' = LSU factors according to Eurostat
                        https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Livestock_unit_(LSU)
    interpolate : Bool, default False
            If True interpolate between defined years

    '''

    # Make sure species, breed and animal are in group by
    # If not add and aggregate after calculating LSU
    reaggregate = False
    if groupby == 'none':
        groupby = []
    if groupby != 'all':
        if isinstance(groupby, str):
            groupby = [groupby]
        groupby_orig = groupby.copy()
        if 'species' not in groupby:
            reaggregate = True
            groupby += ['species']
        if 'breed' not in groupby:
            reaggregate = True
            groupby += ['breed']
        if 'animal' not in groupby:
            reaggregate = True
            groupby += ['animal']

    # Get data
    res = session.get_attr(
        module = 'AnimalHerd',
        attr = 'heads',
        groupby = groupby,
        scn = scn,
        years = years,
        interpolate = interpolate
    )

    # Get species, breed and animal level values
    sp_br_an = list(zip(
        res.columns.get_level_values('species'),
        res.columns.get_level_values('breed'),
        res.columns.get_level_values('animal')
    ))

    # Calculate livestock units
    if method == 'factors':
        res = res.mul([_LSU_from_factors(x) for x in sp_br_an], axis=1)
    else:
        raise ValueError("Only method='factors' allowed")
    
    if reaggregate:
        if groupby_orig == []:
            res = res.sum(axis=1)
        else:
            res = res.T.groupby(groupby_orig).sum().T

    return res

def _LSU_from_factors(x):
    '''Gives livestock units (LSU) per head based on a tuple of (species, breed, animal)
    according to https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Livestock_unit_(LSU)'''
    sp,br,an = x
    if sp == 'cattle':
        if an == 'cows':
            if br == 'dairy':
                return 1
            else:
                return 0.8
        elif an == 'calves':
            return 0.4
        elif an == 'breeding bulls':
            return 1
        else:
            return 0.8
    elif sp == 'sheep':
        return 0.1
    elif sp == 'pigs':
        if an == 'piglets':
            return 0.027
        elif an == 'sows':
            return 0.5
        else:
            return 0.3
    elif sp == 'horses':
        return 0.8
    elif sp == 'poultry':
        if an == 'broilers':
            return 0.007
        else:
            return 0.014
    else:
        return np.nan