import numpy as np
import pandas as pd

def get_crop_div(session, groupby='all', scn='all', years = 'all', method='Shannon', crop_group='crop_group', land_use='cropland', interpolate=False):
    '''Gives crop diversity index

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
        Method to calculate crop diversity
            'Shannon' = Shannon Diversity Index
            'Hill' = Hill numbers of order q=1 (i.e. exp(Shannon))
    crop_group : str, default 'crop_group'
        Aggregation level to use in calculating diversity
    land_use : str, default 'cropland'
        Land use to calculate crop diversity for
    interpolate : Bool, default False
            If True interpolate between defined years
    '''

    if groupby == 'all':
        groupby = ['prod_system', 'region']
    elif groupby == 'none':
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]
    if isinstance(groupby, list):
        groupby = {l:None for l in groupby}

    if crop_group == 'crop':
        crop_group = None

    if 'crop' in groupby:
        raise ValueError("'crop' can't be included in groupby")
    else:
        groupby_get = {'crop':['land_use',crop_group], **groupby}
        groupby_after_get = []
        for k,v in groupby.items():
            if isinstance(v,list):
                for sv in v:
                    groupby_after_get += [sv]
            else:
                if v is None:
                    groupby_after_get += [k]
                else:
                    groupby_after_get += [v]
    if len(groupby_after_get)==0:
        # If no groupby create function to group everything
        groupby_after_get = lambda x: True

    if method == 'Shannon':
        fun = _shannon
    elif method == 'Hill':
        fun = _hill
    else:
        raise ValueError("Only method='Shannon' or 'Hill' allowed")

    res = (
        session.get_attr(
            module = 'CropProduction',
            attr = 'area',
            groupby = groupby_get,
            scn = scn,
            years = years,
            interpolate = interpolate
        )
        .loc[:,land_use]
        # Calculate Diversity Index for each group level
        .T.groupby(groupby_after_get).apply(fun).T
    )

    if res.shape[1] == 1:
        return res.iloc[:,0]

    return res

def _shannon(x):
    """
    Shannon diversity index

    Parameters
    ----------
    x : pd.DataFrame | pd.Series
        Land use by crop(group)

    Returns
    -------
    float
    """
    
    if isinstance(x, pd.Series):
        total = x.sum()
        if total <= 0:
            return np.nan
    
        p = x / total
        p = p[p > 0]
    
        if len(p) == 0:
            return np.nan
        return -(p * np.log(p)).sum()

    if isinstance(x, pd.DataFrame):
        # apply to each (scn, year) column
        return x.apply(_shannon, axis=0)

    raise TypeError("x must be a pandas Series or DataFrame")

def _hill(x, q=1):
    """
    Hill numbers / Effective number of species

    Parameters
    ----------
    x : pd.DataFrame | pd.Series
        Land use by crop(group)
    q : float, default 1
        Order of diversity.
        q = 0 -> richness
        q = 1 -> exp(Shannon)
        q = 2 -> inverse Simpson

    Returns
    -------
    float
    """

    if isinstance(x, pd.Series):
        total = x.sum()
        if total <= 0:
            return np.nan
    
        p = x / total
        p = p[p > 0]
    
        if len(p) == 0:
            return np.nan
    
        if np.isclose(q, 1.0):
            return np.exp(-(p * np.log(p)).sum())
    
        return (p.pow(q).sum()) ** (1.0 / (1.0 - q))

    if isinstance(x, pd.DataFrame):
        # apply to each (scn, year) column
        return x.apply(lambda col: _hill(col, q=q), axis=0)

    raise TypeError("x must be a pandas Series or DataFrame")