import numpy as np

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
    else:
        raise ValueError("Only method='Shannon' allowed")


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
        # Calculate proportion of cropland use per crop_group in for each group
        .T.groupby(groupby_after_get).transform(lambda x: x/x.sum()).T
        # Calculate Shannon Diversity Index for each region
        .T.groupby(groupby_after_get).apply(fun).T
    )

    return res

def _shannon(x):
    return -(x * np.log(x.replace({0:np.nan}))).sum()