import numpy as np

def get_crop_div(session, groupby='all', scn='all', years = 'all', method='Shannon', crop_group='crop_group', land_use='cropland', interpolate=False):
    '''
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

    if method == 'Shannon':
        fun = _shannon
    else:
        raise ValueError("Only method='Shannon' allowed")


    res = (
        session.get_attr('CropProduction','area', groupby_get)
        .loc[:,land_use]
        # Calculate proportion of cropland use per crop_group in for each group
        .T.groupby(groupby_after_get).transform(lambda x: x/x.sum()).T
        # Calculate Shannon Diversity Index for each region
        .T.groupby(groupby_after_get).apply(fun).T
    )

    return res

def _shannon(x):
    return -(x * np.log(x.replace({0:np.nan}))).sum()