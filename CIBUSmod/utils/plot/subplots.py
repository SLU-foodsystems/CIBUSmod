import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def subplots(
    data,
    index = None,
    columns = None,
    ncols = 3,
    size = (6,6),
    kind='bar',
    **kwargs
):

    if (index is not None) and (columns is not None):
        raise ValueError('Can´t supply both index and columns')

    if index is not None:
        if isinstance(index,str):
            index = [index]
        levels = index if len(index)>1 else index[0]
        axis = 0
    elif columns is not None:
        if isinstance(columns,str):
            columns = [index]
        levels = columns if len(columns)>1 else columns[0]
        axis = 1
    else:
        raise ValueError('index or columns must be supplied')
        
    idx = data.index.droplevel(list(set(data.index.names) - set(index))).unique()
    nplots = len(idx)
    nrows = int(np.ceil(nplots/ncols))

    fig, axs = plt.subplots(nrows,ncols, figsize=(size[0]*ncols,size[1]*nrows))

    for i,ax in zip(idx,axs.flatten()):
        pd = data.xs(key=i,level=levels,axis=axis)
        pd.plot(kind=kind,ax=ax,**kwargs)
        if isinstance(i,str):
            ax.set_title(i)
        else:
            ax.set_title(', '.join(i))
        
    return fig, axs