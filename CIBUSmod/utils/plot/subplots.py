import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def subplots(
    data,
    index = None,
    columns = None,
    plot_fn = None,
    ncols = None,
    size = (5,5),
    **kwargs
):
    '''Function to plot subplots based on a pandas.DataFrame or Series

    Parameters
    ----------
    index, columns : (list of) str (default None)
        Index and/or column levels to use for constructing subplot panels.
    plot_fn : plot function (default None)
        Must accept a pandas.DataFrame or Series as its first argument and
        the keyword argument 'ax' for supplying a matplotlib Axes object.
        If None, pandas .plot() method is used.
    ncols : int (default None)
        Number of columns in subplots.
        If None, ncols is selected automatically.
    size : (float, float)
        Size of each subplot panel.
    **kwargs
        Keyword arguments that are passed on to plot_fn or .plot() method.

    Returns
    -------
    (Figure, array of Axes)
    
    '''

    if index is not None:
        if isinstance(index, str):
            index = [index]
        idxs = data.index.droplevel(list(set(data.index.names) - set(index))).unique()
    else:
        idxs = ['']
        index = []
    
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        cols = data.columns.droplevel(list(set(data.columns.names) - set(columns))).unique()
    else:
        cols = ['']
        columns = []

    nplots = len(idxs) * len(cols)
    # Pick a decent value for ncols
    if ncols is None:
        if len(cols)>1:
            largest_divisor = 1
            for i in range(2, min(len(cols), 6) + 1):
                if len(cols) % i == 0 and i < 6:
                    largest_divisor = i
            ncols = largest_divisor
            if ncols < nplots/ncols:
                ncols = min(int(np.ceil(np.sqrt(len(cols)))), 6)
        else:
            ncols = min(int(np.ceil(np.sqrt(len(idxs)))), 6)  
    
    ncols = min(ncols, nplots)
    nrows = int(np.ceil(nplots/ncols))
    
    fig, axs = plt.subplots(nrows,ncols, figsize=(size[0]*ncols,size[1]*nrows))
    
    for idx_col,ax in zip(product(idxs, cols), axs.flatten()):
        idx, col = idx_col
        if isinstance(idx, str):
            idx = (idx,)
        if isinstance(col, str):
            col = (col,)
    
        plot_df = data
        
        if len(index)>0:
            for i, lvl in enumerate(index):
                if plot_df.index.nlevels>1:
                    plot_df = plot_df.xs(key=idx[i], level=lvl, axis=0)
                else:
                    plot_df = plot_df.xs(key=idx[i], axis=0)
    
        if len(columns)>0:
            axis = 1 if isinstance(plot_df, pd.DataFrame) else 0
            for i, lvl in enumerate(columns):
                if (axis==0 and plot_df.index.nlevels>1) or (axis==1 and plot_df.columns.nlevels>1):
                    plot_df = plot_df.xs(key=col[i], level=lvl, axis=axis)
                else:
                    plot_df = plot_df.xs(key=col[i], axis=axis)
                        
        if plot_fn is not None:
            plot_fn(plot_df, ax=ax, **kwargs)
        else:
            plot_df.plot(ax=ax, **kwargs)

        ax.set_title('\n'.join([lab for lab in idx+col if lab != '']))

    return fig, axs