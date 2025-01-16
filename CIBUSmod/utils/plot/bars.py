import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill

from .utils import SetColorCycle, wrapText, wrapXTicks

def bar(
    data,
    ax = None,

    stacked = True,
    group_levels = None,
    group_spacing = 0.5,
    bar_width = 0.8,
    
    cmap = 'tab10',
    reverse_cmap = False,
    edgecolor = 'black',
    linewidth = 0.5,
    
    totmarker = 'o',
    totmarker_size = 10,
    
    axis_padding = 10,
    
    sort_groups = False,
    sort_xlabels = False,
    sort_categories = False,
    
    xlabel = '',
    xlabel_fontsize = 11,
    ylabel = '',
    ylabel_fontsize = 11,
    ticklabels_fontsize = 10,
    
    grouptitle = '',
    grouptitle_fontsize = 11,

    grouplabels_fontsize = 10,
    grouplabels_vertical = False,

    ylim = None
):
    '''Plots a grouped and stacked bar chart from a pandas.DataFrame. Columns are taken
    as categories (i.e. color), inner most index level as groups and remaining index levels
    as x labels'''

    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig = ax.get_figure()
    
    data = data.fillna(0)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    if group_levels is not None:
        has_groups = True
        if not pd.api.types.is_list_like(group_levels):
            group_levels = [group_levels]
    else:
        has_groups = False
        # Add dummy group level
        data = pd.concat({'dummy': data}, names=['group'])
        group_levels = ['group']
    
    xlab_levels = data.index.droplevel(group_levels).names
    data.index = pd.MultiIndex.from_tuples([
        (ng, g) for ng, g in zip(
            data.index.droplevel(xlab_levels), data.index.droplevel(group_levels)
        )
    ], names = ['group', 'xlab'])
   
    sorter = pd.concat([
        data.sum(axis=1).groupby('group', sort=False).transform('sum'),
        data.sum(axis=1)
    ], axis=1)

    def sort_key(x):
        if x.name == 'group':
            return x.map({idx:i for i,idx in enumerate(data.index.unique('group'))})
        elif x.name == 'xlab':
            return x.map({idx:i for i,idx in enumerate(data.index.unique('xlab'))})
        else:
            return x * -1
    
    by = []
    if sort_groups:
        by += [0,'group']
    else:
        by += ['group']
    if sort_xlabels:
        by += [1,'xlab']
    else:
        by += ['xlab']
    sorter = sorter.sort_values(by=by, key=sort_key)
    data = data.reindex(sorter.index)
    
    groups = data.index.unique('group')
    
    if sort_categories:
        sorter = data.abs().sum().sort_values(ascending=False).index
        data = data.reindex(sorter, axis=1)
    cols = data.columns
    ncols = len(cols)

    # Set color map
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    SetColorCycle(ax, ncols, cmap, reverse_cmap)

    y = data.where(data>0,0)
    y_neg = data.where(data<0,0)
    has_neg = (y_neg < 0).any().any()

    if stacked:
        y_base = pd.DataFrame(
            np.concatenate([
                np.zeros((len(y),1)),
                y.cumsum(axis=1).iloc[:,:-1].values
            ], axis=1),
            index = y.index,
            columns = y.columns
        )
        y_neg_base = pd.DataFrame(
            np.concatenate([
                np.zeros((len(y),1)),
                y_neg.cumsum(axis=1).iloc[:,:-1].values
            ], axis=1),
            index = y_neg.index,
            columns = y_neg.columns
        )
    else:
        y_base = y_neg_base = pd.DataFrame(
            0.0,
            index = y.index,
            columns = y.columns
        )
       
    x = pd.Series(
        np.nan,
        index = y.index
    )
    
    for i, group in enumerate(x.index.get_level_values('group')):
        if i > 0:
            x.iloc[i] = x.iloc[i-1] + 1 + (group_spacing if group != prev_group else 0)
            prev_group = group
        else:
            x.iloc[i] = 0.5
            prev_group = group

    x_shift = pd.Series(0.0, index=y.columns)
    if not stacked:
        x_shift.loc[:] = np.linspace(-0.5 + 1/ncols/2, 0.5 - 1/ncols/2, ncols) * bar_width

    for i, col in enumerate(cols):
        ax.bar(
            x = x + x_shift.iloc[i],
            height = y.iloc[:,i],
            bottom = y_base.iloc[:,i],
            width = bar_width if stacked else 1/ncols*bar_width**2,
            label = '\n'.join(col) if isinstance(col, tuple) else col,
            edgecolor = edgecolor,
            linewidth = linewidth
        )
    
    if has_neg:
        SetColorCycle(ax, ncols, cmap, reverse_cmap)
        for i, col in enumerate(cols):
            ax.bar(
                x = x + x_shift.iloc[i],
                height = y_neg.iloc[:,i],
                bottom = y_neg_base.iloc[:,i],
                width = bar_width if stacked else 1/ncols*bar_width,
                edgecolor = edgecolor,
                linewidth = linewidth
            )
        if stacked and ncols > 1:
            ax.scatter(
                x = x,
                y = (y+y_neg).sum(axis=1),
                color = 'black',
                marker = totmarker,
                s = totmarker_size
            )
    
    # Set axis range with padding
    
    xmin = x.min() - 0.5
    xmax = x.max() + 0.5
    ax.set_xlim(xmin, xmax)

    if ylim is not None:
        ymin = ylim[0]
        ymax = ylim[1]
    else:
        if stacked:
            ymax = y.sum(axis=1).max() * 1.01
            ymin = y_neg.sum(axis=1).min() * 1.01 if has_neg else -ymax * 0.01
        else:
            ymax = y.max(axis=1).max() * 1.01
            ymin = y_neg.min(axis=1).min() * 1.01 if has_neg else -ymax * 0.01
    ax.set_ylim(ymin, ymax)

    group_axs = []
    for group in groups:
        xg = x.loc[group]
        axg = ax.inset_axes(
            bounds = [xg.min() - 0.5,
                      ymin,
                      xg.max() - xg.min() + 1,
                      ymax-ymin],
            transform = ax.transData,
            zorder = 0
        )
        
        axg.set_xlim(xg.min() - 0.5, xg.max() + 0.5)
        if has_groups:

            group_label = ', '.join(group) if isinstance(group, tuple) else group

            axg.tick_params(axis='x', which='major', labelsize = grouplabels_fontsize, labelrotation = 90 if grouplabels_vertical else 0, bottom=False, top=False, labelbottom=False, labeltop=True)
            axg.tick_params(axis='x', which='minor', direction='in', width=0.8, bottom=False, top=True)
            if not grouplabels_vertical:
                wrapXTicks(
                    axg.set_xticks(
                        ticks = [xg.mean()],
                        labels = [group_label]
                    )
                )
            else:
                axg.set_xticks(
                        ticks = [xg.mean()],
                        labels = [group_label]
                    )
            axg.set_xticks(
                minor = True,
                ticks = [xg.min() - 0.5, xg.max() + 0.5],
                labels = ['', '']
            )
        else:
            axg.set_xticks([])
            axg.spines['top'].set_visible(False)
        
        axg.spines[['top','bottom']].set_position(('outward', axis_padding))
        axg.spines['bottom'].set_bounds((xg.min(), xg.max()))
        axg.spines[['right','left']].set_visible(False)

        axg.set_yticks([])
        
        group_axs += [axg]
       
    # Apply axis padding and adjust spine ranges
    ax.spines[['left','bottom']].set_position(('outward', axis_padding))
    yticks = [t for t in ax.get_yticks() if t <= ymax and t >= ymin]
    ax.set_yticks(yticks)
    ax.spines['left'].set_bounds((min(yticks), max(yticks)))
    
    ax.spines[['top','right','bottom']].set_visible(False)
    
    # Add zero line
    if has_neg:
        ax.axhline(color = 'black', linewidth = 0.5, linestyle = '--')

    # Set x ticks
    xtick_labels = [', '.join(l) if isinstance(l, tuple) else str(l) for l in x.index.droplevel('group')]
    ax.set_xticks(
        ticks = x,
        labels = xtick_labels,
        rotation = 90
    )

    if ncols > 1:
        ax.legend(
            frameon = False,
            fontsize = 9,
            ncols = max(1, np.floor(ncols/5)),
            reverse = True if stacked else False
        )

    # Set titles and labels
    wrapText(ax.set_title(grouptitle, fontsize=grouptitle_fontsize, pad=10))
    wrapText(ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=10))
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.tick_params(labelsize=ticklabels_fontsize)

    return ax, group_axs

def bar_stacked_grouped(data,**kwargs):
    '''Replaced by 'bar' '''
    if 'group_levels' not in kwargs:
        kwargs['group_levels'] = 0
    return bar(data,**kwargs)

def waterfall(
    data,
    ax,
    breaks = [],
    color_pos = '#F5897A',
    color_neg = '#83D382',
    color_line = 'grey',
    width = 0.5,
    label = 'percent',
    label_size = 11,
    **kwargs
):

    breaks = [data.index[0]] + breaks + [data.index[-1]]

    new_idx = pd.Index(
        [
            i+j for i in data.index
            for j in (['_',''] if (i in breaks) & (i != data.index[0]) else [''])
        ],
        name=data.index.name
    )

    data = data.reindex(new_idx)
    for i in range(len(data)):
        if data.iloc[i,:].isna().all():
            data.iloc[i,:] = data.iloc[i+1,:]

    data_main = data.copy()
    data_main.loc[~data.index.isin(breaks)] = np.nan

    data_chg = data.sum(axis=1).rename('tot').to_frame()
    data_chg['_hide'] = np.nan
    data_chg['_pos'] = np.nan
    data_chg['_neg'] = np.nan
    data_chg['_lab'] = np.nan

    for i in range(1,len(data_chg)-1):
        # Calculate differance
        dif = data_chg.iloc[i,0] - data_chg.iloc[i-1,0]
        
        #
        if label == 'percent':
            data_chg.iloc[i,4] = dif / data_chg.iloc[i-1,0] * 100
        elif label == 'absolute':
            data_chg.iloc[i,4] = dif
        else:
            pass
        
        if dif > 0:
            data_chg.iloc[i,2] = dif
            data_chg.iloc[i,1] = \
            data_chg.iloc[i-1,0]
        elif dif < 0:
            data_chg.iloc[i,3] = -dif
            data_chg.iloc[i,1] = \
            data_chg.iloc[i-1,0] + dif

    data_main.plot(
        kind='bar',
        stacked=True,
        width=width,
        ax=ax,
        **kwargs
    )
    
    data_chg.loc[:,['_hide','_pos','_neg']].plot(
        kind='bar',
        stacked=True,
        width=width,
        ax=ax,
        color = [(0,0,0,0),color_pos,color_neg],
        legend = False
    )
    
    y_adj = data_main.max().max()*0.02
    if label in ['percent','absolute']:
        for i in range(len(data_chg)):
            x=i
            val = data_chg.iloc[i,4]
            lab = f'{val:.1f}' if abs(val) < 10 else f'{val:.0f}'
            if ~np.isnan(data_chg.iloc[i,2]):
                y = data_chg.iloc[i,1] + data_chg.iloc[i,2] + y_adj
                s='+'+lab+('%' if label == 'percent' else '')
                ax.text(
                    x,y,s,
                    ha = 'center',
                    va = 'bottom',
                    size = label_size
                )
            elif ~np.isnan(data_chg.iloc[i,3]):
                y = data_chg.iloc[i,1] - y_adj
                s=lab+('%' if label == 'percent' else '')
                ax.text(
                    x,y,s,
                    ha = 'center',
                    va = 'top',
                    size = label_size
                )
    
    for i in range(len(data)-1):
        y = [data_chg.iloc[i,0]]*2
        x = [i-width/2+0.02,i+1+width/2-0.02]
        ax.plot(
            x,y,
            color=color_line,
            linewidth=1
        )

def marimekko(
    data,
    ax = None,
    
    x_column = None,
    
    cmap = 'tab10',
    reverse_cmap = False,
    edgecolor = 'black',
    linewidth = 0.5,
    
    axis_padding = 10,

    xlim_pad = 0.01,
    ylim_pad = 0.01,
    xlim = None,
    ylim = None,
    
    xlabel = '',
    xlabel_fontsize = 11,
    ylabel = '',
    ylabel_fontsize = 11,
    ticklabels_fontsize = 10,

    sort_xcategories = True,
    sort_categories = True,
    
    xcategorylabels_threshold = 0.02,
    xcategorylabels_wrap = False,
    xcategorylabels_fontsize = 9,
    xcategorylabels_rotation = 45,
    xcategorylabels_halign = 'right'
):

    if ax is None:
        ax = plt.gca()
    
    # Set color map
    # Set color map
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    SetColorCycle(ax, len(data.columns), cmap, reverse_cmap)
    
    if x_column is None:
        x = data.sum(axis=1)
        y = data.div(x, axis=0)
    else:
        x = data.loc[:,x_column]
        y = data.drop(x_column, axis=1)
        
    # Sort categories
    if sort_categories:
        sorter = y.sum().sort_values(ascending=False).index
        y = y.loc[:,sorter]
    
    # Sort x axis categories
    if sort_xcategories:
        if x_column is None:
            sorter = y.sort_values(y.columns.tolist(), ascending=False).index
        else:
            sorter = y.sum(axis=1).sort_values(ascending=False).index
    x = x.loc[sorter]
    y = y.loc[sorter]
    
    x_sum = x.sum()
    x_base = pd.Series(
        np.concatenate([np.zeros(1), x.cumsum().values[:-1]]),
        index = x.index
    )
    x_mid = x_base + x/2
        
    y_base = pd.DataFrame(
        np.concatenate([np.zeros((len(y),1)), y.cumsum(axis=1).values[:,:-1]], axis=1),
        index = y.index,
        columns = y.columns
    )
    y_sum = y.sum(axis=1).max()
    
    # Plot bars
    for col in y.columns:
        ax.bar(x=x_base, height=y.loc[:,col], width=x, bottom=y_base.loc[:,col], align='edge', edgecolor=edgecolor, linewidth=linewidth, label=col)
      
    

    # Set axes ranges
    if not xlim:
        xlim = (-x_sum*0.01, x_sum*(1+xlim_pad))
    ax.set_xlim(xlim[0],xlim[1])
    if not ylim:
        ylim = (-y_sum*0.01, y_sum*(1+ylim_pad))
    ax.set_ylim(ylim[0],ylim[1])

    # Style axes and spines
    ax.tick_params(axis='x', which='major', labelsize = ticklabels_fontsize,
                   bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.tick_params(axis='x', which='minor', labelsize = xcategorylabels_fontsize,
                   length=5, width=0.8,
                   bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.spines[['left','top']].set_position(('outward', axis_padding))
    xticks = [t for t in ax.get_xticks() if t <= xlim[1] and t >= 0]
    yticks = [t for t in ax.get_yticks() if t <= ylim[1] and t >= 0]
    ax.spines['top'].set_bounds((min(xticks), max(xticks)))
    ax.spines['left'].set_bounds((min(yticks), max(yticks)))
    ax.spines[['bottom','right']].set_visible(False)

    # Create category labels
    ticks = [x for x,w in zip(x_mid,x) if w > x_sum * xcategorylabels_threshold]
    labels = [fill(l, xcategorylabels_wrap, break_long_words=False) if xcategorylabels_wrap else l
                for l,w in zip(x.index,x) if w > x_sum * xcategorylabels_threshold]
    ax.set_xticks(
        minor=True,
        ticks = ticks,
        labels = labels,
        rotation = xcategorylabels_rotation,
        ha = xcategorylabels_halign,
        rotation_mode = 'anchor'
    )
    
    ax.legend(
        frameon = False,
        fontsize = 9,
        reverse = True
    )
    
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.xaxis.set_label_position('top') 
    ax.tick_params(axis='y', labelsize=ticklabels_fontsize)

    return ax



