import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bar_stacked_grouped(
    data,
    ax,
    cmap,
    width = 0.7,
    labbel_size = 11,
    min_alpha = 0.5
):
    '''Plots a grouped and stacked bar chart from a pandas.DataFrame. Columns are taken
    as categories (i.e. color), inner most index level as groups and remaining index levels
    as x labels'''
    
    colors=plt.get_cmap(cmap).colors[0:len(data.columns)]
    for i in range(len(data.columns)):
        df = data.iloc[:,i::].sum(axis=1).unstack()
        df.columns = ['_']*(len(df.columns)-1) + [data.columns[i]]
        df.plot(kind='bar',width=width, ax=ax, color=colors[i], edgecolor='grey',legend=False)
    
    # Labels
    df = data.sum(axis=1).unstack()
    x = np.linspace(-width/4,width/4,len(df.columns))
    y = df.iloc[0,:].values + (df.max().max()*0.02)
    s = df.columns.values
    for i in range(len(x)):
        plt.text(
            x[i],y[i],s[i],
            rotation = 'vertical',
            rotation_mode = 'anchor',
            va='center_baseline',
            ha='left',
            size=labbel_size
        )
    
    # Make shade
    hide_colors = [(1,1,1,a) for a in np.linspace(0,(1-min_alpha),len(df.columns))]
    df.columns = ['_']*len(df.columns)
    df.plot(kind='bar', width=width, color=hide_colors, legend=False, ax=ax)

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
                plt.text(
                    x,y,s,
                    ha = 'center',
                    va = 'bottom',
                    size = label_size
                )
            elif ~np.isnan(data_chg.iloc[i,3]):
                y = data_chg.iloc[i,1] - y_adj
                s=lab+('%' if label == 'percent' else '')
                plt.text(
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