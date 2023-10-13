import pandas as pd
import numpy as np

def waterfall(
    data,
    ax,
    breaks = [],
    color_pos = '#F5897A',
    color_neg = '#83D382',
    color_line = 'grey',
    width = 0.5,
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
    data_chg['_hide'] = 0
    data_chg['_pos'] = 0
    data_chg['_neg'] = 0

    for i in range(1,len(data_chg)-1):
        dif = data_chg.iloc[i,0] - data_chg.iloc[i-1,0]
        if dif >= 0:
            data_chg.iloc[i,2] = dif
            data_chg.iloc[i,1] = \
            data_chg.iloc[i-1,0]
        else:
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
    
    data_chg.iloc[:,[1,2,3]].plot(
        kind='bar',
        stacked=True,
        width=width,
        ax=ax,
        color = [(0,0,0,0),color_pos,color_neg],
        legend = False
    )

    for i in range(len(data)-1):
        y = [data_chg.iloc[i,0]]*2
        x = [i-width/2+0.02,i+1+width/2-0.02]
        ax.plot(
            x,y,
            color=color_line,
            linewidth=1
        )