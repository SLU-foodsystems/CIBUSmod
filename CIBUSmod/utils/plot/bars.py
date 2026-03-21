import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill

from .utils import SetColorCycle, wrapText, wrapXTicks

# ============================================================================
# Shared helpers
# ============================================================================

def _prepare_categorical_plot_data(
    data,
    group_levels=None,
    sort_groups=False,
    sort_xlabels=False,
    sort_categories=False
):
    """Prepare data and shared indexing structure for grouped categorical plots.

    Returns
    -------
    data : pd.DataFrame
        Reindexed dataframe with MultiIndex ['group', 'xlab'].
    groups : pd.Index
        Unique group labels in plotting order.
    cols : pd.Index
        Columns in plotting order.
    has_groups : bool
        Whether the user supplied real group levels.
    """
    data = data.fillna(0)
    if isinstance(data, pd.Series):
        data = data.to_frame()

    if group_levels is not None:
        has_groups = True
        if not pd.api.types.is_list_like(group_levels):
            group_levels = [group_levels]
    else:
        has_groups = False
        data = pd.concat({'dummy': data}, names=['group'])
        group_levels = ['group']

    xlab_levels = data.index.droplevel(group_levels).names
    data.index = pd.MultiIndex.from_tuples(
        [
            (ng, g) for ng, g in zip(
                data.index.droplevel(xlab_levels),
                data.index.droplevel(group_levels)
            )
        ],
        names=['group', 'xlab']
    )

    sorter = pd.concat(
        [
            data.sum(axis=1).groupby('group', sort=False).transform('sum'),
            data.sum(axis=1)
        ],
        axis=1
    )

    def sort_key(x):
        if x.name == 'group':
            return x.map({idx: i for i, idx in enumerate(data.index.unique('group'))})
        elif x.name == 'xlab':
            return x.map({idx: i for i, idx in enumerate(data.index.unique('xlab'))})
        else:
            return x * -1

    by = []
    by += [0, 'group'] if sort_groups else ['group']
    by += [1, 'xlab'] if sort_xlabels else ['xlab']

    sorter = sorter.sort_values(by=by, key=sort_key)
    data = data.reindex(sorter.index)

    if sort_categories:
        sorter = data.abs().sum().sort_values(ascending=False).index
        data = data.reindex(sorter, axis=1)

    groups = data.index.unique('group')
    cols = data.columns

    return data, groups, cols, has_groups


def _compute_grouped_x(index, group_spacing=0.5):
    """Compute x positions with gaps between groups."""
    x = pd.Series(np.nan, index=index, dtype=float)

    for i, group in enumerate(x.index.get_level_values('group')):
        if i > 0:
            x.iloc[i] = x.iloc[i - 1] + 1 + (group_spacing if group != prev_group else 0)
            prev_group = group
        else:
            x.iloc[i] = 0.5
            prev_group = group

    return x


def _split_pos_neg(data):
    """Split into positive and negative parts."""
    y_pos = data.where(data > 0, 0.0)
    y_neg = data.where(data < 0, 0.0)
    has_neg = (y_neg < 0).any().any()
    has_pos = (y_pos > 0).any().any()
    mixed_sign = has_pos and has_neg
    return y_pos, y_neg, has_pos, has_neg, mixed_sign


def _compute_stacked_bases(y_pos, y_neg, stacked=True):
    """Compute cumulative bases for positive and negative stacking."""
    if stacked:
        pos_base = pd.DataFrame(
            np.concatenate(
                [
                    np.zeros((len(y_pos), 1)),
                    y_pos.cumsum(axis=1).iloc[:, :-1].values
                ],
                axis=1
            ),
            index=y_pos.index,
            columns=y_pos.columns
        )
        neg_base = pd.DataFrame(
            np.concatenate(
                [
                    np.zeros((len(y_neg), 1)),
                    y_neg.cumsum(axis=1).iloc[:, :-1].values
                ],
                axis=1
            ),
            index=y_neg.index,
            columns=y_neg.columns
        )
    else:
        pos_base = pd.DataFrame(0.0, index=y_pos.index, columns=y_pos.columns)
        neg_base = pd.DataFrame(0.0, index=y_neg.index, columns=y_neg.columns)

    return pos_base, neg_base


def _get_colors(ncols, cmap='tab10', reverse_cmap=False):
    """Get a deterministic list of colors from cmap."""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, ncols))
    if reverse_cmap:
        colors = colors[::-1]
    return colors


def _compute_ylim(y_pos, y_neg, net=None, stacked=True, ylim=None):
    """Compute y-axis limits."""
    has_pos = (y_pos > 0).any().any()
    has_neg = (y_neg < 0).any().any()

    if ylim is not None:
        ymin, ymax = ylim
    else:
        if stacked:
            ymax = y_pos.sum(axis=1).max() * 1.01 if has_pos else 1
            ymin = y_neg.sum(axis=1).min() * 1.02 if has_neg else -ymax * 0.01
        else:
            ymax = y_pos.max(axis=1).max() * 1.01 if has_pos else 1
            ymin = y_neg.min(axis=1).min() * 1.02 if has_neg else -ymax * 0.01

        if net is not None:
            ymax = max(ymax, net.max() * 1.01)
            ymin = min(ymin, net.min() * 1.02)

    return ymin, ymax


def _draw_group_axes(
    ax,
    x,
    groups,
    has_groups,
    ymin,
    ymax,
    axis_padding=10,
    grouplabels_fontsize=10,
    grouplabels_vertical=False
):
    """Draw inset group axes/brackets above the x-axis."""
    group_axs = []

    for group in groups:
        xg = x.loc[group]
        axg = ax.inset_axes(
            bounds=[
                xg.min() - 0.5,
                ymin,
                xg.max() - xg.min() + 1,
                ymax - ymin
            ],
            transform=ax.transData,
            zorder=0
        )

        axg.set_xlim(xg.min() - 0.5, xg.max() + 0.5)

        if has_groups:
            group_label = ', '.join([i for i in group if len(i) > 0]) if isinstance(group, tuple) else group

            axg.tick_params(
                axis='x',
                which='major',
                labelsize=grouplabels_fontsize,
                labelrotation=90 if grouplabels_vertical else 0,
                bottom=False,
                top=False,
                labelbottom=False,
                labeltop=True
            )
            axg.tick_params(
                axis='x',
                which='minor',
                direction='in',
                width=0.8,
                bottom=False,
                top=True
            )

            if not grouplabels_vertical:
                wrapXTicks(
                    axg.set_xticks(
                        ticks=[xg.mean()],
                        labels=[group_label]
                    )
                )
            else:
                axg.set_xticks(
                    ticks=[xg.mean()],
                    labels=[group_label]
                )

            axg.set_xticks(
                minor=True,
                ticks=[xg.min() - 0.5, xg.max() + 0.5],
                labels=['', '']
            )
        else:
            axg.set_xticks([])
            axg.spines['top'].set_visible(False)

        axg.spines[['top', 'bottom']].set_position(('outward', axis_padding))
        axg.spines['bottom'].set_bounds((xg.min(), xg.max()))
        axg.spines[['right', 'left']].set_visible(False)
        axg.set_yticks([])

        group_axs.append(axg)

    return group_axs


def _style_common_axes(
    ax,
    x,
    ymin,
    ymax,
    has_neg=False,
    axis_padding=10,
    xlabel='',
    xlabel_fontsize=11,
    ylabel='',
    ylabel_fontsize=11,
    ticklabels_fontsize=10,
    grouptitle='',
    grouptitle_fontsize=11,
    xtick_interval=None
):
    """Apply shared axis styling, ticks, labels, and titles."""
    ax.spines[['left', 'bottom']].set_position(('outward', axis_padding))

    yticks = [t for t in ax.get_yticks() if ymin <= t <= ymax]
    ax.set_yticks(yticks)
    if len(yticks) > 1:
        ax.spines['left'].set_bounds((min(yticks), max(yticks)))

    ax.spines[['top', 'right', 'bottom']].set_visible(False)

    if has_neg:
        for group in x.index.unique('group'):
            xg = x.loc[group]
    
            # one segment per group, matching the group span
            if isinstance(xg, pd.Series):
                xmin_g = xg.min() - 1
                xmax_g = xg.max() + 1
            else:
                xmin_g = float(xg) - 1
                xmax_g = float(xg) + 1
    
            ax.hlines(
                y=0,
                xmin=xmin_g,
                xmax=xmax_g,
                color='black',
                linewidth=0.5,
                linestyle='--'
            )

    xtick_labels_all = [
        ', '.join(l) if isinstance(l, tuple) else str(l)
        for l in x.index.droplevel('group')
    ]

    if xtick_interval is None or xtick_interval <= 1:
        xticks = x.values
        xtick_labels = xtick_labels_all
    else:
        tick_positions = []
        tick_labels = []

        groups = x.index.unique('group')

        for group in groups:
            xg = x.loc[group]

            # x.loc[group] can be scalar-like if only one row exists in group
            if not isinstance(xg, pd.Series):
                xg = pd.Series([xg], index=pd.Index([x.loc[group].index if hasattr(x.loc[group], 'index') else 0]))

            labels_g = [
                ', '.join(l) if isinstance(l, tuple) else str(l)
                for l in xg.index
            ]

            idx = list(range(0, len(xg), xtick_interval))
            if len(xg) > 0 and idx[-1] != len(xg) - 1:
                idx.append(len(xg) - 1)

            tick_positions.extend(xg.iloc[idx].tolist())
            tick_labels.extend([labels_g[i] for i in idx])

        xticks = tick_positions
        xtick_labels = tick_labels

    ax.set_xticks(
        ticks=xticks,
        labels=xtick_labels,
        rotation=90
    )

    wrapText(ax.set_title(grouptitle, fontsize=grouptitle_fontsize, pad=10))
    wrapText(ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, labelpad=10))
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.tick_params(labelsize=ticklabels_fontsize)


# ============================================================================
# Bar plot
# ============================================================================

def bar(
    data,
    ax=None,

    stacked=True,
    group_levels=None,
    group_spacing=0.5,
    bar_width=0.8,
    legend=True,

    cmap='tab10',
    reverse_cmap=False,
    edgecolor='black',
    linewidth=0.5,

    totmarker='o',
    totmarker_size=10,

    axis_padding=10,

    sort_groups=False,
    sort_xlabels=False,
    sort_categories=False,

    xlabel='',
    xlabel_fontsize=11,
    ylabel='',
    ylabel_fontsize=11,
    ticklabels_fontsize=10,

    xtick_interval=None,

    grouptitle='',
    grouptitle_fontsize=11,

    grouplabels_fontsize=10,
    grouplabels_vertical=False,

    ylim=None,

    return_group_axes = False
):
    """Plots a bar chart from a pandas.DataFrame. Columns are taken as
    categories (i.e. color). Bars can be grouped by specifying index levels in
    `group_levels`. Remaining index levels are used as x labels.
    """
    if ax is None:
        ax = plt.gca()

    data, groups, cols, has_groups = _prepare_categorical_plot_data(
        data=data,
        group_levels=group_levels,
        sort_groups=sort_groups,
        sort_xlabels=sort_xlabels,
        sort_categories=sort_categories
    )

    ncols = len(cols)
    y_pos, y_neg, has_pos, has_neg, mixed_sign = _split_pos_neg(data)
    pos_base, neg_base = _compute_stacked_bases(y_pos, y_neg, stacked=stacked)

    x = _compute_grouped_x(data.index, group_spacing=group_spacing)

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    SetColorCycle(ax, ncols, cmap_obj, reverse_cmap)

    x_shift = pd.Series(0.0, index=cols)
    if not stacked:
        x_shift.loc[:] = np.linspace(-0.5 + 1 / ncols / 2, 0.5 - 1 / ncols / 2, ncols) * bar_width

    for i, col in enumerate(cols):
        ax.bar(
            x=x + x_shift.iloc[i],
            height=y_pos.iloc[:, i],
            bottom=pos_base.iloc[:, i],
            width=bar_width if stacked else 1 / ncols * bar_width**2,
            label='\n'.join(col) if isinstance(col, tuple) else col,
            edgecolor=edgecolor,
            linewidth=linewidth
        )

    if has_neg:
        SetColorCycle(ax, ncols, cmap_obj, reverse_cmap)
        for i, col in enumerate(cols):
            ax.bar(
                x=x + x_shift.iloc[i],
                height=y_neg.iloc[:, i],
                bottom=neg_base.iloc[:, i],
                width=bar_width if stacked else 1 / ncols * bar_width,
                edgecolor=edgecolor,
                linewidth=linewidth
            )

        if stacked and ncols > 1:
            ax.scatter(
                x=x,
                y=(y_pos + y_neg).sum(axis=1),
                color='black',
                marker=totmarker,
                s=totmarker_size
            )

    xmin = x.min() - 0.5
    xmax = x.max() + 0.5
    ax.set_xlim(xmin, xmax)

    ymin, ymax = _compute_ylim(
        y_pos=y_pos,
        y_neg=y_neg,
        net=(y_pos + y_neg).sum(axis=1) if has_neg else None,
        stacked=stacked,
        ylim=ylim
    )
    ax.set_ylim(ymin, ymax)

    group_axs = _draw_group_axes(
        ax=ax,
        x=x,
        groups=groups,
        has_groups=has_groups,
        ymin=ymin,
        ymax=ymax,
        axis_padding=axis_padding,
        grouplabels_fontsize=grouplabels_fontsize,
        grouplabels_vertical=grouplabels_vertical
    )

    if legend and ncols > 1:
        ax.legend(
            frameon=False,
            fontsize=9,
            ncols=max(1, int(np.floor(ncols / 5))),
            reverse=True if stacked else False
        )

    _style_common_axes(
        ax=ax,
        x=x,
        ymin=ymin,
        ymax=ymax,
        has_neg=has_neg,
        axis_padding=axis_padding,
        xlabel=xlabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel=ylabel,
        ylabel_fontsize=ylabel_fontsize,
        ticklabels_fontsize=ticklabels_fontsize,
        grouptitle=grouptitle,
        grouptitle_fontsize=grouptitle_fontsize,
        xtick_interval=xtick_interval
    )

    if return_group_axes:
        return ax, group_axs
    else:
        return ax


# ============================================================================
# Area plot
# ============================================================================

def area(
    data,
    ax=None,

    stacked=True,
    group_levels=None,
    group_spacing=0.5,
    bar_width=0.8,   # kept for API compatibility with bar(); not used
    legend=True,

    cmap='tab10',
    reverse_cmap=False,
    edgecolor='black',
    linewidth=0.5,

    totmarker='o',
    totmarker_size=10,

    axis_padding=10,

    sort_groups=False,
    sort_xlabels=False,
    sort_categories=False,

    xlabel='',
    xlabel_fontsize=11,
    ylabel='',
    ylabel_fontsize=11,
    ticklabels_fontsize=10,

    xtick_interval=None,

    grouptitle='',
    grouptitle_fontsize=11,

    grouplabels_fontsize=10,
    grouplabels_vertical=False,

    ylim=None,

    return_group_axes = False
):
    """Plots an area chart from a pandas.DataFrame.

    Positive and negative values are stacked separately. If both positive and
    negative values are present, a black net-total line is added.
    """
    if ax is None:
        ax = plt.gca()

    data, groups, cols, has_groups = _prepare_categorical_plot_data(
        data=data,
        group_levels=group_levels,
        sort_groups=sort_groups,
        sort_xlabels=sort_xlabels,
        sort_categories=sort_categories
    )

    ncols = len(cols)
    x = _compute_grouped_x(data.index, group_spacing=group_spacing)

    y_pos, y_neg, has_pos, has_neg, mixed_sign = _split_pos_neg(data)
    pos_base, neg_base = _compute_stacked_bases(y_pos, y_neg, stacked=stacked)

    # Use the same color logic as `bar`
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    SetColorCycle(ax, ncols, cmap_obj, reverse_cmap)
    colors = [ax._get_lines.get_next_color() for _ in range(ncols)]
    SetColorCycle(ax, ncols, cmap_obj, reverse_cmap)  # reset cycle after reading colors

    legend_handles = []
    legend_labels = []

    # Draw each category separately, and each group separately,
    # so the filled areas are detached between groups.
    legend_handles = []
    legend_labels = []

    # Draw each category separately, and each group separately,
    # so the filled areas are detached between groups.
    for i, col in enumerate(cols):
        color = colors[i]
        label = '\n'.join(col) if isinstance(col, tuple) else str(col)
        label_added = False

        for group in groups:
            idx = pd.IndexSlice[group, :]

            xg = x.loc[idx].values

            pos_vals = y_pos.loc[idx, col].values
            neg_vals = y_neg.loc[idx, col].values

            # positive part
            if np.any(pos_vals != 0):
                y0 = pos_base.loc[idx, col].values
                y1 = (pos_base.loc[idx, col] + y_pos.loc[idx, col]).values

                h = ax.fill_between(
                    xg,
                    y0,
                    y1,
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    label=label if not label_added else None
                )

                if not label_added:
                    legend_handles.append(h)
                    legend_labels.append(label)
                    label_added = True

            # negative part
            if np.any(neg_vals != 0):
                y0n = neg_base.loc[idx, col].values
                y1n = (neg_base.loc[idx, col] + y_neg.loc[idx, col]).values

                h = ax.fill_between(
                    xg,
                    y0n,
                    y1n,
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    label=label if not label_added else None
                )

                if not label_added:
                    legend_handles.append(h)
                    legend_labels.append(label)
                    label_added = True

    # Net line: also draw group-by-group so it does not bridge gaps
    net = data.sum(axis=1)
    if mixed_sign:
        for group in groups:
            idx = pd.IndexSlice[group, :]
            ax.plot(
                x.loc[idx].values,
                net.loc[idx].values,
                color='black',
                linewidth=1.2,
                marker=totmarker,
                markersize=np.sqrt(totmarker_size),
                zorder=5
            )

    xmin = x.min() - 0.5
    xmax = x.max() + 0.5
    ax.set_xlim(xmin, xmax)

    ymin, ymax = _compute_ylim(
        y_pos=y_pos,
        y_neg=y_neg,
        net=net if mixed_sign else None,
        stacked=stacked,
        ylim=ylim
    )
    ax.set_ylim(ymin, ymax)

    group_axs = _draw_group_axes(
        ax=ax,
        x=x,
        groups=groups,
        has_groups=has_groups,
        ymin=ymin,
        ymax=ymax,
        axis_padding=axis_padding,
        grouplabels_fontsize=grouplabels_fontsize,
        grouplabels_vertical=grouplabels_vertical
    )

    if legend and ncols > 1 and len(legend_handles) > 0:
        order = slice(None, None, -1) if stacked else slice(None)
        ax.legend(
            handles=legend_handles[order],
            labels=legend_labels[order],
            frameon=False,
            fontsize=9,
            ncols=max(1, int(np.floor(ncols / 5)))
        )

    _style_common_axes(
        ax=ax,
        x=x,
        ymin=ymin,
        ymax=ymax,
        has_neg=has_neg,
        axis_padding=axis_padding,
        xlabel=xlabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel=ylabel,
        ylabel_fontsize=ylabel_fontsize,
        ticklabels_fontsize=ticklabels_fontsize,
        grouptitle=grouptitle,
        grouptitle_fontsize=grouptitle_fontsize,
        xtick_interval=xtick_interval
    )

    if return_group_axes:
        return ax, group_axs
    else:
        return ax

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



