import os
from typing import Union, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
import geopandas as gpd

# Supress shapely deprecation warnings due to problems with geopandas vs shapely
import warnings

from matplotlib import pyplot as plt
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Read map from gpkg
map_file = os.path.join(os.path.dirname(__file__), 'swe_regions.gpkg')
# pkg_resources.resource_stream(__name__,'swe_regions.gpkg')
sko = gpd.read_file(map_file, layer='sko').set_index("sko").rename_axis('region')
po8 = gpd.read_file(map_file, layer='po8')
po8['region'] = po8['Nr'] + ' ' + po8['Namn']
po8 = po8[['region','geometry']].set_index('region')
kommun = gpd.read_file(map_file, layer='kommun').drop(columns=['KnKod','KnNamn']).set_index("Kommun").rename_axis('region')
lan = gpd.read_file(map_file, layer='län').drop(columns=['LnKod','LnNamn']).set_index("Län").rename_axis('region')
MAP = {
    'sko' : sko,
    'po8' : po8,
    'kommun' : kommun,
    'län' : lan
}

def map_from_series(ser, reg='sko', cmap_zero_midpoint = False, **kwargs):
    '''
    Parameters
    ----------
    ser : pandas.Series
        A series values to produce the map. Must have 'region' as index
    reg : str
        Defines what 'region' refers to ('sko', 'po8', 'kommun' or 'län')
    cmap_zero_midpoint : 'shrink', 'shift' or False (default)
        Puts the middle of the color map range on zero
        If 'shrink' the color map's range is shrunk on the postive or negative side
        If 'shift' the colormap's center is shifted but the full range is used on
        both the positive and negative side
    **kwargs
        passed on to geopandas.GeoDataFrame.plot()
    '''

    default_style = {
        'edgecolor' : 'none',
        'column' : 'values',
        'linewidth' : 0.1,
        'legend' : True,
    }

    for key in default_style:
        if key not in kwargs:
            kwargs[key] = default_style[key]

    if cmap_zero_midpoint:
        if cmap_zero_midpoint not in ['shrink','shift']:
            raise ValueError("'cmap_zero_midpoint' must be one of 'shrink' or 'shift'")
        try:
            cmap = kwargs.pop('cmap')
        except KeyError:
            raise TypeError("Argument 'cmap' must be specified when 'cmap_zero_midpoint' is True")


        try:
            max_neg = kwargs.pop('vmin')
        except KeyError:
            max_neg = abs(ser.min())
        try:
            max_pos = kwargs.pop('vmax')
        except KeyError:
            max_pos = ser.max()

        max_one_dir = max(max_neg, max_pos)
        range = max_neg + max_pos

        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]

        remapped_cmap = remappedColorMap(
            cmap = cmap,
            start = 0.5 - max_neg / max_one_dir * 0.5 if cmap_zero_midpoint == 'shrink' else 0,
            midpoint = max_neg / range,
            stop = 0.5 + max_pos / max_one_dir * 0.5 if cmap_zero_midpoint == 'shrink' else 1
        )

        kwargs['vmin'] = -max_neg
        kwargs['vmax'] = max_pos
        kwargs['cmap'] = remapped_cmap

    to_plot = MAP[reg].join(ser.rename('values'))

    to_plot.plot(**kwargs)


def map_from_soilseries(ax: matplotlib.axes.Axes,
                        ser: Union[pd.Series, pd.DataFrame],
                        min: Optional[float] = None,
                        max: Optional[float] = None,
                        reg: str = 'sko',
                        verbose: bool = False,
                        font_size: int = 15,
                        **kwargs: Any
                        ) -> matplotlib.axes.Axes:
    """
    Plots values from a pandas Series or DataFrame on a geographic map specified by the region ('reg') parameter. It leverages geopandas for mapping, allowing for customization through additional keyword arguments.

    This function has been adapted to include a call to `clear_colorbars` to ensure that any duplicate colorbars generated in previous iterations are removed, providing cleaner, more accurate visualizations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the map will be plotted.
    ser : Union[pd.Series, pd.DataFrame]
        The data to plot. If a Series, it must have 'region' as its index. If a DataFrame, it must contain 'region' and 'values' columns.
    min : Optional[float], default=None
        The minimum value for scaling the color map. If None, it's automatically determined.
    max : Optional[float], default=None
        The maximum value for scaling the color map. If None, it's automatically determined.
    reg : str, default='sko'
        Specifies the geographic region type for mapping. Supports 'sko', 'po8', 'kommun', 'län'.
    verbose : bool, default=False
        If True, prints additional information during the function execution.
    font_size : int, default=15
        Base font size for plot text elements. The actual font size scales with figure dimensions.
    **kwargs : Any
        Additional keyword arguments passed on to geopandas.GeoDataFrame.plot() for customizing the map appearance.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the map plotted.

    Notes
    -----
    - The function automatically manages colorbars to prevent duplication across multiple calls.
    - It's designed to integrate seamlessly with geospatial data processing and visualization workflows in Python.
    """
    if verbose:
        print('---- map_from_soilseries ----')
        print(f'ser:\n{ser}\n type(ser) {type(ser)}')
        print(f'MAP[reg]:\n{MAP[reg]}\n type MAP[reg]: {type(MAP[reg])}')

    # Remove duplicate colorbars that were dynamically generated in for loops
    clear_colorbars(ax)

    to_plot = pd.concat([MAP[reg], ser], axis=1)
    to_plot['values'] = np.ma.masked_invalid(to_plot['values'])

    if verbose:
        print(f'to_plot:\n{to_plot}\n type(to_plot) {type(to_plot)}')

    default_style = {
        'edgecolor': 'none',
        'column': 'values',
        'linewidth': 0.1,
        'legend': True,
    }
    for key in default_style:
        kwargs.setdefault(key, default_style[key])

    if 'title' in kwargs:
        title = kwargs.pop('title')
    if 'label' in kwargs:
        label = kwargs.pop('label')
        if 'legend_kwds' not in kwargs:
            kwargs['legend_kwds'] = {'label': label}
    if 'cmap' in kwargs:
        colourmap = kwargs.pop('cmap')
    else:
        colourmap = 'RdBu'

    if kwargs.get('kind') is not None:
        for key in ['vmin', 'vmax', 'legend_kwds']:
            kwargs.pop(key, None)

    if kwargs.get('kind') == 'box':
        kwargs.pop('edgecolor', None)
        ax.set_xlabel(kwargs.get('xlabel', ''), fontsize=font_size)  # Remove the x-axis label
        ax.set_xticks([])  # Remove x-axis ticks

    if kwargs.get('kind') in ['line', 'box']:
        kwargs.pop('linewidth', None)

    if min and max is not None:
        if kwargs.get('kind') in ['barh', 'hist']:
            ax.set(xlim=[min, max])
        if kwargs.get('kind') == 'box':
            ax.set(ylim=[min, max])

    colmap = plt.get_cmap(colourmap)
    colmap.set_bad('white', 1.0)

    kwargs['cmap'] = colmap

    gdf = gpd.GeoDataFrame(to_plot, geometry='geometry')
    plot = gdf.plot(ax=ax, **kwargs)

    if kwargs.get('kind') is None:
        ax.set_xlabel(kwargs.get('xlabel',''), fontsize=font_size)  # Remove the x-axis label
        ax.set_ylabel(kwargs.get('ylabel', ''), fontsize=font_size)  # Remove the y-axis label
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
    if title:
        ax.set_title(title, fontsize=font_size)

    return(plot)


def clear_colorbars(ax: matplotlib.axes.Axes) -> None:
    """
    Removes duplicate colorbar Axes from a matplotlib figure.

    This function iterates over all Axes objects within the figure that contains the
    specified Axes (`ax`). If an Axes object is identified as a colorbar (determined by
    the presence of 'colorbar' in its label), the function checks if its label (specifically,
    the ylabel, which is used here as an identifier for colorbars) has already been encountered.
    If so, it is considered a duplicate and is removed from the figure. This helps prevent
    clutter from duplicate colorbars when repeatedly plotting onto the same Axes or figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object from which the figure will be accessed to identify and remove
        duplicate colorbar Axes. This is not necessarily a colorbar Axes itself but
        belongs to the figure being cleaned.

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> # Plotting operations that may inadvertently create duplicate colorbars
    >>> clear_colorbars(ax)
    This will remove any duplicate colorbars associated with the figure containing 'ax'.

    Note
    ----
    This function uses the ylabel of the colorbar Axes as the primary means to identify
    duplicates. This approach assumes that duplicate colorbars will have the same ylabel,
    which may not always be the case depending on the plotting logic used.
    """
    all_axes = ax.figure.axes
    colorbar_labels = set()

    for cbar_ax in all_axes:
        if 'colorbar' in cbar_ax.get_label():
            label = cbar_ax.get_ylabel()
            if label in colorbar_labels:
                cbar_ax.remove()
            else:
                colorbar_labels.add(label)

def clear_colorbars_old(ax):
    print(f'cbar is being removed from {ax}')
    print(f'type(ax.figure.axes): {type(ax.figure.axes)}\n len(ax.figure.axes): {len(ax.figure.axes)}\n list(ax.figure.axes): {list(ax.figure.axes)}')
    for cbar in ax.figure.axes:
        print(f'type(cbar): {type(cbar)}')
        print(f'cbar evaluates to {cbar}')
        if isinstance(cbar, matplotlib.colorbar.Colorbar):
            cbar.remove()

def plot_regions(gdf_name='sko', **kwargs):

    from bokeh.models import GeoJSONDataSource, HoverTool
    from bokeh.plotting import figure, show
    from bokeh.io import output_notebook

    if gdf_name == 'sko':
        gdf = sko
        region = 'SKO'
    elif gdf_name == 'po8':
        gdf = po8
        region = 'PO8'
    elif gdf_name == 'kommun':
        gdf = kommun
        region = 'Kommun'
    elif gdf_name == 'Region':
        gdf = 'lan'
        region = 'Län'

    # Reproject to Sweref99 TM (EPSG:3006)
    gdf = gdf.to_crs(epsg=3006)

    # convert geodataframe to json, compatible with bokeh
    gdf_json = gdf.reset_index().to_json()

    output_notebook() # For Jupyter notebooks. Use output_file("output.html") for scripts

    # Convert GeoDataFrame to GeoJSON DataSource
    geo_source = GeoJSONDataSource(geojson=gdf_json)

    # Initialize figure
    p = figure()
    p.patches('xs', 'ys', source=geo_source, fill_alpha=0.7, line_color="black", line_width=0.5)

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [(region, '@region')]  # Adjust '@region_column_name' to your column name
    p.add_tools(hover)

    show(p)



def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    # This function is sourced from https://github.com/TheChymera/chr-helpers/blob/d05eec9e42ab8c91ceb4b4dcc9405d38b7aed675/chr_matplotlib.py
    # Authors: Paul H, Horea Christian
    # Licence: GNU GENERAL PUBLIC LICENSE, Version 3
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin))
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False),
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap
