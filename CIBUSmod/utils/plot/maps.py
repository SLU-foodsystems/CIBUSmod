import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd

# Supress shapely deprecation warnings due to problems with geopandas vs shapely
import warnings
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
    # 'po8' : po8,
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
        If 'shirk' the color map's range is shurnk on the postive or negative side
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