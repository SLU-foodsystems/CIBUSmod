from bokeh.models import GeoJSONDataSource, HoverTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

import geopandas as gpd
# import pkg_resources
import os
import pandas as pd
import plotly.express as px



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

def map_from_series(ser, reg='sko', **kwargs):
    '''
    Parameters
    ----------
    ser : pandas.Series
        A series values to produce the map. Must have 'region' as index
    reg : str
        Defines what 'region' refers to ('sko', 'po8', 'kommun' or 'län')
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

    to_plot = MAP[reg].join(ser.rename('values'))

    to_plot.plot(**kwargs)


def map_from_soilseries(ax, ser, min=None, max=None, reg='sko', verbose=False, font_size=15, **kwargs):
    '''
    Plots values from a pandas Series or DataFrame on a geographic map specified by 'reg'.

    Alternative version of 'map_from_series'. Used in soil plotting functions

    Parameters
    ----------
    ser : pandas.Series or pandas.DataFrame
        Values to produce the map. If a Series, must have 'region' as index.
        If a DataFrame, it must contain a 'region' column and a 'values' column.
    reg : str
        Specifies the geographic region for mapping ('sko', 'po8', 'kommun', 'län').
    **kwargs
        Additional arguments passed on to geopandas.GeoDataFrame.plot().
    '''
    if verbose:
        print('---- map_from_soilseries ----')
        print(f'ser:\n{ser}\n type(ser) {type(ser)}')
        print(f'MAP[reg]:\n{MAP[reg]}\n type MAP[reg]: {type(MAP[reg])}')

    to_plot = pd.concat([MAP[reg], ser], axis=1)

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

def plot_regions(gdf_name='sko', **kwargs):

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



