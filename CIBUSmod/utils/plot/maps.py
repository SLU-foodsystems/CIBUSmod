import geopandas as gpd
# import pkg_resources
import os

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
    to_plot = MAP[reg].join(ser.rename('values'))
    to_plot.plot(
        edgecolor = 'none',
        column='values',
        linewidth = 0.1,
        legend = True,
        **kwargs
    )