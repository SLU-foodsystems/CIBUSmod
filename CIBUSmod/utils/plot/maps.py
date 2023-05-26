import geopandas as gpd
import pkg_resources

# Supress shapely deprecation warnings due to problems with geopandas vs shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Read map from gpkg
map_file = pkg_resources.resource_stream(__name__,'swe_regions.gpkg')
MAP = {
    'sko' : gpd.read_file(map_file, layer='sko').set_index("sko").rename_axis('region')
}

def map_from_series(ser, reg='sko', **kwargs):
    '''
    Parameters
    ----------
    ser : pandas.Series
        A series values to produce the map. Must have 'region' as index
    reg : str
        Defines what 'region' refers to (only 'sko' possible at the moment)
    **kwargs
        passed on to geopandas.GeoDataFrame.plot()
    '''
    
    ser.name = 'values'
    to_plot = MAP[reg].join(ser)

    to_plot.plot(edgecolor = 'none', column='values', linewidth = 0.1, legend = True, **kwargs)