# import os
# import pandas as pd
# import numpy as np
import geopandas as gpd
# import matplotlib.pyplot as plt

# För att undvika varningar som beror på något knas med geopandas vs shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import pkg_resources

# Read map from gpkg
map_file = pkg_resources.resource_stream(__name__,'swe_regions.gpkg')
MAP = gpd.read_file(map_file, layer='sko').set_index("sko").rename_axis('region')

def map_from_series(ser, ax):
    
    ser.name = 'values'
    to_plot = MAP.join(ser)

    MAP.plot(ax=ax, facecolor = '#DDDDDD')
    to_plot.plot(ax=ax, edgecolor = 'none', column='values', linewidth = 0.1, legend = True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)