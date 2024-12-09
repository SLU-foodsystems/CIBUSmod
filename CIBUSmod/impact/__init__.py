import os

IMPACT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')

from .general import get_emissions, to_ICBM
from .climate import get_GHG, get_deltaT
from .biodiv import get_crop_div
from .livestock import get_LSU