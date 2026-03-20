import os

root = os.path.join(os.path.dirname(__file__), '..')

from .utils.retriever import ParameterRetriever
from .utils.session_db import Session

# Import main modules
from .main_modules.regions import Regions
from .main_modules.demand_and_conversions import \
    DemandAndConversions
from .main_modules.crop_prod import CropProduction
from .main_modules.animal_herd import \
    CattleHerd, PigHerd, BroilerHerd, LayerHerd, HorseHerd, \
    SheepHerd, GoatHerd, \
    AquacultureHerd, FisheriesHerd, \
    make_herds, concat_herds
from .main_modules.waste_and_circularity import WasteAndCircularity

# Import mgmt modules
from .mgmt_modules.manure_mgmt import ManureMgmt
from .mgmt_modules.feed_mgmt import FeedMgmt
from .mgmt_modules.plant_nutrient_mgmt import PlantNutrientMgmt
from .mgmt_modules.machinery_and_energy_mgmt import MachineryAndEnergyMgmt
from .mgmt_modules.inputs_mgmt import InputsMgmt
from .mgmt_modules.crop_residue_mgmt import CropResidueMgmt
from .mgmt_modules.byprod_mgmt import ByProductMgmt
from .mgmt_modules.cover_crops_mgmt import CoverCropsMgmt

# Import LULUC modules
from .LULUC_modules.soil_C import SoilC

# Import geo distiributor
from .optimisation.geo_dist import GeoDistributor
from .optimisation.feed_dist import FeedDistributor

# Import module with impact assessment functions
from . import impact

# Import module with helper functions
from .utils import helpers

# Import module with plotting functions
from .utils import plot

# Import soil_modules functions and SoilData class
from .soil_modules import data_processing
from .soil_modules import icbm_funcs
from .soil_modules.soil_class import SoilData
from .soil_modules.soil_class import SoilDataExplore
from .soil_modules.soil_class import ScenarioTempCalc
from .soil_modules import soil_utils

from .utils.misc import get_git_info
print("""
 -----------------------
|   Imported CIBUSmod   |
 -----------------------""", end='')
try:
    GIT_COMMIT, GIT_BRANCH, GIT_DIRTY, GIT_REMOTE = \
    get_git_info(root).values()
    print(f"""
commit : {GIT_COMMIT}
branch : {GIT_BRANCH}
dirty  : {GIT_DIRTY}
remote : {GIT_REMOTE}
    """)
except:
    print("""
Could not access git.
""")