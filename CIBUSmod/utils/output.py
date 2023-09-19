import warnings
import numpy as np
import pandas as pd

from ..production_systems.demand_and_conversions import DemandAndConversions
from ..production_systems.regions import Regions
from ..production_systems.crop_prod import CropProduction
from ..production_systems.animal_herd import AnimalHerd

from .output_data_manip import concat_herds

class Output(pd.DataFrame):
    '''A pandas.DataFrame augmented with methods to store and retrieve
    model output data.'''

    @classmethod
    def from_file(cls, path):
        return pd.read_pickle(path)

    def __init__(self):

        # Create empty DataFrame
        super().__init__(
            index = pd.MultiIndex(
                levels=[[]]*2,
                codes=[[]]*2,
                names=['scn','year']
            ),
            columns = pd.Index(
                [],
                name='object'
            )
        )

    def store(
            self,
            scn,
            year,
            *args
        ):
        
        for arg in args:
            pass

            if isinstance(arg, DemandAndConversions):
                # Demand
                self.loc[(scn,year),'dem'] = \
                arg.make_static()
            elif isinstance(arg, Regions):
                # Regions
                self.loc[(scn,year),'reg'] = \
                arg.make_static()
            elif isinstance(arg, CropProduction):
                # Crops
                self.loc[(scn,year),'crp'] = \
                arg.make_static()
            elif _isiterable(arg):
                if np.all([isinstance(h,AnimalHerd) for h in arg]):
                    # Animals
                    self.loc[(scn,year),'ani'] = \
                    concat_herds(
                        arg.apply(lambda x: x.make_static())
                    )
                else:
                    print('Iterable of non-AnimalHerd objects ignored')
            else:
                print(type(arg),'ignored')

    def save_file(self, path):
        self.to_pickle(path)

def _isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False