import warnings
import numpy as np
import pandas as pd

from ..main_modules.demand_and_conversions import DemandAndConversions
from ..main_modules.regions import Regions
from ..main_modules.crop_prod import CropProduction
from ..main_modules.animal_herd import AnimalHerd

from .output_data_manip import \
    concat_herds, \
    get_attr, get_GHG, to_ICBM

class Output(pd.DataFrame):
    '''A pandas.DataFrame augmented with methods to store and retrieve
    model output data.'''

    @classmethod
    def from_file(cls, path):
        return pd.read_pickle(path)

    def __init__(self, df=None):
        
        if df is not None:
            super().__init__(df)
        else:
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

    def info(self):
        
        print(
f'''*-----------------*
| CIBUSmod Output |
*-----------------*

Scenarios
---------'''
        )
        for scn in self.index.get_level_values('scn').unique():
            years = self.loc[scn].index.get_level_values('year')
            nyears = len(years)
            if nyears>1:
                years = [years[0], years[-1]]
            print(
f'''{scn}: {' --> '.join(years)} {'('+str(nyears)+' years)' if nyears>1 else ''}'''
            )

        for i,module in enumerate(self.columns):
            print(
f'''
{module}
{'-'*len(module)}
Data attributes:
{', '.join(self.iloc[0,i].data_attr)}
'''
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
                self.loc[(scn,year),'DemandAndConversions'] = \
                arg.make_static()
            elif isinstance(arg, Regions):
                # Regions
                self.loc[(scn,year),'Regions'] = \
                arg.make_static()
            elif isinstance(arg, CropProduction):
                # Crops
                self.loc[(scn,year),'CropProduction'] = \
                arg.make_static()
            elif _isiterable(arg):
                if np.all([isinstance(h,AnimalHerd) for h in arg]):
                    # Animals
                    self.loc[(scn,year),'AnimalHerd'] = \
                    concat_herds(
                        arg.apply(lambda x: x.make_static())
                    )
                else:
                    print('Iterable of non-AnimalHerd objects ignored')
            else:
                print(type(arg),'ignored')
    
    get_attr = get_attr

    def save_file(self, path):
        self.to_pickle(path)

def _isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False