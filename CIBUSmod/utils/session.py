import warnings
import numpy as np
import pandas as pd
import os

from .retriever import ParameterRetriever
from ..main_modules.demand_and_conversions import DemandAndConversions
from ..main_modules.regions import Regions
from ..main_modules.crop_prod import CropProduction
from ..main_modules.animal_herd import AnimalHerd, concat_herds

from .output_data_manip import \
    get_attr, get_emissions, get_GHG, to_ICBM

class Session(object):
    '''
    
    Parameters
    ----------
    name : str
        Session name
    data_path : str
        '''

    def __init__(self, name, data_path, from_file=False):
        
        self.name = name

        self.data_path = _path_from_str(data_path)
        ParameterRetriever.set_data_folder(self.data_path)

        if from_file:
            self.read_output()
        else:
            # Create empty DataFrame
            self.output = pd.DataFrame(
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

        self.scenarios = {}

    def __getitem__(self, scn):
        res = self.scenarios[scn].copy()
        res.pop('years')
        return res

    def __repr__(self):
        
        str0 = f'''*------------------*
| CIBUSmod SESSION |
*------------------*
'''

        str1 = '''Scenarios
---------
'''
        scns = list(self.scenarios.keys())
        scns += [
            s for s
            in self.output.index.get_level_values('scn').unique()
            if s not in scns
        ]
        for scn in scns:
            try:
                years = self.scenarios[scn]['years']
                in_scenarios = True
                if scn in self.output.index.get_level_values('scn'):
                    in_output = True
                else:
                    in_output = False
            except KeyError:    
                years = self.output.loc[scn].index.get_level_values('year')
                in_scenarios = False
                in_output = True
            nyears = len(years)
            if nyears>1:
                years = [years[0], years[-1]]
            str1 += (
f'''{scn if in_scenarios else '('+scn+')'}: {' --> '.join(years)} {'('+str(nyears)+' years)' if nyears>1 else ''}{' [calculated]' if in_output else ''}{' [only in output]' if not in_scenarios else ''}
'''
            )

        str2 = ''
        for i,module in enumerate(self.output.columns):
            str2 += (
f'''{module}:
{', '.join(self.output.iloc[0,i].data_attr)}

'''
            )
            
        return '\n'.join([str0, str1, str2])

    def add_scenario(self, name, scenario=None, modules='all', pars='all', years=None):

        if not isinstance(years,list):
            years = [years]
        
        self.scenarios.update(
            {
                name : {
                    'scenario' : scenario,
                    'modules' : modules,
                    'pars' : pars,
                    'years' : years
                }
            }
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
                self.output.loc[(scn,year),'DemandAndConversions'] = \
                arg.make_static()
            elif isinstance(arg, Regions):
                # Regions
                self.output.loc[(scn,year),'Regions'] = \
                arg.make_static()
            elif isinstance(arg, CropProduction):
                # Crops
                self.output.loc[(scn,year),'CropProduction'] = \
                arg.make_static()
            elif _isiterable(arg):
                if np.all([isinstance(h,AnimalHerd) for h in arg]):
                    # Animals
                    self.output.loc[(scn,year),'AnimalHerd'] = \
                    concat_herds(
                        arg.apply(lambda x: x.make_static())
                    )
                else:
                    print('Iterable of non-AnimalHerd objects ignored')
            else:
                print(type(arg),'ignored')

    def iterate(self):
        for scn in self.scenarios:
            for year in self.scenarios[scn]['years']:
                yield (scn,year)

    def read_output(self):
        path = os.path.join(self.data_path, 'output')
        self.output = pd.read_pickle(os.path.join(path, 'out_'+self.name+'.bz2'))

    def save_output(self):
        path = os.path.join(self.data_path, 'output')
        
        # Create output folder if it does not exist
        if not os.path.isdir(path):
            os.mkdir(path)

        # Prompt if file already exists
        if os.path.isfile(os.path.join(path, 'out_'+self.name+'.bz2')):
            ui = input('File already exists, overwrite? (y/n)')
            if ui.upper() == 'Y':
                write = True
            else:
                write = False
        else:
            write = True

        if write:
            self.output.to_pickle(os.path.join(path, 'out_'+self.name+'.bz2'))


    ########################################
    #                                      #
    #     FUNCTIONS TO GET OUTPUT DATA     #
    #                                      #
    ########################################

    docs = get_attr.__doc__ \
    .replace("    output : pandas.DataFrame\n        CIBUSmod outputs\n", "")
    def get_attr(
        self,
        module,
        attr,
        groupby = 'all',
        interpolate = False,
        keep_duplicate_levels = 'index',
        suffixes = ('_idx','_col')
    ):
        
        res = get_attr(
            output = self.output,
            module = module,
            attr = attr,
            groupby = groupby,
            interpolate = interpolate,
            keep_duplicate_levels = keep_duplicate_levels,
            suffixes = suffixes
        )

        return res
    get_attr.__doc__ = docs
    
    docs = get_emissions.__doc__ \
    .replace("    output : pandas.DataFrame\n        CIBUSmod outputs\n", "")
    def get_emissions(
        self,
        interpolate=False
    ):
        
        res = get_emissions(
            output = self.output,
            interpolate = interpolate
        )

        return res
    get_emissions.__doc__ = docs

    docs = get_GHG.__doc__ \
    .replace("    output : pandas.DataFrame\n        CIBUSmod outputs\n", "")
    def get_GHG(
        self,
        CO2eq=True,
        interpolate=False
    ):
        
        res = get_GHG(
            output = self.output,
            CO2eq = CO2eq,
            interpolate = interpolate
        )

        return res
    get_GHG.__doc__ = docs
        
    docs = to_ICBM.__doc__ \
    .replace("    output : pandas.DataFrame\n        CIBUSmod outputs\n", "")
    def to_ICBM(
        self
    ):
        
        res = to_ICBM(
            output = self.output
        )

        return res
    to_ICBM.__doc__ = docs

def _isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    
def _path_from_str(str):
    path = ''
    for word in str.split('/'):
        path = os.path.join(path, word)
    return path