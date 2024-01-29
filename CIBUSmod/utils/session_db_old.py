import warnings
import sqlite3
import numpy as np
import pandas as pd
import os

from .retriever import ParameterRetriever
from .data_attr import DataAttr
from ..main_modules.demand_and_conversions import DemandAndConversions
from ..main_modules.regions import Regions
from ..main_modules.crop_prod import CropProduction
from ..main_modules.animal_herd import AnimalHerd, concat_herds

class Session(object):
    '''Class that handles the definition of scenarios and storing/retrieving
    output data from a database (SQLite) file.
    
    Parameters
    ----------
    name : str
        Session name
    data_path : str

    When a new Session object is initialised it checks if the file '<data_path>/output/<name>.sqlite' exists
    and if so connects to that database and any outputs within it.

    Printing the Session object will show information on defined scenarios and output data.

    Main methods
    ------------
    .add_scenario()     Defines a new scenario.
    .remove_scenario()  Removes a scenario (including any associated output data).
    .iterate()          Used to iterate over scenarios and years.
    .store()            Stores a model run in the output database.
    .clean()            Cleans up the database file (see note)
    .get_attr()         Get output data

    Note: The database file grows quite large if there are many scenarios and years. Perhaps the data stored in
    the output database should be limited but for now most things calculated in a model run is stored as
    output. Removing scenarios does not immediatly reduce the database file size. If many scenarios have been
    added and removed use .clean() to defragment the database and (potentially) reduce its filesize.
    
    '''

    def __init__(self, name, data_path):
        
        self.name = name

        self.data_path = _path_from_str(data_path)
        self.db_path = os.path.join(self.data_path, 'output', self.name+'.sqlite')
        ParameterRetriever.set_data_folder(self.data_path)

        # Create output folder if it does not already exist
        output_folder = os.path.join(self.data_path, 'output')
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        if os.path.isfile(self.db_path):
            self.scenarios = _db_read_scn(self.db_path)
        else:
            self.scenarios = {}

        self.cashe = CasheDict(max_size=1000) # Dict that stores tables retrieved from database for faster access

    def __getitem__(self, scn):
        res = self.scenarios[scn].copy()
        res.pop('years')
        return res

    def __repr__(self):
        
        str0 = f'''+------------------+
| CIBUSmod SESSION |
+------------------+
Name: {self.name}
'''

        str1 = '''SCENARIOS
=========
'''
        scns = list(self.scenarios.keys())
        tables = _db_get_tables(self.db_path)

        for scn in scns:
            years = self.scenarios[scn]['years']
            nyears = len(years)
            if any([f'${scn}' in t for t in tables]):
                output_status = 'has output'
            else:
                output_status = 'no output'
            if nyears>1:
                years = [years[0], years[-1]]
            str1 += (
f'''{scn}: {' --> '.join(years)} {'('+str(nyears)+' years)' if nyears>1 else ''}[{output_status}]
'''
            )

        str2 = '''OUTPUT DATA
===========
'''
        modules = [t.replace('__metadata','') for t in tables if '__metadata' in t]
        for i,module in enumerate(modules):
            data_attr = DataAttr(0)
            data_attr.metadata = _db_read_metadata(module, self.db_path)
            str2 += (
f'''{module}
{'-'*len(module)}
{data_attr}

'''
            )
            
        return '\n'.join([str0, str1, str2])

    def add_scenario(self, name, scenario=None, modules='all', pars='all', years='nd'):
        '''Adds a scenario to the Session object
        
        Parameters
        ----------
        name : str
            Name of scenario (not necessarily the same as the name of the scenario Excel sheet)
        scenario : None or (list of) str with scenario Excel sheet name(s), default None
            Name of scenaro Excel sheet(s) to use. If None default data are used
            If a list is supplied data is uppdated based on all scenario Excel sheets but if
            the same parameter is updated in several Excel sheets only the latest one in the
            list will have an effect.
        modules : 'all' or (list of) str with module names, default 'all'
            Modules to be updated. If 'all', all modules will be updated otherwise only the
            modules specified will be updated according to the scenario Excel file(s)
        pars : 'all' or (list of) str with parameter names
                or dict with module:[parameter(s)], default 'all'
            Parameters to be updated. If 'all', all parameters are updated otherwise only the
            specified parameters are updated (see examples)

            Example 1:
            pars = ['par_A', 'par_B']
            This will only update 'par_A' and 'par_B' across all modules

            Example 2:
            pars = {'Mod1' : ['par_A', 'par_B'], 'Mod2' : 'par_C'}
            This will only update 'par_A' and 'par_B' in module 'Mod1', only update 'par_C'
            in 'Mod2' and all parameters in any other module.
        years : (list of) str
            Years to be run

        '''

        if name in self.scenarios:
            print(f'A scenario with the name {name} already exists use .update_scenario() or .remove_scenario() first.')
        
        if not isinstance(years, list):
            years = [years]
        years = [str(y) for y in years]
        
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

        _db_write_scn(self.scenarios, self.db_path)
        
        return None
    
    def update_scenario(self, name, scenario=None, modules=None, pars=None, years=None):
        '''Updates a scenario in the Session object

        Parameters
        ----------
        name : str
            Name of scenario (not necessarily the same as the name of the scenario Excel sheet)
        scenario : None or (list of) str with scenario Excel sheet name(s), optional
            Name of scenaro Excel sheet(s) to use. If None default data are used
            If a list is supplied data is uppdated based on all scenario Excel sheets but if
            the same parameter is updated in several Excel sheets only the latest one in the
            list will have an effect.
        modules : 'all' or (list of) str with module names, optional
            Modules to be updated. If 'all', all modules will be updated otherwise only the
            modules specified will be updated according to the scenario Excel file(s)
        pars : 'all' or (list of) str with parameter names
                or dict with module:[parameter(s)], optional
            Parameters to be updated. If 'all', all parameters are updated otherwise only the
            specified parameters are updated (see examples)

            Example 1:
            pars = ['par_A', 'par_B']
            This will only update 'par_A' and 'par_B' across all modules

            Example 2:
            pars = {'Mod1' : ['par_A', 'par_B'], 'Mod2' : 'par_C'}
            This will only update 'par_A' and 'par_B' in module 'Mod1', only update 'par_C'
            in 'Mod2' and all parameters in any other module.
        years : (list of) str, optional
            Years to be run
        '''
        
        if name not in self.scenarios:
            raise KeyError(f'No scenario with the name {name}')

        if scenario is not None:
            self.scenarios[name]['scenario'] = scenario

        if modules is not None:
            self.scenarios[name]['modules'] = modules

        if pars is not None:
            self.scenarios[name]['pars'] = pars

        if years is not None:
            if not isinstance(years, list):
                years = [years]
            years = [str(y) for y in years]
            self.scenarios[name]['years'] = years




    def remove_scenario(self, name):
        '''Removes named scenario including all output data
        
        Parameters
        ----------
        name : str
            Name of scenario to be removed
        
        '''

        if name in self.scenarios:
            if name in _db_get_scn_in_data(self.db_path):
                ui = input('This scenario has output data that will also be removed. Proceed? (Y/n)')
                if ui.capitalize() == 'Y':
                    # Remove all output data from scenario
                    modules = _db_get_modules_in_data(self.db_path)
                    years = _db_get_years_in_data(name, self.db_path)
                    
                    for module in modules:
                        attrs = _db_get_attr_in_data(module, self.db_path)
                        for attr in attrs:
                            for year in years:
                                try:
                                    _db_drop_data(
                                        module = module,
                                        attr = attr,
                                        scn = name,
                                        year = year,
                                        db_path = self.db_path
                                    )
                                except sqlite3.OperationalError:
                                    pass

                    # If no more scenarios with output data also drop metadata
                    if len(_db_get_scn_in_data(self.db_path)) == 0:
                        _db_drop_all_metadata(self.db_path)
                else:
                    return None
                    
            del self.scenarios[name]
            _db_write_scn(self.scenarios, self.db_path)
            
        else:
            raise ValueError(f'Scenario {name} does not exist')

        return None

    def clean(self):
        '''Cleans up database file by dropping any tables not related to any defined
        scenario and the VACUUM;'''

        print(f'Cleaning {self.db_path}. This may take a while...')

        # Look for any tables that do not relate to defined scenarios
        tables = _db_get_tables(self.db_path)
        scns = self.scenarios

        tables_keep = ['scenarios']
        for module in _db_get_modules_in_data(self.db_path):
            tables_keep += [f'{module}__metadata']
            attrs = _db_read_metadata(module, self.db_path).keys()
            for attr in attrs:
                for scn in scns:
                    years = self.scenarios[scn]['years']
                    for year in years:
                        tables_keep += [f'{module}${attr}${scn}${year}', f'{module}${attr}${scn}${year}__idxcol']

        tables_drop = set(tables) - set(tables_keep)

        # Drop tables
        if len(tables_drop)>0:
            _db_drop_tables(tables_drop, self.db_path)
        # Vacuum
        _db_vacuum(self.db_path)

        print('Done!')
        return None

    def iterate(self, subset='no output'):
        '''Iterates over scenarios and years
        
        Parameters
        ----------
        subset : 'all', 'no output' (default) or (list of) scenario name(s)
            If 'all', all scenarios are iterated over
            If 'no output', only scenarios without outputs are iterated over
            If (list of) scenario name(s) are provided those scenarios are iterated over
        
        Example:
        for scn, year in obj.iterate():
        
            <-- CIBUSmod run code -->
            
            obj.store(
                scn = scn,
                year = year,
                ...
            )
            
        '''
        
        if subset == 'no output':
            w_output = _db_get_scn_in_data(self.db_path)
            scns = [s for s in self.scenarios if s not in w_output]
        elif subset == 'all':
            scns = self.scenarios.keys()
        elif isinstance(subset, str):
            scns = [subset]
        elif _isiterable(subset):
            scns = subset
            
        for scn in scns:
            for year in self.scenarios[scn]['years']:
                yield (scn,year)

    def store(self, scn, year, *args):
        '''Write output data to database file.
        
        Parameters
        ----------
        scn : str
            Scenario name
        year : str
            Year
        *args : CIBUSmod main moduels / iterable of AnimalHerd objects
            Pass in the modules to store output data for
            
        '''
        
        if scn not in self.scenarios:
            raise ValueError(f'Scenario "{scn}" not  defined')
        if year not in self.scenarios[scn]['years']:
            raise ValueError(f'Year {year} not defined for scenario "{scn}"')
        
        print(f"Writing outputs to '{self.db_path}'")
       
        for arg in args:

            if _isiterable(arg):
                if len(arg)>0 and np.all([isinstance(h,AnimalHerd) for h in arg]):
                    arg = concat_herds(arg)
                else:
                    warnings.warn('Passed iterable of non-AnimalHerd objects ignored')
                    continue

            if hasattr(arg, 'module_name') and hasattr(arg, 'data_attr'):

                module = arg.module_name
                
                # Only include 'scalable' data attributes (i.e. those that can be aggregated)
                data_attr_dict = {k : v for k, v in arg.data_attr.dict.items() if v['scalable']}

                _db_write_metadata(
                    metadata = data_attr_dict,
                    module = module,
                    db_path = self.db_path
                )
                
                for attr in data_attr_dict:
                    data_to_write = _get_check_and_clean(arg, module, attr)
                    _db_write_data(
                        data=data_to_write,
                        module=module,
                        attr=attr,
                        scn=scn,
                        year=year,
                        db_path=self.db_path
                    )
                    # Remove from cashe
                    if (module, attr, scn, year) in self.cashe:
                        del self.cashe[(module, attr, scn, year)]

            else:
                warnings.warn(f'Passed object of type {type(arg)} ignored')
        
        print("Outputs stored!")

        return None
    
    def get_table(
            self,
            module,
            attr,
            scn,
            year
    ):
        '''Get a table from the database or from the cashed tables
        
        Parameters
        ----------
        module : str
        attr : str
        scn : str
        year : str
        
        Returns
        -------
        pandas.DataFrame, pandas.Series or float'''

        try:
            res = self.cashe[(module, attr, scn, year)]
        except KeyError:
            res = _db_read_data(
                module = module,
                attr = attr,
                scn = scn,
                year = year,
                db_path = self.db_path
            )
            if res is not None:
                self.cashe[(module, attr, scn, year)] = res        
        
        return res

    def get_attr(
        self,
        module,
        attr,
        groupby = 'all',
        scn = 'all',
        years = 'all',
        interpolate = False,
        keep_duplicate_levels = 'index',
        suffixes = ('_idx','_col')
    ):
        '''Get specified data attribute from output.
        
        Parameters
        ----------
        module : str
            Module to get output from: 'DemandAndConversions', 'Regions', 'CropProduction' or 'AnimalHerd'
        attr : str
            data attribute to get
        groupby : str, list or dict, default 'all'
            If str or list data is grouped and aggregated by these index/column levels.
            If 'all' data is not aggregated
            If 'none'  data is summed over all index/columns
            If a dict is supplied relation tables are used
        scn : (list of) str
        years : (list of) str
        interpolate : Bool, default True
            If True interpolate between defined years
        keep_duplicate_levels: {'index','columns','both'}, default 'index'
            If the same groupby level is in both index and columns of data attribute
            then keep level on the specified axis. If 'both', both levels are
            retained and renamed with 'suffixes'
        suffixes : itterable of len 2, default ('_idx','_col')
            Suffixes to use for index and column levels if 'keep_duplicate_levels' is 'both'
            
        Returns
        -------
        pandas.DataFrame or Series with scenario (scn) and year as index and <groupby>
        as columns.
        '''
        
        short_hands = {
            'D':'DemandAndConversions', 'R':'Regions',
            'C':'CropProduction', 'A':'AnimalHerd'
        }
        if module not in short_hands.values():
            try:
                module = short_hands[module.upper()]
            except KeyError:
                raise ValueError('Invalid module name')
            
        if scn == 'all':
            scn = self.scenarios
        else:
            if isinstance(scn, str):
                scn = [scn]
            scn = [s for s in scn if s in self.scenarios]


        if years != 'all' and isinstance(years, str):
            years = [years]

        scn_year_in_output = _db_get_tables_index(self.db_path).get_loc_level((module, attr))[1]
        
        # Create index for data to be returned
        data_index = pd.MultiIndex.from_tuples(
            [(scn, year) for scn in scn if scn in scn_year_in_output.get_level_values('scn') for year in self.scenarios[scn]['years'] if (years == 'all' or year in years) and year in scn_year_in_output.get_loc_level(scn)[1]],
            names = ['scn', 'year']
        )
        if len(data_index)==0:
            raise ValueError('None of the selected scenarios/years in session or no output data')
        data_index_expected = data_index.copy()
        
        # Get data from first available scn and year
        x = None
        for scn, year in data_index:
            x = self.get_table(
                module = module,
                attr = attr,
                scn = scn,
                year = year
            )
            if x is not None:
                break
        if x is None:
            raise ValueError('No output data for any of the seleceted scenarios/years')
    
        if groupby == 'all':
            groupby = list(x.index.names)
            if isinstance(x,pd.DataFrame):
                groupby += [lvl for lvl in x.columns.names if lvl not in groupby]
        if groupby == 'none':
            groupby = []
        
        if isinstance(groupby,str):
            groupby = [groupby]
        if isinstance(groupby,dict):
            rel = {k:v for k,v in groupby.items() if v is not None and k != v}
            groupby = list(groupby)
        else:
            rel = {}
            
        # Check for duplicate groupby levels in both index and
        # columns and if 'keep_duplicate_levels' is 'both', add
        # suffixes in data and groupby list
        if isinstance(x, pd.DataFrame):
            idx_col_same = \
            [lvl for lvl in groupby if lvl in x.index.names and lvl in x.columns.names]
        else:
            idx_col_same = []
        if len(idx_col_same)>0:
            if keep_duplicate_levels == 'both':
                new_groupby = []
                idx_rename = {}
                col_rename = {}
                idx_drop = None
                col_drop = None
                for lvl in groupby:
                    if lvl in idx_col_same:
                        idx_rename.update({lvl:lvl+suffixes[0]})
                        col_rename.update({lvl:lvl+suffixes[1]})
                        new_groupby += [lvl+suffixes[0]]
                        new_groupby += [lvl+suffixes[1]]
                    else:
                        new_groupby += [lvl]
                groupby = new_groupby
            elif keep_duplicate_levels == 'index':
                idx_rename = None
                col_rename = None
                idx_drop = None
                col_drop = idx_col_same
            elif keep_duplicate_levels == 'columns':
                idx_rename = None
                col_rename = None
                idx_drop = idx_col_same
                col_drop = None
            else:
                raise ValueError("'keep_duplicate_levels' must be one of {'index','columns','both'}")
        else:
            idx_rename = None
            col_rename = None
            idx_drop = None
            col_drop = None

        d = []
        for scn, year in data_index:
            # Get attribute
            x = self.get_table(
                module = module,
                attr = attr,
                scn = scn,
                year = year
            )
            if x is None:
                data_index = data_index.drop((scn, year))
                continue
            
            # Drop or add suffixes to handle duplicate levels in index and columns
            if idx_rename is not None:
                x = x.rename_axis(index=idx_rename, columns=col_rename)
            if idx_drop is not None:
                x = x.droplevel(idx_drop)
            if col_drop is not None:
                x = x.droplevel(col_drop, axis=1)
            
            # Get index levels to group by
            ig = [g for g in groupby if g in x.index.names]
            if isinstance(x, pd.DataFrame):
                # Get column levels to group by
                cg = [g for g in groupby if g in x.columns.names]
            else:
                cg = None
            
            for lvl in [g for g in ig if g in rel]:
                # Rename index based on relation table
                x = x.rename(ParameterRetriever.get_rel(lvl,rel[lvl]), level=lvl)
            if cg is not None:
                for lvl in [g for g in cg if g in rel]:
                    # Rename columns based on relation table
                    x = x.rename(ParameterRetriever.get_rel(lvl,rel[lvl]), axis=1, level=lvl)
            
            if len(ig)>0:
                # Group by index levels and aggregate
                x = x.groupby(ig if len(ig)>1 else ig[0]).sum()
            else:
                # Aggregate across all index levels
                x = x.sum()
    
            if isinstance(x,pd.DataFrame) and cg is not None:
                if len(cg)>0:
                    # Group by column levels and aggregate
                    x = x.groupby(cg if len(cg)>1 else cg[0], axis=1).sum()
                else:
                    # Aggregate across all column levels
                    x = x.sum(axis=1)
            elif isinstance(x,pd.Series):
                if cg is not None:
                    if len(cg)>0:
                        # Group by column (now index) levels and aggregate
                        x = x.groupby(cg if len(cg)>1 else cg[0]).sum()
                    else:
                        # Aggregate across all column (now index) levels
                        x = x.sum()
    
            if isinstance(x,pd.DataFrame):
                i_lvls = x.index.nlevels
                c_lvls = x.columns.nlevels
                if i_lvls == 1 and isinstance(x.index,pd.MultiIndex):
                    # Fix problem with single-level MultiIndex stacking by
                    # converting to Index
                    x.index = x.index.get_level_values(0)
                if c_lvls == 1 and isinstance(x.columns,pd.MultiIndex):
                    # Fix problem with single-level MultiIndex stacking by
                    # converting to Index
                    x.columns = x.columns.get_level_values(0)
                # Stack or unstack dataframe to series (take shortest way)
                if i_lvls<=c_lvls:
                    x = x.unstack(list(range(i_lvls)))
                else:
                    x = x.stack(list(range(c_lvls)))
            if not isinstance(x,pd.Series):
                # If float returned create series
                x = pd.Series(x)
    
            d.append(x)

        # Get col index
        data_cols = d[0].index
        # Drop index to speed up concatenation
        d_ = [df.reset_index(drop=True) for df in d]
        # Combine and transpose
        data = pd.concat(d_, axis=1).T

        # Add in index and columns
        data.index = data_index
        data.columns = data_cols

        if len(data_index) < len(data_index_expected):
            warnings.warn(f'Some scenarios/years not found in output data.\n{data_index_expected.difference(data_index)}')
    
        if len(data.columns) == 1:
            # If only one column. Make series with attr as name
            data = data.iloc[:,0]
            data.name = attr
            
        if isinstance(data, pd.DataFrame) and data.columns.nlevels>1:
            # Reorder column levels as specified in groupby
            data = data.reorder_levels([g for g in groupby if g in data.columns.names], axis=1)
    
        if interpolate:
            # Interpolate to yearly data
    
            # Create new index with all years represented
            new_idx = pd.MultiIndex.from_tuples(
                [
                    (scn,str(year))
                    for scn in data.index.get_level_values('scn').unique()
                    for year in range(
                        min(data.loc[scn].index.astype(int)),
                        max(data.loc[scn].index.astype(int))+1
                    )
                ],
                names = ['scn','year']
            )
            # Reindex and interpolate
            data = data.reindex(new_idx).interpolate()
    
        return data
    
def _path_from_str(str):
    path = ''
    for word in str.split('/'):
        path = os.path.join(path, word)
    return path

def _isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def _get_check_and_clean(module, module_name, attr, zero_tol=1e-6):

    data = module.data_attr.get(attr).copy()
    
    if isinstance(data, pd.Series|pd.DataFrame):
        
        if data.isna().any().any():
            warnings.warn(f'NaNs in {module.par.name}.{attr}. Set to zero')
            data = data.fillna(0)
        if (data < -zero_tol).any().any():
            warnings.warn(f'Negative values of down to {data.min().min()} {module.data_attr[attr]["unit"]} in {module_name}.{attr}. Set to zero')
        data = data.where(data >= zero_tol, 0)
        
    elif isinstance(data, np.float_):
        
        if np.isnan(data):
            warnings.warn(f'NaNs in {module.par.name}.{attr}. Set to zero')
            data = 0
        if data < -zero_tol:
            warnings.warn(f'Negative value of {data} {module.data_attr[attr]["unit"]} in {module_name}.{attr}. Set to zero')
        if data < zero_tol:
            data = 0

    else:
        raise TypeError('Data should be pandas.Series, pandas.DataFrame or numpy.float')
        
    return data

class CasheDict(dict):
    # Implementation slightly modified from
    # https://gist.github.com/bencharb/729971d4a9e4633ea08a
    
    default_max_size = 100
    def __init__(self, *args, **kwargs):
        self.max_size = kwargs.pop('max_size', self.default_max_size)
        super(CasheDict, self).__init__(*args, **kwargs)
        
    def __setitem__(self, key, val):
        if key not in self:
            max_size = self.max_size-1  # so the dict is sized properly after adding a key
            self._prune_dict(max_size)
        super(CasheDict, self).__setitem__(key, val)
        
    def update(self, **kwargs):
        super(CasheDict, self).update(**kwargs)
        self._prune_dict(self.max_size)

    def _prune_dict(self, max_size):
        if len(self) >= max_size:
            diff = len(self) - max_size 
            for k in list(self)[:diff]:
                del self[k]
            
# SQLITE DATABASE FUNCTIONS

def _db_write_scn(scn_dict, db_path):
    
    scn_dict_flat = {
        k : {
            k : 'DICT'+'DICT'.join([k+'__'+'::'.join(v) if not isinstance(v, str) else k+'__'+v for k,v in v.items()]) if isinstance(v, dict) else '::'.join(v) if not isinstance(v, str|None) else 'None' if v is None else v
            for k, v in scn_dict[k].items()
        }
        for k in scn_dict
    }

    # Write to db
    con = sqlite3.connect(db_path)
    pd.DataFrame(scn_dict_flat).to_sql('scenarios', con, if_exists="replace")
    con.close()

    return None

def _db_read_scn(db_path):

    # Connect and read
    con = sqlite3.connect(db_path)
    scn_dict_flat = (
        pd.read_sql_query(f"SELECT * from scenarios", con)
        .set_index('index')
        .to_dict()
    )
    con.close()
    
    scn_dict = {
        k : {
            k : {
                kv.split('__')[0] : kv.split('__')[1].split('::')
                if '::' in kv.split('__')[1] else kv.split('__')[1]
                for kv in v.split('DICT') if kv != ''
            } if 'DICT' in v else v.split('::') if '::' in v else v if v != 'None' else None
            for k, v in scn_dict_flat[k].items()
        }
        for k in scn_dict_flat
    }

    for scn in scn_dict:
        if not isinstance(scn_dict[scn]['years'], list):
            scn_dict[scn]['years'] = [scn_dict[scn]['years']]
    
    return scn_dict

def _db_write_metadata(metadata, module, db_path):
    table = f'{module}__metadata'
    con = sqlite3.connect(db_path)
    pd.DataFrame(metadata).to_sql(table, con, if_exists="replace")
    con.close()

def _db_read_metadata(module, db_path):

    table = f'"{module}__metadata"'
    
    con = sqlite3.connect(db_path)
    
    metadata = (
        pd.read_sql_query(f'SELECT * from {table}', con)
        .set_index('index')
        .to_dict()
    )

    con.close()

    return metadata

def _db_write_data(data, module, attr, scn, year, db_path):
    
    data = data.copy()
    table = f'{module}${attr}${scn}${year}'
    
    idxcol = dict()
    if not isinstance(data, pd.DataFrame|pd.Series):
        if isinstance(data, np.float_):
            data = pd.Series(data)
            idxcol.update({'index_names' : 'index'})
        else:
            raise TypeError(f'{module} {attr} not a pandas.DataFrame, pandas.Series or numpy.Float')
    else:
        # Get index names
        idxcol.update({'index_names' : '::'.join([str(n) for n in data.index.names])})
        if isinstance(data, pd.DataFrame):
            # Get column names
            idxcol.update({'columns_names' : '::'.join([str(n) for n in data.columns.names])})
            # Flatten MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['::'.join(c) for c in data.columns]
            # Drop zero columns to avoid >2000 cols sqlite limit
            # and store columns to reindex when reading
            idxcol.update({'columns' : '::::'.join(data.columns)})
            data = data.loc[:, (data != 0).any(axis=0)]

    # Write data and metadata
    con = sqlite3.connect(db_path)
    data.to_sql(table, con, if_exists="replace")
    pd.Series(idxcol).to_sql(f'{table}__idxcol', con, if_exists="replace")
    con.close()

    return None

def _db_read_data(module, attr, scn, year, db_path):

    table = f'{module}${attr}${scn}${year}'
    
    con = sqlite3.connect(db_path)
    try:
        idxcol = (
            pd.read_sql_query(f'SELECT * from "{table}__idxcol"', con)
            .set_index('index')
            .loc[:,'0']
            .to_dict()
        )
        data = pd.read_sql_query(f'SELECT * from "{table}"', con)
    except:
        # Return None if table could not be read
        con.close()
        return None
    
    if 'columns_names' not in idxcol:
        data = (
            data
            .set_index(idxcol['index_names'].split('::'))
            .iloc[:,0]
        )
    else:
        data = (
            data
            .set_index(idxcol['index_names'].split('::'))
        )
        # Restore columns droped while writing to db
        data = data.reindex(idxcol['columns'].split('::::'), axis=1, fill_value=0)
        # Restore column (Multi)Index
        if '::' in idxcol['columns_names']:
            data.columns = pd.MultiIndex.from_tuples(
                [tuple(c.split('::')) for c in data.columns],
                names = idxcol['columns_names'].split('::')
            )
        else:
            data.columns = pd.Index(
                data.columns,
                name = idxcol['columns_names']
            )
        
    con.close()
    
    return data

def _db_drop_data(module, attr, scn, year, db_path):

    _db_drop_tables(
        tables = [
            f'{module}${attr}${scn}${year}',
            f'{module}${attr}${scn}${year}__idxcol'
        ],
        db_path = db_path
    )

    return None

def _db_drop_tables(tables, db_path):

    if isinstance(tables, str):
        tables = [tables]

    # Connect to sqlite
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Drop tables
    for table in tables:
        table = f'"{table}"'
        cur.execute(f"DROP TABLE {table}")
        print(f'Dropping table {table}')
    
    # commit close
    con.commit()
    con.close()

    return None

def _db_drop_all_metadata(db_path):

    # Get metadata tables
    tables = [t for t in _db_get_tables(db_path) if '__metadata' in t]

    # Connect to sqlite
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    for table in tables:
        # Drop table
        cur.execute(f"DROP TABLE {table}")

    # commitand close
    con.commit()
    con.close()

    return None

def _db_vacuum(db_path):

    # Connect to sqlite, vacuum and close
    con = sqlite3.connect(db_path)
    con.execute("VACUUM;")
    con.close()
    
def _db_get_tables(db_path):
    
    # Connect and read
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""SELECT name FROM sqlite_master 
    WHERE type='table';""")
    res = [t[0] for t in cur.fetchall()]
    con.close()

    return res

def _db_get_tables_index(db_path):
    '''Returns a pandas.MultiIndex of all data attribute tables
    with the index levels (module, attr, scn, year)'''
    tables = _db_get_tables(db_path)
    res = pd.MultiIndex.from_tuples(
        [tuple(t.split('$')) for t in tables if '__idxcol' not in t and '__metadata' not in t and t != 'scenarios'],
        names = ['module', 'attr', 'scn', 'year']
    ).sortlevel()[0]
    return res

def _db_get_modules_in_data(db_path):
    tables = _db_get_tables(db_path)
    return [t.replace('__metadata','') for t in tables if '__metadata' in t]

def _db_get_attr_in_data(module, db_path):
    tables = _db_get_tables(db_path)
    return list(np.unique(np.array([t.split('$')[1] for t in tables if '$' in t and t.split('$')[0] == module])))

def _db_get_scn_in_data(db_path):
    tables = _db_get_tables(db_path)
    return list(np.unique(np.array([t.split('$')[2] for t in tables if '$' in t])))

def _db_get_years_in_data(scn, db_path):
    tables = _db_get_tables(db_path)
    return list(np.unique(np.array([t.split('$')[3] for t in tables if '$' in t and t.split('$')[2] == scn and '__idxcol' not in t])))