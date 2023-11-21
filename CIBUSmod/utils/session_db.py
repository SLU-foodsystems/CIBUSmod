# THIS W.I.P. TRYING TO USE A SQLITE DATABASE TO STORE OUTPUTS INSTEAD OF HAVING
# EVERYTHING IN MEMORY. RUNS INTO PROBLES WITH SOME DATAFRAMES HAVING > 2000 COLUMNS
# POT. SOLUTIONS:
#    1) Each AnimalHerd object as separate module and concat after reading from db
#    2) Rethink data structure and drop zero columns (some work but might save a lot of space)


import warnings
import sqlite3
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

    def __init__(self, name, data_path):
        
        self.name = name

        self.data_path = _path_from_str(data_path)
        self.db_path = os.path.join(self.data_path, 'output', self.name+'.sqlite')
        ParameterRetriever.set_data_folder(self.data_path)

        if os.path.isfile(self.db_path):
            self.scenarios = _db_read_scn(self.db_path)
        else:
            self.scenarios = {}

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
f'''{scn if in_scenarios else '('+scn+')'}: {' --> '.join(years)} {'('+str(nyears)+' years)' if nyears>1 else ''}{' [has output]' if in_output else ''}{' [is defined]' if in_scenarios else ''}
'''
            )

        str2 = '''OUTPUT DATA
===========
'''
        for i,module in enumerate(self.output.columns):
            str2 += (
f'''{module}
{'-'*len(module)}
{self.output.iloc[0,i].data_attr}

'''
            )
            
        return '\n'.join([str0, str1, str2])

    def add_scenario(self, name, scenario=None, modules='all', pars='all', years='nd'):

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

        _db_write_scn(self.scenarios, self.db_path)
        
        return None

    def remove_scenario(self, name):
        
        del self.scenarios[name]
        _db_write_scn(self.scenarios, self.db_path)
        
        return None

    def store(
            self,
            scn,
            year,
            *args
        ):

        scn = 'base year'
        year = '2020'
        
        for arg in args:

            if isinstance(arg, DemandAndConversions):
                # Demand
                module = 'DemandAndConversions'
                _db_write_metadata(
                    metadata=arg.data_attr.dict,
                    module=module,
                    db_path=self.db_path
                )
                for attr in arg.data_attr:
                    if arg.data_attr[attr]['scalable']:
                        print(module,attr)
                        _db_write_data(
                            data=arg.data_attr.get(attr),
                            module=module,
                            attr=attr,
                            scn=scn,
                            year=year,
                            db_path=self.db_path
                        )

            elif isinstance(arg, Regions):
                # Regions
                module = 'Regions'
                _db_write_metadata(
                    metadata=arg.data_attr.dict,
                    module=module,
                    db_path=self.db_path
                )
                for attr in arg.data_attr:
                    if arg.data_attr[attr]['scalable']:
                        print(module,attr)
                        _db_write_data(
                            data=arg.data_attr.get(attr),
                            module=module,
                            attr=attr,
                            scn=scn,
                            year=year,
                            db_path=self.db_path
                        )
                    
            elif isinstance(arg, CropProduction):
                # Crops
                module = 'CropProduction'
                _db_write_metadata(
                    metadata=arg.data_attr.dict,
                    module=module,
                    db_path=self.db_path
                )
                for attr in arg.data_attr:
                    if arg.data_attr[attr]['scalable']:
                        print(module,attr)
                        _db_write_data(
                            data=arg.data_attr.get(attr),
                            module=module,
                            attr=attr,
                            scn=scn,
                            year=year,
                            db_path=self.db_path
                        )
                    
            elif _isiterable(arg):
                if np.all([isinstance(h,AnimalHerd) for h in arg]):
                    # Animals
                    module = 'AnimalHerd'
                    all_herds = concat_herds(
                        arg.apply(lambda x: x.make_static())
                    )
                    _db_write_metadata(
                        metadata=all_herds.data_attr.dict,
                        module=module,
                        db_path=self.db_path
                    )
                    for attr in all_herds.data_attr:
                        if all_herds.data_attr[attr]['scalable']:
                            print(module,attr)
                            _db_write_data(
                                data=all_herds.data_attr.get(attr),
                                module=module,
                                attr=attr,
                                scn=scn,
                                year=year,
                                db_path=self.db_path
                            )
                    
                else:
                    print('Iterable of non-AnimalHerd objects ignored')
            else:
                print(type(arg),'ignored')

    def iterate(self):
        for scn in self.scenarios:
            for year in self.scenarios[scn]['years']:
                yield (scn,year)

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

    # Write data and metadata
    con = sqlite3.connect(db_path)
    data.to_sql(table, con, if_exists="replace")
    pd.Series(idxcol).to_sql(f'{table}__idxcol', con, if_exists="replace")
    con.close()

    return None

def _db_read_data(module, attr, scn, year, db_path):

    table = f'{module}${attr}${scn}${year}'
    
    con = sqlite3.connect(db_path)
    
    idxcol = (
        pd.read_sql_query(f'SELECT * from "{table}__idxcol"', con)
        .set_index('index')
        .loc[:,'0']
        .to_dict()
    )
    
    if 'columns_names' not in idxcol:
        data = (
            pd.read_sql_query(f'SELECT * from "{table}"', con)
            .set_index(idxcol['index_names'].split('::'))
            .iloc[:,0]
        )
    else:
        data = (
            pd.read_sql_query(f'SELECT * from "{table}"', con)
            .set_index(idxcol['index_names'].split('::'))
        )
        data.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split('::')) for c in data.columns],
            names = idxcol['columns_names'].split('::')
        )
        
    con.close()
    
    return data

def _db_get_tables(db_path):
    
    # Connect and read
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""SELECT name FROM sqlite_master 
    WHERE type='table';""")
    res = cur.fetchall()
    con.close()

    return res