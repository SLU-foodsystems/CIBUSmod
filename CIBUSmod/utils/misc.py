import functools
import os
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..main_modules.animal_herd import AnimalHerd

# Functions to set and get nested attributes
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class DataAttr(object):
    '''Class that assigns data attributes on main modules and keeps track of metadata.
    
    Parameters
    ----------
    parent : A CIBUSmod main module
    '''

    def __init__(self, parent):
        self.parent = parent
        self.dict = dict()

    def __repr__(self):
        
        # Get column widths for printing
        tot_w = 140
        if len(self.dict) > 0:
            name_w = max([len(n) for n in self.dict]) + 3
            unit_w = max([len(self.dict[n]['unit']) for n in self.dict]) + 3
            orig_w = max([len(self.dict[n]['orig']) for n in self.dict]) + 3
        else:
            name_w = 30
            unit_w = 20
            orig_w = 30
        desc_w = tot_w - name_w - unit_w - orig_w

        fmt_str = "{:<"+str(name_w)+"} {:<"+str(unit_w)+"} {:<"+str(orig_w)+"} {:<"+str(desc_w)+"}\n"

        rep_str = fmt_str.format('ATTR', 'UNIT', 'ORIG', 'DESC')

        for key, value in self.dict.items():
            name = key
            unit, orig, desc = value['unit'], value['orig'], value['desc']
            
            for i in range(max(
                round(len(name)/(name_w-2)+0.5),
                round(len(unit)/(unit_w-2)+0.5),
                round(len(orig)/(orig_w-2)+0.5),
                round(len(desc)/(desc_w-2)+0.5)
            )):
                rep_str += fmt_str.format(
                    name[(name_w-2)*i:(name_w-2)*(i+1)],
                    unit[(unit_w-2)*i:(unit_w-2)*(i+1)],
                    orig[(orig_w-2)*i:(orig_w-2)*(i+1)],
                    desc[(desc_w-2)*i:(desc_w-2)*(i+1)]
                )
                
        return rep_str

    def __getitem__(self, item):
        return self.dict[item]

    def __len__(self):
        return len(self.dict)

    def __iter__(self):
        return iter(self.dict)

    def keys(self):
        return self.dict.keys()

    def items(self):
        return self.dict.items()

    def values(self):
        return self.dict.values()

    def add(
        self,
        data,
        name:str,
        unit:str = '',
        orig:str = '',
        desc:str = '',
        scalable:bool = True
    ):
        '''Sets data attribute on main module (parent) and stores meta-data.

        Parameters
        ----------
        data : Any (usually pandas.DataFrame or pandas.Series)
            Data to store in data attribute
        name : str
            Name of data attribute
        unit : str, default ''
            Unit
        orig : str, default ''
            Origin of the data. A CIBUSmod module name where it is calculated
        desc : str, default ''
            Short description of data
        scalable : Bool, default True
            If scalable is True data is scaled in .scale() methods otherwise not

        Returns
        -------
        None
        '''
        # Set attribute in parent
        rsetattr(self.parent, name, data)
        # Update dict
        self.dict.update({
            name : {
                'unit' : unit,
                'orig' : orig,
                'desc' : desc,
                'scalable' : scalable
            }
        })

        return None
    
    def get(self, attr:str):
        '''Get data attribute
        
        Parameters
        ----------
        attr : str
            Data attribute to get
            
        Returns
        -------
        Data attribute, usually a pandas.DataFrame or pandas.Series'''

        res = rgetattr(self.parent, attr)

        return res

class Container(object):
    '''Class that is only for storing arbitrary attributes in a nested way.'''
    def new(self):
        obj = type(self).__new__(self.__class__)
        return obj

# Some helper functions for pandas objects

# Function that aligns and multiplies two dataframes
def multiply_aligned(
        left : pd.DataFrame,
        right : pd.DataFrame
    ) -> pd.DataFrame:
    '''Note: 'left' should be "bigger" than 'right'. I.e. contain
    more or the same number of levels in index and/or columns
    than 'right'''
    
    aligned = left.align(right)
    multiplied = aligned[0]*aligned[1]
    # Make sure that column and index levels are ordered as in left
    if len(left.columns.names)>1:
        multiplied = multiplied.reorder_levels(left.columns.names, axis=1)
    if len(left.index.names)>1:
        multiplied = multiplied.reorder_levels(left.index.names, axis=0)
    # Make sure index and columns are ordered as in left
    multiplied = multiplied.reindex(index=left.index, columns=left.columns)
    return multiplied

# Function that converts a pandas.MultiIndex to a dict {level : level values}
def multiindex_to_dict(idx : pd.MultiIndex) -> dict:
    '''Creates a dict from pandas.MultiIndex'''
    return {lvl:[val for val in idx.get_level_values(lvl)] for lvl in idx.names}

# Invert a dictionary
def inv_dict(x : dict) -> dict:
    '''Invert dictionary'''
    inv_x = {}
    for k,v in x.items():
        inv_x[v] = inv_x.get(v,[]) + [k]
    return inv_x

def index_to_multi(obj: pd.Index | pd.DataFrame | pd.Series) -> pd.MultiIndex | pd.DataFrame | pd.Series:
    '''Converts pandas.Index/DataFrame/Series to (have) a 1 level pandas.MultiIndex'''
    if isinstance(obj, pd.Index):
        return pd.MultiIndex.from_tuples(
            [(i,) for i in obj],
            names=obj.names
        )
    elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        if not isinstance(obj.index, pd.Index):
            raise TypeError('DataFrame or Series must have a pandas.Index, not a pandas.MultiIndex')
        obj = obj.copy()
        obj.index = pd.MultiIndex.from_tuples(
            [(i,) for i in obj.index],
            names=obj.index.names
        )
        return obj

def fix_herds(herds : "AnimalHerd | list | pd.Series") -> pd.Series:
    '''Convert herds to pd.Series if list or AnimalHerd object supplied and check index
    across AnimalHerd objects'''
    if isinstance(herds, pd.Series):
        herds = herds
    else:
        if not isinstance(herds, list):
            herds = [herds]
        herds = pd.Series(
            data=herds,
            index=pd.MultiIndex.from_tuples(
                [(h.species,h.breed,h.prod_system,h.sub_system) for h in herds],
                names=['species','breed','prod_system','sub_system']
            )
        )
    check_index(herds)
    return herds

def check_index(herds : pd.Series) -> None:
    '''Raises Exception if all AnimalHerd indexes are not the same'''
    if len(herds)>0:
        for n in range(len(herds)-1):
            if (herds[n].index != herds[n+1].index).any():
                raise Exception('Indexes does not match across herds!')