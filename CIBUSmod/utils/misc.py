import functools
from itertools import product

import pandas as pd
import numpy as np

from typing import TYPE_CHECKING, Literal, Iterable, Sequence, Hashable

if TYPE_CHECKING:
    from ..main_modules.animal_herd import AnimalHerd

# Functions to set and get nested attributes
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rdelattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    return delattr(rgetattr(obj, pre) if pre else obj, post)

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
    # Convert to a pd.Series if not done already
    if not isinstance(herds, pd.Series):
        if not isinstance(herds, list):
            herds = [herds]
        herds = pd.Series(
            data=herds,
            index=pd.MultiIndex.from_tuples(
                [(h.species, h.breed, h.prod_system, h.sub_system) for h in herds],
                names=["species", "breed", "prod_system", "sub_system"],
            ),
        )
    check_index(herds)
    return herds


def check_index(herds: pd.Series) -> None:
    """Raises Exception if all AnimalHerd indexes are not the same"""
    if len(herds) > 0:
        for n in range(len(herds) - 1):
            if (herds.iloc[n].index != herds.iloc[n + 1].index).any():
                raise Exception("Indexes does not match across herds!")


def extend_index(
    levels: Sequence[Iterable[Hashable]],
    names: Iterable[str],
    index: pd.MultiIndex,
    mode: Literal["append", "prepend"] = "append",
) -> pd.MultiIndex:
    """
    Extend index with new values
    """
    # Return existing index if user does provide new levels/names
    if len([*levels]) == 0 or len([*names]) == 0:
        return index

    # Return new index if the original had no values
    if any(map(lambda lvl: len(lvl) == 0, index.levels)):
        return pd.MultiIndex.from_product(levels, names=names)

    if mode == "append":
        combine = lambda xs, ys: xs + ys
    elif mode == "prepend":
        combine = lambda xs, ys: ys + xs
    else:
        msg = f"Unexpected mode: {mode}."
        raise ValueError(msg)

    return pd.MultiIndex.from_tuples(
        [combine(tup, new_els) for tup in index.values for new_els in product(*levels)],
        names=combine(index.names, names),
    )
