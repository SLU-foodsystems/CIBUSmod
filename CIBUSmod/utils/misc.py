import functools

# Functions to set and get nested attributes
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class Container(object):
    '''Class that is only for storing arbitrary attributes in a nested way.'''
    def new(self):
        obj = type(self).__new__(self.__class__)
        return obj

# Function that aligns
def multiply_aligned(left,right):
    # Note: 'left' should be "bigger" than 'right'. I.e. contain
    # more or the same number of levels in index and/or columns
    # than 'right'
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