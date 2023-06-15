import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr

class Regions(object):
    '''Class that handles region attributes

    Parameters
    ----------
    x0 : 
    par : 

    Attributes set on init
    ----------------------

    Attributes set by CropProduction.calculate()
    --------------------------------------------
    '''

    def __init__(self,x0,par):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = set()
        
        self.x0 = x0
        self.par = par
        self.index = x0['crp'].index.get_level_values('region').unique()

    def get_region_attributes(self):
        pass