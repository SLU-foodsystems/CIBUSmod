import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr
from ..utils.misc import Container

class CropProduction(object):
    '''Class that handles crop production
    
    Parameters
    ----------
    par : ParameterRetriever object
    index : pandas.Index or pandas.MultiIndex
        Index for the rows. This is also passed on to the ParameterRetriever
        
    Attributes set on init
    ----------------------
    index : pandas.Index or padnas.MultiIndex
        Index for rows
        
    Attributes set by CropProduction.calculate()
    --------------------------------------------
    area : pandas.DataFrame
        Total crop area [ha] for most crops and [m2] for greenhouse crops 
    harvest : pandas.DataFrame
        Total harvest of crops [kg DM] for cereals, pulses, oilseeds, forages etc. and [kg wet weight] for vegetables, berries, fruit etc.
    production : pandas.DataFrame
        Total production of "crop products" (e.g. the crops "Wheat, winter" and "Wheat, spring" both produce the crop product "wheat") [kg DM] or [kg wet weight]
    by_products : pandas.DataFrame
        As above but for by-products (e.g. straw)
    '''

    # List of attributes in class
    # Note: remember to update if more attributes are included!
    data_attr = ['area','harvest','production','by_products','seed_demand']

    def __init__(self,par,index):
        
        self.par = par
        self.index = index

    def calculate(self,verbose=False):
        '''Calculates crop production based on a vector ('x') of crop areas.
        Index of 'x' is retained in the output and can be used as filters for the ParameterRetriever.

        Parameters
        ----------
        verbose : Bool

        Returns
        -------
        Nothing. Stores output in pandas.DataFrames in the attrubutes: 'area', 'harvest', 'production' and 'by_products'
        '''

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        for i in self.index.names:
            self.par.set(
                **{i : self.index.get_level_values(i).values}
            )

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='CropProduction')

        self.area = pd.Series(np.ones(len(self.index)), index=self.index)

        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        vprint('Calculating harvest ...')
        self.harvest = self.area * p('yield')

        vprint('Calculating production ...')
        self.calculate_production()

        vprint('Calculating seed demand ...')
        self.calculate_seed_demand()

        # calculate NPK req
        # calculate tractor energy
        # calculate other inputs

        vprint(type='end')

    def scale(self,new_x):
        '''Rescales all output based on new_x (i.e. areas) and returns a StaticCropProduction object
        
        Parameters
        ----------
        new_x : numpy.array or pandas.Series
        
        '''

        # Check so that x length and index match CropProduction object
        if len(new_x) != len(self.index):
            raise TypeError('Length of x does not match length of index!')
        if hasattr(new_x,'index'):
            if (new_x.index != self.index).any():
                raise TypeError('Index of x does not match index!')

        obj = StaticCropProduction()

        old_x = self.area

        obj.index = self.index.copy()
        obj.par = self.par # Note: This is not a copy
        
        for attr in self.data_attr:
            try:
                rsetattr(obj, attr, rgetattr(self, attr).mul(new_x/old_x, axis=0))
            except:
                pass
        
        return obj

    def calculate_production(self):
        
        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        cps = self.par.get_unique('crop_prod') # Get crop products
        bps = self.par.get_unique('by_prod') # Get by-products

        # Calculate crop product production
        production = pd.DataFrame(index=self.index, columns=pd.Index(cps, name='crop_prod'))
        production = self.par.get_from_frame('crop_to_prod', production).mul(self.harvest, axis=0)
        self.par.remove('crop_prod')

        # Calculate by-product generation
        by_products = pd.DataFrame(index=self.index, columns=pd.Index(bps, name='by_prod'))
        by_products = self.par.get_from_frame('crop_to_prod', by_products).mul(self.harvest, axis=0)
        self.par.remove('by_prod')

        self.production, self.by_products = (production, by_products)

    def calculate_seed_demand(self):

        # Get crop products
        cps = self.par.get_unique('crop_prod', qry='parameter == "seed"') 

        # Create dataframe
        seed_demand = pd.DataFrame(index=self.index, columns=pd.Index(cps, name='crop_prod'))
        self.seed_demand = self.par.get_from_frame('seed', seed_demand).mul(self.area, axis=0)

class StaticCropProduction(Container):
    '''Class used to create static copys of animal her objects. These stores all attributes except 'par'
    but does not inherit any methods'''

    def __repr__(self):
        return CropProduction.__repr__(self)