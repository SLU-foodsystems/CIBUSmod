import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr
from ..utils.misc import Container

from ..mgmt_modules.plant_nutrient_mgmt import Fertiliser

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

    def __init__(self,par,index):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = set()
        
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

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='CropProduction')

        # Set areas to ones
        self.area = pd.Series(np.ones(len(self.index)), index=self.index)
        self.data_attr.update(['area'])

        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        vprint('Calculating harvest ...')
        self.par.clear()
        self.par.set(**self.index.to_frame().to_dict('list'))
        self.harvest = self.area * p('yield') # [kg]
        self.harvest_dm = self.harvest * p('crop_dm') # [kg DM]
        self.data_attr.update(['harvest','harvest_dm'])

        vprint('Calculating production ...')
        self.calculate_production()

        vprint('Calculating crop residues ...')
        self.calculate_crop_residues()

        vprint('Calculating seed demand ...')
        self.calculate_seed_demand()

        # calculate other inputs???

        vprint(type='end')

    def scale(self,new_x):
        '''Scales all data attributes based on new_x (i.e. areas)
        
        Parameters
        ----------
        new_x : numpy.array or pandas.Series
        
        '''

        # Check so that new_x length and index match CropProduction object
        if len(new_x) != len(self.index):
            raise TypeError('Length of x does not match length of index!')
        if hasattr(new_x,'index'):
            if (new_x.index != self.index).any():
                raise TypeError('Index of x does not match index!')

        old_x = self.area
        
        for attr in self.data_attr:
            if rgetattr(self, attr) is not None:
                rsetattr(self, attr, rgetattr(self, attr).mul(new_x/old_x, axis=0))
    
    def make_static(self):
        '''Returns a StaticCropProduction object that retains all data attributes but
        has no methods or ParameterRetriever'''
        
        obj = StaticCropProduction()

        obj.index = self.index.copy()
        obj.fertiliser = Fertiliser()
        obj.data_attr = self.data_attr.copy()

        for attr in obj.data_attr:
            if rgetattr(self, attr) is not None:
                rsetattr(obj, attr, rgetattr(self, attr).copy())
            else:
                rsetattr(obj, attr, None)

        return obj

    def calculate_production(self):

        self.par.clear()

        # Get crop products
        cps = self.par.get_unique('crop_prod') 
        
        # Calculate crop product production
        production = self.par.get_from_frame(
            'crop_to_prod',
            pd.DataFrame(
                index=self.index,
                columns=pd.Index(cps, name='crop_prod')
            )
        ).mul(self.harvest, axis=0)

        self.production = production # [kg]
        self.data_attr.update(['production'])

    def calculate_crop_residues(self):
        self.par.clear()
        p = self.par.get

        # Calculate above and below ground crop residues from 
        # share of harvested DM that are residues multiplied
        # by DM-fraction and harvest 
        self.par.set(**self.index.to_frame().to_dict('list'))
        crop_residues = (
            pd.DataFrame(
                np.array([
                    p('ag_resid'),
                    p('bg_resid'),
                ]).T,
                index = self.index,
                columns = pd.Index(['above ground','below ground'], name='residue')
            )
            .mul(p('crop_dm'), axis=0)
            .mul(self.harvest, axis=0)
        )

        # Store crop residue dataframe [kg DM]
        self.crop_residues = crop_residues
        
        # Create series to keep track of harvested crop residues [kg DM]
        self.harvested_crop_residues = \
        pd.Series(
            0,
            index=self.index,
        )
        
        self.data_attr.update(['crop_residues', 'harvested_crop_residues'])

    def calculate_seed_demand(self):

        self.par.clear()

        # Get crop products
        cps = self.par.get_unique('crop_prod', qry='parameter == "seed"') 

        # Create dataframe
        seed_demand = pd.DataFrame(index=self.index, columns=pd.Index(cps, name='crop_prod'))
        self.seed_demand = self.par.get_from_frame('seed', seed_demand).mul(self.area, axis=0)
        self.data_attr.update(['seed_demand'])

class StaticCropProduction(Container):
    '''Class used to create static copys of animal her objects. These stores all attributes except 'par'
    but does not inherit any methods'''

    def __repr__(self):
        return CropProduction.__repr__(self)