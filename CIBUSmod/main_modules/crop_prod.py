import pandas as pd
import numpy as np

from CIBUSmod.utils.retriever import ParameterRetriever
from CIBUSmod.main_modules.regions import Regions

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr

class CropProduction(object):
    '''Main module that handles crop production

    Parameters
    ----------
    par : ParameterRetriever object
    index : pandas.Index or pandas.MultiIndex
        Index for the rows. This is also passed on to the ParameterRetriever
    '''

    module_name = 'CropProduction'

    def __init__(self, par: ParameterRetriever, regions: Regions = None, index: pd.MultiIndex = None):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = DataAttr(self)

        self.par = par

        if index is not None and regions is not None:
            raise ValueError("Provide only one of index or regions")
        if index is not None:
            self.regions = None
            self.index = index
        elif regions is not None:
            self.regions = regions
            self.index = regions.data_attr.get('x0_crops').index
        else:
            raise ValueError("One of regions or index must be proveided")

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

        # Update index
        if self.regions is not None:
            self.index = self.regions.data_attr.get('x0_crops').index

        # Set areas to ones
        self.data_attr.add(
            pd.Series(np.ones(len(self.index)), index=self.index),
            name = 'area',
            unit = 'ha or m2',
            orig = 'CropProduction',
            desc = 'Crop areas in ha (or m2 for greenhouse crops)'
        )

        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        vprint('Calculating harvest ...')

        self.par.clear()
        self.par.set(**self.index.to_frame().to_dict('list'))
        harvest = self.data_attr.get('area') * p('yield') # [kg]
        harvest_dm = harvest * p('crop_dm') # [kg DM]

        # Add data attributes
        self.data_attr.add(
            harvest,
            name = 'harvest',
            unit = 'kg/year',
            orig = 'CropProduction',
            desc = 'Total potential crop harvest in "natural" water content (see CropProduction parameter sheet)'
        )
        self.data_attr.add(
            harvest_dm,
            name = 'harvest_dm',
            unit = 'kg DM/year',
            orig = 'CropProduction',
            desc = 'Total potential crop harvest in dry matter'
        )

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

        old_x = self.data_attr.get('area')

        for attr in self.data_attr:
            if self.data_attr[attr]['scalable']:
                if self.data_attr.get(attr) is not None:
                    self.data_attr.add(
                        self.data_attr.get(attr).mul(new_x/old_x, axis=0),
                        name = attr,
                        **self.data_attr[attr]
                    )

    def make_static(self):
        '''Returns a StaticCropProduction object that retains all data attributes but
        has no methods or ParameterRetriever'''

        obj = StaticCropProduction()

        obj.index = self.index.copy()

        obj.data_attr = DataAttr(obj)

        for attr in self.data_attr:
            if self.data_attr.get(attr) is not None:
                obj.data_attr.add(data=self.data_attr.get(attr).copy(), name=attr, **self.data_attr[attr])
            else:
                obj.data_attr.add(data=None, name=attr, **self.data_attr[attr])

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
        ).mul(self.data_attr.get('harvest'), axis=0)

        # Add data attribute
        self.data_attr.add(
            production,
            name = 'production',
            unit = 'kg/year',
            orig = 'CropProduction',
            desc = 'Total production of crop products'
        )

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
                    p('ag_resid') * p('frac_renew'),
                    p('bg_resid') * p('frac_renew'),
                ]).T,
                index = self.index,
                columns = pd.Index(['above ground','below ground'], name='residue')
            )
            .mul(p('crop_dm'), axis=0)
            .mul(self.data_attr.get('harvest'), axis=0)
        )

        # Add data attribute
        self.data_attr.add(
            crop_residues,
            name = 'crop_residues',
            unit = 'kg DM/year',
            orig = 'CropProduction',
            desc = 'Generated above and below ground crop residues whether or not harvested'
        )

    def calculate_seed_demand(self):

        self.par.clear()

        # Get crop products
        cps = self.par.get_unique('crop_prod', qry='parameter == "seed"')

        # Create dataframe and calculate seed demand
        df = pd.DataFrame(index=self.index, columns=pd.Index(cps, name='crop_prod'))
        seed_demand = self.par.get_from_frame('seed', df).mul(self.data_attr.get('area'), axis=0)

        # Add data attribute
        self.data_attr.add(
            seed_demand,
            name = 'seed_demand',
            unit = 'kg/year',
            orig = 'CropProduction',
            desc = 'Demand for seeds'
        )

class StaticCropProduction():
    '''Class used to create static copys of animal herd objects. These stores all attributes except 'par'
    but does not inherit any methods'''

    index: pd.Index | pd.MultiIndex
    data_attr: DataAttr

    def __repr__(self):
        return CropProduction.__repr__(self)
