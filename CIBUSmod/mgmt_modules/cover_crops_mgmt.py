import pandas as pd
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init

if TYPE_CHECKING:
    from CIBUSmod.main_modules.crop_prod import CropProduction
    from CIBUSmod.utils.retriever import ParameterRetriever

class CoverCropsMgmt(object):

    '''Management module that handles cover crops.

    Parameters
    ----------
    crops : CropProduction object
    par : ParameterRetriever object
    '''

    def __init__(
        self,
        crops : "CropProduction",
        par : "ParameterRetriever"
    ):

        self.par = par
        self.crops = crops

    def calculate(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='CoverCropsMgmt')

        vprint('Calculating cover crops area ...')
        self.calculate_cover_crops_area()

        vprint('Calculating above and below ground cover crop residues ...')
        self.calculate_cover_crop_residues()

        vprint(type='end')
        
        return None

    def calculate_cover_crops_area(self):
        self.par.clear()
        # Get cover crops
        ccs = self.par.get_unique('cover_crop')
        # Get crop areas
        areas = self.crops.data_attr.get('area')
        # Create data frame for cover crop areas
        CC_area = pd.DataFrame(
            index = areas.index,
            columns = pd.Index(ccs, name = 'cover_crop')
        )
        # Calculate cover crop areas
        CC_area = (self.par.get_from_frame(
            'share_preceded_by_cover_crop',
            CC_area
        )/100).mul(areas, axis=0)

        # Store data attribute
        self.crops.data_attr.add(
            CC_area,
            name = 'cover_crops.area',
            unit = 'ha',
            orig = "CoverCropsMgmt",
            desc = "Area of cover crops"
        )

    def calculate_cover_crop_residues(self):
        self.par.clear()

        # Get cover crop area
        CC_area = self.crops.data_attr.get('cover_crops.area')

        # Calculate above and below ground residues from cover crops
        ag_residues = CC_area * self.par.get_from_frame('ag_biomass', CC_area)
        bg_residues = CC_area * self.par.get_from_frame('bg_biomass', CC_area)

        CC_residues = pd.concat([
            pd.concat({'above ground': ag_residues}, names=['residue'], axis=1),
            pd.concat({'below ground': bg_residues}, names=['residue'], axis=1)
        ], axis=1)

        # Store data attributes
        self.crops.data_attr.add(
            CC_residues,
            name = 'cover_crops.residues',
            unit = 'kg DM/year',
            orig = "CoverCropsMgmt",
            desc = "Above and below ground residues from cover crops"
        )

    def get_residual_N(self):
        '''Method to return the residual nitrogen (N) from cover crops [kg N].
        Returns a pd.DataFrame of the same form as df with (prod_system, region) as index'''
        self.par.clear()
        
        # Get cover crop areas
        CC_area = self.crops.data_attr.get('cover_crops.area')
        
        # Calculate residual nitrogen from cover crops per production system and region
        CC_N_resid = (
            (CC_area * self.par.get_from_frame('resid_N', CC_area))
            .sum(axis=1)
            .groupby(['prod_system', 'region']).sum()
        )

        return CC_N_resid
    
    def get_soil_loss_adjust(self, df):
        '''Method to return the adjustment factors for soil losses due to cover crops.
        Returns a pd.DataFrame of the same form as df'''
        return self._get_adjust_factors('soil_loss_adjust', df)

    def get_leach_adjust(self, df):
        '''Method to return the adjustment factors for leaching due to cover crops.
        Returns a pd.DataFrame of the same form as df'''
        return self._get_adjust_factors('leach_adjust', df)
    
    def _get_adjust_factors(self, of, df):
        self.par.clear()
        
        # Get crop and cover crop areas
        crop_area = self.crops.data_attr.get('area')
        CC_area = self.crops.data_attr.get('cover_crops.area')
        
        # Calculate leaching adjustment factors
        adjust_factors = pd.DataFrame(
            0.0,
            index = df.index,
            columns = df.columns
        )
        for cc in CC_area.columns:
            self.par.set(cover_crop = cc)
            # Add inverse of leach adjust factor times cover crop area
            adjust_factors += (1-self.par.get_from_frame(of, df)).mul(CC_area.loc[:,cc], axis=0)
        # Divide by total crop area and invert (Set NaN to 1 for cases where crop area is 0)
        adjust_factors = (1 - adjust_factors.div(crop_area, axis=0)).fillna(1)
        
        return adjust_factors
    
