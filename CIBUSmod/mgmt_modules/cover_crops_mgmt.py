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

    def calculate(self):
        return None

    def get_cover_crops_area(self):
        self.par.clear()
        # Get cover crops
        ccs = self.par.get_unique('cover_crop')
        # Get crop areas
        areas = self.crops.data_attr.get('area')
        # Create data frame for cover crop areas
        cover_crop_areas = pd.DataFrame(
            index = areas.index,
            columns = pd.Index(ccs, name = 'cover_crop')
        )
        # Calculate cover crop areas
        cover_crop_areas = (self.par.get_from_frame(
            'share_preceded_by_cover_crop',
            cover_crop_areas
        )/100).mul(areas, axis=0)

        return cover_crop_areas