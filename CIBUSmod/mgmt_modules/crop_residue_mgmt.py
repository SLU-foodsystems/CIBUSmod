import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.misc import fix_herds, index_to_multi

if TYPE_CHECKING:
    from ..main_modules.crop_prod import CropProduction
    from ..main_modules.animal_herd import AnimalHerd
    from ..utils.retriever import ParameterRetriever

class CropResidueMgmt():
    '''Management module that handles the allocation of crop residues to different uses
    
    Parameters
    ----------
    herds : (pandas.Series of) AnimalHerd object(s)
    crops : CropProduction object
    par : ParameterRetriever object
    '''

    def __init__(
            self,
            crops : "CropProduction",
            herds : "pd.Series | AnimalHerd",
            par : "ParameterRetriever"
        ):
        
        self.par = par

        self.crops = crops
        self.herds = fix_herds(herds)

    def calculate(self, verbose=False) -> None:
        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='CropResidueMgmt')

        vprint('Calculating harvestable crop residues ...')
        self.calculate_harvestable_crop_residues()

        vprint('Allocating harvestable crop residues to uses ...')
        self.allocate_crop_residues_to_uses()

        vprint(type='end')

        return None

    def calculate_harvestable_crop_residues(self) -> None:
        
        # Get harvestable crop residue products ('feed')
        res = self.par.get_unique('crop_resid', qry='parameter == "crop_resid_harvestable"')

        # Get above ground crop residues
        crop_residues = self.crops.crop_residues['above ground']

        df = pd.DataFrame(
            index = crop_residues.index,
            columns = pd.Index(
                res,
                name = 'crop_resid'
            )
        )

        crop_residues_harvestable = (
            self.par.get_from_frame('crop_resid_harvestable', df)
            .mul(crop_residues, axis=0)
        )
        
        # Add data attribute
        self.crops.data_attr.add(
            crop_residues_harvestable,
            name = 'crop_residues_harvestable',
            unit = 'kg DM/year',
            orig = 'CropResidueMgmt',
            desc = 'Potentially harvestable crop residues'
        )

        return None

    def allocate_crop_residues_to_uses(self) -> None:

        # Get crop residues used for feed
        demand_for_feed = (
            pd.concat([h.feed.crop_residue_demand for h in self.herds], axis=1)
            .xs('domestic', level='origin', axis=1)
            .T.groupby(['crop_resid']).sum().T
            # WIP
        )
        
        # Get crop residues used for bedding
        demand_for_bedding = (
            pd.concat([h.bedding_material for h in self.herds], axis=1)
            .T.groupby(['feed']).sum().T
            .rename_axis(columns={'feed':'crop_resid'})
        )
        
        # Get crop residues used for energy
        demand_for_energy = 0
        # TO BE IMPLEMNTED!
        
        # Calculate total crop residue demand
        total_demand = demand_for_feed + demand_for_bedding + demand_for_energy
        
        # Calculate allocation factors to allocate harvestable crop residues to use on regional level
        crop_residues_alloc = self.crops.crop_residues_harvestable.groupby(['region']).transform(lambda x: x/x.sum())
        
        # Calculate crop residue harvest per crop, prod_system and region
        crop_residues_harvest = (
            index_to_multi(total_demand)
            .reindex(crop_residues_alloc.index.reorder_levels(['region','prod_system','crop']))
            .reorder_levels(['crop','prod_system','region'])
            .mul(
                crop_residues_alloc
            )
        )
        
        # Set harvest to harvestable if harvest exceeds harvestable
        crop_residues_harvest = crop_residues_harvest.where(
            crop_residues_harvest <= self.crops.crop_residues_harvestable,
            self.crops.crop_residues_harvestable
        )
        
        # Calculate remaining demand that needs to be met nationally and remaining harvestable
        # crop residues
        remaining_demand = total_demand.sum() - crop_residues_harvest.sum()
        remaining_harvestable = self.crops.crop_residues_harvestable - crop_residues_harvest
        
        # Calculate allocation factors to allocate harvestable crop residues to use on national level
        crop_residues_alloc_nat = remaining_harvestable.transform(lambda x: x/x.sum())
        
        # Add remaining demand to harvest
        crop_residues_harvest += remaining_demand * crop_residues_alloc_nat
        
        assert np.isclose(
            total_demand.sum().astype(float),
            crop_residues_harvest.sum().astype(float)
        )
        assert (crop_residues_harvest<=self.crops.crop_residues_harvestable).all().all()
        
        # Add data attribute
        self.crops.data_attr.add(
            crop_residues_harvest,
            name = 'crop_residues_harvest',
            unit = 'kg DM/year',
            orig = 'CropResidueMgmt',
            desc = 'Harvested crop residues'
        )

        return None