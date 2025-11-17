import pandas as pd
import numpy as np
import warnings
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.misc import fix_herds, index_to_multi, elem_to_name

if TYPE_CHECKING:
    from ..main_modules.demand_and_conversions import DemandAndConversions
    from ..main_modules.crop_prod import CropProduction
    from ..main_modules.animal_herd import AnimalHerd
    from ..utils.retriever import ParameterRetriever

class CropResidueMgmt():
    '''Management module that handles the allocation of crop residues to different uses.

    Parameters
    ----------
    herds : (pandas.Series of) AnimalHerd object(s)
    crops : CropProduction object
    par : ParameterRetriever object
    '''

    def __init__(
            self,
            demand : "DemandAndConversions",
            crops : "CropProduction",
            herds : "pd.Series | AnimalHerd",
            par : "ParameterRetriever"
        ):

        self.par = par

        self.demand = demand
        self.crops = crops
        self.herds = fix_herds(herds)

    def calculate(self, verbose=False) -> None:
        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='CropResidueMgmt')

        vprint('Calculating harvestable crop residues ...')
        self.calculate_harvestable_crop_residues()

        vprint('Allocating harvestable crop residues to uses ...')
        self.allocate_crop_residues_to_uses()

        vprint('Calculating N and C in above and below ground crop residues ...')
        self.calculate_residues_NC()

        vprint(type='end')

        return None

    def calculate_harvestable_crop_residues(self) -> None:

        # Get harvestable crop residue products ('feed')
        res = self.par.get_unique('crop_resid', qry='parameter == "crop_resid_harvestable"')

        # Get above ground crop residues
        crop_residues = self.crops.data_attr.get('crop_residues')['above ground']

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
            pd.concat([h.data_attr.get('feed.crop_residue_demand') for h in self.herds if 'heads' in h.data_attr], axis=1)
            .T.groupby(['crop_resid']).sum().T
            # WIP
        )

        # Get crop residues used for bedding
        demand_for_bedding = (
            pd.concat([h.data_attr.get('bedding_material') for h in self.herds if 'heads' in h.data_attr], axis=1)
            .T.groupby(['feed']).sum().T
            .rename_axis(columns={'feed':'crop_resid'})
        )

        # Get crop residues used for other purposes
        demand_other = self.demand.data_attr.get('crop_resid_demand').sum(axis=1).groupby('crop_resid').sum()

        # Calculate total regional crop residue demand (i.e. for feed and bedding)
        regional_demand = demand_for_feed + demand_for_bedding

        # Calculate allocation factors to allocate harvestable crop residues to use on regional level
        crop_residues_alloc = self.crops.data_attr.get('crop_residues_harvestable').groupby(['region']).transform(lambda x: x/x.sum())

        # Calculate crop residue harvest per crop, prod_system and region
        crop_residues_harvest = (
            index_to_multi(regional_demand)
            .reindex(crop_residues_alloc.index.reorder_levels(['region','prod_system','crop']))
            .reorder_levels(['crop','prod_system','region'])
            .mul(
                crop_residues_alloc
            )
        )

        # Set harvest to harvestable if harvest exceeds harvestable
        crop_residues_harvest = crop_residues_harvest.where(
            crop_residues_harvest <= self.crops.data_attr.get('crop_residues_harvestable'),
            self.crops.data_attr.get('crop_residues_harvestable')
        )

        # Calculate remaining feed and bedding demand that needs to be met nationally and
        # add in total demand for other purposes
        remaining_demand = (regional_demand.sum() - crop_residues_harvest.sum()).add(demand_other, fill_value=0)
        # Calculate remaining harvestable crop residues
        remaining_harvestable = self.crops.data_attr.get('crop_residues_harvestable') - crop_residues_harvest

        # Calculate allocation factors to allocate harvestable crop residues to use on national level
        crop_residues_alloc_nat = remaining_harvestable.transform(lambda x: x/x.sum())

        # Add remaining demand to harvest
        crop_residues_harvest += remaining_demand * crop_residues_alloc_nat

        assert np.isclose(
            (regional_demand.sum().add(demand_other, fill_value=0)).astype(float),
            crop_residues_harvest.sum().astype(float)
        )
        # Check that demand does not exceed what is harvestable
        # NOTE: Might be good to include some automatic adjustment of "other" demand
        # if this happens.
        if not (crop_residues_harvest<=self.crops.data_attr.get('crop_residues_harvestable')).all().all():
            # Get crop residues with too high demand
            crs = (crop_residues_harvest>self.crops.data_attr.get('crop_residues_harvestable')).any()
            crs = crs.index[crs].to_list()
            warnings.warn(f"\nCropResidueMgmt: Crop residue demand exceeds availability for:\n{crs}")
            # Set harvest to harvestable
            crop_residues_harvest = self.crops.data_attr.get('crop_residues_harvestable')

        # Add data attribute
        self.crops.data_attr.add(
            crop_residues_harvest,
            name = 'crop_residues_harvest',
            unit = 'kg DM/year',
            orig = 'CropResidueMgmt',
            desc = 'Harvested crop residues'
        )

        return None
    
    def calculate_residues_NC(self):
        # Calculate N in crop residues
        self.crops.par.clear()
        self.crops.par.set(**self.crops.index.to_frame().to_dict('list'))
        p = self.crops.par.get

        # Get crop residues
        crop_residues = self.crops.data_attr.get('crop_residues').copy()

        # Subtract harvested crop residues
        crop_residues.loc[:,['above ground']] = (
            crop_residues.loc[:,['above ground']]
            .sub(self.crops.data_attr.get('crop_residues_harvest').sum(axis=1), axis=0)
        )

        for element in ['N','C']:
            # Get N/C in crop residue DM and multiply by total crop residues left in field
            df = (
                pd.DataFrame(
                    np.array([
                        p('ag_resid_'+element),
                        p('bg_resid_'+element),
                    ]).T,
                    index = self.crops.index,
                    columns = pd.Index(['above ground','below ground'], name='residue')
                )
                .mul(crop_residues)
            )
            
            # Add data attribute
            self.crops.data_attr.add(
                df,
                name = 'fertiliser.crop_residues_'+element,
                unit = f'kg {element}/year',
                orig = 'CropResidueMgmt',
                desc = f'{elem_to_name[element].title()} in above and below ground crop residues left in the field (i.e. not harvested)'
            )
