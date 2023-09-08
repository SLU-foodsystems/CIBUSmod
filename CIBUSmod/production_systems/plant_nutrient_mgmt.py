import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr, multiply_aligned
from ..utils.output_data_manip import concat_herds
from ..utils.misc import Container

class PlantNutrientMgmt():
    '''Class that that calculates ammount of plant nutrients needed for crop production
    and balances this with manure generation, etc.
    
    Parameters
    ----------
    crops : CropProduction object
    herds : (pandas.Series of) AnimalHerd object(s)
    demand : DemandAndConversions object
    par : ParameterRetriever object
    '''

    def __init__(self,crops,herds,demand,par):

        self.par = par
        self.crops = crops
        self.demand = demand

        if isinstance(herds, pd.Series):
            self.herds = herds
        else:
            self.herds = pd.Series(
                data=herds,
                index=pd.MultiIndex.from_tuples(
                    [(herds.species,herds.breed,herds.prod_system,herds.sub_system)],
                    names=['species','breed','prod_system','sub_system']
                )
            )
            
    def calculate(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='PlantNutrientMgmt')
    
        # Create feed objects in herds
        self.crops.fertiliser = Fertiliser()

        vprint('Calculating crop N requirements ...')
        self.calculate_TAN_req()

        vprint('Distributing manure ...')
        self.distribute_manure()

        vprint(type='end')
             
    def calculate_TAN_req(self):
        idx=pd.IndexSlice

        # Get crop yields
        yields = (self.crops.harvest / self.crops.area).fillna(0)

        # Calculate ley share per region
        lu_rel = self.crops.par.get_rel('crop','land_use') # crop --> land_use
        cg_rel = self.crops.par.get_rel('crop','crop_group') # crop --> crop_group

        ley_share = self.crops.area.to_frame()
        ley_share.loc[:,'lu'] = ley_share.index.get_level_values('crop').map(lu_rel)
        ley_share.loc[:,'cg'] = ley_share.index.get_level_values('crop').map(cg_rel)
        ley_share = (
            ley_share.set_index(['lu','cg'], append=True)[0]
            .loc[idx[:,:,:,'cropland',:]]
            .groupby(['cg','prod_system','region'])
            .sum()
        )
        ley_share = (ley_share / ley_share.groupby(['prod_system','region']).transform('sum')).loc['Ley']

        # Propagate across uses (same yield/ley_share for all uses)
        pdf = pd.DataFrame(
            data=1,
            index=yields.index,
            columns=pd.Index(['food','non-food','feed','none'], name='use')
        )

        yields = pdf.mul(yields, axis=0)
        ley_share = pdf.mul(ley_share, axis=0)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!! USE NEW 'crops.production_per_use' ATTRIBUTE INSTEAD !!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Get share of each crop_prod used for food, non-food and feed
        # Note: Use shares are calculated on national level. Potential
        # future development is to try and differentiate use shares
        # across regions.
        use_per_crop_prod = pd.concat([
            self.demand.crop_prod_demand,

            self.crops.seed_demand.groupby('prod_system')
            .sum().stack().rename('seed'),

            concat_herds(self.herds)
            .feed.crop_product_demand.sum()
            .groupby(['prod_system','crop_prod']).sum().rename('feed')
        ], axis=1).rename_axis('use', axis=1)
        use_per_crop_prod = use_per_crop_prod.div(use_per_crop_prod.sum(axis=1), axis=0).fillna(0)
        use_per_crop_prod['none'] = 1 - use_per_crop_prod.sum(axis=1)

        # Get share of crop production that supplies each crop_prod
        crop_prod_per_crop = (
            self.crops.production
            .div(self.crops.production.sum(axis=1), axis=0).fillna(0)
        )

        # Add crop_prod='none' for crops not producing anything
        crop_prod_per_crop['none'] = 1 - crop_prod_per_crop.sum(axis=1)
        # Add missing crop_prods in use_shares with use='none'
        for cp in [cp for cp in crop_prod_per_crop.columns.unique() if cp not in use_per_crop_prod.index.get_level_values('crop_prod')]:
                for ps in use_per_crop_prod.index.get_level_values('prod_system').unique():
                    use_per_crop_prod.loc[(ps,cp),:] = [0]*(len(use_per_crop_prod.columns)-1) + [1]

        # Calculate share of crop area per use
        use_per_crop = crop_prod_per_crop.mul(use_per_crop_prod.unstack()).groupby('use', axis=1).sum()
        assert np.isclose(use_per_crop.sum(axis=1),1).all()

        # Caluclate crop area per use
        area_per_use = use_per_crop.mul(self.crops.area, axis=0)

        # Append po8 regions to index
        region2po8 = self.par.get_rel('region','po8')
        yields = yields.set_index(yields.index.get_level_values('region').map(region2po8).rename('po8'), append=True)
        ley_share = ley_share.set_index(ley_share.index.get_level_values('region').map(region2po8).rename('po8'), append=True)
        area_per_use = area_per_use.set_index(area_per_use.index.get_level_values('region').map(region2po8).rename('po8'), append=True)
        
        # Recommendation [kg TAN/ha]
        TAN_rec = (
            self.par.get_from_frame('N_rec_a',yields) * ((yields/1000) ** 2) +
            self.par.get_from_frame('N_rec_b',yields) * yields/1000 +
            self.par.get_from_frame('N_rec_m',yields)
        )
        # Adjustment for ley [-]
        TAN_ley_adj = (
            self.par.get_from_frame('N_ley_a',ley_share) * ley_share +
            self.par.get_from_frame('N_ley_m',ley_share)
        )

        TAN_req = (TAN_rec * TAN_ley_adj * area_per_use).droplevel('po8')

        self.crops.fertiliser.TAN_req = TAN_req
        self.crops.data_attr.update(['fertiliser.TAN_req'])

    def distribute_manure(self):
        # Generated manure is allocated to different crop areas based
        # on TAN requirements while ensuring that manure P application
        # does not exceed a certain threshold (kg P/ha) defined by
        # parameter 'manure_P_max' (THE LATTER NOT YET IMPLEMENTED).
        #
        # The following rules are used to distribute manure to different
        # crop areas.
        #
        # 1.    Distribute manure deposited by animals in production system X
        #       while grazing to 'grazing crop' Y based on share of grazed
        #       biomass produced by 'grazing crop' Y and used by animal
        #       production system X.
        #
        # 2.	Distribute organic manure deposited in stables to organic areas
        #       in the region based on TAN requirements up to 100% of
        #       crop TAN requirements.
        #
        # 3.	Distribute conventional manure and any organic manure remaining
        #       after (2) to organic areas in the region based on TAN requirements
        #       up to X% of crop TAN requirements, where X is defined by parameter
        #       'manure_TAN_max'.
        #
        # NOTE: This method of distributing manure does not produce reliable results
        # in terms of ammount of manure applied on different crops, but focus on distribution
        # across production systems. E.g. it is likely to assume that manure on cattle
        # farms is primarily applied on the farms' own crop area (i.e. fodder crops)

        herds = concat_herds(self.herds)

        # Create dataframe for results
        manure_TAN_application = pd.DataFrame(
            0,
            columns = herds.manure.N_to_spread.columns,
            index = self.crops.area.index
        )

        # 1. MANURE TO GRAZING AREAS ---------------------------->
        # Get crops used for grazing
        grazing_crops = self.crops.par.get_unique('crop', 'f_crop_prod=="grazing"')
        # Get menure TAN deposited while grazing
        manure_TAN_grazing = herds.manure.TAN_to_spread.xs('grazing', level='MMS', axis=1, drop_level=False)
        # Share of grazed biomass per "grazing crop"
        share_grazed_per_grazing_crop = (
            self.crops.production_per_use
            .filter(regex="feed.*")
            .loc[grazing_crops]
            .groupby(['prod_system','region'])
            .transform(lambda x: x/x.sum())
            .fillna(0)
        )
        # Split columns to multiindex with species, breed, sub_system
        share_grazed_per_grazing_crop.columns = pd.MultiIndex.from_tuples(
            [
                tuple(
                    s.replace('feed ','')
                    .replace('(','')
                    .replace(')','')
                    .split(', ')
                ) for s in 
                share_grazed_per_grazing_crop.columns
            ], 
            names=['species','breed','sub_system']
        )

        for ps in (
                manure_TAN_grazing
                .columns
                .get_level_values('prod_system')
                .unique()
            ):
            # Distribute manure on grazing crops
            manure_TAN_per_grazing_crop = (
                manure_TAN_grazing
                .xs(ps, level='prod_system', axis=1, drop_level=False)
                .multiply(
                    share_grazed_per_grazing_crop
                    .xs(ps, level='prod_system', drop_level=False)
                    .reindex(manure_TAN_grazing.columns, axis=1),
                    axis=0
                )
            )

            # Add to result dataframe
            manure_TAN_application = \
            manure_TAN_application.add(manure_TAN_per_grazing_crop, fill_value=0)

        # Create dataframes to track manure TAN available to spread and TAN
        # requirements that are not yet met.
        manure_TAN = herds.manure.TAN_to_spread.drop('grazing', level='MMS', axis=1)
        TAN_req = self.crops.fertiliser.TAN_req.sum(axis=1)
        manure_TAN_remaining = manure_TAN.copy()
        TAN_req_remaining = TAN_req.copy()

        # 2. ORGANIC MANURE TO ORGANIC AREAS -------------------------->

        # TAN requirements to be covered and manure to be used in this step
        TAN_to_cover = TAN_req_remaining.xs('organic', level='prod_system', drop_level=False)
        manure_TAN_to_use = manure_TAN_remaining.xs('organic', level='prod_system', axis=1, drop_level=False)

        manure_TAN_to_spread = _distribute_manure_TAN(TAN_to_cover, manure_TAN_to_use)

        manure_TAN_remaining, TAN_req_remaining, manure_TAN_application = \
        _update_manure_TAN_frames(
            manure_TAN_to_spread,
            manure_TAN_remaining,
            TAN_req_remaining, manure_TAN_application
        )

        # 3. ALL MANURE TO ORGANIC AREAS UP TO X% TAN --------------------->

        # Calculate TAN requirements to be covered in this step
        self.par.clear()

        TAN_not_to_cover = TAN_req.xs('organic', level='prod_system', drop_level=False)
        TAN_not_to_cover = (
            TAN_not_to_cover *
            (1-self.par.get(
                'manure_TAN_max',
                **TAN_not_to_cover.index.to_frame().to_dict('list')
            )/100)
        )

        TAN_to_cover = (
            TAN_req_remaining.xs('organic', level='prod_system', drop_level=False) -
            TAN_not_to_cover
        )
        TAN_to_cover[TAN_to_cover<0] = 0

        manure_TAN_to_use = manure_TAN_remaining

        manure_TAN_to_spread = _distribute_manure_TAN(TAN_to_cover, manure_TAN_to_use)

        manure_TAN_remaining, TAN_req_remaining, manure_TAN_application = \
        _update_manure_TAN_frames(
            manure_TAN_to_spread,
            manure_TAN_remaining,
            TAN_req_remaining, manure_TAN_application
        )

        # 4. ALL MANURE TO CONVENTIONAL AREAS --------------------->

        # Calculate TAN requirements to be covered in this step
        TAN_to_cover = TAN_req_remaining
        manure_TAN_to_use = manure_TAN_remaining

        manure_TAN_to_spread = _distribute_manure_TAN(TAN_to_cover, manure_TAN_to_use)

        manure_TAN_remaining, TAN_req_remaining, manure_TAN_application = \
        _update_manure_TAN_frames(
            manure_TAN_to_spread,
            manure_TAN_remaining,
            TAN_req_remaining, manure_TAN_application
        )

        # FINALIZE ----------------------------------------------->

        # Calculate share of manure applied per crop
        share_manure_per_crop = (
            manure_TAN_application
            .groupby('region')
            .transform(lambda x: x/x.sum())
            .fillna(0)
        )

        # Apply shares to manure dataframes
        self.crops.fertiliser.manure_TAN = manure_TAN_application # [kg TAN]
        self.crops.fertiliser.manure_N = herds.manure.N_to_spread.multiply(share_manure_per_crop) # [kg N]
        # self.crops.fertiliser.manure_P =  # [kg P]
        # self.crops.fertiliser.manure_K =  # [kg K]
        # self.crops.fertiliser.manure_VS =  # [kg VS]
        
        self.crops.data_attr.update(['fertiliser.manure_TAN','fertiliser.manure_N'])
                
class Fertiliser(Container):
    '''Class to store fertiliser attributes in CropProduction obejcts'''

def _distribute_manure_TAN(TAN_to_cover, manure_TAN_to_use):

    # Calculate share of manure to be spread
    share_manure_to_spread = (
        TAN_to_cover.groupby('region').sum()
        /
        manure_TAN_to_use.sum(axis=1)
    ).fillna(0)
    share_manure_to_spread[share_manure_to_spread>1] = 1
    share_manure_to_spread[share_manure_to_spread<0] = 0

    # Calculate distribution key for distributing manure to crops
    # within each region
    dist_key = (
        TAN_to_cover
        .groupby('region')
        .transform(lambda x:x/x.sum())
    )

    # Calculate manure to spread
    manure_TAN_to_spread = (
        manure_TAN_to_use
        .multiply(share_manure_to_spread, axis=0)
        .multiply(dist_key, axis=0)
    )

    return manure_TAN_to_spread

def _update_manure_TAN_frames(
        manure_TAN_to_spread,
        manure_TAN_remaining,
        TAN_req_remaining,
        manure_TAN_application):
    
    return (
        # Update manure TAN remaining
        manure_TAN_remaining.subtract(
            manure_TAN_to_spread.groupby('region').sum(),
            fill_value = 0
        ),

        # Update TAN requirements remaining
        TAN_req_remaining.subtract(
            manure_TAN_to_spread.sum(axis=1),
            fill_value = 0
        ),

        # Update applied manure
        manure_TAN_application.add(
            manure_TAN_to_spread,
            fill_value = 0
        )
    )