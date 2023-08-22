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
        self.calculate_N_req()

        vprint('Distributing manure ...')
        self.distribute_manure()

        vprint(type='end')
             
    def calculate_N_req(self):
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
        
        # Recommendation [kg N/ha]
        N_rec = (
            self.par.get_from_frame('N_rec_a',yields) * ((yields/1000) ** 2) +
            self.par.get_from_frame('N_rec_b',yields) * yields/1000 +
            self.par.get_from_frame('N_rec_m',yields)
        )
        # Adjustment for ley [-]
        N_ley_adj = (
            self.par.get_from_frame('N_ley_a',ley_share) * ley_share +
            self.par.get_from_frame('N_ley_m',ley_share)
        )

        N_req = (N_rec * N_ley_adj * area_per_use).droplevel('po8')

        self.crops.fertiliser.N_rec = N_rec
        self.crops.fertiliser.N_ley_adj = N_ley_adj
        self.crops.fertiliser.N_req = N_req
        self.crops.data_attr.update(['fertiliser.N_req'])

    def distribute_manure(self):
        pass
        
        # 1) Manure per production system to crops grown for feed
        #    in that production system.

        # 2) Surpluss manure to organic crops distributed according
        #    to N requirements
                
class Fertiliser(Container):
    '''Class to store fertiliser attributes in CropProduction obejcts'''