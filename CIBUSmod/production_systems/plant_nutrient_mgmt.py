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

    def __init__(self,demand,regions,crops,herds,par):

        self.par = par
        self.demand = demand
        self.regions = regions
        self.crops = crops

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

        # TO BE ADDED: Sewedge sludge application (and other recirculated?)

        vprint('Calculating mineral N application ...')
        self.calculate_mineral_N_application()

        vprint('Calculating N in crop residues ...')
        self.calculate_N_in_crop_residues()

        vprint('Calculating N application losses ...') # Only NH3 YES?
        self.calculate_N_application_losses(of='mineral_N')
        self.calculate_N_application_losses(of='manure_TAN')

        vprint('Calculating N soil losses ...')
        self.calculate_N_soil_losses(of='mineral_N')
        self.calculate_N_soil_losses(of='manure_N')
        self.calculate_N_soil_losses(of='crop_residues_N')
        self.calculate_organic_soil_N_losses()

        # TO BE ADDED: Soil N2O emissions
        # TO BE ADDED: Other gasous emissions
        # TO BE ADDED: Leaching

        vprint(type='end')
             
    def calculate_TAN_req(self):
        idx=pd.IndexSlice

        # Get crop yields
        yields = (self.crops.harvest / self.crops.area).fillna(0)

        # Calculate ley share per region
        lu_rel = self.par.get_rel('crop','land_use') # crop --> land_use
        cg_rel = self.par.get_rel('crop','crop_group') # crop --> crop_group

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

        # Calculate area per use
        share_per_use = (
            self.crops.production_per_use
            .div(
                self.crops.production_per_use.sum(axis=1),
                axis=0
            )
            .rename_axis('use', axis=1)
            .fillna(0)
        )
        share_per_use.columns = (
            share_per_use.columns
            .str.replace('feed.*','feed', regex=True)
            .str.replace('export','food') # Assume food use for exported crops
        )
        share_per_use = share_per_use.groupby('use', axis=1).sum()
        share_per_use['none'] = 1 - share_per_use.drop('none', axis=1).sum(axis=1)

        area_per_use = share_per_use.mul(self.crops.area, axis=0)

        # Propagate across uses (same yield/ley_share for all uses)
        pdf = pd.DataFrame(
            data=1,
            index=yields.index,
            columns=area_per_use.columns
        )

        yields = pdf.mul(yields, axis=0)
        ley_share = pdf.mul(ley_share, axis=0)

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

        self.crops.fertiliser.TAN_req = TAN_req.sum(axis=1)
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
        TAN_req = self.crops.fertiliser.TAN_req
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
                
    def calculate_mineral_N_application(self):
        # Mineral N application is assumed to cover additional
        # TAN requirements after manure and other organic
        # fertilisers have been applied. 

        self.par.clear()

        # Get share of fertiliser types
        fertiliser_type_shares = \
        self.par.get_from_frame(
            'mineral_N_fertiliser_share',
            pd.DataFrame(
                index=self.crops.index,
                columns=pd.Index(
                    self.par.get_unique(
                        'fertiliser_type',
                        qry='parameter == "mineral_N_fertiliser_share"'
                    ), name='fertiliser_type'
                )
            )
        )/100

        # Calculate TAN to apply 
        TAN_to_apply = (
            self.crops.fertiliser.TAN_req - 
            # self.crops.fertiliser.organic_TAN.sum(axis=1) - !!! TO BE ADDED !!!
            self.crops.fertiliser.manure_TAN.sum(axis=1)
        ).clip(lower=0) # set to zero if manure supplies more than requirement

        # Calculate mineral N fertiliser application
        self.crops.fertiliser.mineral_N = \
        fertiliser_type_shares.mul(
            TAN_to_apply,
            axis=0
        ).fillna(0)
        self.crops.data_attr.update(['fertiliser.mineral_N'])

    def calculate_N_in_crop_residues(self):
        # Calculate N in crop residues
        self.crops.par.clear()
        self.crops.par.set(**self.crops.index.to_frame().to_dict('list'))
        p = self.crops.par.get

        self.crops.fertiliser.crop_residues_N = (
            pd.DataFrame(
                np.array([
                    p('ag_resid_N'),
                    p('bg_resid_N'),
                ]).T,
                index = self.crops.index,
                columns = pd.Index(['above ground','below ground'], name='residue')
            )
            .mul(self.crops.crop_residues)
        )
        self.crops.data_attr.update(['fertiliser.crop_residues_N'])

    def calculate_N_application_losses(self, of):
        # Application losses of NH3-N calculated according to
        # Tier 2 method described in 'Informative Inventory
        # Report Sweden 2023 Submitted under the Convention
        # on Long-Range Transboundary Air Pollution (UNECE CLRTAP)'
        #
        # NOTE: Potentially country specific method should be
        # implemented to account for manure spreading technology
        # and timing.

        self.par.clear()

        # Get TAN application
        TAN_appl = getattr(self.crops.fertiliser, of)

        if of=='manure_TAN':
            # Aggregate manure 
            TAN_appl = (
                TAN_appl
                .groupby(['species','breed','animal','MMS'], axis=1)
                .sum()
            )

        # Get compounds lost
        compounds = \
        self.par.get_unique(
            'compound',
            qry='parameter == "application_losses"'
        )

        if not isinstance(TAN_appl.columns,pd.MultiIndex):
            # Make multiindex to fix problem with
            # pandas reindex from single index
            TAN_appl.columns = pd.MultiIndex.from_tuples([
                (cols,) for cols in TAN_appl.columns
            ], names=TAN_appl.columns.names)

        # Add compounds to columns
        TAN_appl = TAN_appl.reindex(
            pd.MultiIndex.from_tuples([
                cols + tuple([cp])
                for cols in TAN_appl.columns
                for cp in compounds
            ], names=TAN_appl.columns.names+['compound']),
            axis=1
        )

        # Get application losses (% of TAN) per fertiliser or animal/MMS
        application_losses = pd.Series(
            self.par.get(
                'application_losses',
                **TAN_appl.columns.to_frame().to_dict('list')
            )/100,
            index = TAN_appl.columns
        )

        # Calculate N loss
        N_loss = TAN_appl.mul(
            application_losses,
            axis=1
        )

        # Store resulting N application losses [kg N]
        attr_name = of.replace('TA','') + '_application_loss'
        setattr(
            self.crops.fertiliser,
            attr_name,
            N_loss
        )
        self.crops.data_attr.update(['fertiliser.'+attr_name])

    def calculate_N_soil_losses(self, of):
        
        self.par.clear()

        # Get soil loss parameter name
        par_name = 'soil_losses_' + of.replace('_N','')

        # Get N application
        N_appl = getattr(self.crops.fertiliser, of)

        if of=='manure_N':
            # Aggregate manure 
            N_appl = (
                N_appl
                .groupby(['species','breed','animal','MMS'], axis=1)
                .sum()
            )

        # Get compounds lost
        compounds = \
        self.par.get_unique(
            'compound',
            qry=f'parameter == "{par_name}"'
        )

        if not isinstance(N_appl.columns,pd.MultiIndex):
            # Make multiindex to fix problem with
            # pandas reindex from single index
            N_appl.columns = pd.MultiIndex.from_tuples([
                (cols,) for cols in N_appl.columns
            ], names=N_appl.columns.names)

        # Add compounds to columns
        N_appl = N_appl.reindex(
            pd.MultiIndex.from_tuples([
                cols + tuple([cp])
                for cols in N_appl.columns
                for cp in compounds
            ], names=N_appl.columns.names+['compound']),
            axis=1
        )

        # Get emission factors [% of N]
        EF = (
            pd.DataFrame(
                1,
                columns = N_appl.columns,
                index = N_appl.index
            )
            .mul(
                self.par.get(par_name, **N_appl.columns.to_frame().to_dict('list'))/100,
                axis=1
            )
        )

        # IMPLEMENT REGIONALISED EFs HERE!!!

        # Apply emission factors
        N_loss = N_appl * EF

        # Store resulting N soil losses [kg N]
        attr_name = of + '_soil_loss'
        setattr(
            self.crops.fertiliser,
            attr_name,
            N_loss
        )
        self.crops.data_attr.update(['fertiliser.'+attr_name])

    def calculate_organic_soil_N_losses(self):
        
        self.par.clear()
        self.regions.par.clear()

        # Get crop areas and append land_use
        areas = self.crops.area.rename('area').to_frame()
        rel = self.par.get_rel('crop','land_use')
        areas['land_use'] = [rel[c] for c in areas.index.get_level_values('crop')]
        areas = areas.set_index('land_use', append=True)['area']

        # Get compounds lost
        compounds = \
        self.par.get_unique(
            'compound',
            qry=f'parameter == "soil_losses_organic_soils"'
        )

        # Construct dataframe
        organic_soil_N_loss = pd.DataFrame(
            index = areas.index,
            columns = pd.Index(compounds, name='compound')
        )

        # Calculate emissions
        organic_soil_N_loss = (
            self.regions.par.get_from_frame('share_org_soil', organic_soil_N_loss)/100 *
            self.par.get_from_frame('soil_losses_organic_soils', organic_soil_N_loss)
        ).mul(areas, axis=0).droplevel('land_use')

        self.crops.fertiliser.organic_soil_N_loss = organic_soil_N_loss
        self.crops.data_attr.update(['fertiliser.organic_soil_N_loss'])


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