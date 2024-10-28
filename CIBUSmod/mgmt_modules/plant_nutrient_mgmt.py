import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.misc import multiply_aligned, inv_dict, index_to_multi
from ..main_modules.animal_herd import concat_herds

if TYPE_CHECKING:
    from ..main_modules.demand_and_conversions import DemandAndConversions
    from ..main_modules.regions import Regions
    from ..main_modules.crop_prod import CropProduction
    from ..main_modules.waste_and_circularity import WasteAndCircularity
    from ..utils.retriever import ParameterRetriever

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

    def __init__(
            self,
            demand: "DemandAndConversions",
            regions: "Regions",
            crops: "CropProduction",
            waste: "WasteAndCircularity",
            herds: pd.Series,
            par: "ParameterRetriever"
        ):

        self.par = par
        self.demand = demand
        self.regions = regions
        self.crops = crops
        self.waste = waste

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

        vprint('Calculating crop NPK requirements ...')
        self.calculate_TAN_req()
        self.calculate_PK_req(element='P')
        self.calculate_PK_req(element='K')

        vprint('Distributing manure ...')
        self.distribute_manure()

        vprint('Distributing non-manure organic fertilisers ...')
        self.distribute_organic_fertilisers()

        vprint('Calculating mineral NPK application ...')
        self.calculate_mineral_NPK_application(element='N')
        self.calculate_mineral_NPK_application(element='P')
        self.calculate_mineral_NPK_application(element='K')

        self.calculate_manure_application_area()

        vprint('Calculating N in crop residues ...')
        self.calculate_N_in_crop_residues()

        vprint('Calculating N application losses ...') # Only NH3 YES?
        self.calculate_N_application_losses(of='mineral_N')
        self.calculate_N_application_losses(of='manure_TAN')
        self.calculate_N_application_losses(of='organic_TAN')

        vprint('Calculating N soil losses ...')
        self.calculate_N_soil_losses(of='mineral_N')
        self.calculate_N_soil_losses(of='manure_N')
        self.calculate_N_soil_losses(of='organic_N')
        self.calculate_N_soil_losses(of='crop_residues_N')
        self.calculate_organic_soil_N_losses()

        vprint('Calculating N leaching ...')
        self.calculate_leaching_N()

        # TO BE ADDED: Leaching of P and K

        vprint('Calculating lime application ...')
        self.calculate_lime_application()

        vprint('Calculating liming emissions ...')
        self.calculate_liming_emissions()

        vprint(type='end')

    def calculate_TAN_req(self):
        self.par.clear()
        idx=pd.IndexSlice

        # Get crop yields
        yields = (self.crops.data_attr.get('harvest') / self.crops.data_attr.get('area')).fillna(0)

        # Calculate area per use
        share_per_use = (
            self.crops.data_attr.get('production_per_use')
            .div(
                self.crops.data_attr.get('production_per_use').sum(axis=1),
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
        share_per_use = share_per_use.T.groupby('use').sum().T
        share_per_use['none'] = 1 - share_per_use.drop('none', axis=1).sum(axis=1)

        area_per_use = share_per_use.mul(self.crops.data_attr.get('area'), axis=0)

        # Propagate across uses (same yield/ley_share for all uses)
        pdf = pd.DataFrame(
            data=1,
            index=yields.index,
            columns=area_per_use.columns
        )

        yields = pdf.mul(yields, axis=0)

        # Recommendation [kg TAN/ha]
        TAN_rec = (
            self.par.get_from_frame('N_rec_a',yields) * ((yields/1000) ** 2) +
            self.par.get_from_frame('N_rec_b',yields) * yields/1000 +
            self.par.get_from_frame('N_rec_m',yields)
        )

        TAN_req = (TAN_rec * area_per_use)

        # Calculate residual N from crops [kg N]
        self.par.clear()
        residual_N = (
            self.crops.data_attr.get('area') *
            self.par.get('N_resid_crop', **self.crops.index.to_frame().to_dict('list'))
        ).groupby(['prod_system','region']).sum()

        # Allocate residual N from crops to crops in 'crop_groups' based on
        # TAN requirements
        crop_groups = ['Cereals, winter', 'Cereals, spring', 'Ley', 'Fodder crops']
        sel = [c for k,v in inv_dict(self.par.get_rel('crop','crop_group')).items() for c in v if k in crop_groups]
        residual_N_allocation = (
            TAN_req.sum(axis=1)
            .where(TAN_req.index.get_level_values('crop').isin(sel), 0)
            .groupby(['prod_system', 'region'])
            .transform(lambda x: x / x.sum())
        )

        # Calculate adjustment of TAN requirements for residual N
        TAN_req_adj = residual_N * residual_N_allocation

        # Add data attribute
        self.crops.data_attr.add(
            (TAN_req.sum(axis=1) - TAN_req_adj).clip(lower=0),
            name = 'fertiliser.TAN_req',
            unit = 'kg TAN/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Crop plant available nitrogen (TAN) requirements to be covered by fertiliser/manure application'
        )

        return None

    def calculate_PK_req(self, element):

        # Get crop yields in tonnes and P-class
        yields = (
            (self.crops.data_attr.get('harvest') / self.crops.data_attr.get('area') / 1000)
            .fillna(0)
        )

        # Join in P- or K-class and set as index
        yields = (
            yields
            .to_frame()
            .join(self.regions.data_attr.get(f'soil_{element}_class'))
            .set_index(f'{element}_class', append=True)
            .iloc[:,0]
        )

        self.par.clear()
        self.par.set(**yields.index.to_frame().to_dict('list'))
        p = self.par.get

        rec = (
            yields * p(f'{element}_rec_a') +
            p(f'{element}_rec_m')
        )

        # Set values below zero to zero
        rec = rec.where(rec > 0, 0)

        # Apply adjustment factor
        req = rec * p(f'{element}_adj')

        # Multiply by area
        req = req * self.crops.data_attr.get('area')

        # Drop P/K-class from index
        req = req.droplevel(f'{element}_class')

        # Add data attribute
        self.crops.data_attr.add(
            req,
            name = f'fertiliser.{element}_req',
            unit = f'kg {element}/year',
            orig = 'PlantNutrientMgmt',
            desc = f'{_elem_to_name[element].capitalize()} requirements to be covered by fertiliser/manure application'
        )

        return None

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
        # 4.    Distribute any manure remaining after (3) per animal herd to crop
        #       areas used for feed in the region for that herd based on TAN requirements.
        #
        # 5.    Distribute any remaining manure after (3) to conventional areas
        #       in the region based on TAN requirements.

        herds = concat_herds(self.herds)

        # Create dataframe for results
        manure_TAN_application = pd.DataFrame(
            0.0,
            columns = herds.data_attr.get('manure.N_to_spread').columns,
            index = self.crops.data_attr.get('area').index
        )

        # 1. MANURE TO GRAZING AREAS ---------------------------->
        # Get crops used for grazing
        grazing_crops = self.crops.par.get_unique('crop', 'f_crop_prod=="grazing"')
        # Get menure TAN deposited while grazing
        manure_TAN_grazing = herds.data_attr.get('manure.TAN_to_spread').xs('grazing', level='MMS', axis=1, drop_level=False)
        # Share of grazed biomass per "grazing crop"
        share_grazed_per_grazing_crop = (
            self.crops.data_attr.get('production_per_use')
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
        manure_TAN = herds.data_attr.get('manure.TAN_to_spread').drop('grazing', level='MMS', axis=1)
        TAN_req = self.crops.data_attr.get('fertiliser.TAN_req')
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

        # 4. MANURE PER HERD TO CROP AREAS USED AS FEED --------------------->

        share_crop_prod_per_use = (
            self.crops.data_attr.get('production_per_use')
            .div(self.crops.data_attr.get('production_per_use').sum(axis=1), axis=0)
            # NaNs generated in cases where production is zero
            # Not a problem here so just replace with 0
            .fillna(0)
        )

        # Calculate TAN requirements to be covered in this step
        for sp,br,ss,ps in manure_TAN_remaining.columns.droplevel(['animal','MMS']).unique():
            TAN_to_cover = TAN_req_remaining.mul(
                share_crop_prod_per_use
                .loc[(slice(None),ps,slice(None)),f'feed ({sp}, {br}, {ss})'],
                fill_value=0
            )
            manure_TAN_to_use = manure_TAN_remaining.loc[:,([sp],[br],[ss],[ps])]

            manure_TAN_to_spread = _distribute_manure_TAN(TAN_to_cover, manure_TAN_to_use)

            manure_TAN_remaining, TAN_req_remaining, manure_TAN_application = \
            _update_manure_TAN_frames(
                manure_TAN_to_spread,
                manure_TAN_remaining,
                TAN_req_remaining, manure_TAN_application
            )

        # 5. ALL MANURE TO CONVENTIONAL AREAS --------------------->

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
            .div(manure_TAN_application.groupby('region').sum().replace({0:np.nan}), axis=0)  # 0 -> NaN to avoid div by 0 error
            .fillna(0)
        )

        # Apply shares to manure dataframes and add data attributes
        res = (
            manure_TAN_application
            .rename_axis(columns={'prod_system':'animal_prod_system'}) # Rename to avoid 'prod_system' in both index and column levels
        )
        self.crops.data_attr.add(
            res,
            name = 'fertiliser.manure_TAN',
            unit = 'kg TAN/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Plant available nitrogen (TAN) in applied manure'
        )

        for element in ['N', 'P', 'K', 'C']:
            res = (
                herds.data_attr.get(f'manure.{element}_to_spread')
                .multiply(share_manure_per_crop)
                .rename_axis(columns={'prod_system':'animal_prod_system'}) # Rename to avoid 'prod_system' in both index and column levels
            )
            self.crops.data_attr.add(
                res,
                name = f'fertiliser.manure_{element}',
                unit = f'kg {element}/year',
                orig = 'PlantNutrientMgmt',
                desc = f'{_elem_to_name[element].capitalize()} in applied manure'
            )

    def distribute_organic_fertilisers(self):
        # Non-manure organic fertilisers originating from the WasteAndCircularity module
        # are destributed according to the following procedure:
        #
        # 1.    Organic fertiliser generated in a given region is distributed to organic crops
        #       in that region based on TAN requirements minus TAN in applied manure.
        #
        # 2.    Any remaining organic fertiliser in a region is distributed to conventional crops
        #       in that region based on TAN requirements minus TAN in applied manure.
        #
        # 3.    Organic fertiliser remaining after 1 and 2 is distributed based on remaining TAN
        #       requirements in organic crops nationally.
        #
        # 4.    Organic fertiliser remaining after 3 is distributed based on remaining TAN requirements
        #       in conventional crops nationally.

        # Get TAN requirements remaining after manure application
        # Set to zero if manure covers more than requirements
        TAN_req = (
            self.crops.data_attr.get('fertiliser.TAN_req') -
            self.crops.data_attr.get('fertiliser.manure_TAN').sum(axis=1)
        ).clip(lower=0)

        # Get TAN available in organic fertilisers
        # Note: These are aggregated by treatment to save computation time.
        # The groupby could potentially be removed to preserve the origin
        # of feedstocks
        org_TAN = (
            self.waste.data_attr.get('organic_fertiliser_TAN')
            .T.groupby('treatment').sum().T
        )
        # org_TAN.iloc[:,1] = org_TAN.iloc[:,0]

        # Create dataframe for results
        org_TAN_application = pd.DataFrame(
            0.0,
            columns = org_TAN.columns,
            index = TAN_req.index
        )

        # Create dataframes to track remaining TAN requirements and organic fertiliser
        TAN_req_remaining = TAN_req.copy()
        org_TAN_remaining = org_TAN.copy()

        # 1. ORGANIC FERTILISERS TO ORGANIC AREAS IN REGION -------------------------->
        TAN_to_cover = TAN_req_remaining.xs('organic', level='prod_system', drop_level=False)

        org_TAN_to_spread = _distribute_manure_TAN(TAN_to_cover, org_TAN_remaining)

        org_TAN_remaining, TAN_req_remaining, org_TAN_application = \
        _update_manure_TAN_frames(
            org_TAN_to_spread,
            org_TAN_remaining,
            TAN_req_remaining, org_TAN_application
        )

        # 2. ORGANIC FERTILISERS TO CONVENTIONAL AREAS IN REGION --------------------->
        TAN_to_cover = TAN_req_remaining

        org_TAN_to_spread = _distribute_manure_TAN(TAN_to_cover, org_TAN_remaining)

        org_TAN_remaining, TAN_req_remaining, org_TAN_application = \
        _update_manure_TAN_frames(
            org_TAN_to_spread,
            org_TAN_remaining,
            TAN_req_remaining, org_TAN_application
        )

        # 3. ORGANIC FERTILISERS TO ORGANIC AREAS ACROSS REGIONS --------------------->
        # AND
        # 4. ORGANIC FERTILISERS TO CONVENTIONAL AREAS ACROSS REGIONS ---------------->
        for do_step in [3,4]:

            if do_step == 3:
                TAN_to_cover = TAN_req_remaining.xs('organic', level='prod_system', drop_level=False)
            else:
                TAN_to_cover = TAN_req_remaining

            share_to_use = 1 if TAN_to_cover.sum() > org_TAN_remaining.sum().sum() else TAN_to_cover.sum()/org_TAN_remaining.sum().sum()

            dist_key = (
                TAN_to_cover /
                TAN_to_cover.sum()
            )

            org_TAN_to_spread = pd.DataFrame(
                np.atleast_2d(dist_key.values).T @ np.atleast_2d(org_TAN_remaining.sum().values * share_to_use),
                index = dist_key.index,
                columns = org_TAN_remaining.columns
            )

            org_TAN_remaining, TAN_req_remaining, org_TAN_application = \
            _update_manure_TAN_frames(
                org_TAN_to_spread,
                org_TAN_remaining,
                TAN_req_remaining, org_TAN_application
            )

        # FINALIZE -------------------------------------------------------------------->

        # Calculate share of organic fertiliser applied per crop
        share_per_crop = (
            org_TAN_application
            .div(org_TAN_application.groupby('region').sum().replace({0:np.nan}), axis=0)  # 0 -> NaN to avoid div by 0 error
            .fillna(0)
        )

        self.crops.data_attr.add(
            org_TAN_application,
            name = 'fertiliser.organic_TAN',
            unit = 'kg TAN/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Plant available nitrogen (TAN) in applied non-manure organic fertilisers'
        )

        for element in ['N', 'P', 'K', 'C']:
            res = (
                self.waste.data_attr.get(f'organic_fertiliser_{element}')
                .T.groupby('treatment').sum().T
                .multiply(share_per_crop)
            )
            self.crops.data_attr.add(
                res,
                name = f'fertiliser.organic_{element}',
                unit = f'kg {element}/year',
                orig = 'PlantNutrientMgmt',
                desc = f'{_elem_to_name[element].capitalize()} in applied non-manure organic fertilisers'
            )

    def calculate_mineral_NPK_application(self, element):
        # Mineral N, P or K application is assumed to cover additional
        # TAN, P or K requirements after manure and other organic
        # fertilisers have been applied.

        self.par.clear()
        element_ = 'TAN' if element == 'N' else element

        # Get share of fertiliser types
        fertiliser_type_shares = \
        self.par.get_from_frame(
            f'mineral_{element}_fertiliser_share',
            pd.DataFrame(
                index=self.crops.index,
                columns=pd.Index(
                    self.par.get_unique(
                        'fertiliser_type',
                        qry=f'parameter == "mineral_{element}_fertiliser_share"'
                    ), name='fertiliser_type'
                )
            )
        )/100

        # Get TAN/P/K requirements
        req = self.crops.data_attr.get(f'fertiliser.{element_}_req')

        # Get manure TAN/P/K input
        manure = self.crops.data_attr.get(f'fertiliser.manure_{element_}').drop('grazing', level='MMS', axis=1).sum(axis=1)

        # Get non-manure organic fertiliser TAN/P/K input
        org_fert = self.crops.data_attr.get(f'fertiliser.organic_{element_}').sum(axis=1)

        # Get long term N from manure and other organic fertilisers
        if element == 'N':
            self.par.clear()
            manure_long = (
                self.crops.data_attr.get('fertiliser.manure_N') -
                self.crops.data_attr.get('fertiliser.manure_TAN')
            ).mul(
                self.par.get('N_resid_manure', **self.crops.data_attr.get('fertiliser.manure_N').columns.to_frame().to_dict('list'))/100,
                axis = 1
            ).sum(axis=1)

            self.par.clear()
            org_fert_long = (
                self.crops.data_attr.get('fertiliser.organic_N') -
                self.crops.data_attr.get('fertiliser.organic_TAN')
            ).mul(
                self.par.get('N_resid_organic', **self.crops.data_attr.get(f'fertiliser.organic_N').columns.to_frame().to_dict('list'))/100,
                axis = 1
            ).sum(axis=1)
        else:
            manure_long = 0
            org_fert_long = 0

        # Calculate TAN, P or K to apply
        mineral_fertiliser_to_apply = (
            req - manure - manure_long - org_fert - org_fert_long
        ).clip(lower=0) # set to zero if manure/organic fertilisers supplies more than requirement

        # Calculate mineral fertiliser application per fertiliser type
        mineral_fertiliser_application = \
        fertiliser_type_shares.mul(
            mineral_fertiliser_to_apply,
            axis=0
        ).fillna(0)

        # Add data attribute
        self.crops.data_attr.add(
            mineral_fertiliser_application,
            name = f'fertiliser.mineral_{element}',
            unit = f'kg {element}/year',
            orig = 'PlantNutrientMgmt',
            desc = f'{_elem_to_name[element].capitalize()} in applied fertilisers'
        )

    def calculate_manure_application_area(self):

        self.par.clear()

        manure_N = self.crops.data_attr.get('fertiliser.manure_N').drop('grazing', level='MMS', axis=1).sum(axis=1)
        organic_N = self.crops.data_attr.get('fertiliser.organic_N').sum(axis=1)
        mineral_N = self.crops.data_attr.get('fertiliser.mineral_N').sum(axis=1)
        total_N = (manure_N + organic_N + mineral_N).replace({0:np.nan})

        # Get minimum share of total N from manure on area where manure is applied
        min_share_manure_N_where_applied = pd.Series(
            self.par.get('min_share_manure_N_where_applied', **total_N.index.to_frame().to_dict('list'))/100,
            index = total_N.index
        )

        # Calculate the share of area where manure is applied to reach at
        # least the minimum share of manure N in total N applications.
        # Note: Looks like this may overestimate the area with manure applications
        # as no completely unfertilised areas are accounted for in the model
        share_area_with_manure_application = (manure_N/(total_N*min_share_manure_N_where_applied)).clip(upper=1).fillna(0)

        # Calculate the total area with manure application
        area_with_manure_application = share_area_with_manure_application * self.crops.data_attr.get('area')

        # Add data attribute
        self.crops.data_attr.add(
            area_with_manure_application,
            name = 'fertiliser.manure_application_area',
            unit = 'ha/m2',
            orig = 'PlantNutrientMgmt',
            desc = 'Area where manure is applied. Used in calculating energy requirements for manure spreading'
        )

        return None

    def calculate_N_in_crop_residues(self):
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

        # Get N in crop residue DM and multiply by total crop residues left in field
        crop_residues_N = (
            pd.DataFrame(
                np.array([
                    p('ag_resid_N'),
                    p('bg_resid_N'),
                ]).T,
                index = self.crops.index,
                columns = pd.Index(['above ground','below ground'], name='residue')
            )
            .mul(crop_residues)
        )

        # Add data attribute
        self.crops.data_attr.add(
            crop_residues_N,
            name = 'fertiliser.crop_residues_N',
            unit = 'kg N/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Nitrogen (N) in above and below ground crop residues left in the field (i.e. not harvested)'
        )

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
        TAN_appl = self.crops.data_attr.get(f'fertiliser.{of}')

        if of=='manure_TAN':
            # Aggregate manure
            TAN_appl = (
                TAN_appl
                .T.groupby(['species','breed','animal','MMS']).sum().T
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
        attr_name = 'fertiliser.' + of.replace('TA','') + '_application_loss'
        self.crops.data_attr.add(
            N_loss,
            name = attr_name,
            unit = 'kg N/year',
            orig = 'PlantNutrientMgmt',
            desc = f'Losses of nitrogen (N) from {"mineral fertiliser" if "mineral" in of else "manure"} application'
        )

    def calculate_N_soil_losses(self, of):

        self.par.clear()

        # Get soil loss parameter name
        par_name = 'soil_losses_' + of.replace('_N','')

        # Get N application
        N_appl = self.crops.data_attr.get(f'fertiliser.{of}')

        if of=='manure_N':
            # Aggregate manure
            N_appl = (
                N_appl
                .T.groupby(['species','breed','animal','MMS']).sum().T
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
                1.0,
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
        attr_name = 'fertiliser.' + of + '_soil_loss'
        self.crops.data_attr.add(
            N_loss,
            name = attr_name,
            unit = 'kg N/year',
            orig = 'PlantNutrientMgmt',
            desc = f'Soil losses of nitrogen (N) from {"applied mineral fertilisers" if "mineral" in of else "applied and deposited manure" if "manure" in of else "crop residues"}'
        )

    def calculate_organic_soil_N_losses(self):

        self.par.clear()
        self.regions.par.clear()

        # Get crop areas and append land_use
        areas = self.crops.data_attr.get('area').rename('area').to_frame()
        rel = self.par.get_rel('crop','land_use')
        areas['land_use'] = [rel[c] for c in areas.index.get_level_values('crop')]
        areas = areas.set_index('land_use', append=True)['area']

        # Get compounds lost
        compounds = \
        self.par.get_unique(
            'compound',
            qry='parameter == "soil_losses_organic_soils"'
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

        self.crops.data_attr.add(
            organic_soil_N_loss,
            name = 'fertiliser.organic_soil_N_loss',
            unit = 'kg N/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Soil losses of nitrogen (N) from organic soils'
        )

    def calculate_leaching_N(self):

        # Get compounds leached
        compounds = self.par.get_unique('compound', qry = 'parameter == "N_leaching_frac"')

        # Get N additions from fertiliser, manure and crop residues
        N_add = pd.concat([
            self.crops.data_attr.get('fertiliser.mineral_N')
            .sum(axis=1).rename('fertiliser'),
            self.crops.data_attr.get('fertiliser.manure_N')
            .sum(axis=1).rename('manure'),
            self.crops.data_attr.get('fertiliser.organic_N')
            .sum(axis=1).rename('organic'),
            self.crops.data_attr.get('fertiliser.crop_residues_N')
            .sum(axis=1).rename('crop residues')
        ], axis=1).rename_axis(columns = 'source')

        # Create output dataframe
        df = pd.DataFrame(
            index = N_add.index,
            columns = pd.MultiIndex.from_tuples(
                [(s,c) for s in N_add.columns for c in compounds],
                names = ['source','compound']
            )
        )

        # Add land use class as index
        lu_rel = self.par.get_rel('crop','land_use')
        df['land_use'] = [lu_rel[c] for c in df.index.get_level_values('crop')]
        df = df.set_index('land_use', append=True)

        # Apply leacing factors
        leaching_N = multiply_aligned(
            self.par.get_from_frame('N_leaching_frac', df),
            N_add
        )

        # Drop 'land_use' from index
        leaching_N = leaching_N.droplevel('land_use')

        # Add data attribute
        self.crops.data_attr.add(
            leaching_N,
            name = 'fertiliser.leaching_N',
            unit = 'kg N/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Losses of nitrogen (N) via leaching'
        )

    def calculate_lime_application(self):
        """"Method to calculate lime application based on the liming effect of
        removed harvest, removed crop residues and applied fertilisers and manure"""

        # Calculate lime effect of removed harvest
        self.par.clear()
        crops_CaO = self.crops.data_attr.get('harvest_dm')
        crops_CaO = crops_CaO.mul(
            self.par.get(
                'lime_effect_crops',
                **crops_CaO.index.to_frame().to_dict('list')
            )
        )
        # Set lime effect on non-cropland to zero
        crops_CaO.loc[
            crops_CaO.rename(
                self.par.get_rel('crop','land_use'),
                level='crop'
            )
            .index.get_level_values('crop') != 'cropland'
        ] = 0

        # Calculate lime effect of removed crop residues
        self.par.clear()
        crop_resid_CaO = self.crops.data_attr.get('crop_residues_harvest')
        crop_resid_CaO = crop_resid_CaO.mul(
            self.par.get_from_frame(
                'lime_effect_crops',
                crop_resid_CaO
            )
        )

        # Calculate lime effect of manure application
        self.par.clear()
        manure_CaO = self.crops.data_attr.get('fertiliser.manure_N')
        manure_CaO = manure_CaO.mul(
            self.par.get(
                'lime_effect_manure',
                **manure_CaO.columns.to_frame().to_dict('list')
            ), axis = 1
        )

        # Calculate lime effect of N fertiliser application
        self.par.clear()
        fert_N_CaO = self.crops.data_attr.get('fertiliser.mineral_N')
        fert_N_CaO = fert_N_CaO.mul(
            self.par.get(
                'lime_effect_fertiliser_N',
                **fert_N_CaO.columns.to_frame().to_dict('list')
            ), axis = 1
        )

        # Calculate lime effect of P fertiliser application
        self.par.clear()
        fert_P_CaO = self.crops.data_attr.get('fertiliser.mineral_P')
        fert_P_CaO = fert_P_CaO.mul(
            self.par.get(
                'lime_effect_fertiliser_P',
                **fert_P_CaO.columns.to_frame().to_dict('list')
            ), axis = 1
        )

        # Calculate lime effect of K fertiliser application
        self.par.clear()
        fert_K_CaO = self.crops.data_attr.get('fertiliser.mineral_K')
        fert_K_CaO = fert_K_CaO.mul(
            self.par.get(
                'lime_effect_fertiliser_K',
                **fert_K_CaO.columns.to_frame().to_dict('list')
            ), axis = 1
        )

        # Calculate total lime effect
        total_CaO = (
            crops_CaO +
            crop_resid_CaO.sum(axis=1) +
            fert_N_CaO.sum(axis=1) +
            fert_P_CaO.sum(axis=1) +
            fert_K_CaO.sum(axis=1) +
            manure_CaO.sum(axis=1)
        )

        # Create DF for lime application
        lime_application = pd.DataFrame(
            1.0,
            index = total_CaO.index,
            columns = pd.Index(self.par.get_unique('fertiliser_type', qry='parameter == "liming_agent_share"'), name = 'fertiliser_type')
        )

        # Calculate applied lime per liming agend
        lime_application = (
            lime_application *
            # Factor in liming agent shares
            (self.par.get_from_frame(
                'liming_agent_share',
                lime_application
            )/100) *
            # Factor in neutralising value of different liming agent (CaO equivalents)
            (1/(self.par.get_from_frame(
                'liming_agent_CaO_value',
                lime_application
            )/100))
        ).mul(
            # Multiply by total liming requirements
            -total_CaO, axis = 0
        )

        # Remove negative lime applications while
        # maintaining sum of application
        total_appl = lime_application.sum()
        lime_application = lime_application.where(lime_application>0,0)
        lime_application = lime_application.div(lime_application.sum(), axis=1).mul(total_appl, axis=1)

        # Add data attribute
        self.crops.data_attr.add(
            lime_application,
            name = 'fertiliser.liming',
            unit = 'kg/year',
            orig = 'PlantNutrientMgmt',
            desc = 'Applied lime'
        )

    def calculate_liming_emissions(self):
        self.par.clear()

        # Get lime application
        lime_application = self.crops.data_attr.get('fertiliser.liming')

        # Get compounds
        cmps = self.par.get_unique('compound', qry='parameter == "liming_emissions"')

        # Create result DF with 'compound' column level
        liming_emissions = pd.DataFrame(
            1.0,
            index = lime_application.index,
            columns = pd.MultiIndex.from_tuples(
                [col+(cmp,) for col in index_to_multi(lime_application.columns) for cmp in cmps],
                names = lime_application.columns.names + ['compound']
            )
        )

        # Calculate emissions from lime
        liming_emissions = multiply_aligned(liming_emissions, lime_application).mul(
            self.par.get(
                'liming_emissions',
                **liming_emissions.columns.to_frame().to_dict('list')
            )
        )

        # Add data attribute
        self.crops.data_attr.add(
            liming_emissions,
            name = 'fertiliser.liming_emissions',
            unit = 'kg/year',
            orig = 'PlantNutrientMgmt',
            desc = 'CO2 emissions from lime application'
        )

_elem_to_name = {
    'C' : 'carbon (C)',
    'N' : 'nitrogen (N)',
    'P' : 'phosphorous (P)',
    'K' : 'potassium (K)'
}

def _distribute_manure_TAN(TAN_to_cover, manure_TAN_to_use):

    # Calculate share of manure to be spread
    share_manure_to_spread = (
        TAN_to_cover.groupby('region').sum()
        /
        manure_TAN_to_use.sum(axis=1).replace({0:np.nan}) # 0 -> NaN to avoid div by 0 error
    ).fillna(0)
    share_manure_to_spread[share_manure_to_spread>1] = 1
    share_manure_to_spread[share_manure_to_spread<0] = 0

    # Calculate distribution key for distributing manure to crops
    # within each region
    dist_key = (
        TAN_to_cover
        .div(TAN_to_cover.groupby('region').sum().replace({0:np.nan}), axis=0) # 0 -> NaN to avoid div by 0 error
        .fillna(0)
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
