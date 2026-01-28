import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

from ..utils.verbose_print import verbose_init
from ..utils.misc import fix_herds

if TYPE_CHECKING:
    from ..main_modules.demand_and_conversions import DemandAndConversions
    from ..main_modules.regions import Regions
    from ..main_modules.crop_prod import CropProduction
    from ..main_modules.waste_and_circularity import WasteAndCircularity
    from ..utils.retriever import ParameterRetriever

class MachineryAndEnergyMgmt(object):
    '''Class that handles calculation of energy requirements
    in machinery, stables and greenhouses.

    Parameters
    ----------
    regions : Regions object
    crops : CropProduction object
    herds : pandas.Series of AnimalHerd object(s)
    par : ParameterRetriever object
    '''

    def __init__(
            self,
            regions: "Regions",
            crops: "CropProduction",
            waste: "WasteAndCircularity",
            herds: pd.Series,
            par: "ParameterRetriever"
        ):

        self.par = par
        self.regions = regions
        self.crops = crops
        self.waste = waste
        self.herds = fix_herds(herds)

    def calculate(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='MachineryAndEnergyMgmt')

        # Create dataframe to store energy use in crop production [kWh]
        energy_use = pd.DataFrame(
            0.0,
            index = self.crops.index,
            columns = pd.MultiIndex.from_product([
                ['field machinery', 'grain dryers', 'grain dryers auxiliary energy', 'greenhouses'],
                self.par.get_unique('energy_source')
            ], names=['activity','energy_source'])
        )
        # Add data attribute
        self.crops.data_attr.add(
            energy_use,
            name = 'energy_use',
            unit = 'kWh/year',
            orig = 'MachineryAndEnergyMgmt',
            desc = 'Energy use for field machinery, grain dryers and greenhouses'
        )

        vprint('Calculating energy use in field machinery ...')
        self.calculate_field_machinery()

        vprint('Calculating energy use for grain drying ...')
        self.calculate_drying()

        vprint('Calculating energy use in greenhouses ...')
        self.calculate_greenhouses()

        vprint('Calculating energy use in stables ...')
        self.calculate_stables()

        vprint('Calculating fuel use emissions ...')
        self.calculate_combustion_emissions()

        vprint(type='end')

    def calculate_field_machinery(self):
        p = self.par.get
        pf = self.par.get_from_frame
        idx = pd.IndexSlice

        field_operations = pd.Index(
            self.par.get_unique('operation'),
            name = 'operation'
        )
        soil_classes = pd.Index(
            self.par.get_unique('soil_class'),
            name = 'soil_class'
        )
        forage_systems = pd.Index(
            self.par.get_unique('forage_system'),
            name = 'forage_system'
        )

        # Get field operatiobns matrix specifying nr of times
        # each operation is performed
        self.par.clear()
        A = pf(
            'field_operations',
            pd.DataFrame(index=self.crops.index, columns=field_operations)
        )

        # Calculate tractor energy requirements
        self.par.clear()

        F_air = ( # Force air friction [N]
            p('density_air') *
            (p('field_speed')/3.6)**2 *
            p('tractor_surface_coefficient') *
            p('tractor_front_area')
        )

        F_rr = ( # Force road friction [N]
            p('tractor_weight') * 9.81 *
            p('friction_coefficient_field')
        )

        F_acc = ( # Force acceleration [N]
            p('tractor_weight') *
            p('mean_acceleration')
        )

        F_slope = ( # Force slope [N]
            p('tractor_weight') * 9.81 *
            np.sin(p('mean_slope'))
        )

        F_MR = ( # ??? [N]
            p('tractor_weight') * 9.81 *
            (
                1 / p('tractor_ground_resistance') +
                0.04 +
                0.05 *
                (
                    p('slippage') /
                    np.sqrt(p('tractor_ground_resistance'))
                )
            ) /
            1000
        )

        E_tractor = ( # [kWh/ha]
            (F_air + F_rr + F_acc + F_slope + F_MR) *
            (p('field_speed')/3.6) / 1000 /
            p('work_rate')
        )[0]

        # APPLICATION ENERGY REQUIREMENTS
        self.par.clear()
        M = pd.DataFrame(
            index = pd.MultiIndex.from_product([soil_classes,forage_systems]),
            columns = field_operations
        )

        # Calculate application energy requirements per hectare ...
        # ... for soil texture dependent applications
        E_soil_dep = ( # [kWh/ha]
            pf('application_soil_cor',M) *
            (
                pf('application_par_A',M) +
                pf('application_par_B',M) * pf('field_speed',M) +
                pf('application_par_C',M) * pf('field_speed',M)**2
            ) *
            pf('working_depth',M) *
            (pf('field_speed',M) / 3.6) / 1000 /
            (pf('field_efficiency',M) * pf('field_speed',M) / 10)
        )
        E_soil_dep.loc[:,:] = np.where(E_soil_dep>0,E_soil_dep+E_tractor,np.nan) # add tractor energy and set zero to NaN
        # ... based on power requirements
        E_from_power = ( # [kWh/ha]
            pf('P_per_width',M) /
            (pf('field_efficiency',M) * pf('field_speed',M) / 10)
        )
        E_from_power.loc[:,:] = np.where(E_from_power>0,E_from_power+E_tractor,np.nan) # add tractor energy and set zero to NaN
        # ... based on energy requirements per ha and/or per ton harvest
        E_from_energy = pf('E_per_area',M) # [kWh/ha]
        # combine
        E_per_area = E_soil_dep
        E_per_area.update(E_from_power, overwrite=False)
        E_per_area.update(E_from_energy, overwrite=False)
        E_per_area = (E_per_area * pf('forage_system_share',M)/100).groupby('soil_class').sum()

        # Calculate energy requirements ...
        self.par.clear()
        M = pd.DataFrame(
            index = soil_classes,
            columns = field_operations
        )
        # ... per ton harvested crop
        E_per_mass = (pf('E_per_mass',M) / 1000).replace(0,np.nan) # [kWh/kg]
        E_per_mass.update(
            pf('E_per_mass_distance',M) * pf('mean_distance_field_to_farm',M) / 1000,
            overwrite=False
        )
        # ... per ton harvested straw
        E_per_mass_straw = pf('E_per_mass_straw',M) / 1000 # [kWh/kg]
        # ... per area with manure application
        E_per_area_manure = pf('E_per_area_manure',M) # [kWh/ha]

        # Calculate final energy
        E_final = A.copy()
        # Series to get regions with soil class
        sc = pd.Series(self.regions.data_attr.get('soil_texture').index.values, index = self.regions.data_attr.get('soil_texture'))

        for soil_class in sc.index.unique():
            E_final.loc[idx[:,:,sc[soil_class]],:] = (
                (A * E_per_area.loc[soil_class,:]).mul(self.crops.data_attr.get('area'), axis=0) +
                (A * E_per_mass.loc[soil_class,:]).mul(self.crops.data_attr.get('harvest'), axis=0) +
                (A * E_per_mass_straw.loc[soil_class,:]).mul(self.crops.data_attr.get('crop_residues_harvest').sum(axis=1), axis=0) +
                (A * E_per_area_manure.loc[soil_class,:]).mul(self.crops.data_attr.get('fertiliser.manure_application_area'), axis=0)
            ).astype(float)

        # Calculate energy requirements per energy source and store
        self.par.clear()
        M = self.crops.data_attr.get('energy_use').loc[:,idx['field machinery',:]]

        self.crops.data_attr.get('energy_use').loc[:,idx['field machinery',:]] = (
            (pf('energy_source_share',M)/100)
            .mul(E_final.sum(axis=1), axis=0) /
            pf('drivetrain_efficiency',M)
        )

    def calculate_drying(self):
        self.par.clear()
        self.crops.par.clear()

        # Get garin dryer energy use dataframes
        dryers_DF = self.crops.data_attr.get('energy_use').loc[:,['grain dryers']].copy()
        dryers_aufiliary_DF = self.crops.data_attr.get('energy_use').loc[:,['grain dryers auxiliary energy']].copy()

        # Get salvaged harvest
        harvest = self.crops.data_attr.get('production').sum(axis=1)
        orig_index = harvest.index.copy()
        # Calculate as DM
        harvest_DM = harvest * self.crops.par.get('crop_dm', **harvest.index.to_frame().to_dict('list'))
        # Get water content at havest and
        field_WC = pd.Series(
            self.par.get('water_content_at_harvest', **harvest_DM.index.to_frame().to_dict('list')),
            index = harvest.index
        )
        # Drop crops without grain drying (i.e. water_content_at_harvest=-99)
        dryers_DF = dryers_DF.loc[field_WC!=-99]
        dryers_aufiliary_DF = dryers_aufiliary_DF.loc[field_WC!=-99]
        harvest_DM = harvest_DM.loc[field_WC!=-99]
        harvest = harvest.loc[field_WC!=-99]
        field_WC = field_WC.loc[field_WC!=-99]
        # Get water conent after drying
        dried_WC = pd.Series(
            self.par.get('water_content_after_drying', **harvest_DM.index.to_frame().to_dict('list')),
            index = harvest.index
        )
        assert (field_WC>dried_WC).all()

        water_removed = ( # total kg waer removed
            (harvest_DM / (1 - field_WC/100)) -
            (harvest_DM / (1 - dried_WC/100))
        )

        dryers_energy_use = (
            (self.par.get_from_frame('energy_source_share', dryers_DF)/100) *
            self.par.get_from_frame('dryer_energy_per_mass_water', dryers_DF) / # kWh/kg water removed
            (self.par.get_from_frame('dryer_efficiency', dryers_DF)/100)
        ).mul(water_removed, axis=0)

        dryers_auxiliary_energy_use = (
            (self.par.get_from_frame('energy_source_share', dryers_aufiliary_DF)/100) *
            self.par.get_from_frame('dryer_auxiliary_energy', dryers_aufiliary_DF)
        ).mul(harvest, axis=0)

        # Update energy_use data attribute
        self.crops.data_attr.get('energy_use').update(
            pd.concat([
                dryers_energy_use,
                dryers_auxiliary_energy_use
            ], axis=1)
        )

        return None

    def calculate_greenhouses(self):
        pf = self.par.get_from_frame
        idx = pd.IndexSlice

        self.par.clear()
        M = self.crops.data_attr.get('energy_use').loc[:,idx['greenhouses',:]]
        self.crops.data_attr.get('energy_use').loc[:,idx['greenhouses',:]] = (
            pf('greenhouse_energy_use',M)
            .mul(self.crops.data_attr.get('area'), axis=0) *
            (pf('energy_source_share',M)/100)
        )

    def calculate_stables(self):

        pf = self.par.get_from_frame
        self.par.clear()

        # Get stable energy use activities and energy sources
        acs = self.par.get_unique('activity',qry="parameter.isin(['stable_energy_use_per_head','stable_energy_use_per_inserted_head','stable_energy_use_per_prod'])")
        ess = self.par.get_unique('energy_source')

        for herd in self.herds:

            # Set species and breed filters for ParameterRetriever
            self.par.set(
                species = herd.species,
                breed = herd.breed
            )

            # Create dataframes of heads, inserted heads and production
            # and calculate energy use
            if 'heads' in herd.data_attr:
                heads = herd.data_attr.get('heads').fillna(0).reindex(
                    columns = pd.MultiIndex.from_tuples(
                        [(ps,an,ac,es) for ps,an in herd.data_attr.get('heads').columns for ac in acs for es in ess],
                        names=['prod_system','animal','activity','energy_source']
                    )
                )
                energy_use_per_head = (
                    heads *
                    pf('stable_energy_use_per_head',heads)
                )
            else:
                energy_use_per_head = 0

            if 'inserted_n' in herd.data_attr:
                inserted_heads = herd.data_attr.get('inserted_n').fillna(0).reindex(
                    columns = pd.MultiIndex.from_tuples(
                        [(ps,an,ac,es) for ps,an in herd.data_attr.get('inserted_n').columns for ac in acs for es in ess],
                        names=['prod_system','animal','activity','energy_source']
                    )
                )
                energy_use_per_inserted_head = (
                    inserted_heads *
                    pf('stable_energy_use_per_inserted_head',inserted_heads)
                )
            else:
                energy_use_per_inserted_head = 0

            prod = herd.data_attr.get('production').reindex(
                columns = pd.MultiIndex.from_tuples(
                    [(ps,an,ap,ac,es) for ps,an,ap in herd.data_attr.get('production').columns for ac in acs for es in ess],
                    names=['prod_system','animal','animal_prod','activity','energy_source']
                )
            )
            energy_use_per_prod = (
                (prod * pf('stable_energy_use_per_prod',prod))
                .T.groupby(['prod_system','animal','activity','energy_source']).sum().T
            )

            # Calculate energy use
            energy_use = (
                energy_use_per_head +
                energy_use_per_inserted_head +
                energy_use_per_prod
            )

            # Apply energy source share factors
            energy_use = energy_use * (pf('energy_source_share', energy_use)/100)

            # Add data attribute
            herd.data_attr.add(
                energy_use,
                name = 'energy_use',
                unit = 'kWh/year',
                orig = 'MachineryAndEnergyMgmt',
                desc = 'Energy use in stables'
            )

    def calculate_combustion_emissions(self):

        for module in [self.crops, self.waste] + list(self.herds):
            self.par.clear()

            # Get energy use. kWh --> TJ
            energy_use = module.data_attr.get('energy_use') * 1000 * 3600 / 1e12

            # Get compounds
            cps = self.par.get_unique('compound', qry='parameter == "combustion_EF"')

            energy_use = energy_use.reindex(
                pd.MultiIndex.from_tuples(
                    [cols + (cp,) for cols in energy_use.columns for cp in cps],
                    names = energy_use.columns.names + ['compound'],
                ),
                axis=1
            )

            # Calculate emissions
            emissions = energy_use.mul(
                self.par.get('combustion_EF', **energy_use.columns.to_frame().to_dict('list')),
                axis=1
            )

            # Add data attribute
            module.data_attr.add(
                emissions,
                name = 'energy_use_emissions',
                unit = 'kg/year',
                orig = 'MachineryAndEnergyMgmt',
                desc = 'Direct combustion emissions from energy use (excl. supply chain)'
            )
