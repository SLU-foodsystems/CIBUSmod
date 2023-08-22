import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import Container

class MachineryAndEnergyMgmt(object):
    '''Class that handles calculation of energy requirements
    in machinery, stables and greenhouses.

    Parameters
    ----------
    regions : Regions object
    crops : CropProduction object
    herds : (pandas.Series of) AnimalHerd object(s)
    par : ParameterRetriever object
    '''

    def __init__(self, regions, crops, herds, par):

        self.par = par
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
        vprint = verbose_init(verbose, id_str='MachineryAndEnergyMgmt')

        # Create dataframe to store energy use [kWh]
        self.crops.energy_use = pd.DataFrame(
            index = self.crops.index,
            columns = pd.MultiIndex.from_product([
                ['field machinery', 'grain drying', 'greenhouses'],
                self.par.get_unique('energy_source')
            ], names=['activity','energy_source'])
        )
        self.crops.data_attr.update(['energy_use'])

        vprint('Calculating energy use in field machinery ...')
        self.calculate_field_machinery()

        vprint('Calculating energy use for grain drying (not implemented) ...')
        self.calculate_field_machinery()

        vprint('Calculating energy use in greenhouses ...')
        self.calculate_greenhouses()

        vprint('Calculating energy use in stables (not implemented) ...')
        self.calculate_stables()
    
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

        E_tractor = float( # [kWh/ha]
            (F_air + F_rr + F_acc + F_slope + F_MR) *
            (p('field_speed')/3.6) / 1000 /
            p('work_rate')
        )

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

        # Calculate final energy
        E_final = A.copy()
        # Series to get regions with soil class
        sc = pd.Series(self.regions.soil['class'].index.values, index = self.regions.soil['class'])

        for soil_class in soil_classes:
            E_final.loc[idx[:,:,sc[soil_class]],:] = (
                (A * E_per_area.loc[soil_class,:]).mul(self.crops.area, axis=0) +
                (A * E_per_mass.loc[soil_class,:]).mul(self.crops.harvest, axis=0) +
                (A * E_per_mass_straw.loc[soil_class,:]).mul(self.crops.by_products['straw'], axis=0)
            )

        # Calculate energy requirements per energy source and store
        self.par.clear()
        M = self.crops.energy_use.loc[:,idx['field machinery',:]]

        self.crops.energy_use.loc[:,idx['field machinery',:]] = (
            (pf('energy_source_share',M)/100)
            .mul(E_final.sum(axis=1), axis=0) /
            pf('drivetrain_efficiency',M)
        )
        
    def calculate_drying(self):
        pass

    def calculate_greenhouses(self):
        pf = self.par.get_from_frame
        idx = pd.IndexSlice

        self.par.clear()
        M = self.crops.energy_use.loc[:,idx['greenhouses',:]]
        self.crops.energy_use.loc[:,idx['greenhouses',:]] = (
            pf('greenhouse_energy_use',M)
            .mul(self.crops.area, axis=0) *
            (pf('energy_source_share',M)/100)
        )



    def calculate_stables(self):
        pass

    

class EnergyUse(Container):
    '''Class to store energy use attributes'''