#!/usr/bin/python3

"""This module contains the soil_modules organic carbon data and methods.

Calculate the SOC data and related CO2 fluxes from input in
external modules.

The total CO2 fluxes and SOC data of a scenario are directly
accessible through a SoilData class instance after having run the
soil_modules calculations using the SoilCalc class.

Each SoilData class instance contains the complete input_inventory of
C inputs, SOC time-series data and CO2 fluxes for a scenario as an
xarray inside a netcdf file
"""

import inspect
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import pickle
from scipy import stats
import seaborn as sns
from typing import Dict, TypeVar, Type, List, Any, Tuple, Optional, Union
import xarray as xr

import CIBUSmod.soil_modules.soil_utils as soil_utils
from CIBUSmod.soil_modules.soil_params import C_CONTENT_CROPS
import CIBUSmod.soil_modules.icbm_funcs as icbm_funcs
from CIBUSmod.soil_modules import temp_path
from CIBUSmod.soil_modules import export_path

from CIBUSmod.utils import plot

T = TypeVar('T', bound='SoilData')

class SoilData:
    """Class to calculate and keep track of the data related to soil carbon and CO2 fluxes in a given scenario"""

    def __init__(self, session_df_name: str, session_name: str, import_path: str = temp_path,
                 verbose: bool = False) -> None:
        # Automatically set up instance upon initialization

        # input variables
        self.input_df, self.scenario, self.name = self.set_df_and_name(session_df_name, session_name, import_path)

        # Initialize output variables
        self.initialize_output_variables()
        self._set_startyear(verbose=verbose)

        # Perform save operations
        self.save_inventory()
        self.save_instance_state()

    @staticmethod
    def set_df_and_name(df_file_name: str, name_file_name: str, import_path: str) -> (pd.DataFrame, str, str):
        try:
            soil_input_df = soil_utils.read_csv_preserved(f'{import_path}/{df_file_name}')
            scenario = str(soil_input_df.index.get_level_values('scn').unique()[0])
            name = soil_utils.load_data(name_file_name, import_path)
        except Exception as e:
            print('Unexpected problem.')
            # Print a statement indicating the problem
            raise ValueError("input_data  must be a pd.DataFrame containing an index level named 'scn'") from e
        return (soil_input_df, scenario, name)

    def initialize_output_variables(self):
        # Initialize output variables to None or appropriate default values
        self.startyear = None  # type: int
        self.ss_input_df = None  # type: pd.Dataframe
        self.soc_ha_df = None  # type: pd.Dataframe
        self.soc_sko_df = None  # type: pd.Dataframe
        self.historic_ha_df = None  # type: pd.Dataframe
        self.historic_sko_df = None  # type: pd.Dataframe
        self.input_inventory = None  # type: xr.Dataset
        self.soc_inventory = None  # type: xr.Dataset
        self.historic_inventory = None  # type: xr.Dataset
        self.total_soc_inventory = None # type: xr.Dataset
        # help variables
        self._residue_col_name = None  # type: str
        self._input_grouped_df = None  # type: pd.Dataframe
        self._c_input_ha_df = None  # type: pd.Dataframe
        self._c_input_sko_df = None  # type: pd.Dataframe
        self._ss_input_ha_df = None  # type: pd.Dataframe
        self._ss_input_sko_df = None  # type: pd.Dataframe
        self._h_value_dict = None  # type: Dict
        self._scn_ha_sel = None  # type: list
        self._scn_sko_sel = None  # type: list
        self._ss_ha_sel = None  # type: list
        self._ss_sko_sel = None  # type: list
        self._spinup_groupby_df = None  # type: pd.DataFrame

    def __getstate__(self):
        # Copy the object's state using dict.copy() to avoid modifying the original state
        state = self.__dict__.copy()
        # Remove DataFrame and DataSet attributes before pickling
        excluded_keys = ['input_df',
                         'ss_input_df',
                         'soc_ha_df',
                         'soc_sko_df',
                         'historic_ha_df',
                         'historic_sko_df',
                         'input_inventory',
                         'soc_inventory',
                         'historic_inventory']
        for key in excluded_keys:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        # Restore instance attributes (except for excluded ones)
        self.__dict__.update(state)

    def _add_prefixes(self, verbose=False):
        """Add input fraction prefixes to manure input, ensuring it's only done once."""
        # Check for a specific prefixed column.
        prefixed_columns = [col for col in self.input_df.columns if col.startswith('i_ag_manure')]
        if prefixed_columns:
            # If any prefixed columns are found, assume '_add_prefixes' has been applied and skip.
            if verbose:
                print(">>> '_add_prefixes()' skipped: already applied. <<<")
            return

        if verbose:
            print(">>> Executing '_add_prefixes()' <<<")
        # Add prefixes to all the manure fractions so that all c input fractions follow the same naming convention
        self.input_df = soil_utils.add_prefix('manure', 'i_ag', self.input_df)
        if verbose:
            print(">>> '_add_prefixes()' executed succesfully <<<")
            print(f"input_df.columns now: {self.input_df.columns}")

    def _change_year_name(self, new_name='input_year', verbose=False):
        """Change the column name "year" to 'new_name'(default="input_year")"""
        if verbose:
            print(">>> Executing '_change_year_name()'<<<")
        if isinstance(self.input_df.index, pd.MultiIndex):
            self.input_df.index = self.input_df.index.set_names(new_name, level='year')
        else:
            self.input_df = self.input_df.rename(columns={'year': new_name})
        if verbose:
            print(">>> '_change_year_name()' executed successfully <<<")


    def _set_startyear(self, new_name='input_year', verbose=False) -> None:
        """
        Changes the name of the 'year' index level or column to 'input_year', if necessary.
        Then sets 'startyear' to the first value of 'input_year' in 'input_df'.

        Parameters
        ----------
        - new_name (str, optional): The new name for the 'input_year' column if 'year' is present.
          Defaults to 'input_year'

        Returns
        -------
        None

        """
        # make sure that the 'year' column name is unambiguous
        if isinstance(self.input_df.index, pd.MultiIndex):
            if new_name not in self.input_df.index.names \
                    and 'year' in self.input_df.index.names:
                self._change_year_name(new_name=new_name, verbose=verbose)
        else:
            if new_name not in self.input_df.columns \
                    and 'year' in self.input_df.columns:
                self._change_year_name(new_name=new_name, verbose=verbose)
        # make sure the 'input_year' column is in datetime64 format
        self._make_datetime(verbose=verbose)
        self.startyear = self.input_df.index.get_level_values(new_name).year[0]

    def _make_datetime(self, verbose=False):
        """Change the "input_year" column dtype to datetime64"""
        if verbose:
            print(">>> Executing '_make_datetime()'<<<")
        if isinstance(self.input_df.index, pd.MultiIndex):
            idx = list(self.input_df.index.names)
            self.input_df = self.input_df.reset_index()
            self.input_df['input_year'] = pd.to_datetime(self.input_df['input_year'], format='%Y')
            self.input_df = self.input_df.set_index(idx)
        else:
            self.input_df['input_year'] = pd.to_datetime(self.input_df['input_year'], format='%Y')
        if verbose:
            print(">>> '_make_datetime()' executed succesfully <<<")

    def _calc_input_ha(self, verbose=False):
        """
        Calculate the yield and input per unit area, given total yield ("harvest_kg_dm") and area ("area_ha")

        Also assign yield and input per sko
        """
        # Check if computations have already been applied
        if "areayield" in self.input_df.columns and "areayield_residues" in self.input_df.columns:
            if verbose:
                print(">>> '_calc_input_ha()' skipped: already applied. <<<")
            return

        if verbose:
            print(">>> Executing '_calc_input_ha()'<<<")

        # Calculate yield per unit area
        self.input_df["areayield"] = self.input_df["harvest_kgdm"] / self.input_df["area_ha"] * C_CONTENT_CROPS

        if "crop_residues_kgdm" in self.input_df.columns:
            self.input_df["areayield_residues"] = self.input_df["crop_residues_kgdm"] / self.input_df[
                "area_ha"] * C_CONTENT_CROPS
        else:
            self.input_df["areayield_residues"] = 0

        if verbose:
            print(">>> '_calc_input_ha()' executed succesfully <<<")

    def _calc_amnd_ha(self, verbose=False):
        """
        Calculate the amnendment input per unit area, given total manure ("_kgc") and area ("area_ha")
        Avoids re-running computations if already applied.
        """
        if verbose:
            print(">>> Executing '_calc_amnd_ha()'<<<")

        # Calculate and add columns with the manure input per ha
        tot_manure_cols = soil_utils.get_filtered_namelist(['kgc'], ['manure', 'crop'], self.input_df)
        new_col_names = [f'{i[:-4]}_ha' for i in tot_manure_cols]

        # Check if any of the new columns already exist
        if all(col in self.input_df.columns for col in new_col_names):
            if verbose:
                print(">>> '_calc_amnd_ha()' skipped: already applied. <<<")
            return

        # Calculate and add new columns with the input per ha
        for i in tot_manure_cols:
            col_name = f'{i[:-4]}_ha'
            self.input_df[col_name] = self.input_df[i] / self.input_df['area_ha']
            if verbose:
                print(f"--- Inserted {col_name}---")
        if verbose:
            print(">>> '_calc_amnd_ha()' executed succesfully <<<")

    def _make_multiindex(self, idx, verbose=False):
        """create multiindex dataframe"""
        if verbose:
            print(">>> Executing '_make_multiindex()'<<<")
        scn_input_idx = idx # ['scn', 'crop', 'prod_system', 'region', 'input_year']
        # Create a multiindex df
        self.input_df = self.input_df.set_index(scn_input_idx)
        if verbose:
            print(">>> '_make_multiindex()' executed succesfully <<<")


    def _calc_crop_inputs(self, straw_removed=False, verbose=False, looped=False):
        """Calculate the carbon inputs above and below ground based on allocation methods and factors"""
        if verbose:
            print(">>> Executing '_calc_crop_inputs()'<<<")
        allo_dict = soil_utils.alloc_helper(name_df=False,
                                            allo1_df=False,
                                            allo2_df=False,
                                            allo3_df=False,
                                            scenario_dict=False,
                                            verbose=verbose)
        mapper = (allo_dict['crop_andren2004_map'], allo_dict['crop_jacobs_map'], allo_dict['crop_hanna_map'])
        sources = ('Andren2004', 'Jacobs2020', 'Hanna')
        allo_dfs = (allo_dict['c_allom_andren_df'], allo_dict['c_alloc_jacobs_df'], allo_dict['c_alloc_hanna_df'])

        if looped is False:
            self.input_df = soil_utils.calculate_c_inputs_vectorized(self.input_df,
                                                                     mapper,
                                                                     sources,
                                                                     allo_dfs,
                                                                     straw_removed=straw_removed,
                                                                     verbose=verbose)
        else:
            self.input_df = soil_utils.calculate_c_inputs(self.input_df,
                                                          mapper,
                                                          sources,
                                                          allo_dfs,
                                                          straw_removed=straw_removed,
                                                          verbose=verbose)


        #  Drop unwanted 'level_0' or 'index' columns if they were added
        if 'level_0' in self.input_df.columns or 'index' in self.input_df.columns:
            self.input_df = self.input_df.drop(columns=['level_0', 'index'], errors='ignore')

        if verbose:
            print(">>> '_calc_crop_inputs()' executed succesfully <<<")
            print(f"input_df.columns now: {self.input_df.columns}")


    def _make_spinup_df(self, verbose=False):
        """
        Generate a DataFrame with input values for spinup modeling.

        Parameters:
        - verbose (bool, optional): If True, print execution messages.

        Returns:
        - pd.DataFrame: DataFrame with input values for spinup modeling.
        """
        if verbose:
            print('---Executing _make_spinup_df()---')
        scn_multi_df = self.input_df
        year0 = self.startyear

        # create spinup_ha_df and spinup_ha_multiidx_df based on scenario df
        self.ss_input_df = self.input_df.query(f'input_year == {self.startyear}')
        self.ss_input_df = self.ss_input_df.droplevel(['input_year'])

        if verbose:
            print('---Leaving _make_spinup_df()---')


    def calc_scn_inputs(self, verbose=False, looped=False):
        """
        Calculate the carbon inputs used for soil organic carbon modelling in CIBUSmod.

        Parameters:
        - verbose (bool, optional): If True, print execution messages. Defaults to False.

        Returns:
        None

        Note:
        This method generates a multi-index DataFrame for CIBUSmod scenarios based on the input DataFrame.
        It performs various calculations and transformations on the input data to create a scenario input DataFrame.
        The resulting DataFrame is stored in the instance variable `input_df`. Additionally, if `save_inventory` is True,
        the input_inventory dataset is saved to a netcdf-file. The method also sets the instance variables `startyear` and `input_inventory`.

        Example usage:
        ```python
        scenario_instance = SoilData()
        scenario_instance.calc_scn_inputs(new_name='input_year', save_inventory=True, verbose=True)
        ```
        """
        soil_utils.colored_rule(color='cyan', height=2)
        print("Calculating scenario inputs...")
        if verbose:
            print('---Executing calc_scn_inputs()---')
        _residue_col = self.input_df.filter(like='residues', axis=1)
        if len(_residue_col.columns) == 1:
            self._residue_col_name = _residue_col.columns[0]
            if '_harvest' in self._residue_col_name:
                new_col_name = self._residue_col_name.replace('_harvest', "")
                self.input_df.rename(columns={self._residue_col_name: new_col_name}, inplace=True)
        # self.input_df = soil_utils.make_idx_continuous(self.input_df)
        idx = self.input_df.index.names
        self.input_df.reset_index(inplace=True)
        self.input_df = soil_utils.make_df_lower(self.input_df)
        if not isinstance(self.input_df, pd.DataFrame):
            return print('> Input dataframe not set. Please supply a valid input dataframe and retry')
        self._calc_input_ha(verbose=verbose)
        self._calc_amnd_ha(verbose=verbose)
        self._calc_crop_inputs(verbose=verbose, looped=looped)
        self._add_prefixes(verbose=verbose)
        self._make_multiindex(idx, verbose=verbose)
        # Set instance variables
        self.input_inventory = self.input_df.to_xarray()
        if verbose:
            print('info: Finished creating scenario input_df')
            print('    > Continuing with the creation of spinup input_df')
        self._make_spinup_df()
        if verbose:
            print('---calc_scn_inputs() executed successfully---')
        soil_utils.colored_rule(color='green', height=2)

    def _calculate_soc(self, grouping, verbose=False):
        """
        Calculate the SOC pools for both the ha and sko dataframes
        """
        if verbose:
            print('---Executing _calculate_soc()---')
        if isinstance(self._c_input_ha_df, pd.DataFrame) and isinstance(self._c_input_sko_df, pd.DataFrame):
            print("'_c_input_ha_df' and '_c_input_sko_df' set. Continuing")
        else:
            self._make_scn_area_dfs(grouping, verbose=verbose)
            if verbose:
                print(
                    f"'_c_input_ha_df' and '_c_input_sko_df' generated")
        if not self._h_value_dict:
            if verbose:
                print("-> 'h_value_dict' not set.")
            temp, self._h_value_dict = soil_utils.h_map_helper(verbose=verbose)
        # extract a list of the scenario columns by which icbm is to be run
        if verbose:
            print("Extracting filtered_namelist per ha")
        self._scn_ha_sel = list(
            set(soil_utils.get_filtered_namelist(['i_a', 'i_b', 'ha'], ['manure', 'crop'], self._c_input_ha_df)))
        if verbose:
            print("Extracting filtered_namelist per sko")
        self._scn_sko_sel = list(
            set(soil_utils.get_filtered_namelist(['i_a', 'i_b', 'sko'], ['manure', 'crop'], self._c_input_sko_df)))
        if verbose:
            print('---Leaving _calculate_soc()---')

    def _calculate_historic_soc(self,
                                verbose=False
                                ):
        """
        Calculate the historic SOC pools at year 0 for both the ha and sko dataframes
        """
        if verbose:
            print('---Executing _calculate_historic_soc()---')
        # Assign a name to the spinup scenario_name variable
        if isinstance(self._ss_input_ha_df, pd.DataFrame) and isinstance(self._ss_input_sko_df, pd.DataFrame):
            print("'_ss_input_ha_df' and '_ss_input_sko_df' set. Continuing")
        else:
            if verbose:
                print("> One or both of '_ss_input_ha_df' and '_ss_input_sko_df' are unset")
                print("info: Generating 'spinup_ha_df' and 'spinup_sko_df' from 'input_df'")
            self._make_spinup_area_dfs(verbose=verbose)
            if verbose:
                print(f"'_ss_input_ha_df' and '_ss_input_sko_df' set using input_df for {self.scenario}")

        if not self._h_value_dict:
            if verbose:
                print("> 'h_value_dict' not set.")
            temp, self._h_value_dict = soil_utils.h_map_helper(verbose=verbose)

        # extract a list of the spinup columns by which icbm is to be run
        if not self._ss_ha_sel:
            if verbose:
                print("Extracting filtered_namelist per ha")
            self._ss_ha_sel = list(
                set(soil_utils.get_filtered_namelist(['i_a', 'i_b'], ['manure', 'crop', 'ha'], self._ss_input_ha_df)))
        if not self._ss_sko_sel:
            if verbose:
                print("Extracting filtered_namelist per sko")
            self._ss_sko_sel = list(
                set(soil_utils.get_filtered_namelist(['i_a', 'i_b'], ['manure', 'crop', 'ha'], self._ss_input_sko_df)))
        print('---Leaving _calculate_historic_soc()---')

    def _scn_icbm_calculations(self, verbose=False):
        """
        Calculate the soc timeseries dataframe

        using _c_input_df, _scn_sel and h_value_dict as input for scenario
        """
        if verbose:
            print('---Executing _scn_icbm_calculations()---')
            print("Calculating SOC timeseries per ha")
        self.soc_ha_df = icbm_funcs.input_df_to_soc_df(self._c_input_ha_df, self._scn_ha_sel, self._h_value_dict,
                                                       year_label="input_year", verbose=verbose)
        if verbose:
            print("Calculating SOC timeseries per sko")
        self.soc_sko_df = icbm_funcs.input_df_to_soc_df(self._c_input_sko_df, self._scn_sko_sel, self._h_value_dict,
                                                        year_label="input_year", verbose=verbose)
        if verbose:
            print("Done calculating SOC timeseries")
        # Create soc xarray datasets
        if verbose:
            print("Creating xarray soc dataset")
        soc_ha_ds = self.soc_ha_df.to_xarray()
        soc_sko_ds = self.soc_sko_df.to_xarray()
        self.soc_inventory = soc_ha_ds.merge(soc_sko_ds)
        if verbose:
            print('---_scn_icbm_calculations() executed succesfully---')

    def _historic_icbm_calculations(self, verbose=False):
        """
        Calculate the historic soc timeseries dataframes

        using _ss_input df, _ss_sel and h_value_dict as input
        """
        if verbose:
            print('---Executing _historic_icbm_calculations()---')
            print(f"info: Calculating SOC SS values per ha in {self.startyear}")
        spinup_ss_ha_df = icbm_funcs.spinup_df_to_ss_soc_df(self._ss_input_ha_df,
                                                            self._ss_ha_sel,
                                                            self._h_value_dict)
        if verbose:
            print(f"info: Calculating SOC SS values per sko in {self.startyear}")
        spinup_ss_sko_df = icbm_funcs.spinup_df_to_ss_soc_df(self._ss_input_sko_df,
                                                             self._ss_sko_sel,
                                                             self._h_value_dict)
        if verbose:
            print("info: Calculating SOC timeseries per ha")
        self.historic_ha_df = icbm_funcs.input_df_to_soc_df(spinup_ss_ha_df,
                                                            self._ss_ha_sel,
                                                            self._h_value_dict,
                                                            historic=True)
        if verbose:
            print("info: Calculating SOC timeseries per sko")
        self.historic_sko_df = icbm_funcs.input_df_to_soc_df(spinup_ss_sko_df,
                                                             self._ss_sko_sel,
                                                             self._h_value_dict,
                                                             historic=True)
        if verbose:
            print("info: Finished calculating historic SOC timeseries")
        # Create historic soc timeseries to xarray dataset
        if verbose:
            print("Creating xarray historic soc dataset")
        historic_ha_ds = self.historic_ha_df.to_xarray()
        historic_sko_ds = self.historic_sko_df.to_xarray()
        self.historic_inventory = historic_ha_ds.merge(historic_sko_ds)
        if verbose:
            print('---_historic_icbm_calculations() executed succesfully---')

    def _make_scn_area_dfs(self, grouping: list=['scn', 'prod_system', 'region', 'input_year'], verbose: bool=False) -> None:
        """Create scenario dataframes with yields per ha and per sko."""
        if verbose:
            print('---Executing _make_scn_area_dfs()---')
        scenario_name = self.scenario
        # Group the scn df by scn, prod system, region and year and calculate total input and area of all crops
        scn_multi_groupby_idx = grouping
        self._input_grouped_df = self.input_df.groupby(scn_multi_groupby_idx).sum()
        # Select the columns that should hold the weighted average
        wt_at_cols = soil_utils.get_filtered_namelist(['_ha'], ['manure', 'crop'], self._input_grouped_df)
        tot_cols = soil_utils.get_filtered_namelist(['_kgc'], ['manure', 'crop'], self._input_grouped_df)
        # Calculate the weighted average per ha from total input per sko / total area per sko
        for n, i in enumerate(wt_at_cols):
            self._input_grouped_df[i] = self._input_grouped_df[tot_cols[n]] / self._input_grouped_df['area_ha']
        # Make separate df's for ha and sko input, include sko input and area info
        self._c_input_ha_df = self._input_grouped_df.loc[:, wt_at_cols]
        self._c_input_ha_df['area_ha'] = self._input_grouped_df['area_ha']
        self._c_input_ha_df['harvest_kgdm'] = self._input_grouped_df['harvest_kgdm']
        self._c_input_sko_df = self._input_grouped_df.loc[:, tot_cols]
        self._c_input_sko_df['area_ha'] = self._input_grouped_df['area_ha']
        self._c_input_sko_df['harvest_kgdm'] = self._input_grouped_df['harvest_kgdm']
        if verbose:
            print('---Leaving _make_scn_area_dfs()---')

    def _make_spinup_area_dfs(self, verbose=False):
        """
        Create spinup dataframes with yields per ha and per sko
        """
        if verbose:
            print('---Executing _make_spinup_area_dfs()---')
        # Group the spinup df by prod system and region to calculate total input and areas of all crops
        spinup_groupby_idx = ['prod_system', 'region']
        self._spinup_groupby_df = self.ss_input_df.groupby(spinup_groupby_idx).sum()
        # Select the columns that should hold the weighted average
        wt_at_cols = soil_utils.get_filtered_namelist(['_ha'], ['manure', 'crop'], self._spinup_groupby_df)
        tot_cols = soil_utils.get_filtered_namelist(['_kgc'], ['manure', 'crop'], self._spinup_groupby_df)
        # Create a temporary df and calculate the weighted average per ha from total input per sko / total area per sko
        tempframe = self._spinup_groupby_df.copy()
        for n, i in enumerate(wt_at_cols):
            tempframe[i] = tempframe[tot_cols[n]] / tempframe['area_ha']
        # Rename the temporary df
        spinup_all_c_inputs_df = tempframe.copy(deep=True)
        # Make separate df's for ha and total input, include total input and area info
        self._ss_input_ha_df = spinup_all_c_inputs_df.loc[:, wt_at_cols]
        self._ss_input_ha_df['area_ha'] = spinup_all_c_inputs_df['area_ha']
        self._ss_input_ha_df['harvest_kgdm'] = spinup_all_c_inputs_df['harvest_kgdm']
        self._ss_input_sko_df = spinup_all_c_inputs_df.loc[:, tot_cols]
        self._ss_input_sko_df['area_ha'] = spinup_all_c_inputs_df['area_ha']
        self._ss_input_sko_df['harvest_kgdm'] = spinup_all_c_inputs_df['harvest_kgdm']
        if verbose:
            print('---Leaving _make_spinup_area_dfs()---')

    def calc_soc_timeseries(self, group: list = ['scn', 'prod_system', 'region', 'input_year'], verbose=False,
                            looped=False):
        """Calculate the SOC timeseries and create a soc_inventory dataset"""
        soil_utils.colored_rule(color='cyan', height=2)
        print('Calculating SOC timeseries...')
        if looped:
            self._calculate_soc(grouping=group, verbose=verbose)
            self._scn_icbm_calculations(verbose=verbose)
        else:
            self._calculate_soc_new(grouping=group, verbose=verbose)
            self._scn_icbm_calculations_new(verbose=verbose)
        soil_utils.colored_rule(color='green', height=2)

    def _calculate_soc_new(self, grouping, verbose=False):
        """
        Calculate the SOC pools for both the ha and sko dataframes
        """
        if verbose:
            print('---Executing _calculate_soc()---')

        # Check if calculations have already been performed
        if hasattr(self, '_soc_calculated') and self._soc_calculated:
            if verbose:
                print("SOC calculations already performed. Skipping...")
            return

        # Assuming _make_scn_area_dfs and h_map_helper are already optimized for vectorized operations
        self._make_scn_area_dfs_new(grouping, verbose=verbose)
        if verbose:
            print(f"'_c_input_ha_df' and '_c_input_sko_df' generated")
        if not self._h_value_dict:
            if verbose:
                print("-> 'h_value_dict' not set.")
            _, self._h_value_dict = soil_utils.h_map_helper_new(verbose=verbose)

        # Vectorized extraction of filtered namelists
        if verbose:
            print("Extracting filtered_namelist per ha")
        self._scn_ha_sel = list(
            set(soil_utils.get_filtered_namelist_new(['i_a', 'i_b', 'ha'], ['manure', 'crop'], self._c_input_ha_df)))
        if verbose:
            print("Extracting filtered_namelist per sko")
        self._scn_sko_sel = list(
            set(soil_utils.get_filtered_namelist_new(['i_a', 'i_b', 'sko'], ['manure', 'crop'], self._c_input_sko_df)))

        # Mark that SOC calculations have been performed to avoid reapplication
        self._soc_calculated = True

        if verbose:
            print('---Leaving _calculate_soc()---')

    def _make_scn_area_dfs_new(self, grouping: list = ['scn', 'prod_system', 'region', 'input_year'],
                               verbose: bool = False) -> None:
        """Create scenario dataframes with yields per ha and per sko."""
        if verbose:
            print('---Executing _make_scn_area_dfs()---')

        # Group the DataFrame by the specified columns and calculate the sum.
        self._input_grouped_df = self.input_df.groupby(grouping).sum()

        # Calculate weighted averages for manure and crop inputs per ha.
        # This utilizes vectorized operations across the entire DataFrame without looping.
        wt_at_cols = [col for col in self._input_grouped_df.columns if '_ha' in col]
        tot_cols = [col for col in self._input_grouped_df.columns if '_kgc' in col]

        # Use a vectorized approach to compute weighted averages
        for wt_at, tot in zip(wt_at_cols, tot_cols):
            self._input_grouped_df[wt_at] = self._input_grouped_df[tot] / self._input_grouped_df['area_ha']

        # Extract specific columns for ha and sko DataFrames
        self._c_input_ha_df = self._input_grouped_df[wt_at_cols + ['area_ha', 'harvest_kgdm']].copy()
        self._c_input_sko_df = self._input_grouped_df[tot_cols + ['area_ha', 'harvest_kgdm']].copy()

        if verbose:
            print('---Leaving _make_scn_area_dfs()---')

    def calc_historic_soc_timeseries(self, verbose=False):
        """Calculate the historic SOC timeseries and update the soc_inventory dataset"""
        soil_utils.colored_rule(color='cyan', height=2)
        print('Calculating historic SOC timeseries...')
        self._calculate_historic_soc(verbose=verbose)
        self._historic_icbm_calculations(verbose=verbose)
        soil_utils.colored_rule(color='green', height=2)


    # Methods to load and save inventory and state variables

    def save_inventory(self, dataset=None):
        """
        Saves the specified inventory dataset(s) to NetCDF and CSV files.

        This method supports saving individual datasets, a list of datasets, or all predefined datasets if none specified.
        It saves xarray Datasets to NetCDF files and pandas DataFrames to CSV files, using a naming convention based on the dataset name.

        Parameters:
        - dataset: Optional[str, list] - Name(s) of the dataset(s) to be saved ('input', 'soc', 'historic', 'total_soc').
                                          If None, all datasets will be saved.
        """
        # Default dataset names if none are specified
        if dataset is None:
            dataset = ['input', 'soc', 'historic', 'total_soc']
            print(f'Dataset set to {dataset}')

        # Recursively save each dataset if a list is provided
        if isinstance(dataset, list):
            for ds in dataset:
                self.save_inventory(ds)
            return

        # Save individual dataset
        self._save_dataset(dataset)

    def _save_dataset(self, dataset_name, temp_path=temp_path):
        """
        Helper method to save an individual dataset to the appropriate file format.

        Parameters:
        - dataset_name: str - The name of the dataset to save.
        """
        scn_ds_name = f'{self.scenario}_{dataset_name}_ds'
        dataset_attr = f'{dataset_name}_inventory'

        # Save xarray Dataset to NetCDF
        if hasattr(self, dataset_attr) and isinstance(getattr(self, dataset_attr), xr.Dataset):
            getattr(self, dataset_attr).to_netcdf(f"{temp_path}/{scn_ds_name}.nc")
            print(f"{dataset_attr} saved as {scn_ds_name}.nc in {temp_path}")

        # Save pandas DataFrames to CSV, checking for existence
        df_names = [f'{dataset_name}_df', f'ss_{dataset_name}_df', f'{dataset_name}_ha_df', f'{dataset_name}_sko_df']
        for df_name in df_names:
            if hasattr(self, df_name) and isinstance(getattr(self, df_name), pd.DataFrame):
                scn_df_name = f'{self.scenario}_{df_name}'
                #csv_path = f'{temp_path}/{df_name}.csv'
                soil_utils.to_csv_preserved(getattr(self, df_name), save_as=scn_df_name, save_path=temp_path)
                print(f"{df_name} saved as {scn_df_name} in {temp_path}")

    def load_inventory(self, dataset=None, temp_path=temp_path):
        """
        Loads inventory data based on specified dataset names.

        This method supports loading single datasets, a list of datasets, or all datasets if none specified.
        It updates the instance attributes for input, SOC (Soil Organic Carbon), and historic data inventories
        along with their corresponding DataFrames from CSV files.

        Parameters:
        - dataset: Optional[str, list] - Name(s) of the dataset(s) to be loaded ('inputs', 'soc', 'historic', 'total_soc').
                                          If None, all datasets will be loaded.
        """
        ds_loaded = []  # Tracks loaded xarray datasets
        df_loaded = []  # Tracks loaded CSV dataframes
        # Load specific datasets or all if none specified
        if dataset is None:
            dataset = ['inputs', 'soc', 'historic', 'total_soc']
            for ds in dataset:
                self.load_inventory(ds)
        elif isinstance(dataset, list):  # Recursively load each dataset in list
            for ds in dataset:
                self.load_inventory(ds)
        else:  # Load individual dataset
            if dataset == 'inputs':
                # Load 'inputs' dataset
                scn_input_ds_name = f'{self.scenario}_input_ds'
                self.input_inventory = xr.load_dataset(f"{temp_path}/{scn_input_ds_name}.nc")
                ds_loaded.append('input_inventory')
                # Load associated CSVs if they exist
                self._load_csv_files(temp_path, ['input_df', 'ss_input_df'], df_loaded)
            elif dataset == 'soc':
                # Load 'soc' dataset
                try:
                    scn_soc_ds_name = f'{self.scenario}_soc_ds'
                    self.soc_inventory = xr.load_dataset(f"{temp_path}/{scn_soc_ds_name}.nc")
                    ds_loaded.append('soc_inventory')
                    # Load associated CSVs if they exist
                    self._load_csv_files(temp_path, ['soc_ha_df', 'soc_sko_df'], df_loaded)
                except(FileNotFoundError):
                    pass
            elif dataset == 'historic':
                try:
                    # Load 'historic' dataset
                    scn_historic_ds_name = f'{self.scenario}_historic_ds'
                    self.historic_inventory = xr.load_dataset(f'{temp_path}/{scn_historic_ds_name}.nc')
                    ds_loaded.append('historic_inventory')
                    # Load associated CSVs if they exist
                    self._load_csv_files(temp_path, ['historic_ha_df', 'historic_sko_df'], df_loaded)
                except(FileNotFoundError):
                    pass
            elif dataset == 'total_soc':
                try:
                    # Load 'total_soc' dataset
                    scn_total_soc_ds_name = f'{self.scenario}_total_soc_ds'
                    self.total_soc_inventory = xr.load_dataset(f'{temp_path}/{scn_total_soc_ds_name}.nc')
                    ds_loaded.append('total_soc_inventory')
                except(FileNotFoundError):
                    pass
        # Print loaded datasets and dataframes
        self._print_loaded_items(ds_loaded, 'dataset')
        self._print_loaded_items(df_loaded, 'dataframe')

    def _load_csv_files(self, temp_path, csv_names, loaded_list):
        """
        Helper method to load CSV files as DataFrames if they exist.

        Parameters:
        - temp_path: str - The path where CSV files are located.
        - csv_names: list - A list of CSV file base names to load.
        - loaded_list: list - A list to append the names of successfully loaded CSV files.
        """
        for csv_name in csv_names:
            scn_input_df_name = f'{self.scenario}_{csv_name}'
            csv_path = f'{temp_path}/{scn_input_df_name}.csv'
            if os.path.exists(csv_path):
                setattr(self, csv_name, soil_utils.read_csv_preserved(csv_path))
                loaded_list.append(csv_name)

    def _print_loaded_items(self, items, item_type):
        """
        Helper method to print loaded items.

        Parameters:
        - items: list - A list of loaded item names.
        - item_type: str - A description of the item type ('dataset' or 'dataframe').
        """
        newline = '\n '
        if items:
            print(f"The following {item_type} variables have been set:{newline}{newline.join(items)}")
        else:
            print(f"No {item_type} variables have been loaded.")

    def check_attributes_status(self, access='public'):
        """
        Checks the status of all attributes in the given class instance.

        Parameters:
        - access: Selects whether to show 'public' or 'private' attributes. (default: 'public')

        Returns:
        A dictionary where keys are attribute names and values are tuples containing
        the current type of the attribute and a boolean indicating if it is set (not None).
        """
        attrs_status = {}
        for attribute in dir(self):
            # Filter based on access level
            if access == 'public' and attribute.startswith('_'):
                continue
            elif access == 'private' and not attribute.startswith('_'):
                continue
            attr_value = getattr(self, attribute)
            # Skip methods and magic methods
            if inspect.ismethod(attr_value) or attribute.startswith('__'):
                continue
            # Check if the attribute is set (not None)
            is_set = attr_value is not None
            attrs_status[attribute] = (type(attr_value).__name__, is_set)
        return f"The following attributes are set for {access} variables", attrs_status

    def save_instance_state(self, temp_path=temp_path):
        with open(f'{temp_path}/{self.scenario}.pickle', 'wb') as file:
            pickle.dump(self, file)
        print(f'Saved instance variable states to {temp_path}/{self.scenario}')

    @classmethod
    def load_instance_state(cls: Type[T], scenario_name: str, temp_path=temp_path) -> T:
        with open(f'{temp_path}/{scenario_name}.pickle', 'rb') as file:
            instance = pickle.load(file)
        return instance


    # Methods to print status messages
    def print_public_parameter_status(self):
        col1_length = len(max(self.check_attributes_status('public')[1].keys())) + 2
        attribute = ' \n'.join([f"{key}: {' ' * (col1_length - len(key))}{value}" for key, value in self.check_attributes_status('public')[1].items()])
        print(f"The following public attributes have been set for {self.scenario}:\n{attribute}")


    def print_private_parameter_status(self):
        col1_length = len(max(self.check_attributes_status('private')[1].keys())) + 2
        attribute = ' \n'.join([f"{key}: {' ' * (col1_length - len(key))}{value}" for key, value in self.check_attributes_status('private')[1].items()])
        print(f"The following private attributes have been set for {self.scenario}:\n{attribute}")

    def _add_total_soc(self):
        """
        Adds a 'tot_soc' data array to the class's inventory datasets.

        This method applies the 'assign_tot_soc' function to both 'soc_inventory'
        and 'historic_inventory' datasets of the class. It calculates 'tot_soc' as
        the sum of 'y_pool' and 'o_pool' data arrays within these datasets. The
        resulting 'tot_soc' data array represents the total soil organic carbon.

        Returns:
        --------
        None: Modifies the 'soc_inventory' and 'historic_inventory' datasets in-place.
        """
        soil_utils.assign_tot_soc(self.soc_inventory)
        soil_utils.assign_tot_soc(self.historic_inventory)

    def _add_co2_flux(self):
        """
        Adds a 'co2_flux' data array to the class's inventory datasets.

        This method applies the 'assign_co2_flux' function to both 'soc_inventory'
        and 'historic_inventory' datasets of the class. It calculates 'co2_flux' as
        the difference in 'tot_soc' (total soil organic carbon) between two
        consecutive years. The 'co2_flux' data array represents the carbon dioxide
        flux due to changes in soil organic carbon over time.

        Returns:
        --------
        None: Modifies the 'soc_inventory' and 'historic_inventory' datasets in-place.
        """
        soil_utils.assign_co2_flux(self.soc_inventory)
        soil_utils.assign_co2_flux(self.historic_inventory)


    def total_merge(self):
        """
        Calculates total SOC fluxes by merging the historic and new input soc inventories.

        This is done by first adding total SOC and annual CO2 fluxes to the historic_inventory and the soc_inventory.
        These are then combined buy coordinate into a new inventory dataset called 'total_soc_inventory'

        Parameters
        ----------
        None: In order to align the historic_inventory with the soc_inventory the dimensions 'scn' and 'input_year' are added to the historic_inventory.
                'scn' is set to self.scenario, while 'input_year' is set to self.startyear -1.

        Returns
        -------
        None: The self.total_soc_inventory is set, and 'tot_soc' and 'co2_flux' are added to the historic_inventory and soc_inventory datasets.
        """
        # Calculate and assign total SOC and annual co2 flux vectors to the soc_inventory and the historic_inventory.
        self._add_total_soc()
        self._add_co2_flux()

        # Align the datasets to be merged and combine them by coordinates
        try:
            historic_inputyear = pd.to_datetime(self.startyear - 1, format='%Y')
            self.historic_inventory = self.historic_inventory.expand_dims(
                {'scn': [self.scenario.lower()], 'input_year': [historic_inputyear]}, axis=None)
            self.total_soc_inventory = xr.combine_by_coords([self.soc_inventory, self.historic_inventory])
        except ValueError as e:
            print(f'The inventories have most likely already been merged: {e}')


    # Plotting functions
    def plot_single_region_timeseries(self,
                                      fractions: List[str] = ['new', 'hist', 'tot'],
                                      sko: int = 111,
                                      system: str = 'conventional',
                                      scenario: str = 'fai',
                                      selection: str = '_ha'
                                      ) -> None:
        """
        Plots the SOC inventory data for a single region or per ha, based on specified criteria.

        This method iterates through a given list of fraction types and aggregates
        the SOC data across all fractions and input years for the specified region (`sko`),
        production system (`system`), and scenario (`scenario`). It then plots this aggregated
        data. The method is intended for use with 'new', 'historical' ('hist'), and
        'total' ('tot') SOC data.

        Parameters:
        - fractions: The types of fractions to plot. Defaults to ['new', 'hist', 'tot'].
        - sko: The region code to plot data for. Defaults to 111.
        - system: The production system, e.g., 'conventional'. Defaults to 'conventional'.
        - scenario: The scenario name, e.g., 'fai'. Defaults to 'fai'.
        - selection: The area selection to be plotted, e.g. '_ha' or '_kgc'. Defaults to '_ha'.
        """

        mask = self.total_soc_inventory['fraction'].str.contains(selection)
        output = []
        for frac in fractions:
            if frac == 'new':
                new_soc = (
                    self.soc_inventory.tot_soc.sel(
                        {'region': sko, 'prod_system': system, 'scn': scenario}
                    ).where(mask, drop=True).sum('fraction').sum('input_year')
                )
                new_soc.plot()
                output.append(new_soc)
            if frac == 'hist':
                old_soc = (
                    self.historic_inventory.tot_soc.sel(
                        {'region': sko, 'prod_system': system, 'scn': scenario}
                    ).where(mask, drop=True).sum('fraction').sum('input_year')
                )
                old_soc.plot()
                output.append(old_soc)
            if frac == 'tot':
                total_soc = (
                    self.total_soc_inventory.tot_soc.sel(
                        {'region': sko, 'prod_system': system, 'scn': scenario}
                    ).where(mask, drop=True).sum('fraction').sum('input_year')
                )
                total_soc.plot()
                output.append(total_soc)
        return output

    def plot_all_regions_map(self,
                             reg='sko',
                             system='conventional',
                             vers='_ha',
                             initial_year = '2020',
                             final_year = '2050',
                             **kwargs):

        scenario = self.scenario.lower()
        mask = self.total_soc_inventory['fraction'].str.contains(vers)

        initial_soc_level = (self.total_soc_inventory.tot_soc.
                             sel({'output_year': initial_year, 'prod_system': system, 'scn': scenario}).
                             where(mask, drop=True).
                             sum(['fraction', 'input_year', 'output_year']))
        final_soc_level = (self.total_soc_inventory.tot_soc.
                           sel({'output_year': final_year, 'prod_system': system, 'scn': scenario}).
                           where(mask, drop=True).
                           sum('fraction').sum('input_year').sum('output_year'))

        stock_change = (final_soc_level / initial_soc_level - 1)
        #regions = stock_change.region.data
        #soc_change = stock_change.data
        stock_change_series = pd.Series(stock_change.data, index=stock_change.region.data.astype(str))
        stock_change_series.index.name = 'region'
        stock_change_series.name = 'values'
        print(stock_change_series)
        #stock_change_series = stock_change_df.loc[:, 'SOC stock change'] * 100
        a, b, c =plot.maps.map_from_soilseries(stock_change_series, reg, **kwargs)
        return a, b, c


class SoilDataExplore:
    """Class to explore and visualize the data related to soil carbon and CO2 fluxes produced with a SoilData class instance"""

    def __init__(self, scenario_prefix: str, import_path: str = temp_path, verbose: bool = False) -> None:
        # Initialize instance variables
        self.initialize_instance_variables(verbose)
        self.load_instance_state(scenario_prefix, import_path, verbose)
        self.load_inventory(scenario_prefix, dataset=['inputs', 'soc', 'historic', 'total_soc'], verbose=verbose)

        if verbose:
            print(f'Initialization of {self.name} instance of SoilData class complete')


    def initialize_instance_variables(self, verbose=False):
        if verbose:
            print('---Initializing variables---')
        # Initialize output variables to None or appropriate default values
        self.name = None # type: string
        self.scenario = None # type: string
        self.startyear = None  # type: int
        self.input_inventory = None  # type: xr.Dataset
        self.soc_inventory = None  # type: xr.Dataset
        self.historic_inventory = None  # type: xr.Dataset
        self.total_soc_inventory = None # type: xr.Dataset
        if verbose:
            print('+++Variables initialized---')


    def load_instance_state(self, scenario_name: str, temp_path: str=temp_path, verbose: bool=False) -> None:
        if verbose:
            print('---executing load_instance_state----')
        with open(f'{temp_path}/{scenario_name}.pickle', 'rb') as file:
            instance = pickle.load(file)
        self.name = instance.name
        self.scenario = instance.scenario
        self.startyear = instance.startyear
        if verbose:
            print('+++load_instance_state finished+++')


    def load_inventory(self, prefix: str, dataset: list=None, temp_path: str=temp_path, verbose: bool=False) -> None:
        """
        Loads inventory data based on specified dataset names.

        This method supports loading single datasets, a list of datasets, or all datasets if none specified.
        It updates the instance attributes for input, SOC (Soil Organic Carbon), and historic data inventories
        along with their corresponding DataFrames from CSV files.

        Parameters:
        -----------
        - dataset: Optional[str, list] - Name(s) of the dataset(s) to be loaded ('inputs', 'soc', 'historic', 'total_soc').
                                          If None, all datasets will be loaded.
        """
        if verbose:
            print('---Loading inventory---')
        ds_loaded = []  # Tracks loaded xarray datasets
        # Load specific datasets or all if none specified
        if dataset is None:
            dataset = ['inputs', 'soc', 'historic', 'total_soc']
            for ds in dataset:
                self.load_inventory(prefix, ds)
        elif isinstance(dataset, list):  # Recursively load each dataset in list
            for ds in dataset:
                self.load_inventory(prefix, ds)
        else:  # Load individual dataset
            if dataset == 'inputs':
                # Load 'inputs' dataset
                try:
                    scn_input_ds_name = f'{prefix}_input_ds'
                    self.input_inventory = xr.load_dataset(f"{temp_path}/{scn_input_ds_name}.nc")
                    ds_loaded.append('input_inventory')
                except(FileNotFoundError):
                    pass
            elif dataset == 'soc':
                # Load 'soc' dataset
                try:
                    scn_soc_ds_name = f'{prefix}_soc_ds'
                    self.soc_inventory = xr.load_dataset(f"{temp_path}/{scn_soc_ds_name}.nc")
                    ds_loaded.append('soc_inventory')
                except(FileNotFoundError):
                    pass
            elif dataset == 'historic':
                try:
                    # Load 'historic' dataset
                    scn_historic_ds_name = f'{prefix}_historic_ds'
                    self.historic_inventory = xr.load_dataset(f'{temp_path}/{scn_historic_ds_name}.nc')
                    ds_loaded.append('historic_inventory')
                except(FileNotFoundError):
                    pass
            elif dataset == 'total_soc':
                try:
                    # Load 'total_soc' dataset
                    scn_total_soc_ds_name = f'{prefix}_total_soc_ds'
                    self.total_soc_inventory = xr.load_dataset(f'{temp_path}/{scn_total_soc_ds_name}.nc')
                    ds_loaded.append('total_soc_inventory')
                except(FileNotFoundError):
                    pass
        # Print loaded datasets
        self._print_loaded_items(ds_loaded, 'dataset')
        if verbose:
            print('+++Inventory loaded+++')



    def _print_loaded_items(self, items, item_type):
        """
        Helper method to print loaded items.

        Parameters:
        -----------
        - items: list - A list of loaded item names.
        - item_type: str - A description of the item type ('dataset' or 'dataframe').
        """
        newline = '\n '
        if items:
            print(f"The following {item_type} variables have been set:{newline}{newline.join(items)}")
        else:
            print(f"No {item_type} variables have been loaded.")


    def _load_csv_files(self, temp_path: str, csv_names: list, loaded_list: list) -> None:
        """
        Helper method to load CSV files as DataFrames if they exist.

        Parameters:
        -----------
        - temp_path: str - The path where CSV files are located.
        - csv_names: list - A list of CSV file base names to load.
        - loaded_list: list - A list to append the names of successfully loaded CSV files.
        """
        for csv_name in csv_names:
            scn_input_df_name = f'{self.scenario}_{csv_name}'
            csv_path = f'{temp_path}/{scn_input_df_name}.csv'
            if os.path.exists(csv_path):
                setattr(self, csv_name, soil_utils.read_csv_preserved(csv_path))
                loaded_list.append(csv_name)


    def check_attributes_status(self, access: str='public') -> tuple[str, dict[str, tuple[type, bool]]]:
        """
        Checks the status of all attributes in the given class instance.

        Parameters:
        -----------
        - access: Selects whether to show 'public' or 'private' attributes. (default: 'public')

        Returns:
        --------
        A dictionary where keys are attribute names and values are tuples containing
        the current type of the attribute and a boolean indicating if it is set (not None).
        """
        attrs_status = {}
        for attribute in dir(self):
            # Filter based on access level
            if access == 'public' and attribute.startswith('_'):
                continue
            elif access == 'private' and not attribute.startswith('_'):
                continue
            attr_value = getattr(self, attribute)
            # Skip methods and magic methods
            if inspect.ismethod(attr_value) or attribute.startswith('__'):
                continue
            # Check if the attribute is set (not None)
            is_set = attr_value is not None
            attrs_status[attribute] = (type(attr_value).__name__, is_set)
        return f"The following attributes are set for {access} variables", attrs_status


    # Methods to print status messages
    def print_public_parameter_status(self):
        """
        Prints the current status of public attributes for the instance, specifically for the scenario set.

        This method retrieves the public attributes of the instance as determined by the scenario configuration. It formats
        the attribute names and their values neatly for display. The method is useful for debugging and verification purposes,
        allowing users to quickly check the initialization and current state of private attributes.

        Attributes are obtained by calling `check_attributes_status` with 'private' as the argument, which should return a
        dictionary of private attribute names and their current values.

        No parameters are required for this method.

        Returns:
        --------
           None. This method prints the status of private attributes directly to the console.
        """
        col1_length = len(max(self.check_attributes_status('public')[1].keys())) + 2
        attribute = ' \n'.join([f"{key}: {' ' * (col1_length - len(key))}{value}" for key, value in
                                self.check_attributes_status('public')[1].items()])
        print(f"The following public attributes have been set for {self.scenario}:\n{attribute}")


    def print_private_parameter_status(self):
        """
        Prints the current status of private attributes for the instance, specifically for the scenario set.

        This method retrieves the private attributes of the instance as determined by the scenario configuration. It formats
        the attribute names and their values neatly for display. The method is useful for debugging and verification purposes,
        allowing users to quickly check the initialization and current state of private attributes.

        Attributes are obtained by calling `check_attributes_status` with 'private' as the argument, which should return a
        dictionary of private attribute names and their current values.

        No parameters are required for this method.

        Returns:
        --------
           None. This method prints the status of private attributes directly to the console.
        """
        col1_length = len(max(self.check_attributes_status('private')[1].keys())) + 2
        attribute = ' \n'.join([f"{key}: {' ' * (col1_length - len(key))}{value}" for key, value in
                                self.check_attributes_status('private')[1].items()])
        print(f"The following private attributes have been set for {self.scenario}:\n{attribute}")


    def show_variable_values(self, variable_name='fraction'):
        return(list(self.total_soc_inventory[variable_name].values))


    # Plotting functions
    def plot_single_region_timeseries(self,
                                      fractions: List[str] = ['new', 'hist', 'tot'],
                                      sko: int = 111,
                                      system: str = 'conventional',
                                      scenario: str = 'fai',
                                      selection: str = '_ha',
                                      plot_config:  dict[str, Any]={'label': ['New SOC', 'Historic SOC', 'Total SOC']},
                                      label_config: dict[str, Any]={'xlabel': 'Time [Year]',
                                                                    'ylabel': 'SOC content [kg C]'},
                                      save_as=False,
                                      save_path=export_path,
                                      ) -> None:
        """
        Plots time series of SOC inventory data for a specified region, system, and scenario,
        allowing comparison across old and new SOC.

        This method aggregates SOC data for specified fractions and plots them as time series.
        It supports custom plot configurations and labeling, facilitating detailed analysis
        and visualization of SOC changes over time.

        Parameters:
        -----------
        - fractions (List[str]): Types of SOC fractions to plot, defaulting to ['new', 'hist', 'tot'].
        - sko (int): Region code for the data, default is 111.
        - system (str): Production system type, default is 'conventional'.
        - scenario (str): Scenario name, default is 'fai'.
        - selection (str): Area selection for plotting ('_ha' or '_kgc'), default is '_ha'.
        - plot_config (dict[str, Any]): Configuration for plot appearance, including labels.
        - label_config (dict[str, Any]): Configuration for axis labels and plot title.
        - save_as (str): Filename to save the plot, if specified; otherwise, no file is saved.
        - save_path (str): Path where the plot will be saved if `save_as` is specified.

        Returns:
        --------
        - A list of xarray.DataArray objects containing aggregated SOC data for the specified fractions.

        Note: The function adjusts the plot title based on the `selection` parameter to reflect the
        chosen area selection. Custom labels and titles can be specified through `label_config` and `plot_config`.
        """


        mask = self.total_soc_inventory['fraction'].str.contains(selection)
        fig, ax = plt.subplots()
        output = []
        if selection == '_ha':
            variant = 'HA'
        elif selection == '_kgc':
            variant = 'SKO'
        else:
            variant = ''

        label_config['title'] = f'SOC time series per {variant}. \nRegion {sko}, System: {system}'
        for frac in fractions:
            if frac == 'new':
                new_soc = (
                    self.soc_inventory.tot_soc.sel(
                        {'region': sko, 'prod_system': system, 'scn': scenario}
                    ).where(mask, drop=True).sum('fraction').sum('input_year')
                )
                output.append(new_soc)
            if frac == 'hist':
                old_soc = (
                    self.historic_inventory.tot_soc.sel(
                        {'region': sko, 'prod_system': system, 'scn': scenario}
                    ).where(mask, drop=True).sum('fraction').sum('input_year')
                )
                output.append(old_soc)
            if frac == 'tot':
                total_soc = (
                    self.total_soc_inventory.tot_soc.sel(
                        {'region': sko, 'prod_system': system, 'scn': scenario}
                    ).where(mask, drop=True).sum('fraction').sum('input_year')
                )
                output.append(total_soc)
        xr_array_list_plotter(ax, [data for data in output], plot_kwargs=plot_config, label_kwargs=label_config)

        plt.show()
        if save_as:
            fig.savefig(f'{save_path}/{save_as}.svg', format='svg')
            fig.savefig(f'{save_path}/{save_as}.png', format='png')

        return output


    def plot_fraction_total_timeseries_comparison(self,
                                                  sko: int = 111,
                                                  system: str = 'conventional',
                                                  scenario: str = 'fai',
                                                  selection: str = '_ha',
                                                  fraction: str = '_ha',
                                                  fraction_name: str = 'fraction',
                                                  label_config: dict[str, Any] = {'xlabel': 'Time [Year]',
                                                                                  'ylabel': 'SOC content [kg C]'},
                                                  min_year: Union[int, None] = None,
                                                  max_year: Union[int, None] = None,
                                                  save_as=False,
                                                  save_path=export_path,
                                                  ) -> None:
        """
        Plots the SOC inventory data for a single region or per ha, based on specified criteria.

        This method iterates through a given list of fraction types and aggregates
        the SOC data across all fractions and input years for the specified region (`sko`),
        production  system (`system`), and scenario (`scenario`). It then plots this aggregated
        data. The method is intended for use with 'new', 'historical' ('hist'), and
        'total' ('tot') SOC data.

        Parameters:
        -----------
        - fractions: The types of fractions to plot. Defaults to ['new', 'hist', 'tot'].
        - sko: The region code to plot data for. Defaults to 111.
        - system: The production system, e.g., 'conventional'. Defaults to 'conventional'.
        - scenario: The scenario name, e.g., 'fai'. Defaults to 'fai'.
        - selection: The area selection to be plotted, e.g. '_ha' or '_kgc'. Defaults to '_ha'.
        """

        mask1 = self.total_soc_inventory['fraction'].str.contains(selection)

        output = []
        total_soc = (
            self.total_soc_inventory.tot_soc.sel(
                {'region': sko, 'prod_system': system, 'scn': scenario}
            ).where(mask1, drop=True).sum('fraction').sum('input_year')
        )
        output.append(total_soc)

        mask2 = self.total_soc_inventory['fraction'].str.contains(fraction)
        fraction_soc = (
            self.total_soc_inventory.tot_soc.sel(
                {'region': sko, 'prod_system': system, 'scn': scenario}
            ).where(mask2, drop=True).sum('fraction').sum('input_year')
        )
        output.append(fraction_soc)


        fig, ax = plt.subplots()

        if selection == '_ha':
            variant = 'HA'
        elif selection == '_kgc':
            variant = 'SKO'
        else:
            variant = ''

        label_config['title'] = f'SOC time series per {variant}. \nRegion {sko}, System: {system}'
        plot_config = {'label': ['Total', f'{fraction_name}']}

        xr_array_list_plotter(ax, [data for data in output], min_year=min_year, max_year=max_year, plot_kwargs=plot_config, label_kwargs=label_config)

        plt.show()

        if save_as:
            fig.savefig(f'{save_path}/{save_as}.svg', format='svg')
            fig.savefig(f'{save_path}/{save_as}.png', format='png')

        return output

    def plot_all_regions_map(self,
                             reg: str = 'sko',
                             system: str = 'conventional',
                             vers: str = '_ha',
                             initial_year: str = '2020',
                             final_year: str = '2050',
                             percentage: bool = True,
                             colormap='YlOrBr_r',
                             width: float = 10,
                             height_scaling: float = 1.3,
                             standard_font: int = 15,
                             save_as: Optional[str] = None,
                             save_path: str = export_path,
                             **kwargs: Dict[str, Any]) -> pd.Series:
        """
        Generates a map visualizing the change in soil organic carbon (SOC) stocks across all regions between two specified years.

        Parameters:
        -----------
        - reg (str): Geographic region type for mapping ('sko', 'po8', 'kommun', 'län'). Default is 'sko'.
        - system (str): Farming system type ('conventional', 'organic', etc.). Default is 'conventional'.
        - vers (str): Version or type of SOC stocks data to plot, typically indicating the data source or calculation method. Default is '_ha'.
        - initial_year (str): The starting year for SOC stocks comparison. Default is '2020'.
        - final_year (str): The ending year for SOC stocks comparison. Default is '2050'.
        - percentage (bool): If True, displays SOC stock changes as percentages. Otherwise, displays raw values. Default is True.
        - width (float): Width of the figure in inches. Default is 10.
        - height_scaling (float): Factor to scale figure height based on the width, maintaining aspect ratio. Default is 1.3.
        - standard_font (int): Base font size for plot text elements. Scales with figure width. Default is 15.
        - save_as (Optional[str]): Filename to save the plot as SVG and PNG. If None, the plot is not saved. Default is None.
        - save_path (str): Directory path to save the plot if 'save_as' is provided. Uses 'export_path' by default.
        - **kwargs (Dict[str, Any]): Additional keyword arguments passed to the plotting function.

        Returns:
        --------
        pd.Series: A pandas Series containing the calculated SOC stock changes for each region.

        This method visualizes SOC stock changes across regions, providing insights into environmental and land management impacts over time.

        Note:
                * There is currently no mapping between 'sko' and other regions.
                  Changing the 'reg' parameter will cause the method to throw an error.
                * Changing 'width' and 'standard_font' does not affect the legend and its label, only the map and title.
                  To find a good balane between map and legen, trial and error using these two parameters is recommended.
        """
        standard_width = 10
        scaling_factor = width / standard_width
        height = width * height_scaling
        font_size = scaling_factor * standard_font

        scenario = self.scenario.lower()
        mask = self.total_soc_inventory['fraction'].str.contains(vers)

        initial_soc_level = (self.total_soc_inventory.tot_soc.
                             sel({'output_year': initial_year, 'prod_system': system, 'scn': scenario}).
                             where(mask, drop=True).
                             sum(['fraction', 'input_year', 'output_year']))
        final_soc_level = (self.total_soc_inventory.tot_soc.
                           sel({'output_year': final_year, 'prod_system': system, 'scn': scenario}).
                           where(mask, drop=True).
                           sum('fraction').sum('input_year').sum('output_year'))

        stock_change = (final_soc_level / initial_soc_level - 1)
        # regions = stock_change.region.data
        # soc_change = stock_change.data
        if percentage:
            stock_change_series = pd.Series(stock_change.data * 100, index=stock_change.region.data.astype(str))
        else:
            stock_change_series = pd.Series(stock_change.data, index=stock_change.region.data.astype(str))
        stock_change_series.index.name = 'region'
        stock_change_series.name = 'values'
        # stock_change_series = stock_change_df.loc[:, 'SOC stock change'] * 100
        fig, ax = plt.subplots(figsize=(width, height))
        plot.maps.map_from_soilseries(ax, stock_change_series, reg, font_size=font_size, cmap=colormap, **kwargs)
        plt.show()
        if save_as:
            fig.savefig(f'{save_path}/{save_as}.svg', format='svg')
            fig.savefig(f'{save_path}/{save_as}.png', format='png')
        return stock_change_series


    def plot_soc_co2_regions_map(self,
                                 reg: str = 'sko',
                                 system: str = 'conventional',
                                 vers: str = '_ha',
                                 initial_year: str = '2020',
                                 final_year: str = '2050',
                                 colormaps: List[str] = None,
                                 percentage: bool = True,
                                 width: float = 10,
                                 height_scaling: float = 1.3,
                                 standard_font: int = 15,
                                 save_as: bool = False,
                                 save_path: str = export_path,
                                 verbose: bool = False
                                 ) -> Tuple[pd.Series, pd.Series]:
        """
        Plots two side-by-side maps showing changes in soil organic carbon (SOC) stocks and cumulative CO2 emissions betweem specified initial and final years, allowing for visual comparison of changes over time within regions.

        Parameters:
        -----------
        - reg: Specifies the geographic region type for mapping ('sko', 'po8', 'kommun', 'län'). Default is 'sko'.
        - system: Farming system type ('conventional', 'organic', etc.). Default is 'conventional'.
        - vers: Version or type of SOC stocks data to plot, typically indicating the data source or calculation method. Default is '_ha'.
        - initial_year: The initial year for the SOC stocks and CO2 emissions comparison. Default is '2020'.
        - final_year: The final year for the SOC stocks and CO2 emissions comparison. Default is '2050'.
        - percentage: If True, SOC stock changes are displayed as percentages; otherwise, raw values are used. Default is True.
        - width: Width of the figure in inches. Default is 10.
        - height_scaling: Scaling factor to determine figure height based on width to maintain aspect ratio. Default is 1.3.
        - standard_font: Base font size for plot text elements. Actual font size scales with figure width. Default is 15.
        - save_as: If True, specifies the filename to save the plot as SVG and PNG. Default is False.
        - save_path: Directory path where the plot will be saved if save_as is True. Uses 'export_path' by default.
        - verbose: If True, prints additional information during plotting. Default is False.

        Returns:
        --------
        Tuple[pd.Series, pd.Series]: A tuple containing two pandas Series with SOC stock changes and CO2 emissions data, indexed by region.

        This method visualizes the distribution and changes in SOC stocks and CO2 emissions across specified regions for two points in time,
         offering insights into soil organic carbon stock evolution, and environmental impact trends under different farming systems.

        Note:
                * There is currently no mapping between 'sko' and other regions.
                  Changing the 'reg' parameter will cause the method to throw an error.
                * Changing 'width' and 'standard_font' does not affect the legend and its label, only the map and title.
                  To find a good balane between map and legen, trial and error using these two parameters is recommended.
        """
        standard_width = 10
        scaling_factor = width / standard_width
        height = width * height_scaling
        font_size = scaling_factor * standard_font

        scenario = self.scenario.lower()
        # Extract the correct soc data arrays
        mask = self.total_soc_inventory['fraction'].str.contains(vers)

        if system == 'all':
            initial_soc_level = (self.total_soc_inventory.tot_soc.
                                 sel({'output_year': initial_year, 'scn': scenario}).
                                 where(mask, drop=True).
                                 sum(['fraction', 'input_year', 'output_year', 'prod_system']))
            final_soc_level = (self.total_soc_inventory.tot_soc.
                               sel({'output_year': final_year, 'scn': scenario}).
                               where(mask, drop=True).
                               sum(['fraction', 'input_year', 'output_year', 'prod_system']))

        else:
            initial_soc_level = (self.total_soc_inventory.tot_soc.
                                 sel({'output_year': initial_year, 'prod_system': system, 'scn': scenario}).
                                 where(mask, drop=True).
                                 sum(['fraction', 'input_year', 'output_year']))
            final_soc_level = (self.total_soc_inventory.tot_soc.
                               sel({'output_year': final_year, 'prod_system': system, 'scn': scenario}).
                               where(mask, drop=True).
                               sum('fraction').sum('input_year').sum('output_year'))

        # calculate and create the dataseries with soc stock change and co2 emissions
        stock_change = (final_soc_level / initial_soc_level - 1)
        co2_emissions = (initial_soc_level - final_soc_level) * 3.6
        if percentage:
            stock_change_series = pd.Series(stock_change.data * 100, index=stock_change.region.data.astype(str))
            co2_emissions_series = pd.Series(co2_emissions.data, index=co2_emissions.region.data.astype(str))
        else:
            stock_change_series = pd.Series(stock_change.data, index=stock_change.region.data.astype(str))
            co2_emissions_series = pd.Series(co2_emissions.data, index=co2_emissions.region.data.astype(str))
        stock_change_series.index.name = 'region'
        stock_change_series.name = 'values'
        co2_emissions_series.index.name = 'region'
        co2_emissions_series.name = 'values'
        # calculate the total area changed per region dataseries
        if system == 'all':
            area1 = self.input_inventory.area_ha.sel({'input_year': initial_year}).sum(['scn', 'crop', 'input_year', 'prod_system'])
            area2 = self.input_inventory.area_ha.sel({'input_year': final_year}).sum(['scn', 'crop', 'input_year', 'prod_system'])
        else:
            area1 = self.input_inventory.area_ha.sel({'input_year': initial_year, 'prod_system': system}).sum(['scn', 'crop', 'input_year'])
            area2 = self.input_inventory.area_ha.sel({'input_year': final_year, 'prod_system': system}).sum(['scn', 'crop', 'input_year'])
        area_change = area2-area1
        area_change_series  = pd.Series(area_change, index=area_change.region.data.astype(str))
        area_change_series.index.name = 'region'
        area_change_series.name = 'values'
        # calculate the percentage area change per region dataseries
        area_change_fraction = area2 / area1 - 1
        area_change_perc_series = pd.Series(area_change_fraction * 100, index=area_change_fraction.region.data.astype(str))
        area_change_perc_series.index.name = 'region'
        area_change_perc_series.name = 'values'

        # calculate totals for Sweden
        mask_swe = self.total_soc_inventory['fraction'].str.contains('_kgc')

        if system == 'all':
            initial_soc_level_swe = (self.total_soc_inventory.tot_soc.
                                     sel({'output_year': initial_year, 'scn': scenario}).
                                     where(mask_swe, drop=True).
                                     sum(['fraction', 'input_year', 'output_year', 'region', 'prod_system']))
            final_soc_level_swe = (self.total_soc_inventory.tot_soc.
                                   sel({'output_year': final_year, 'scn': scenario}).
                                   where(mask_swe, drop=True).
                                   sum(['fraction', 'input_year', 'output_year', 'region', 'prod_system']))
        else:
            initial_soc_level_swe = (self.total_soc_inventory.tot_soc.
                                     sel({'output_year': initial_year, 'prod_system': system, 'scn': scenario}).
                                     where(mask_swe, drop=True).
                                     sum(['fraction', 'input_year', 'output_year', 'region']))
            final_soc_level_swe = (self.total_soc_inventory.tot_soc.
                                   sel({'output_year': final_year, 'prod_system': system, 'scn': scenario}).
                                   where(mask_swe, drop=True).
                                   sum(['fraction', 'input_year', 'output_year', 'region']))

        absolute_CO2_emissions_swe = (initial_soc_level_swe - final_soc_level_swe) * 3.6
        relative_soc_stock_change_swe = (final_soc_level_swe / initial_soc_level_swe) - 1

        area1_swe = area1.sum()
        area2_swe = area2.sum()

        absolute_area_change_swe = area2_swe - area1_swe
        relative_area_change_swe = area2_swe / area1_swe

        ## plotting
        # stock_change_series = stock_change_df.loc[:, 'SOC stock change'] * 100
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(width, height))

        colormap = ['YlOrBr_r', 'YlGnBu', 'Spectral', 'Spectral']
        if colormaps is not None:
            if colormaps[0] != '':
                colormap[0] = colormaps[0]
            if colormaps[1] != '':
                colormap[1] = colormaps[1]
            if colormaps[2] != '':
                colormap[2] = colormaps[2]
            if colormaps[3] != '':
                colormap[3] = colormaps[3]

        kwargs1 = {'title': f'SOC stock changes {initial_year}-{final_year}\n{system} agriculture', 'cmap': colormap[0],
                   'legend_kwds':{'label': 'Percentage change in SOC stocks, top soil (%)'}}
        kwargs2 = {'title': f'CO2-emissions {initial_year}-{final_year}\n{system} agriculture', 'cmap': colormap[1],
                   'legend_kwds':{'label': 'Cumulative CO2 emissions, top soil [kg]'}}
        kwargs3 = {'title': f'Agricultural land use change {initial_year}-{final_year}\n{system} agriculture', 'cmap': colormap[2],
                   'legend_kwds': {'label': f'Change in hectares (ha) of {system} agricultural land'}}
        kwargs4 = {'title': f'Agricultural land use change {initial_year}-{final_year}\n{system} agriculture', 'cmap': colormap[3],
                   'legend_kwds': {'label': f'Change in percentage (%) of {system} agricultural land'}}

        plot.maps.map_from_soilseries(axs[0, 0], stock_change_series, reg, font_size=font_size, verbose=verbose, **kwargs1)
        plot.maps.map_from_soilseries(axs[0, 1], co2_emissions_series, reg, font_size=font_size, verbose=verbose, **kwargs2)
        plot.maps.map_from_soilseries(axs[1, 0], area_change_series, reg, font_size=font_size, verbose=verbose, **kwargs3)
        plot.maps.map_from_soilseries(axs[1, 1], area_change_perc_series, reg, font_size=font_size, verbose=verbose, **kwargs4)
        plt.tight_layout
        plt.show()
        if save_as:
            fig.savefig(f'{save_path}/{save_as}.svg', format='svg')
            fig.savefig(f'{save_path}/{save_as}.png', format='png')

        output_dict = {'soc_stock_change_series': stock_change_series,
                       'co2_emission_series': co2_emissions_series,
                       'area_change_series': area_change_series,
                       'area_change_perc_series': area_change_perc_series,
                       'absolute_CO2_emission_swe': absolute_CO2_emissions_swe,
                       'relative_soc_stock_change_swe': relative_soc_stock_change_swe,
                       'absolute_area_change_swe': absolute_area_change_swe,
                       'relative_area_change_swe': relative_area_change_swe
                       }
        return output_dict


    def plot_soc_stock_maps(self,
                            reg: str = 'sko',
                            system: str = 'conventional',
                            vers: str = '_ha',
                            initial_year: str = '2020',
                            final_year: str = '2050',
                            colormap='YlOrBr_r',
                            width: float = 10,
                            height_scaling: float = 1.3,
                            standard_font: int = 15,
                            save_as: str = False,
                            save_path: str = export_path,
                            verbose: bool = False,
                            **kwargs) -> Tuple[pd.Series, pd.Series]:
        """
        Plots two side-by-side maps showing soil organic carbon (SOC) stocks for two given years (initial and final),
        allowing for visual comparison of SOC distribution changes over time within specified regions.

        Parameters:
        -----------
        - reg (str): Specifies the geographic region type for mapping ('sko', 'po8', 'kommun', 'län'). Default is 'sko'.
        - system (str): Farming system type ('conventional', 'organic', etc.). Default is 'conventional'.
        - vers (str): Version or type of SOC stocks data to plot, typically indicating the data source or calculation method. Default is '_ha'.
        - initial_year (str): The year for the first SOC stock map. Default is '2020'.
        - final_year (str): The year for the second SOC stock map. Default is '2050'.
        - width (float): Width of the figure in inches. Default is 10.
        - height_scaling (float): Scaling factor to determine figure height based on width to maintain aspect ratio. Default is 1.3.
        - standard_font (int): Base font size for plot text elements. Actual font size scales with figure width. Default is 15.
        - save_as (bool or str): If not False, specifies the filename to save the plot as an SVG. Default is False.
        - save_path (str): Directory path where the plot will be saved if save_as is specified. Default uses 'export_path'.
        - verbose (bool): If True, prints additional information during plotting. Default is False.

        Returns:
        --------
        - initial_soc_stock (pd.Series): A pandas Series containing SOC stocks data for the initial year, indexed by region.
        - final_soc_stock (pd.Series): A pandas Series containing SOC stocks data for the final year, indexed by region.

        This method visualizes the distribution and changes in SOC stocks across specified regions for two points in time,
        offering insights into soil health and carbon sequestration trends under different farming systems.

        Note:
            * There is currently no mapping between 'sko' and other regions.
              Changing the 'reg' parameter will cause the method to throw an error.
            * Changing 'width' and 'standard_font' does not affect the legend and its label, only the map and title.
              To find a good balane between map and legen, trial and error using these two parameters is recommended.
        """

        standard_width = 10
        scaling_factor = width / standard_width
        height = width * height_scaling
        font_size = scaling_factor * standard_font

        scenario = self.scenario.lower()
        mask = self.total_soc_inventory['fraction'].str.contains(vers)

        if system == 'both':
            initial_org_soc_level = (self.total_soc_inventory.tot_soc.
                                     sel({'output_year': initial_year, 'prod_system': 'organic', 'scn': scenario}).
                                     where(mask, drop=True).
                                     sum(['fraction', 'input_year', 'output_year']))
            final_org_soc_level = (self.total_soc_inventory.tot_soc.
                                   sel({'output_year': final_year, 'prod_system': 'organic', 'scn': scenario}).
                                   where(mask, drop=True).
                                   sum('fraction').sum('input_year').sum('output_year'))
            initial_conv_soc_level = (self.total_soc_inventory.tot_soc.
                                      sel({'output_year': initial_year, 'prod_system': 'conventional', 'scn': scenario}).
                                      where(mask, drop=True).
                                      sum(['fraction', 'input_year', 'output_year']))
            final_conv_soc_level = (self.total_soc_inventory.tot_soc.
                                    sel({'output_year': final_year, 'prod_system': 'conventional', 'scn': scenario}).
                                    where(mask, drop=True).
                                    sum('fraction').sum('input_year').sum('output_year'))

            initial_org_soc_stock = pd.Series(initial_org_soc_level.data,
                                              index=initial_org_soc_level.region.data.astype(str))
            final_org_soc_stock = pd.Series(final_org_soc_level.data,
                                            index=final_org_soc_level.region.data.astype(str))
            initial_org_soc_stock.index.name = 'region'
            initial_org_soc_stock.name = 'values'
            final_org_soc_stock.index.name = 'region'
            final_org_soc_stock.name = 'values'

            initial_conv_soc_stock = pd.Series(initial_conv_soc_level.data,
                                              index=initial_conv_soc_level.region.data.astype(str))
            final_conv_soc_stock = pd.Series(final_conv_soc_level.data,
                                            index=final_conv_soc_level.region.data.astype(str))
            initial_conv_soc_stock.index.name = 'region'
            initial_conv_soc_stock.name = 'values'
            final_conv_soc_stock.index.name = 'region'
            final_conv_soc_stock.name = 'values'

            cbar_min = min(initial_org_soc_stock.min(),
                           final_org_soc_stock.min(),
                           initial_conv_soc_stock.min(),
                           final_conv_soc_stock.min())
            cbar_max = max(initial_org_soc_stock.max(),
                           final_org_soc_stock.max(),
                           initial_conv_soc_stock.max(),
                           final_conv_soc_stock.max())

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(width, height*2))

            kwargs1 = {'title': f'SOC stocks in {initial_year} - conventional\n(selection: {vers})',
                       'cmap': colormap,
                       'vmin': cbar_min,
                       'vmax': cbar_max,
                       'legend_kwds': {'label': 'SOC stocks [kg]'}}
            kwargs2 = {'title': f'SOC stocks in {final_year} - conventional\n(selection: {vers})',
                       'cmap': colormap,
                       'vmin': cbar_min,
                       'vmax': cbar_max,
                       'legend_kwds': {'label': 'SOC stocks [kg]'}}
            kwargs3 = {'title': f'SOC stocks in {initial_year} - organic\n(selection: {vers})',
                       'cmap': colormap,
                       'vmin': cbar_min,
                       'vmax': cbar_max,
                       'legend_kwds': {'label': 'SOC stocks [kg]'}}
            kwargs4 = {'title': f'SOC stocks in {final_year} - organic\n(selection: {vers})',
                       'cmap': colormap,
                       'vmin': cbar_min,
                       'vmax': cbar_max,
                       'legend_kwds': {'label': 'SOC stocks [kg]'}}

            kwargs1.update(kwargs)
            kwargs2.update(kwargs)
            kwargs3.update(kwargs)
            kwargs4.update(kwargs)

            # if kwargs.get('kind') is not None:
                # to_remove =['vmin', 'vmax']
                # if kwargs.get('legend') is None:
                #     to_remove.append('legend_kwds')
                # for key in to_remove:
                #     kwargs1.pop(key, None)
                #     kwargs2.pop(key, None)
                #     kwargs3.pop(key, None)
                #     kwargs4.pop(key, None)
            if kwargs.get('kind') in ['barh', 'hist']:
                axs[0, 0].set(xlim=[cbar_min, cbar_max])
                axs[0, 1].set(xlim=[cbar_min, cbar_max])
                axs[1, 0].set(xlim=[cbar_min, cbar_max])
                axs[1, 1].set(xlim=[cbar_min, cbar_max])

            plot.maps.map_from_soilseries(axs[0,0], initial_conv_soc_stock, reg, font_size=font_size, verbose=verbose, **kwargs1)
            plot.maps.map_from_soilseries(axs[0,1], final_conv_soc_stock, reg, font_size=font_size, verbose=verbose, **kwargs2)
            plot.maps.map_from_soilseries(axs[1,0], initial_org_soc_stock, reg, font_size=font_size, verbose=verbose, **kwargs3)
            plot.maps.map_from_soilseries(axs[1,1], final_org_soc_stock, reg, font_size=font_size, verbose=verbose, **kwargs4)
            plt.tight_layout
            plt.show()
        else:
            initial_soc_level = (self.total_soc_inventory.tot_soc.
                                 sel({'output_year': initial_year, 'prod_system': system, 'scn': scenario}).
                                 where(mask, drop=True).
                                 sum(['fraction', 'input_year', 'output_year']))
            final_soc_level = (self.total_soc_inventory.tot_soc.
                               sel({'output_year': final_year, 'prod_system': system, 'scn': scenario}).
                               where(mask, drop=True).
                               sum('fraction').sum('input_year').sum('output_year'))

            initial_soc_stock = pd.Series(initial_soc_level.data, index=initial_soc_level.region.data.astype(str))
            final_soc_stock = pd.Series(final_soc_level.data, index=final_soc_level.region.data.astype(str))
            initial_soc_stock.index.name = 'region'
            initial_soc_stock.name = 'values'
            final_soc_stock.index.name = 'region'
            final_soc_stock.name = 'values'

            cbar_min = min(initial_soc_stock.min(), final_soc_stock.min())
            cbar_max = max(initial_soc_stock.max(), final_soc_stock.max())

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(width, height))

            kwargs1 = {'title': f'SOC stocks in {initial_year}\n(selection: {vers})',
                       'cmap': colormap,
                       'vmin': cbar_min,
                       'vmax': cbar_max,
                       'legend_kwds':{'label': 'SOC stocks [kg]'}}
            kwargs2 = {'title': f'SOC stocks in {final_year}\n(selection: {vers})',
                       'cmap': colormap,
                       'vmin': cbar_min,
                       'vmax': cbar_max,
                       'legend_kwds':{'label': 'SOC stocks [kg]'}}

            kwargs1.update(kwargs)
            kwargs2.update(kwargs)

            if kwargs.get('kind') is True:
                for key in ['cmap', 'vmin', 'vmax']:
                    kwargs1.pop(key, None)
                    kwargs2.pop(key, None)

            plot.maps.map_from_soilseries(axs[0], initial_soc_stock, reg, font_size=font_size, verbose=verbose, **kwargs1)
            plot.maps.map_from_soilseries(axs[1], final_soc_stock, reg, font_size=font_size, verbose=verbose, **kwargs2)
            plt.tight_layout
            plt.show()
        if save_as:
            fig.savefig(f'{save_path}/{save_as}.svg', format='svg')
            fig.savefig(f'{save_path}/{save_as}.png', format='png')


    def plot_regions(self, region_type: str='sko'):
        '''
        Plots an interactive map of Sweden which shows the identification code used in the geodataframes and inventories when mouse is howvered over a region.

        Parameters
        ----------
        region_type: A string indicating the type of region to be plotted. Available options are 'sko' (default), 'po8, 'kommun' and 'Region'.

        Returns
        -------
        None: Outputs an interactive map to the notebook

        '''
        plot.maps.plot_regions(gdf_name=region_type)


    def outlier_detect(self, x: pd.Series,
                       y: pd.Series=None,
                       type: str='box',
                       xlabel='Unset label',
                       ylabel='Unset label',
                       threshold=3):
        '''
        Choose between a number of plots that can help detect the presence of outliers

        Parameters
        ----------
        x:      Main dataseries to be used for plotting. Mandatory
        y:      (optional) dataseries to be used for plotting in scatter plots
        type:   (optional) The name of the plot type to be plotted. Available options are 'box', 'scatter', 'z'.
        xlabel: (optional) The label for the x-axis in scatterplots.
        ylabel: (optional) The label for the x-axis in scatterplots.
        threshold: (optional) The value used to detect outliers with the Z-method (default: 3)

        Returns
        -------
        None: Generates a plot of the choosen type in the notebook
        '''
        if type == 'box':
            # Create a boxplot
            sns.boxplot(x=x)
        elif type == 'scatter':
            # Create a scatterplot
            if y is not None:
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.scatter(x, y, color='green')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.show()
            else:
                print("'A series is require for the 'y' coordinate when chosing type='scatter'")
        elif type == 'z':
            # Run a Z-test and a do a line plot
            # Z-score outlier detection
            threshold = threshold
            z = np.abs(stats.zscore(x))
            ar = np.where(z > threshold)[0]
            print(ar)
            for i in ar:
                print(
                    f'sko {z.index[i]} has a Z-score of {z.iloc[i]}')  # prints the sko no. and zscore of the identified outliers
                print(f"It's nominal value is {x.iloc[i]}")
            drop_map = z.index[ar]
            new_set = x.drop(drop_map)
            z_new = np.abs(stats.zscore(new_set))
            print(f"For comparison, the summary statistics of the remaining dataset is:\n{new_set.describe()}\nand it's Z-scores are:\n {z_new}")
            z.plot()
        else:
            print("The available options for type are 'box' and 'scatter")

def xr_array_list_plotter(ax, data, min_year=None, max_year=None, plot_kwargs={}, label_kwargs={}):
    for n, item in enumerate(data):
        # Ensure data_series is a pandas Series
        if isinstance(item, xr.DataArray):
            label = plot_kwargs['label'][n]
            local_plot_kwargs = plot_kwargs.copy()
            local_plot_kwargs['label'] = label
            # Plot using the series index as x-values and the series values as y-values
            ax.plot(item.output_year.data, item.data, **local_plot_kwargs)

    if min_year is not None:
        ax.set_xlim(left=min_year)
    if max_year is not None:
        ax.set_xlim(right=max_year)
    ax.set_xlabel(label_kwargs.get('xlabel', ''))
    ax.set_ylabel(label_kwargs.get('ylabel', ''))
    if 'title' in label_kwargs:
        ax.set_title(label_kwargs['title'])
    ax.legend()
    return ax