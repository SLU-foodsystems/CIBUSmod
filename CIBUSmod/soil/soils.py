#!/usr/bin/python3

"""This module contains the soil organic carbon data and methods.

Calculate the SOC data and related CO2 fluxes from input in
external modules.

The total CO2 fluxes and SOC data of a scenario are directly
accessible through a SoilData class instance after having run the
soil calculations using the SoilCalc class.

Each SoilData class instance contains the complete input_inventory of
C inputs, SOC time-series data and CO2 fluxes for a scenario as an
xarray inside a netcdf file
"""

import inspect
import os.path
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import CIBUSmod.soil.soil_utils as soil_utils
from CIBUSmod.soil.soil_params import C_CONTENT_CROPS
import CIBUSmod.soil.icbm_funcs as icbm_funcs
from ..soil import temp_path

class SoilData:
    """Class to calculate and keep track of the data related to soil carbon and CO2 fluxes in a given scenario"""

    def __init__(self, name: str, input_df: pd.DataFrame, scenario: str) -> None:
        # input variables
        self.name = name
        self.input_df = input_df
        self.scenario = scenario
        # output variables
        self.startyear = None # type: int
        self.ss_input_df = None  # type: pd.Dataframe
        self.soc_ha_df = None # type: pd.Dataframe
        self.soc_sko_df = None  # type: pd.Dataframe
        self.historic_ha_df = None  # type: pd.Dataframe
        self.historic_sko_df = None  # type: pd.Dataframe
        self.co2_fluxes = None  # type: np.ndarray
        self.input_inventory = None  # type: xr.Dataset
        self.soc_inventory = None # type: xr.Dataset
        self.historic_inventory = None  # type: xr.Dataset
        # help variables
        self._residue_col_name = None  # type: str
        self._input_grouped_df = None  # type: pd.Dataframe
        self._c_input_ha_df = None  # type: pd.Dataframe
        self._c_input_sko_df = None  # type: pd.Dataframe
        self._ss_input_ha_df = None  # type: pd.Dataframe
        self._ss_input_sko_df = None  # type: pd.Dataframe
        self._h_value_dict = None  # type: Dict
        self._scn_ha_sel = None # type: list
        self._scn_sko_sel = None  # type: list
        self._ss_ha_sel = None  # type: list
        self._ss_sko_sel = None  # type: list
        self._spinup_groupby_df = None # type: pd.DataFrame

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
        """Add input fraction prefixes to manure input"""
        if verbose:
            print(">>> Executing '_add_prefixes()' <<<")
        # Add prefixes to all the manure fractions so that all c input fractions follow the same naming convention
        self.input_df = soil_utils.add_prefix('manure', 'i_ag', self.input_df)
        if verbose:
            print(">>> '_add_prefixes()' executed succesfully <<<")
        

    def _change_year_name(self, new_name='input_year', verbose=False):
        """Change the column name "year" to 'new_name'(default="input_year")"""
        if verbose:
            print(">>> Executing '_change_year_name()'<<<")
        if isinstance(self.input_df.index, pd.MultiIndex):
            self.input_df.index.set_names(['input_year'], level='year', inplace=True)
        else:
            self.input_df = self.input_df.rename(columns={'year': new_name})
        if verbose:
            print(">>> '_change_year_name()' executed succesfully <<<")


    def _make_datetime(self, verbose=False):
        """Change the "input_year" column dtype to datetime64"""
        if verbose:
            print(">>> Executing '_make_datetime()'<<<")
        if isinstance(self.input_df.index, pd.MultiIndex):
            idx = list(self.input_df.index.names)
            self.input_df = self.input_df.reset_index()
            pd.to_datetime(self.input_df.input_year, format='%Y')
            self.input_df = self.input_df.set_index(idx)
            self.input_df.index.dtypes
        else:
            self.input_df.input_year = pd.to_datetime(self.input_df.input_year, format='%Y')
        if verbose:
            print(">>> '_make_datetime()' executed succesfully <<<")


    def _calc_input_ha(self, verbose=False):
        """
        Calculate the yield and input per unit area, given total yield ("harvest_kg_dm") and area ("area_ha")

        Also assign yield and input per sko
        """
        if verbose:
            print(">>> Executing '_calc_input_ha()'<<<")
        # Calculate yield per unit area
        self.input_df["areayield"] = self.input_df["harvest_kgdm"] / self.input_df["area_ha"] * C_CONTENT_CROPS
        if "crop_residues_harvest_kgdm" in self.input_df.columns:
            self.input_df["areayield_residues"] = self.input_df["crop_residues_harvest_kgdm"] / self.input_df["area_ha"] * C_CONTENT_CROPS
        else:
            self.input_df["areayield_residues"] = 0

        if verbose:
            print(">>> '_calc_input_ha()' executed succesfully <<<")


    def _calc_amnd_ha(self, verbose=False):
        """ Calculate the amnendment input per unit area, given total manure ("_kgc") and area ("area_ha")"""
        if verbose:
            print(">>> Executing '_calc_amnd_ha()'<<<")
        # Calculate and add columns with the manure input per ha
        tot_manure_cols = soil_utils.get_filtered_namelist(['kgc'], ['manure', 'crop'], self.input_df)
        newframe = self.input_df.copy()
        # insert new columns with the input per ha
        for i in tot_manure_cols:
            col_name = f'{i[:-4]}_ha'
            newframe[col_name] = newframe[i] / newframe['area_ha']
            if verbose:
                print(f"--- Inserted {col_name}---")        
        self.input_df = newframe.copy(deep=True)
        if verbose:
                print(">>> '_calc_amnd_ha()' executed succesfully <<<")


    def _make_multiindex(self, verbose=False):
        """create multiindex dataframe"""
        if verbose:
            print(">>> Executing '_make_multiindex()'<<<")
        scn_input_idx = ['scn', 'crop', 'prod_system', 'region', 'input_year']
        # Create a multiindex df
        self.input_df = self.input_df.set_index(scn_input_idx)
        if verbose:
                print(">>> '_make_multiindex()' executed succesfully <<<")


    def _calc_crop_inputs(self, straw_removed=False, verbose=False):
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
    
        self.input_df = soil_utils.calculate_c_inputs(self.input_df,
                                                      mapper,
                                                      sources,
                                                      allo_dfs,
                                                      straw_removed=straw_removed,
                                                      verbose=verbose)
        if verbose:
            print(">>> '_calc_crop_inputs()' executed succesfully <<<")


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

    
    def calc_scn_inputs(self,
                        new_name='input_year',
                        verbose=False
                        ):
        """
        Calculate the carbon inputs used for soil organic carbon modelling in CIBUSmod.

        Parameters:
        - new_name (str, optional): The new name for the 'input_year' column if 'year' is present.
          Defaults to 'input_year'.
        - save_inventory (bool, optional): If True, save the input_inventory dataset to a netcdf-file.
          Defaults to False.
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
                self.input_df.rename(columns={ self._residue_col_name: new_col_name}, inplace=True)
        #self.input_df = soil_utils.make_idx_continuous(self.input_df)
        self.input_df.reset_index(inplace=True)
        self.input_df = soil_utils.make_df_lower(self.input_df)
        if not isinstance(self.input_df, pd.DataFrame):
            return print('> Input dataframe not set. Please supply a valid input dataframe and retry')
        self._calc_input_ha(verbose=verbose)
        self._calc_amnd_ha(verbose=verbose)
        self._calc_crop_inputs(verbose=verbose)
        self._add_prefixes(verbose=verbose)
        # make sure that the 'year' column name is unambiguous
        if isinstance(self.input_df.index, pd.MultiIndex):
            if 'input_year' not in self.input_df.index.names \
            and 'year' in self.input_df.index.names:
                self._change_year_name(new_name=new_name, verbose=verbose)
        else:
            if 'input_year' not in self.input_df.columns \
            and 'year' in self.input_df.columns:
                self._change_year_name(new_name=new_name, verbose=verbose)
        # make sure the 'input_year' column is in datetime64 format
        self._make_datetime(verbose=verbose)
        self._make_multiindex(verbose=verbose)
        # Set instance variables
        self.startyear = self.input_df.index.get_level_values('input_year').year[0]
        self.input_inventory = self.input_df.to_xarray()
        if verbose:
            print('info: Finished creating scenario input_df')
            print('    > Continuing with the creation of spinup input_df')
        self._make_spinup_df()
        if verbose:
            print('---calc_scn_inputs() executed successfully---')
        soil_utils.colored_rule(color='green', height=2)

    def _calculate_soc(self, save_ds=False, verbose=False):
        """
        Calculate the SOC pools for both the ha and sko dataframes
        """
        if verbose:
            print('---Executing _calculate_soc()---')
        if isinstance(self._c_input_ha_df, pd.DataFrame) and isinstance(self._c_input_sko_df, pd.DataFrame):
            print("'_c_input_ha_df' and '_c_input_sko_df' set. Continuing")
        else:
            self._make_scn_area_dfs(verbose=verbose)
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
        self._scn_ha_sel = list(set(soil_utils.get_filtered_namelist(['i_a', 'i_b', 'ha'], ['manure', 'crop'], self._c_input_ha_df)))
        if verbose:
            print("Extracting filtered_namelist per sko")
        self._scn_sko_sel = list(set(soil_utils.get_filtered_namelist(['i_a', 'i_b', 'sko'], ['manure', 'crop'], self._c_input_sko_df)))
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
        scenario_name = 'spinup_soc'
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
            # try:
            if not self._h_value_dict:
                temp, self._h_value_dict = soil_utils.h_map_helper(verbose=verbose)

        # extract a list of the spinup columns by which icbm is to be run
        if not self._ss_ha_sel:
            if verbose:
                print("Extracting filtered_namelist per ha")
            self._ss_ha_sel = list(set(soil_utils.get_filtered_namelist(['i_a', 'i_b'], ['manure', 'crop', 'ha'], self._ss_input_ha_df)))
        if not self._ss_sko_sel:
            if verbose:
                print("Extracting filtered_namelist per sko")
            self._ss_sko_sel = list(set(soil_utils.get_filtered_namelist(['i_a', 'i_b'], ['manure', 'crop', 'ha'], self._ss_input_sko_df)))
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

    def _make_scn_area_dfs(self, verbose=False):
        """Create scenario dataframes with yields per ha and per sko."""
        if verbose:
            print('---Executing _make_scn_area_dfs()---')
        scenario_name = self.scenario
        # Group the scn df by scn, prod system, region and year and calculate total input and area of all crops
        scn_multi_groupby_idx = ['scn', 'prod_system', 'region', 'input_year']
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

    def _make_spinup_area_dfs(self,verbose=False):
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


    def calc_soc_timeseries(self, verbose=False):
        """Calculate the SOC timeseries and create a soc_inventory dataset"""
        soil_utils.colored_rule(color='cyan', height=2)
        print('Calculating SOC timeseries...')
        self._calculate_soc(verbose=verbose)
        self._scn_icbm_calculations(verbose=verbose)
        soil_utils.colored_rule(color='green', height=2)

    def calc_historic_soc_timeseries(self, verbose=False):
        """Calculate the historic SOC timeseries and update the soc_inventory dataset"""
        soil_utils.colored_rule(color='cyan', height=2)
        print('Calculating historic SOC timeseries...')
        self._calculate_historic_soc(verbose=verbose)
        self._historic_icbm_calculations(verbose=verbose)
        soil_utils.colored_rule(color='green', height=2)


    def save_inventory(self, dataset=None):
        # check if dataset is set, otherwise set to all datasets
        if dataset is None:
            dataset = ['inputs', 'soc', 'historic']
            print(f'dataset set to {dataset}')
            for ds in dataset:
                self.save_inventory(ds)
        elif type(dataset) is list:
            for ds in dataset:
                self.save_inventory(ds)
        else:
            if dataset == 'inputs':
                # Save the input_inventory dataset to a netcdf-file
                scn_input_ds_name = f'{self.scenario}_input_ds'
                self.input_inventory.to_netcdf(f"{temp_path}/{scn_input_ds_name}.nc")
                if isinstance(self.ss_input_df, pd.DataFrame):
                    soil_utils.to_csv_preserved(self.ss_input_df,
                                                'ss_input_df',
                                                save_type='temp')
                print(f"Scenario dataset saved as {scn_input_ds_name}.nc in {temp_path}")
            if dataset == 'soc':
                # Save the input_inventory dataset to a netcdf-file
                scn_soc_ds_name = f'{self.scenario}_soc_ds'
                self.soc_inventory.to_netcdf(f"{temp_path}/{scn_soc_ds_name}.nc")
                if isinstance(self.soc_ha_df, pd.DataFrame):
                    soil_utils.to_csv_preserved(self.soc_ha_df,
                                                'soc_ha_df',
                                                save_type='temp')
                if isinstance(self.soc_sko_df, pd.DataFrame):
                    soil_utils.to_csv_preserved(self.soc_sko_df,
                                                'soc_sko_df',
                                                save_type='temp')
                print(f"SOC dataset saved as {scn_soc_ds_name}.nc in {temp_path}")
            if dataset == 'historic':
                self.historic_inventory.to_netcdf(f'{temp_path}/historic_soc_ds.nc')
                if isinstance(self.historic_ha_df, pd.DataFrame):
                    soil_utils.to_csv_preserved(self.historic_ha_df,
                                                'historic_ha_df',
                                                save_type='temp')
                if isinstance(self.historic_sko_df, pd.DataFrame):
                    soil_utils.to_csv_preserved(self.historic_sko_df,
                                                'historic_sko_df',
                                                save_type='temp')
                print(f"Historic SOC dataset saved as historic_soc_ds.nc \n in {temp_path}")

    def load_inventory(self, dataset=None):
        # check if dataset is set, otherwise set to all datasets
        if dataset is None:
            dataset = ['inputs', 'soc', 'historic']
            print(f'dataset set to {dataset}')
            for ds in dataset:
                self.load_inventory(ds)
        elif type(dataset) is list:
            for ds in dataset:
                self.load_inventory(ds)
        else:
            if dataset == 'inputs':
                scn_input_ds_name = f'{self.scenario}_input_ds'
                self.input_inventory = xr.load_dataset(f"{temp_path}/{scn_input_ds_name}.nc")
                if os.path.exists(f'{temp_path}/ss_input_df.csv'):
                    self.ss_input_df = soil_utils.read_csv_preserved(f'{temp_path}/ss_input_df.csv')
                print(f"Scenario dataset loaded from {scn_input_ds_name}.nc in {temp_path}")
            if dataset == 'soc':
                # Save the input_inventory dataset to a netcdf-file
                scn_soc_ds_name = f'{self.scenario}_soc_ds'
                self.soc_inventory = xr.load_dataset(f"{temp_path}/{scn_soc_ds_name}.nc")
                if os.path.exists(f'{temp_path}/soc_ha_df.csv'):
                    self.soc_ha_df = soil_utils.read_csv_preserved(f'{temp_path}/soc_ha_df.csv')
                if os.path.exists(f'{temp_path}/soc_sko_df.csv'):
                    self.soc_sko_df = soil_utils.read_csv_preserved(f'{temp_path}/soc_sko_df.csv')
                print(f"SOC dataset loaded from {scn_soc_ds_name}.nc in {temp_path}")
            if dataset == 'historic':
                self.historic_inventory = xr.load_dataset(f'{temp_path}/historic_soc_ds.nc')
                if os.path.exists(f'{temp_path}/.historic_ha_df.csv'):
                    self.historic_ha_df = soil_utils.read_csv_preserved(f'{temp_path}/historic_ha_df.csv')
                if os.path.exists(f'{temp_path}/historic_ha_df.csv'):
                    self.historic_sko_df = soil_utils.read_csv_preserved(f'{temp_path}/historic_sko_df.csv')
                print(f"Historic SOC dataset loaded from historic_soc_ds.nc \n in {temp_path}")


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

    def __getstate__(self):
        # Copy the object's state using dict.copy() to avoid modifying the original state
        state = self.__dict__.copy()
        # Remove DataFrame and DataSet attributes before pickling
        excluded_keys = ['input_df', 'ss_input_df', 'soc_ha_df', 'soc_sko_df', 'historic_ha_df', 'historic_sko_df',
                         'input_inventory', 'soc_inventory', 'historic_inventory']
        for key in excluded_keys:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        # Restore instance attributes (except for excluded ones)
        self.__dict__.update(state)
        # Initialize DataFrames or DataSets here if needed, or leave them to be set elsewhere

    def save_instance_state(self, temp_path=temp_path):
        with open(f'{temp_path}/{self.name}.pickle', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_instance_state(cls, instance_name: str, temp_path=temp_path):
        with open(f'{temp_path}/{instance_name}.pickle', 'rb') as file:
            instance = pickle.load(file)
        return instance