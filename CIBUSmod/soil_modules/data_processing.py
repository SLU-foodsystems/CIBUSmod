#!/usr/bin/env python

"""
Procedures and functions to calculate SOC from CIBUSmod scenario files
"""

import os
import warnings
from typing import Union, List

import pandas as pd
import xarray as xr

import CIBUSmod.soil_modules.icbm_funcs as icbm_funcs
import CIBUSmod.soil_modules.soil_utils as utils

root = os.path.abspath(os.getcwd())
from ..soil_modules import soil_input_path
from ..soil_modules import soil_temp_path




# The formulas below are used to calculate the h_map_df to se when calculating carbon inputs for ICBM:


def crop_map_helper(input_df=False,
                    re_col_name='re',
                    scenario_dict=False,
                    verbose=False
                   ):
    """
    Map CIBUS crop types to corresponding re-crop types using a provided or default dataframe.

    Parameters:
    - input_df (pd.DataFrame, optional): DataFrame containing mapping of CIBUS crop types to re-crop types.
      If not provided, default values are loaded from 'data_input/crop_carbon_map.csv'.
    - re_col_name (str, optional): Column name in input_df specifying re-crop types. Defaults to 're'.

    Returns:
    - pd.DataFrame: DataFrame containing the mapping of CIBUS crop types to re-crop types.
    - dict: Dictionary mapping CIBUS crop types to corresponding re-crop types.
    """

    if verbose:
        print('---Executing crop_map_helper()---')
    if not isinstance(input_df, pd.DataFrame):
        # create a df from default file input
        if verbose:
            print("No input dataframe exists.")
            print(f"Creating 'input_df' from 'crop_carbon_map.csv'in {soil_input_path}")
        input_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/crop_carbon_map.csv', index_col="CIBUS"))
    else:
        pass
    # Create a dict to map CIBUS crop types to the re-crop types
    crop_re_dict = dict(zip(
        list(input_df.index.str.lower()),
        list(input_df[re_col_name])
    ))    
    # extract index from input_df, without duplicates
    index = input_df.index.drop_duplicates(keep=False)
    # copy cinput_df based on c_index
    crop_in_df = input_df.loc[index]

    scenario_dict.update({'crop_re_df': crop_in_df, 'crop_re_dict': crop_re_dict})
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print("   'crop_in_dict'")
        print("   'crop_re_dict'")
                
        print('---Leaving crop_map_helper()---')

    return crop_in_df, crop_re_dict


def h_map_helper(h_in_df=False,
                 crop_in_df=False,
                 amnd_in_df=False,
                 map_col='h_value_type',
                 output_col_name='h_value',
                 scenario_dict=False,
                 verbose=False
                ):
    """
    Create and map H-values for crops and amendments based on input dataframes.

    Parameters:
    - h_in_df (pd.DataFrame): DataFrame containing H-values for different fractions (crops,
      amendments, and crop parts). If not provided, default values are loaded from 'data_input/h_values.csv'.
    - crop_in_df (pd.DataFrame): DataFrame containing crop data to map H-values. Must have a
      column specified by map_col with values corresponding to H-value types.
    - amnd_in_df (pd.DataFrame): DataFrame containing amendment data to map H-values. Must have a
      column specified by map_col with values corresponding to H-value types.
    - map_col (str, optional): Column in crop_in_df and amnd_in_df specifying H-value types.
      Defaults to 'h_value_type'.
    - output_col_name (str, optional): Column name in the resulting H-value DataFrame.
      Defaults to 'h_value'.

    Returns:
    - pd.DataFrame: DataFrame containing mapped H-values for crops and amendments.
    - dict: Dictionary of all mapped H-values.
    """

    if verbose:
        print('---Executing h_map_helper()---')
    if not isinstance(h_in_df, pd.DataFrame):
        # Import default h_values to use for different fractions (crops, amendments and crop parts) as a pandas df
        if verbose:
            print("An h-value mapping dataframe does not exist.")
            print(f"Creating h_map_df from 'h_values.csv' in {soil_input_path}")
        h_map_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/h_values.csv', index_col=["h_value_type", "h_frac"]))
    else:
        # Make sure all text is lower case
        if verbose:
            print("An h-value mapping dataframe exists.")
            print("Assigning existing df to 'h_map_df'")
        h_map_df = utils.make_df_lower(h_in_df)

    if not isinstance(crop_in_df, pd.DataFrame):
        # create a df from default file input
        if verbose:
            print("CIBUS crop mapping dataframe does not exist")
            print("> Calling 'crop_map_helper()'")
        crop_in_df, crop_re_dict = crop_map_helper(scenario_dict=scenario_dict, verbose=verbose)
        if verbose:
            print("'crop_in_df' and 'crop_re_dict' created")
    else:
        pass

    if not isinstance(amnd_in_df, pd.DataFrame):
        # create an input dataframe for amendments from default import
        if verbose:
            print("An amendment mapping dataframe does not exist")
            print(f"Creating 'amnd_map_df' from 'amnd_map.csv' in {soil_input_path}")
        amnd_map_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/amnd_map.csv', index_col="CIBUS"))
    else:
        # Make sure all text is lower case
        if verbose:
            print("An amendment mapping dataframe exists.")
            print("Assigning existing df to 'amnd_map_df'")
        amnd_map_df = utils.make_df_lower(amnd_in_df)

    #### calculate the h-value dataframe by combiningcrops and amendments
    h_value_df_crops = utils.map_cin_h_to_dataframe(map_col,
                                                    crop_in_df,
                                                    h_map_df,
                                                    output_col_name)
    h_value_df_amendments = utils.map_cin_h_to_dataframe(map_col,
                                                         amnd_map_df,
                                                         h_map_df,
                                                         output_col_name)
    
    h_value_df = pd.concat([h_value_df_crops, h_value_df_amendments])
    
    # Create a dict from h_value_df
    h_value_dict = h_value_df.to_dict('dict')[h_value_df.columns[0]]
    
    # create separate dicts for different fractions
    h_ag_dict = h_value_df.loc[(slice(None),'ag'),].droplevel(1, axis=0).to_dict()[h_value_df.columns[0]]
    h_bg_dict = h_value_df.loc[(slice(None),'bg'),].droplevel(1, axis=0).to_dict()[h_value_df.columns[0]]
    h_amnd_dict = h_value_df.loc[(slice(None),'amnd'),].droplevel(1, axis=0).to_dict()[h_value_df.columns[0]]
    if verbose:
        print("'h_value_df' and 'h_value_dict' created")
        
    scenario_dict.update({'h_value_df': h_value_df,
                          'h_value_dict': h_value_dict,
                          'h_ag_dict': h_ag_dict,
                          'h_bg_dict': h_bg_dict,
                          'h_amnd_dict': h_amnd_dict})
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print("   'h_value_df'")
        print("   'h_value_dict'")
        print("   'h_ag_dict'")
        print("   'h_bg_dict'")
        print("   'h_amnd_dict'")        
        print("---Leaving h_map_helper()---")

    return h_value_df, h_value_dict

    
# The functions below are used to calculate carbon allocation to ag and bg for different CIBUS crops

def alloc_helper(name_df=False, 
                 allo1_df=False,
                 allo2_df=False,
                 allo3_df=False,
                 scenario_dict=False,
                 verbose=False
                ):
    """
    Helper function to allocate allometric function parameters or allocation factors
    based on provided DataFrames or default files.

    Parameters:
    - name_df (pd.DataFrame or False, optional): DataFrame containing crop names and their
      corresponding values in different sources (andren2004, jacobs, hanna, cpc, crop_group).
      Defaults to reading 'data_input/name_map_all.csv' if not provided.
    - allo1_df (pd.DataFrame or False, optional): DataFrame containing allometric function
      parameters from Andren (2004). Defaults to reading 'data_input/c_allom_andren2004.csv'
      if not provided.
    - allo2_df (pd.DataFrame or False, optional): DataFrame containing allocation factors 
      from Jacobs et al. (2020). Defaults to reading 'data_input/c_alloc_jacobs2020.csv' if
      not provided.
    - allo3_df (pd.DataFrame or False, optional): DataFrame containing allocation factors
      from Hanna (unpublished). Defaults to reading 'data_input/c_alloc_hanna.csv' if not
      provided.

    Returns:
    dict: A dictionary containing the following key-value pairs:
        - 'c_allom_andren_df': DataFrame with Andren (2004) allometric function parameters.
        - 'c_alloc_jacobs_df': DataFrame with Jacobs et al. (2020) allocation factors.
        - 'c_alloc_hanna_df': DataFrame with Hanna's (unpublished) allocation factors.
        - 'crop_andren2004_map': Dictionary mapping crop names to Andren (2004) values.
        - 'crop_jacobs_map': Dictionary mapping crop names to Jacobs et al. (2020) values.
        - 'crop_hanna_map': Dictionary mapping crop names to Hanna's (unpublished) values.
        - 'crop_cpc_map': Dictionary mapping crop names to CPC values.
        - 'crop_crop_group_map': Dictionary mapping crop names to crop group values.
    """

    if verbose:
        print('---Executing alloc_helper()---')
    
    if not isinstance(name_df, pd.DataFrame):
        name_map_all_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/name_map_all.csv')).dropna(subset='crop')
    else:
        name_map_all_df = utils.make_df_lower(name_df).dropna(subset='crop')
        
    # Create dictionaries between cibus names (crop) and the different sources
    crop_andren2004_map = dict(zip(name_map_all_df.crop, name_map_all_df.andren2004))
    crop_jacobs_map = dict(zip(name_map_all_df.crop, name_map_all_df.jacobs))
    crop_hanna_map = dict(zip(name_map_all_df.crop, name_map_all_df.hanna))
    crop_cpc_map = dict(zip(name_map_all_df.crop, name_map_all_df.cpc))
    crop_crop_group_map = dict(zip(name_map_all_df.crop, name_map_all_df.crop_group))
    
    # Create dfs with the allometric function parameters or allocation factors
    if not isinstance(allo1_df, pd.DataFrame):
        c_allo1_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/c_allom_andren2004.csv', index_col=0, decimal=','))
    else: 
        c_allo1__df = utils.make_df_lower(allo1_df)
    
    if not isinstance(allo2_df, pd.DataFrame):
        c_allo2_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/c_alloc_jacobs2020.csv', index_col=0, usecols=['Crop', 'i_ag', 'i_bg'], decimal=','))
    else:
        c_allo2_df = utils.make_df_lower(allo2_df)
    
    if not isinstance(allo3_df, pd.DataFrame):
        c_allo3_df = utils.make_df_lower(pd.read_csv(f'{soil_input_path}/c_alloc_hanna.csv', index_col=0, usecols=['Crop', 'i_ag', 'i_bg', 'source'], decimal=','))
    else:
        c_allo3_df = utils.make_df_lower(allo3_df)
    
    out_dict = dict(zip(('c_allom_andren_df', 'c_alloc_jacobs_df', 'c_alloc_hanna_df', 'crop_andren2004_map', 'crop_jacobs_map', 'crop_hanna_map', 'crop_cpc_map', 'crop_crop_group_map'), (c_allo1_df, c_allo2_df, c_allo3_df, crop_andren2004_map, crop_jacobs_map, crop_hanna_map, crop_cpc_map, crop_crop_group_map)))

    scenario_dict.update({'alloc_dict': out_dict})
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict' ***")
        print("   'alloc_dict': a dictionary with all the dataframes and mapping dictionaries used to allocate carbon inputs for individual crops.")
        print('---Leaving alloc_helper()---')
    
    return out_dict


# The functions below are used to generate the dataframes with inputs and calculate the C inputs

def make_scn_input_df(scenario_name='FAI',
                      scenario_file=False, 
                      scenario_df=False,
                      C_content_crops=0.5,
                      name_df=False,
                      allo1_df=False,
                      allo2_df=False,
                      allo3_df=False,
                      scenario_dict=False,
                      verbose=False
                     ):
    """
    Create a CIBUSmod scenario dataframe based on input data and allocation parameters.

    This function takes input data for a CIBUSmod scenario, calculates yields and carbon input per unit area,
    and inserts new columns with manure input per hectare based on the total input per scenario.

    Parameters:
    - scenario_file (str or False, optional): Path to the scenario file in CSV format.
      Defaults to 'data_input/FAI.csv' if not provided.
    - scenario_df (pd.DataFrame or False, optional): DataFrame containing scenario data.
      Defaults to reading 'data_input/FAI.csv' or the provided scenario_file if not provided.
    - C_content_crops (float, optional): Carbon content of crops. Defaults to 0.5.
    - name_df (pd.DataFrame or False, optional): DataFrame containing crop names and their
      corresponding values in different sources (andren2004, jacobs, hanna, cpc, crop_group).
      Defaults to reading 'data_input/name_map_all.csv' if not provided.
    - allo1_df (pd.DataFrame or False, optional): DataFrame containing allometric function
      parameters from Andren (2004). Defaults to reading 'data_input/c_allom_andren2004.csv'
      if not provided.
    - allo2_df (pd.DataFrame or False, optional): DataFrame containing allocation factors
      from Jacobs et al. (2020). Defaults to reading 'data_input/c_alloc_jacobs2020.csv' if
      not provided.
    - allo3_df (pd.DataFrame or False, optional): DataFrame containing allocation factors
      from Hanna (unpublished). Defaults to reading 'data_input/c_alloc_hanna.csv' if not
      provided.

    Returns:
    pd.DataFrame: DataFrame containing CIBUSmod scenario inputs.
    """

    if verbose:
        print('---Executing make_scn_input_df()---')
        print(f'scenario name and scenario file is: {scenario_name, scenario_file} in make_scn_input_df')
    
    if not scenario_file:
        if verbose:
            print('> Scenario file not set')
        if not isinstance(scenario_df, pd.DataFrame):
            if verbose:
                print('> Scenario_df not set')
            inputs_yields_df = utils.make_df_lower(pd.read_csv(f'data_input/{scenario_name}.csv'))
            if verbose:
                print(f"Scenario_input dataframe created from data_input/{scenario_name}.csv")
        else:
            if verbose:
                print('> scenario_df set')
            inputs_yields_df = utils.make_df_lower(scenario_df)
    else:
        if verbose:
            print("> Scenario file set")
            print(f"Creating scenario_input dataframe from {scenario_file}")
        inputs_yields_df = utils.make_df_lower(pd.read_csv(scenario_file))
        if verbose:
            print(f'Scenario_input dataframe created from {scenario_file}')
    
    if 'year' in inputs_yields_df.columns:
        # add a datetime64 year value based on the year column, drop year column
        inputs_yields_df["input_year"] = pd.to_datetime(inputs_yields_df["year"], format="%Y")
        inputs_yields_df = inputs_yields_df.drop(columns='year')
    
    # Calculate yield per unit area
    inputs_yields_df["areayield"] = inputs_yields_df["harvest_kgdm"]/inputs_yields_df["area_ha"] * C_content_crops

    ## Insert new columns containing the manure input per ha, based on the total input per sko
    
    # create a list of the columns to convert
    tot_manure_cols = utils.get_filtered_namelist(['kgc'], ['manure', 'crop'], inputs_yields_df)
    
    newframe = inputs_yields_df.copy()
    # insert a new column with the input per ha
    for i in tot_manure_cols:
        col_name = f'{i[:-4]}_ha'
        newframe[col_name] = newframe[i] / newframe['area_ha']
    
    inputs_yields_df = newframe.copy(deep=True)

    # make the indexes continuous to avoid errors in iterative loops based on row number
    # this is good to do, especially when rows have been dropped from a dataframe
    inputs_yields_df = utils.make_idx_continuous(inputs_yields_df)

    allo_dict = alloc_helper(name_df,
                             allo1_df,
                             allo2_df,
                             allo3_df,
                             scenario_dict,
                             verbose)

    mapper = (allo_dict['crop_andren2004_map'], allo_dict['crop_jacobs_map'], allo_dict['crop_hanna_map'])
    sources = ('Andren2004', 'Jacobs2020', 'Hanna')
    allo_dfs = (allo_dict['c_allom_andren_df'], allo_dict['c_alloc_jacobs_df'], allo_dict['c_alloc_hanna_df'])

    scn_inputs_df = utils.calculate_c_inputs(inputs_yields_df,
                                             mapper,
                                             sources,
                                             allo_dfs)
    
    scenario_dict.update({'alloc_dfs': allo_dfs,
                          'alloc_dicts': mapper,
                          'input_df': scn_inputs_df
                         })
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print("   'alloc_dfs': All dataframes dataframes containing the carbon allometric functions and factors for individual crops.")
        print("   'alloc_dict': All dictionaries used to map carbon alloc_dfs to individual crops.")
        print("   'input_df': Dataframe with all yields and carbon inputs expressed mapped to CIBUS crop names.")
        print('---Leaving make_scn_input_df()---')

    return scn_inputs_df, scenario_name


def make_scn_multi_df(scn_inputs_df=False,
                      scenario_name='FAI',
                      scenario_file=False, 
                      scenario_df=False,
                      C_content_crops=0.5,
                      name_df=False,
                      allo1_df=False,
                      allo2_df=False,
                      allo3_df=False,
                      called=False,
                      scenario_dict=False,
                      verbose=False    
                     ):
    """
    Generate a multi-index DataFrame for CIBUSmod scenarios.

    Parameters:
    - scn_inputs_df (pd.DataFrame, optional): DataFrame containing scenario inputs. If not provided, it's generated using scn_input_df_maker.
    - scenario_file (str, optional): Path to the scenario file in CSV format. Ignored if scn_inputs_df is provided.
    - scenario_df (pd.DataFrame, optional): DataFrame containing scenario data. Ignored if scn_inputs_df is provided.
    - C_content_crops (float, optional): Carbon content for crops. Defaults to 0.5.
    - name_df, allo1_df, allo2_df, allo3_df (pd.DataFrame, optional): DataFrames for various allocations and mappings.

    Returns:
    - pd.DataFrame: Multi-index DataFrame with scenario inputs for CIBUSmod.
    - int: Start year for scenarios.
    """
    if verbose:
        print('---Executing make_scn_multi_df()---')

    if scenario_dict:
            scenario_name = scenario_dict['scenario_name']
            scenario_file = scenario_dict['scenario_file']
            if verbose:
                print(f'scenario_name and scenario_file set to {scenario_name, scenario_file}')

    if not isinstance(scn_inputs_df, pd.DataFrame):      
        if verbose:
            print('> scn_inputs_df not set')
        if 'input_df' in scenario_dict.keys():
            print("Setting 'scn_inputs_df' using 'scenario_dict'")
            scn_inputs_df = scenario_dict['input_df']
        else:
            if verbose:
                print('Calling make_scn_inputs_df()')
            
            scn_inputs_df, scenario_name = make_scn_input_df(scenario_name,
                                                             scenario_file, 
                                                             scenario_df,
                                                             C_content_crops,
                                                             name_df,
                                                             allo1_df,
                                                             allo2_df,
                                                             allo3_df,
                                                             scenario_dict=scenario_dict,
                                                             verbose=verbose)
            if verbose:
                print('> make_scn_inputs_df completed<')
    else:
        if verbose:
            print('scn_inputs_df set. \n continuing')

    if not isinstance(scn_inputs_df, pd.MultiIndex):
        # Insert new columns containing the crop total input per sko, based on the input per ha
        # create a list of the columns to convert
        ha_crop_cols = utils.get_filtered_namelist(['ha'], ['crop'], scn_inputs_df)
        
        newframe = scn_inputs_df.copy()
        
        # insert a new column with the input per ha
        for i in ha_crop_cols:
            col_name = f'{i[0:4]}_kgc_{i[8:]}'
            newframe[col_name] = newframe[i] * newframe['area_ha']
    
        #Overwrite the input file variable with the newframe
        scn_inputs_df = newframe.copy(deep=True)
        
        # Add prefixes to all the manure fractions so that all c input fractions follow the same naming convention
        scn_inputs_prefixed_df = utils.add_prefix('manure', 'i_ag', scn_inputs_df)

        # create a selection of columns used to create multiindex for scenarios and spinup 
        scn_input_idx = ['scn', 'crop', 'prod_system', 'region', 'input_year']
        # Create a multiindex df
        scn_multi_df = scn_inputs_prefixed_df.set_index(scn_input_idx)

    else:
        scn_multi_df = scn_inputs_df

    startyear = scn_multi_df.index.get_level_values('input_year').year[0]

    scn_input_ds_name = f'{scenario_name}_input_ds'

    scn_input_ds = scn_multi_df.to_xarray()
    
    scenario_dict.update({'scenario_multi_df': scn_multi_df,
                          'scenario_input_ds': scn_input_ds
                         })

    if not called:
        # Save the scn_multi_df to a csv-file
        utils.to_csv_preserved(scn_multi_df,
                               f'{scenario_name}_input_df',
                               save_type='temp')
        print(f"{scenario_name} multiindex dataframe saved to {scenario_name}_input_df in {soil_temp_path}")

        scn_input_ds.to_netcdf(f'{soil_temp_path}/{scn_input_ds_name}.nc')
        print(f"Scenario dataset saved as {scn_input_ds_name}.nc in {soil_temp_path}")
    else:
        if verbose:
            print('_make_spinup_df called recursively. No scenario files saved')
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print(f"   'scenario_multi_df': Multiindex dataframe for scenario {scenario_name}.")
        print(f"   'scenario_input_ds': Xarray dataset for input in scenario {scenario_name}.")
        print('---Leaving make_scn_multi_df()---')

    warnings.filterwarnings("ignore", message="Warning, first index label is None, conversion of index label not possible")
    warnings.filterwarnings("ignore", message="Warning, index label is None. No conversion of index values (should be auto-generated ints)")
    
    return scn_multi_df, scn_input_ds, startyear


def make_spinup_df(input_df=False,
                   year0=False,
                   scn_inputs_df=False,
                   scenario_name='FAI',
                   scenario_file=False, 
                   scenario_df=False,
                   C_content_crops=0.5,
                   name_df=False,
                   allo1_df=False,
                   allo2_df=False,
                   allo3_df=False,
                   scenario_dict=False,
                   verbose=False  
                  ):
    """
    Generate a DataFrame with input values for spinup modeling.

    Parameters:
    - input_df (pd.DataFrame, optional): DataFrame containing input values. Ignored if startyear is provided.
    - year0 (int, optional): End year for spinup modelling. Ignored if input_df is provided.
    - scn_inputs_df (pd.DataFrame, optional): DataFrame containing scenario inputs. Used if input_df is not provided.
    - scenario_file (str, optional): Path to the scenario file in CSV format. Ignored if scn_inputs_df is provided.
    - scenario_df (pd.DataFrame, optional): DataFrame containing scenario data. Ignored if scn_inputs_df is provided.
    - C_content_crops (float, optional): Carbon content for crops. Defaults to 0.5.
    - name_df, allo1_df, allo2_df, allo3_df (pd.DataFrame, optional): DataFrames for various allocations and mappings.

    Returns:
    - pd.DataFrame: DataFrame with input values for spinup modeling.
    - int: End year used in spinup model
    """

    if verbose:
        print('---Executing _make_spinup_df()---')
    
    if not isinstance(input_df, pd.DataFrame):
        if verbose:
            print('> Input df not set')
        if 'scn_multi_df' in scenario_dict.keys():
            print("Setting 'scn_multi_df' using 'scenario_dict'")
            scn_multi_df = scenario_dict['scn_multi_df']
            
        else:
            if verbose:
                print('> Calling make_scn_multi_df()')
            scn_multi_df, scn_input_ds, year0 = make_scn_multi_df(scn_inputs_df,
                                                                  scenario_name,
                                                                  scenario_file, 
                                                                  scenario_df,
                                                                  C_content_crops,
                                                                  name_df,
                                                                  allo1_df,
                                                                  allo2_df,
                                                                  allo3_df,
                                                                  called=True,
                                                                  scenario_dict=scenario_dict,
                                                                  verbose=verbose)
            if verbose:
                print('>make_scn_multi_df completed<')
        
    else:
        if verbose:
            print('> Input df set')
        scn_multi_df = input_df
        year0 = scn_multi_df.index.get_level_values('input_year').year[0]
    
    # create spinup_ha_df and spinup_ha_multiidx_df based on scenario df
    spinup_multi_df = scn_multi_df.query(f'input_year == {year0}')
    spinup_multi_df = spinup_multi_df.droplevel(['input_year'])

    # Save the ss_input_df to a csv-file
    utils.to_csv_preserved(spinup_multi_df,
                           'spinup_input_df',
                           save_type='temp')
    if verbose:
        print(f"Spinup dataframe saved to 'spinup_indput_df.csv' in {soil_temp_path}")
    
    scenario_dict.update({'spinup_multi_df': spinup_multi_df})
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print(f"   'spinup_multi_df': Multiindex dataframe for spinup modelling.")
        print('---Leaving _make_spinup_df()---')
    
    return spinup_multi_df, year0


def make_scn_area_dfs(input_df=False,
                      scenario_dict=False,
                      verbose=False
                     ):
    """
    Create scenario dataframes with yields per ha and per sko
    """

    if 'scenario_name' not in scenario_dict.keys():
        scenario_name = input('Please input the scenario name:')
    else:
        scenario_name = scenario_dict['scenario_name']
    
    if verbose:
        print('---Executing _make_scn_area_dfs()---')

    if not isinstance(input_df, pd.DataFrame):
        if 'scenario_multi_df' in  scenario_dict.keys():
            input_df = scenario_dict['scenario_multi_df']
        else:
            print("No usable input dataframe exist in 'scenario_dict'. Please provide an input dataframe using 'input_df='")     
    
    # Group the scn df by scn, prod system, region and year and calculate total input and areas of all crops
    scn_multi_groupby_idx = ['scn', 'prod_system', 'region', 'input_year']
    scn_sko_prodsys_sum_df = input_df.groupby(scn_multi_groupby_idx).sum()
    
    # Select the columns that should hold the weighted average
    wt_at_cols = utils.get_filtered_namelist(['_ha'], ['manure', 'crop'], scn_sko_prodsys_sum_df)
    tot_cols = utils.get_filtered_namelist(['_kgc'], ['manure', 'crop'], scn_sko_prodsys_sum_df)
    
    # Create a temporary df and calculate the weighted average per ha from total input per sko / total area per sko
    tempframe = scn_sko_prodsys_sum_df.copy()
    
    for n, i in enumerate(wt_at_cols):
        tempframe[i] = tempframe[tot_cols[n]] / tempframe['area_ha']
    
    # Rename the temporary df
    scn_all_c_inputs_df = tempframe.copy(deep=True)
    
    # Make separate df's for ha and total input, include total input and area info
    scn_ha_c_inputs_df = scn_all_c_inputs_df.loc[:,wt_at_cols]
    scn_ha_c_inputs_df['area_ha'] = scn_all_c_inputs_df['area_ha']
    scn_ha_c_inputs_df['harvest_kgdm'] = scn_all_c_inputs_df['harvest_kgdm']
    scn_sko_c_inputs_df = scn_all_c_inputs_df.loc[:,tot_cols]
    scn_sko_c_inputs_df['area_ha'] = scn_all_c_inputs_df['area_ha']
    scn_sko_c_inputs_df['harvest_kgdm'] = scn_all_c_inputs_df['harvest_kgdm']

    
    scenario_dict.update({'scenario_ha_input_df': scn_ha_c_inputs_df, 
                          'scenario_sko_input_df': scn_sko_c_inputs_df
                         })
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print(f"   'scenario_ha_input_df': Scenario {scenario_name} input dataframe expressed per ha.")
        print(f"   'scenario_sko_input_df': Scenario {scenario_name} input dataframe expressed per sko.")
        print('---Leaving _make_scn_area_dfs()---')
    
    return scn_ha_c_inputs_df, scn_sko_c_inputs_df

        
def _make_spinup_area_dfs(input_df=False,
                         scenario_dict=False,
                         verbose=False
                        ):
    """
    Create spinup dataframes with yields per ha and per sko
    """

    if verbose:
        print('---Executing _make_spinup_area_dfs()---')

    if not isinstance(input_df, pd.DataFrame):
        if 'spinup_multi_df' in  scenario_dict.keys():
            input_df = scenario_dict['spinup_multi_df']
        else:
            print("No usable input dataframe exist in 'scenario_dict'. Please provide an input dataframe using 'input_df='")        

    # Group the spinup df by prod system and region to calculate total input and areas of all crops
    spinup_multi_groupby_idx = ['prod_system', 'region']
    spinup_sko_prodsys_sum_df = input_df.groupby(spinup_multi_groupby_idx).sum()
    
    # Select the columns that should hold the weighted average
    wt_at_cols = utils.get_filtered_namelist(['_ha'], ['manure', 'crop'], spinup_sko_prodsys_sum_df)
    tot_cols = utils.get_filtered_namelist(['_kgc'], ['manure', 'crop'], spinup_sko_prodsys_sum_df)
    
    # Create a temporary df and calculatr the weighted average per ha from total input per sko / total area per sko
    tempframe = spinup_sko_prodsys_sum_df.copy()
    
    for n, i in enumerate(wt_at_cols):
        tempframe[i] = tempframe[tot_cols[n]] / tempframe['area_ha']
    
    # Rename the temporary df
    spinup_all_c_inputs_df = tempframe.copy(deep=True)
    
    # Make separate df's for ha and total input, include total input and area info
    spinup_ha_c_inputs_df = spinup_all_c_inputs_df.loc[:,wt_at_cols]
    spinup_ha_c_inputs_df['area_ha'] = spinup_all_c_inputs_df['area_ha']
    spinup_ha_c_inputs_df['harvest_kgdm'] = spinup_all_c_inputs_df['harvest_kgdm']
    spinup_sko_c_inputs_df = spinup_all_c_inputs_df.loc[:,tot_cols]
    spinup_sko_c_inputs_df['area_ha'] = spinup_all_c_inputs_df['area_ha']
    spinup_sko_c_inputs_df['harvest_kgdm'] = spinup_all_c_inputs_df['harvest_kgdm']

    scenario_dict.update({'spinup_ha_input_df': spinup_ha_c_inputs_df, 
                          'spinup_sko_input_df': spinup_sko_c_inputs_df
                         })
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print("   'spinup_ha_input_df': Spinup input dataframe expressed per ha.")
        print("   'spinup_sko_input_df': Spinup input dataframe expressed per sko.")
        print('---Leaving _make_spinup_area_dfs()---')
    
    return spinup_ha_c_inputs_df, spinup_sko_c_inputs_df
    

def calculate_scn_soc(scn_ha_df=False,
                      scn_sko_df=False,
                      h_value_dict=False,
                      input_df=False,
                      scenario_dict=False,
                      verbose=False
                     ):
    """
    Calculate the scenario SOC pools for both the ha and sko dataframes
    """

    if verbose:
        print('---Executing _calculate_soc()---')

    if 'scenario_name' not in scenario_dict.keys():
        scenario_name = input('Please input the scenario name:')
    else:
        scenario_name = scenario_dict['scenario_name']
    
    if not isinstance(scn_ha_df, pd.DataFrame) or not isinstance(scn_sko_df,pd.DataFrame):
        if verbose:
            print("> One or both of 'scn_ha_df' and 'scn_sko_df' are not set") 
        try:
            if verbose:
                print(">_make_scn_area_dfs() called<")
            scn_ha_df, scn_sko_df = make_scn_area_dfs(input_df=input_df,
                                                      scenario_dict=scenario_dict,
                                                      verbose=verbose)
            if verbose:
                print(f"'scn_ha_df' and 'scn_sko_df' set using input_df for {scenario_name}\n>make_sc_area_dfs() completed") 
        except:
            if not isinstance(input_df, pd.DataFrame):
                if 'scenario_multi_df' in  scenario_dict.keys():
                    input_df = scenario_dict['spinup_multi_df']
                else:
                    print("No usable input dataframe exist in 'scenario_dict'. Please provide an input dataframe using 'input_df='")     
            
    if not h_value_dict:
        if verbose:
            print("> 'h_value_dict' not set.")
        if 'h_value_dict' in scenario_dict.keys():
            if verbose:
                print("'h_value_dict' assigned using 'scenario_dict'.")
            h_value_dict = scenario_dict['h_value_dict']
        else:   
            if verbose:
                print("> Calling 'h_map_helper()'")
            temp, h_value_dict = h_map_helper(scenario_dict=scenario_dict,
                                              verbose=verbose)
            if verbose:
                print(" 'h_map_helper()' finished. Continuing")
    
    # extract a list of the scenario columns by which icbm is to be run
    if verbose:
        print("Extracting filtered_namelist per ha")
    scn_ha_sel = list(set(utils.get_filtered_namelist(['i_a', 'i_b', 'ha'], ['manure', 'crop'], scn_ha_df)))
    if verbose:
        print("Extracting filtered_namelist per sko")
    scn_sko_sel = list(set(utils.get_filtered_namelist(['i_a', 'i_b', 'sko'], ['manure', 'crop'], scn_sko_df)))
    # Calculate the soc timeseries dataframe using input df, icbm_selection and h_value_dict as input for scenario
    if verbose:
        print("Calculating SOC timeseries per ha")
    scn_ha_soc_df = icbm_funcs.input_df_to_soc_df(scn_ha_df, scn_ha_sel, h_value_dict, year_label="input_year")
    if verbose:
        print("Calculating SOC timeseries per sko")
    scn_sko_soc_df = icbm_funcs.input_df_to_soc_df(scn_sko_df, scn_sko_sel, h_value_dict, year_label="input_year")
    if verbose:
        print("Done calculating SOC timeseries")

    # Create and save soc scenarios to xarray datasets
    if verbose:
        print("Creating xarray datasets and saving as netcdf-files")
    scn_ha_soc_ds_name = f'{scenario_name}_ha_soc_ds'
    
    scn_ha_soc_ds = scn_ha_soc_df.to_xarray()
    scn_ha_soc_ds.to_netcdf(f'{soil_temp_path}/{scn_ha_soc_ds_name}.nc')
    if verbose:
        print(f"{scenario_name}-scenario SOC dataset saved as {scn_ha_soc_ds_name}.nc \n in {soil_temp_path}")

    scn_sko_soc_ds_name = f'{scenario_name}_sko_soc_ds'

    scn_sko_soc_ds = scn_sko_soc_df.to_xarray()
    scn_sko_soc_ds.to_netcdf(f'{soil_temp_path}/{scn_sko_soc_ds_name}.nc')
    if verbose:
        print(f"{scenario_name}-scenario SOC dataset saved as {scn_sko_soc_ds_name}.nc \n in {soil_temp_path}")

    scenario_dict.update({'scenario_ha_soc_df': scn_ha_soc_df,
                          'scenario_sko_soc_df':  scn_sko_soc_df,
                          'scenario_ha_soc_ds': scn_ha_soc_ds, 
                          'scenario_sko_soc_ds':  scn_sko_soc_ds 
                         })
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print(f"   'scenario_ha_soc_df': SOC dataframe expressed per ha for scenario {scenario_name}.")
        print(f"   'scenario_sko_soc_df': SOC dataframe expressed per sko for scenario {scenario_name}.")
        print(f"   'scenario_ha_soc_ds': xarray dataset of 'scenario_ha_soc_df'.")
        print(f"   'scenario_sko_soc_ds': xarray dataset of 'scenario_sko_soc_df'.")
        print('---Leaving _calculate_soc()---')
    
    return scn_ha_soc_df, scn_sko_soc_df, scn_ha_soc_ds, scn_sko_soc_ds


def calculate_historic_soc(spinup_ha_df=False,
                           spinup_sko_df=False,
                           h_value_dict=False,
                           input_df=False,
                           scenario_dict=False,
                           verbose=False
                          ):
    """
    Calculate the SS SOC pools at year 0 for both the ha and sko dataframes
    """

    if verbose:
        print('---Executing _calculate_historic_soc()---')

    # Assign a name to the spinup scenario_name variable
    scenario_name = 'spinup_soc'
    
    if not isinstance(spinup_ha_df, pd.DataFrame) or not isinstance(spinup_sko_df,pd.DataFrame):
        if verbose:
            print("> One or both of 'spinup_ha_df' and 'spinup_sko_df' are not set") 

        if 'spinup_multi_df' in  scenario_dict.keys():
            input_df = scenario_dict['spinup_multi_df']
                 
        if isinstance(input_df, pd.DataFrame):
            if verbose:
                print("Generating 'spinup_ha_df' and 'spinup_sko_df' from 'input_df'")
                print(">'_make_spinup_area_dfs()' called<")
            spinup_ha_df, spinup_sko_df = make_spinup_area_dfs(input_df,
                                                               scenario_dict=scenario_dict,
                                                               verbose=verbose)
            if verbose:
                print(f"'spinup_ha_df' and 'spinup_sko_df' set using input_df for {scenario_name}\n>_make_spinup_area_dfs() completed")
        else:
            print("No usable input dataframe exist in 'scenario_dict'. Please provide an input dataframe using 'input_df='")    
    
    if not h_value_dict:
        if verbose:
            print("> 'h_value_dict' not set.")
        #try:
        if 'h_value_dict' in scenario_dict.keys():
            if verbose:
                print("'h_value_dict' assigned using 'scenario_dict'.")
            h_value_dict = scenario_dict['h_value_dict']
        #except:
        else:
            if verbose:
                print("> Calling 'h_map_helper()'")
            temp, h_value_dict = h_map_helper(scenario_dict=scenario_dict,
                                              verbose=verbose)
            if verbose:
                print(" 'h_map_helper()' finished. Continuing")
        
    # extract a list of the spinup columns by which icbm is to be run
    spinup_ha_sel = list(set(utils.get_filtered_namelist(['i_a', 'i_b'], ['manure', 'crop', 'ha'], spinup_ha_df)))
    spinup_sko_sel = list(set(utils.get_filtered_namelist(['i_a', 'i_b'], ['manure', 'crop', 'ha'], spinup_sko_df)))
    
    # Calculate the soc timeseries dataframe using input df, icbm_selection and h_value_dict as input for spinup
    if verbose:
        print(">spinup_df_to_soc_df called to calculate SS SOC per ha")
    spinup_ss_ha_df = icbm_funcs.spinup_df_to_ss_soc_df(spinup_ha_df, spinup_ha_sel, h_value_dict)
    if verbose:
        print(">spinup_df_to_soc_df finished<")
        print(">spinup_df_to_soc_df called to calculate SS SOC per sko")
    spinup_ss_sko_df = icbm_funcs.spinup_df_to_ss_soc_df(spinup_sko_df, spinup_sko_sel, h_value_dict)
    if verbose:
        print(">spinup_df_to_soc_df finished<")

    
    if verbose:
        print(">input_df_to_soc_df called to calculate SOC timeseries per ha")
    spinup_ha_soc_df = icbm_funcs.input_df_to_soc_df(spinup_ss_ha_df,
                                                     spinup_ha_sel,
                                                     h_value_dict,
                                                     historic=True)
    if verbose:
        print(">input_df_to_soc_df finished<")
        print(">input_df_to_soc_df called to calculate SOC timeseries per sko")
    spinup_sko_soc_df = icbm_funcs.input_df_to_soc_df(spinup_ss_sko_df,
                                                      spinup_sko_sel,
                                                      h_value_dict,
                                                      historic=True)
    if verbose:
        print(">input_df_to_soc_df finished<")

    # Create and save spinup soc scenario to xarray dataset
    spinup_ha_soc_ds = spinup_ha_soc_df.to_xarray()
    spinup_ha_soc_ds.to_netcdf(f'{soil_temp_path}/spinup_ha_soc_ds.nc')
    if verbose:
        print(f"Spinup SOC dataset saved as spinup_ha_soc_ds.nc \n in {soil_temp_path}")
    
    spinup_sko_soc_ds = spinup_sko_soc_df.to_xarray()
    spinup_sko_soc_ds.to_netcdf(f'{soil_temp_path}/spinup_sko_soc_ds.nc')
    if verbose:
        print(f"Spinup SOC dataset saved as spinup_sko_soc_ds.nc \n in {soil_temp_path}")

    scenario_dict.update({'spinup_ha_soc_df': spinup_ha_soc_df,
                          'spinup_sko_soc_df':  spinup_sko_soc_df,
                          'spinup_ha_soc_ds': spinup_ha_soc_ds, 
                          'spinup_sko_soc_ds':  spinup_sko_soc_ds 
                         })
    
    if verbose:
        print("*** The following dict keys and their values have been added to 'scenario_dict ***")
        print(f"   'historic_ha_soc_df': dataframe expressed per ha for historic SOC.")
        print(f"   'historic_sko_soc_df': dataframe expressed per sko for historic SOC.")
        print(f"   'historic_ha_soc_ds': xarray dataset of 'historic_ha_soc_df'.")
        print(f"   'historic_sko_soc_ds': xarray dataset of 'historic_sko_soc_df'.")

        print('---Leaving _calculate_historic_soc()---')
    
    return spinup_ha_soc_df, spinup_sko_soc_df, spinup_ha_soc_ds, spinup_sko_soc_ds



"""
    def make_h_map_df(h_map_df=False,
                  crop_map_df=False,
                  amnd_map_df=False,
                  re_col_name='re',
                  map_col='h_value_type',
                  output_col_name='h_value'
                 ):
    ###
    Generate a DataFrame with h-values for crops and amendments based on input DataFrames.

    Parameters:
    - h_map_df (pd.DataFrame, optional): DataFrame containing default h-values for different fractions (crops, amendments, crop parts).
      If not provided, default values are loaded from 'data_input/h_values.csv'.
    - crop_map_df (pd.DataFrame, optional): DataFrame containing crop mapping data. If not provided, default values are loaded from 'data_input/crop_carbon_map.csv'.
    - amnd_map_df (pd.DataFrame, optional): DataFrame containing amendment mapping data. If not provided, default values are loaded from 'data_input/amnd_map.csv'.
    - re_col_name (str, optional): Column name in crop_map_df representing re-crop types. Defaults to 're'.
    - map_col (str, optional): Column name in h_map_df to map values from. Defaults to 'h_value_type'.
    - output_col_name (str, optional): Column name for output h-values. Defaults to 'h_value'.

    Returns:
    - pd.DataFrame: DataFrame containing calculated h-values for crops and amendments.
    ###
    
    print('---Executing makae_h_map_df()---')
    crop_in_df, crop_re_dict = crop_map_helper(input_df=crop_map_df, re_col_name=re_col_name)
    h_value_df, h_value_dict = h_map_helper(h_in_df=h_map_df, crop_in_df=crop_in_df, amnd_in_df=amnd_map_df, map_col=map_col, output_col_name=output_col_name)

    print('---Leaving make_h_map_df()---')
    
    return h_value_df
"""

### Functions below are used with xarrays ###

def assign_tot_soc(xarray_input: Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]],
                   totsoc_label: str = 'tot_soc',
                   year_coord: str = 'output_year'
                  ) -> None:
    """
    Calculate and assign total soil organic carbon (tot_soc) based on young (y_pool) and old (o_pool) pools.

    Parameters:
    -----------
    xarray_input : xr.Dataset or list of xr.Dataset or xr.DataArray or list of xr.DataArray
        The input dataset(s) containing 'y_pool' and 'o_pool' variables.

    Returns:
    --------
    None
        The function modifies the input dataset(s) in-place by adding a 'tot_soc' variable.

    Notes:
    ------
    - If 'xarray_input' is a list, the function will iterate through each element and apply the calculations recursively.
    - The 'tot_soc' variable is calculated as the sum of 'y_pool' and 'o_pool'.
    - The input can be either a single xr.Dataset, a list of xr.Dataset, a single xr.DataArray, or a list of xr.DataArray.
      The function handles each case appropriately.

    Example:
    ---------
    # Single xr.Dataset
    ds = assign_tot_soc(ds)

    # List of xr.Dataset
    ds_list = [ds1, ds2, ds3]
    ds_list = assign_tot_soc(ds_list)

    # Single xr.DataArray
    da = assign_tot_soc(da)

    # List of xr.DataArray
    da_list = [da1, da2, da3]
    da_list = assign_tot_soc(da_list)
    """
    
    if isinstance(xarray_input, list):
        for i in xarray_input:
            assign_tot_soc(i)        
    else:
        tot_soc = xarray_input.y_pool + xarray_input.o_pool
        if isinstance(xarray_input, xr.DataArray):
            xarray_input = xr.concat([xarray_input, tot_soc.rename(totsoc_label)], dim='dummy_dim')
        elif isinstance(xarray_input, xr.Dataset):
            xarray_input[totsoc_label] = tot_soc

    return


def assign_co2_flux(xarray_input,
                    totsoc_label='tot_soc',
                    co2flux_label='co2',
                    year_coord='output_year'
                   ):
    """
    Calculate and assign carbon dioxide flux (co2_flux) based on the differences in total soil organic carbon (tot_soc)
    over time, using the specified labels and coordinates.

    Parameters:
    -----------
    xarray_input : xr.Dataset or list of xr.Dataset or xr.DataArray or list of xr.DataArray
        The input dataset(s) containing relevant variables for calculating CO2 flux.

    totsoc_label : str, optional
        The label for the variable representing total soil organic carbon. Default is 'tot_soc'.

    co2flux_label : str, optional
        The label for the variable representing carbon dioxide flux. Default is 'co2_flux'.

    year_coord : str, optional
        The coordinate along which the differences are calculated. Default is 'output_year'.

    Returns:
    --------
    None
        The function modifies the input dataset(s) in-place by adding a 'co2_flux' variable.

    Notes:
    ------
    - If 'xarray_input' is a list, the function will iterate through each element and apply the calculations recursively.
    - The 'co2_flux' variable is calculated as the difference in 'tot_soc' over the specified 'year_coord' times a conversion factor (3.6).
    - The input can be either a single xr.Dataset, a list of xr.Dataset, a single xr.DataArray, or a list of xr.DataArray.
      The function handles each case appropriately.

    Example:
    ---------
    # Single xr.Dataset
    ds = assign_co2_flux(ds)

    # List of xr.Dataset
    ds_list = [ds1, ds2, ds3]
    ds_list = assign_co2_flux(ds_list)

    # Single xr.DataArray
    da = assign_co2_flux(da)

    # List of xr.DataArray
    da_list = [da1, da2, da3]
    da_list = assign_co2_flux(da_list)
    """
    
    if isinstance(xarray_input, list):
        for i in xarray_input:
            assign_co2_flux(i)        
    else:
        co2_flux = xarray_input[totsoc_label].diff(dim=year_coord)*-3.6
        co2_flux = co2_flux.rename(co2flux_label)
        if isinstance(xarray_input, xr.DataArray):
            xarray_input = xr.concat([xarray_input, co2_flux], dim='dummy_dim')
        elif isinstance(xarray_input, xr.Dataset):
            xarray_input[co2flux_label] = co2_flux

    return