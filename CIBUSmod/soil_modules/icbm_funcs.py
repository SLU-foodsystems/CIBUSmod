#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created on Wed Jun  7 12:47:17 2023
#
# @author: niceri

"""Provide functions needed to model soil carbon pool evolution.

Uses ICBM.
Can handle an arbitrary length and number of C inputs.

The run_icbm function computes the state of two pools:
Y and O, at the end of each consecutive time-step between start year
and endyear

The default time period of calculation (n) is set to 100 timesteps (years).
This can be overriden by supplying a custom value to n.
Alternatively both startyear and endyear can be given.

Each carbon input is associated with a specific h-value.
Both should be supplied together. h defaults to 0.

All parameters in Andrén et al. (1997) are included and can be altered.

The environmental modifier, re, may be submitted as a dictionary to model
variable conditions. If not given, re defaults to 1 for all years.

The equations run in the model at each time step are the following:
Y->O: Y(t)*h*ky*re
Y-> atmosphere: Y(t)*(1-h)*ky*re
Y(t) = Y(t-1)*(1-h)*ky*re+i(t)
O-> atmosphere: O(t)*ko*re
O(t) = O(t-1)*ko*re+Y(t-1)*h*ky*re


The num_to_dict is a helper function that takes
"""

import CIBUSmod as cm
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List, Dict, Union

def spinup_icbm(C_in = False,
             h = False,
             re: float = 1,
             re_default: float = 1,
             ky: float = 0.37,
             ko: float = 0.01,
             startyear: int = False,
             endyear: int = 2021,
             n: int = 1001,
             series = False
               ) -> tuple:
    """
    Execute the ICBM model for an average annual input over 'n' years.

    The `spinup_icbm` function computes the state of two pools, Y and O,
    after each consecutive time-step. Y(0) and O(0) can be set to initial
    values other than 0, but if none are given, 0 will be assumed.

    The model requires two parameter values, `ky` and `ko`, which are dependent on
    the specific soil. If not supplied by the user, they will be set to the
    updated parameter values from Bolinder (2020): ky=0.37, ko=0.01,
    which is a reparametrization of the original ICBM version using 20
    years of additional data from long-term field trials.

    An additional parameter can be submitted for the environmental modifier,
    `re`. This should be a dictionary with year: re values. For any year not
    present in the dictionary, the re value will be set to 1 using the
    helper function `num_to_dict`.

    Parameters:
        C_in (float or int)  -- Average C input of a specific source material (assumed the same every year).
        h (float)            -- The h-value associated with C_in.
        re (float or dict)   -- Default value to create an re-dictionary or {year: re} dict for arbitrary years. (default = 1)
        re_default (float or int) -- Default re assigned to years not supplied in {year: re} dict. (default = 1)
        ky (float)           -- First-order decay rate of the young pool. (default = 0.8)
        ko (float)           -- First-order decay rate of the old pool. (default = 0.01)
        startyear (int)      -- First year of simulation (for plotting and tracking purposes).
                               If False, the parameter n is used as the number of timesteps.
                               (default = False)
        endyear (int)        -- End year of simulation (for plotting and tracking purposes).
                               (default = 2021)
        n (int)              -- Number of timesteps in the simulation, used if endyear is False. [years] (default = 1000)

    Returns:
        tuple with dict of floats: (Y, O) - Y and O pools populated with
        C-values between the given start year and year0 + n.
    """
    
    young = dict()
    old = dict()

    if startyear:
        years = list(range(startyear, endyear))
        re_d = num_to_dict(re, startyear, len(years), re_default)
    else:
        startyear = endyear - n
        years = list(range(startyear, endyear))
        re_d = num_to_dict(re, startyear, n, re_default)

    young[startyear] = 0
    old[startyear] = 0

    for i in years[1::]:
        young[i] = (young[i-1]+C_in) * math.exp(-ky * re_d[i])
        old[i] = (old[i-1] + h * (young[i-1] * (1 - math.exp(-ky * re_d[i]))))\
            * (math.exp(-ko * re_d[i]))
    
    if series:
        return (young, old)
    else:
        return (young[endyear-1], old[endyear-1])

    
def spinup_df_to_ss_soc_df(input_df: pd.DataFrame,
                           column_names: List[str],
                           h_value_dict: Dict[tuple, float],
                           area_label: str = 'area_ha',
                           verbose: bool = False
                          ) -> pd.DataFrame:
    """
    Convert input data into Soil Organic Carbon (SOC) steady-state time series and create a DataFrame.

    This function processes input data from the provided DataFrame and calculates SOC steady-state values using the ICBM model. The resulting SOC steady-state values is structured in a DataFrame with columns for separate pools: Y-pool, and O-pool.

    Args:
        input_df (pd.DataFrame): Input DataFrame containing C input data.
        column_names (list): List of column names from 'input_df' to process.
        h_value_dict (dict): Dictionary containing H-values for different crop and fraction combinations.
        area_label (str): Name of the column in 'input_df' containing area information. (default = 'area_ha')

    Returns:
        pd.DataFrame: A DataFrame with SOC steady-state time series and appropriate columns and MultiIndex.

    Example:
        # Define the input DataFrame 'input_df', column names to process, and 'h_value_dict'
        column_names = ['crop_fraction_1', 'crop_fraction_2']
        h_value_dict = {('generic_annual_crop', 'frac_1'): 0.8, ('generic_annual_crop', 'frac_2'): 0.6}
        soc_result = spinup_df_to_ss_soc_df(input_df, column_names, h_value_dict)

        # 'soc_result' will contain the calculated SOC steady-state time series in a structured DataFrame.
    """  

    if verbose:
        print('---Executing spinup_df_to_ss_soc_df()---')
    # Create empty lists to hold the return results
    y_ss_col = list()
    o_ss_col = list()
    area_col = list()
    idx_vector = list()
    fraction_vector = list()
    
    # Extract the multiindex of input_df
    idx = input_df.index
    
    for x in column_names:
    # Iterate over the columns in the input_df
        # h_val = 1 # help-variable to enable correct creation of h-value dict

        if 'crop' in x:
            # If the column referes to crop. Extract info on the input fraction and crop
            # Set to 'generic_annual crop'
            frac = x[2:4]
            crop = 'generic_annual_crop'
        elif 'manure' in x:
            # If the column referes to amemndement. Set fraction to 'amnd' and crop to 'manure'
            frac = 'amnd'
            crop = 'manure'
        else:
            # If the column does not contain crop or amendment, display error message and break
            print(f'crop is: {x}')
            print('Error, crop was not found in h-value dict')
            # h_val = 0 # set to false to skip the h-value dict step
            continue

        # Look up the h-value for each column using the info about crop and fraction.
        h = h_value_dict[crop, frac]
        
        # Append the current index level to the idx_vector
        idx_vector.append(pd.Index(idx))

        for n, i in enumerate(input_df[x]):
            # For each row in the selected column, extract the C input value and perform the following tasks
            # Calculate a soc time series using icbm_new.spinup_icbm with the C input and h-value
            c_output = spinup_icbm(i, h)
            # set the steady-state (ss) values for the young and old pool
            y_ss_col.append(c_output[0])
            o_ss_col.append(c_output[1])
            # set the area for the current prod_system and region combination
            area_col.append(input_df[area_label][idx[n]])
            # Append the current input fraction to the fraction_vector
            fraction_vector.append([x])
    
    # Turn the output timeseries into a vector to be able to represent as column data
    fraction = np.concatenate(fraction_vector)
    idx_names = input_df.index.names
    output_idx = idx_vector[0]

    if len(idx_vector) > 1:
        for i in range(1,len(idx_vector)):
            output_idx =  output_idx.append(idx_vector[i])
        
    new_idx = pd.MultiIndex.rename(pd.MultiIndex.from_tuples(output_idx), idx_names)
   
    # Store the data and names of the output dataframe in variables 
    cols = [fraction, y_ss_col, o_ss_col, area_col]   
    cols_names = ['fraction', 'y_pool', 'o_pool', 'area_ha']
    
    # Build a dict with the dtypes of the input columns to set the correct dtypes in the output_df
    cols_dtypes = dict()
    for n, i in enumerate(cols):
        cols_dtypes[cols_names[n]] = type(i[0])
        
    # Create an output df
    output_df = pd.DataFrame(data=np.transpose(cols),
                             index=new_idx,
                             columns=cols_names
                            )
    

    # Set the correct dtypes on the output_df
    output_df = output_df.astype(cols_dtypes)
    
    # Create a new multiindex with and fraction included
    output_idx = list(idx_names)
    output_idx.append('fraction')
    output_df.reset_index(inplace=True)
    output_df.set_index(output_idx, inplace=True)
    if verbose:
        print('---spinup_df_to_ss_soc_df() executed succesfully---')
            
    return output_df


def run_icbm(C_in = False,
             h = False,
             re: float = 1,
             re_default: float = 1,
             Y0: float = 0,
             O0: float = 0,
             ky: float = 0.37,
             ko: float = 0.01,
             startyear: int = 2020,
             endyear: int = False,
             n: int = 100,
             diag: bool = False
            ) -> tuple:
    """Execute the ICBM model for a single input in year 0.

    The icbm function computes the state of two pools: Y and O
    after each consecutive time-step. Y(0) and O(0) can be set to initial
    values other than 0, but if none are given, 0 will be assumed.

    The model requires two parameter values which are dependent on
    the specific soil: ky and ko. If these are not supplied by the user,
    they will be set to the updated parameter values from Bolinder (2020),
    which is a reparametrization of the original ICBM version using 20
    years of additional data from long term field trials.
    (ky: 0.8 -> 0.37; ko: 0.006 -> 0.010)

    An additional parameter can be submitted for the environmental modifier,
    re. This should be a dictionary with year: re values. For any year not
    present in the dictionary, the re value will be set to 1 using the
    helper function num_to_dict.

    Parameters:
        C_in (float or int): A single C input of a specific source material.
        h (float): The h-value associated with C_in.
        re (float or dict): Default value to create an re-dictionary or
            {year: re} dict for arbitrary years. (default = 1)
        re_default (float or int): Default re assigned to years not supplied
            in {year: re} dict. (default = 1)
        Y0 (float or int): C content in the Y pool at the beginning of year 0.
            (default = 0)
        O0 (float or int): C content in the O pool at the beginning of year 0.
            (default = 0)
        ky (float): First-order decay rate of the young pool. (default = 0.8)
        ko (float): First-order decay rate of the old pool. (default = 0.006)
        startyear (int): First year of simulation (for plotting and tracking
                                                   purposes).
            (default = 2020)
        endyear (int or False): End year of simulation (for plotting and
                                                        tracking purposes).
            If False, the parameter n is used as the number of timesteps.
            (default = False)
        n (int): Number of timesteps in the simulation,
                 used if endyear is False. [years]
            (default = 100)
        diag (bool): Flag to show diagnostic output. (default = False)

    Returns:
        tuple with dict of floats: (Y, O) - Y and O pools populated with
        C-values between the given start year and year0 + n.
    """
    young = dict()
    old = dict()

    if endyear:
        years = list(range(startyear, endyear))
        re_d = num_to_dict(re, startyear, len(years), re_default)
    else:
        years = list(range(startyear, startyear+n))
        re_d = num_to_dict(re, startyear, n, re_default)

    young[startyear] = (Y0+C_in)*math.exp(-ky*re_d[startyear])
    old[startyear] = (O0+h*(Y0*(1-math.exp(-ky*re_d[startyear]))))\
        * (math.exp(-ko*re_d[startyear]))

    for i in years[1::]:
        young[i] = young[i-1] * math.exp(-ky * re_d[i])
        old[i] = (old[i-1] + h * (young[i-1] * (1 - math.exp(-ky * re_d[i]))))\
            * (math.exp(-ko * re_d[i]))

        # for diagnostics
        if diag:
            print(f'Y[{i}] is {young[i]} and O[{i}] is {old[i]}')
            
    return (young, old)


def num_to_dict(in_num: float or dict,
                year0: int,
                n: int,
                re_default: float = 1) -> dict:
    """Create a continuous year:value dict of desired length.

    num_to_dict takes an argument and converts it to a dictionary with
    annual values from year 0 to year0+n.
    Missing dictionary keys are generated and assigned the re_default value.

    Mandatory arguments
        in_num -- either a dictionary of year:re-value pairs,
                  or a default re-value for all years
        year0  -- base year for the dictionary to be generated
        n      -- number of consecutive annual dictionary posts

    Optional arguments
        re_default  -- the default re-value used for any years missing if
                        'in_num' is supplied as a dictionary

    Returns
        num_dict    -- a continuous time-series dictionary of {year:re-value}
                        between year0 and year0+n
    """
    num_dict = dict()
    keys = list(range(year0, year0+n))

    if type(in_num) == dict:
        for i in keys:
            if i not in in_num:
                num_dict[i] = re_default
            else:
                num_dict[i] = in_num[i]
    else:
        for i in keys:
            num_dict[i] = in_num

    return num_dict


def generate_output_df(C_in: dict,
                       h_in: dict,
                       eval_intervall: list = [2020, 2100],
                       re_in: dict = 1,
                       Y0: float = 0,
                       O0: float = 0,
                       ky: float = 0.37,
                       ko: float = 0.01,
                       tot: bool = False
                      ) -> dict:
    """Create pandas dataframes with C-ppols from input time-series.

    Takes time series of C-inputs and calculates Y and O pools for each input.
    The C-pool time-series generated are stored in pandas dataframes and
    returned as dictionaries with Y and O pools as keys and
    the dataframes as values.
    Also includes the total of Y+O in the Tot dataframe.

    Mandatory arguments
        C_in  --  Annual C inputs {year:value}
        h_in  --  h-values associated with each annual C inputs {year:value}

    Optional Arguments
        re_in           -- default value to create an re-dictionary (float),
                            or {year:re} dict for arbitrary years
                            (missing years defaults to 1)
        Y0              -- C content in the Y pool at the beginning
                            of year 0 (default = 0)
        O0              -- C content in the O pool at the beginning
                            of year 0 (default = 0)
        ky              -- first-order decay rate of young pool
                            (default = 0.37)
        ko              -- first-order decay rate of old pool
                            (default = 0.01)
        eval_intervall  -- First and last year of simulation.
                            (default = [2020, 2100])
        tot             -- flag to include calculation of total SOC pool (Y+O)

    Returns
        output: a pandas dataframe holding the output of run_icbm.
    """
    # Create pandas dataframes to hold outputs of icbm runs
    column_names = list(range(eval_intervall[0], eval_intervall[1]))
    young_df = pd.DataFrame(columns=column_names, index=list(C_in.keys()))
    old_df = pd.DataFrame(columns=column_names, index=list(C_in.keys()))
    tot_df = pd.DataFrame()
    young_df.name = 'Y-pool'
    old_df.name = 'O-pool'
    if tot:
        tot_df.name = 'Tot-C'

    for i in C_in.keys():
        if i >= eval_intervall[1]:
            break

        # send individual input and associated h-values to run_icbm
        C_pools = run_icbm(C_in=C_in[i],
                           h=h_in[i],
                           re=re_in,
                           startyear=i,
                           endyear=eval_intervall[1],
                           Y0=Y0,
                           O0=O0,
                           ky=ky,
                           ko=ko)

        for j in C_pools[0]:
            young_df.loc[i, j] = C_pools[0][j]
            old_df.loc[i, j] = C_pools[1][j]

    if tot:
        tot_df.loc[:] = young_df.loc[:]+old_df.loc[:]
        return {'Y': young_df, 'O': old_df, 'Tot': tot_df}
    else:
        return {'Y': young_df, 'O': old_df}


def co2ify(df_in: pd.DataFrame,
           diag: bool = False,
           save_csv: bool = False
          ) -> pd.DataFrame:
    ''' 
    Convert a dataframe with carbon pool changes to CO2 fluxes.
    
    Subtracts the values of two consecutive years to calculate the C flows between the soil and the atmosphere.
    Converts the C flows to CO2 fluxes by multiplying by 44/12
    
     Mandatory arguments:
        df_in  -- a dataframe with carbon pool time series in rows.

    Optional arguments:
        save_csv  -- Boolean flag to save csv-files of intermediate matrix outputs. (default = False)
        
    Returns:
        (df_out, np_out)   -- a tuple with a dataframe and an np.array, each holding the same CO2 flux values
    '''
  
    columns = df_in.columns                # Save column index for later use
    # Replace NaN with zeros to get subtraction between dfs right and convert dataframe to np.array to simplify matrix ops.
    np_mtr = df_in.fillna(0).to_numpy()    

    # Create np.array with values shifted one step to the right. This respresent the values at t-1.
    np_init = np_mtr  
    np_pre = np.insert(np_init, 0, 0, axis=1)

    # Delete last column to give np.arrays same dimensions
    np_post = np.delete(np_pre, -1, axis=1)  
        
    np_out = np.subtract(np_post, np_mtr)*44/12
    df_out = pd.DataFrame(np_out, columns)

    return (df_out, np_out)


def plot_results(selection,
                 dataframes,
                 labels
                ):
    """
    @params
    selection: list with start and end years for SOC evolution of inputs to be plotted.
    A single year can be given to plot the evolution of a single input year.
    dataframes: a list of dataframes that each represent a single C pool.
    years: list of years of evaluation.
    labels: list of labels for the different pools, to be used in the plot. "-pool" will be appended.
    diag: Can be set to True to see behaviour of function.
    @result
    A plot of the SOC evolution from the inputs between start and end year.
    A curve for each pool (dataframe) is plotted.
    """
    dfs = dict()
    x = 0
    years = dataframes[0].columns.values.tolist()
    for i in dataframes:
        if len(selection) == 1:
            dfs[labels[x]] =  i.loc[selection] # if only one year is given, make a slice of a single year
        else:
            dfs[labels[x]] =  i.loc[selection[0]:selection[1]] # slice the df between (and including) start ad end years
        x += 1     
    
    plt.figure()
    plt.subplot(1,1,1)
    
    n = 0
    for i in dfs:
        plt.plot(years, dfs[i].sum(axis=0), label=(f'{labels[n]}'))
        n += 1
        
    plt.legend()
    #plt.xticks(list(range(years[0],years[-1],10)))

    plt.ylabel('C pools (kg C m-3)')
    plt.title('Initial C')

    
def classify_soil(ler,
                  sand,
                  silt
                 ):
    """Classify soil texture using HYPRES triangeln.
    
    Mandatory arguments
    ler (int):  The percentage of clay in the soil
    sand (int): The percentage of sand in the soil
    silt (int): The percentage of silt in the soil
    
    Returns
    FST (int, str): The soil class according to the FST classes
    mod_FST (str): The soil class according to the mod-FST classes,
                    as used in MACRO-DB
    CIBUS:  The soil fractions used in CIBUS input data
    """
    if ler > 60:
        FST = (5, "very fine")
        mod_FST = "k4"
        CIBUS = "clay"
    elif ler > 35:
        FST = (4, "fine")
        mod_FST = "k4"
        CIBUS = "clay"
    elif silt > 85:
        FST = (3, "")
        mod_FST = "k3"
        CIBUS = "silt"
    elif sand < 40:
        FST = (2, "medium")
        mod_FST = "k2b"
        CIBUS = "silt"
    elif sand > 65 and ler < 18:
        FST = (1, "coarse")
        mod_FST = "k1"
        CIBUS = "sand"
    else:
        FST = (2, "medium")
        mod_FST = "k2a"
        CIBUS = "sand"
        
    print(f'the FST class is {FST} and the mod-FST class is {mod_FST}')
    
    return FST, mod_FST, CIBUS


def classify_organimatter(om):
    """Classify organic matter class according to FST.
     
    Mandatory arguments
    om (int):  The percentage of organic matter in the soil
       
    Returns
    om_class (str): The soil organic matter class according to FST
    om_content (str) : The organic matter content description
    org_c (float): The fraction of organic C in the soil
    """
    if om < 3:
        om_class = "u"
        om_content = "Low"
    elif om < 6:
        om_class = "n"
        om_content = "Medium"
    else:
        om_class = "h"
        om_content = "High"
    org_c = 0.58*om
    
    print(f'The organic matter content is {om_content} and the \
    FST class is {om_class}.\n The amount of organic C is {0.58*om}%')
    
    return om_class, om_content, org_c


def run_icbm2(C_in = False,
             h = False,
             re: float = 1,
             re_default: float = 1,
             Y0: float = 0,
             O0: float = 0,
             ky: float = 0.37,
             ko: float = 0.01,
             startyear: int = 2020,
             endyear: int = False,
             inputyear: int = 2020,
             n: int = 100,
             historic: bool = False
             ) -> tuple:
    """Run the ICBM model for a single input in year 0.

    The icbm function computes the state of two pools: Y and O
    after each consecutive time-step. Y(0) and O(0) can be set to initial
    values other than 0, but if none are given, 0 will be assumed.

    The model requires two parameter values which are dependent on
    the specific soil: ky and ko. If these are not supplied by the user,
    they will be set to the updated parameter values from Bolinder (2020),
    which is a reparametrization of the original ICBM version using 20
    years of additional data from long term field trials.
    (ky: 0.8 -> 0.37; ko: 0.006 -> 0.010)

    An additional parameter can be submitted for the environmental modifier,
    re. This should be a dictionary with year: re values. For any year not
    present in the dictionary, the re value will be set to 1 using the
    helper function num_to_dict.

    Parameters:
        C_in (float or int): A single C input of a specific source material.
        h (float): The h-value associated with C_in.
        re (float or dict): Default value to create an re-dictionary or
            {year: re} dict for arbitrary years. (default = 1)
        re_default (float or int): Default re assigned to years not supplied
            in {year: re} dict. (default = 1)
        Y0 (float or int): C content in the Y pool at the beginning of year 0.
            (default = 0)
        O0 (float or int): C content in the O pool at the beginning of year 0.
            (default = 0)
        ky (float): First-order decay rate of the young pool. (default = 0.8)
        ko (float): First-order decay rate of the old pool. (default = 0.006)
        startyear (int): First year of simulation (for plotting and tracking
                                                   purposes).
            (default = 2020)
        endyear (int or False): End year of simulation (for plotting and
                                                        tracking purposes).
            If False, the parameter n is used as the number of timesteps.
            (default = False)
        inputyear (int): The year in which the first input takes place.
            All pools are set to 0 prior to this year. (default = 2020)
        n (int): Number of timesteps in the simulation,
                 used if endyear is False. [years]
            (default = 100)
        
    Returns:
        A tuple with:
            A 2-dimensional np-array with young soc and old soc as the dimensions.
            A list with the years of the modelled output period.
    """
    young = dict()
    old = dict()

    if endyear:
        years = list(range(startyear, endyear))
        re_d = num_to_dict(re, startyear, len(years), re_default)
    else:
        years = list(range(startyear, startyear+n))
        re_d = num_to_dict(re, startyear, n, re_default)
    
    for year in list(range(startyear, inputyear+1)):
        young[year] = 0
        old[year] = 0
    
    young[inputyear+1] = C_in*math.exp(-ky*re_d[inputyear])
    old[inputyear+1] = 0
    
    if historic:
        young[startyear] = Y0
        old[startyear] = O0
        for i in years[1::]:
            young[i] = young[i-1] * math.exp(-ky * re_d[i])
            old[i] = (old[i-1] + h * (young[i-1] * (1 - math.exp(-ky * re_d[i]))))\
            * (math.exp(-ko * re_d[i]))
            
    else:
        for i in years[(inputyear+1)-startyear+1::]:
            young[i] = young[i-1] * math.exp(-ky * re_d[i])
            old[i] = (old[i-1] + h * (young[i-1] * (1 - math.exp(-ky * re_d[i]))))\
            * (math.exp(-ko * re_d[i]))
       
    # create an np-array with young and old C levels 
    c_pools_np = np.array([list(young.values()), list(old.values())])
            
    return (c_pools_np, years)


def input_df_to_soc_df(input_df: pd.DataFrame,
                       column_names: List[str],
                       h_value_dict: Dict[tuple, float],
                       historic: bool = False,
                       index_frac_name: str = 'fraction',
                       area_label: str = 'area_ha',
                       year_label: object = "input_year",
                       verbose: bool = False,
                       **kwarg: object
                       ) -> pd.DataFrame:
    """
    Convert input data into Soil Organic Carbon (SOC) time series and create a DataFrame.

    This function processes input data from the provided DataFrame and calculates SOC time series using the ICBM model.
    The resulting SOC time series is structured in a DataFrame with columns for input area, output year, fraction, Y-pool, and O-pool.

    Args:
        input_df (pd.DataFrame): Input DataFrame containing C input data.
        column_names (List[str]): List of column names from 'input_df' to process.
        h_value_dict (Dict[tuple, float]): Dictionary containing H-values for different crop and fraction combinations.
        historic (bool, optional): If True, the function processes historic data; otherwise, it processes future projections. Defaults to False.
        index_frac_name (str, optional): The name of the index containing information about crop or amendment. Defaults to 'fraction'.
        area_label (str, optional): The label of the column in 'input_df' containing the area information. Defaults to 'area_ha'.
        **kwarg: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame with SOC time series and appropriate columns and MultiIndex.

    This function iterates over specified columns in 'input_df' and performs the following steps for each column:
    1. Determines the crop and fraction based on the column name.
    2. Looks up the H-value for the crop and fraction from 'h_value_dict.'
    3. Extracts the input year and area from the MultiIndex and DataFrame, respectively.
    4. Calculates SOC time series using the ICBM2 model.
    5. Constructs output year vectors, index vectors, Y-pool, and O-pool vectors.
    6. Combines vectors into a DataFrame with appropriate columns and MultiIndex.
    7. Sets correct data types for DataFrame columns.
    8. Converts output years to datetime64 format and sets the MultiIndex accordingly.

    Example:
        # Define the input DataFrame 'input_df', column names to process, 'h_value_dict', and 'idx_names'
        soc_result_historic = input_df_to_soc_df(input_df, column_names, h_value_dict, idx_names, historic=True)

        # 'soc_result_historic' will contain the calculated SOC time series for historic data.

        # Define the input DataFrame 'input_df', column names to process, 'h_value_dict', and 'idx_names'
        soc_result = input_df_to_soc_df(input_df, column_names, h_value_dict, idx_names)

        # 'soc_result' will contain the calculated SOC time series in a structured DataFrame.
    """ 
    if verbose:
        print('---Executing input_df_to_soc_df()---')

    # Create empty lists to hold the return results
    input_area_vector = list()
    output_year_vector = list()
    Y_vector = list()
    O_vector = list()
    idx_vector = list()
    fraction_vector = list()
    idx = input_df.index
    
    if historic:
        if verbose:
            print('info: calculating historic SOC')
        for n, x in enumerate(input_df.index.get_level_values(index_frac_name)):
            if 'crop' in x:
                #If the column referes to crop. Extract info on the input fraction and crop
                #set to 'generic_annual crop'
                frac = x[2:4]
                crop = 'generic_annual_crop'
            elif 'manure' in x:
                #If the column reffers to amendment. Set fraction to 'amnd' and crop to 'manure'
                frac = 'amnd'
                crop = 'manure'
            else:
                #If the column does not contain crop or amendment, display error message and break
                print('Error, crop was not found in h-value dict')
                break
            
            # Look up the h-value for each column using the info about crop and fraction.
            h = h_value_dict[crop, frac]
            # Set Y0 and O0 varaiable values
            Y0=input_df.iloc[n,0]
            O0=input_df.iloc[n,1]
            # for each row in the selected column, extract the C input value and perform the following tasks
            # Calculate a soc time series using icbm_new.run_icbm2 with the C input, h-value and input year
            c_output = run_icbm2(C_in=0, h=h, Y0=Y0, O0=O0, historic=True)
            # Append the output year list for each icbm result to the output_year_vector
            output_year_vector.append(c_output[1])
            input_area_vector.append([input_df[area_label].iloc[n]] * len(c_output[1]))
            # Append the current index level to the idx_vector
            idx_vector.append(pd.Index(idx[n]))
            # Append the current Y-pool and O-pool outputs to the Y_vector and O_vector
            Y_vector.append(c_output[0][0])
            O_vector.append(c_output[0][1]) 
    else:
        if verbose:
            print('info: calculating scenario SOC')
        for x in column_names:
        # Iterate over the columns in the input_df
            if 'crop' in x:
                #If the column referes to crop. Extract info on the input fraction and crop
                #set to 'generic_annual crop'
                frac = x[2:4]
                crop = 'generic_annual_crop'
            elif 'manure' in x:
                #If the column referes to amemndement. Set fraction to 'amnd' and crop to 'manure'
                frac = 'amnd'
                crop = 'manure'
            else:
                #If the column does not contain crop or amendment, display message and break
                print(f'{x} not an input, continuing')
                break

            # Look up the h-value for each column using the info about crop and fraction.
            h = h_value_dict[crop, frac]

            for n, i in enumerate(input_df[x]):
                # for each row in the selected column, extract the C input value and perform the following tasks
                # extract the input year from the multiindex
                input_year = idx.get_level_values(year_label)[n]
                # Calculate a soc time series using icbm_new.run_icbm2 with the C input, h-value and input year
                c_output = run_icbm2(C_in=i, h=h, startyear=2020, inputyear=input_year.year)
                # Append the output year list for each icbm result to the output_year_vector
                output_year_vector.append(c_output[1])
                input_area_vector.append([input_df[area_label].iloc[n]] * len(c_output[1]))
                # Append the current index level to the idx_vector
                idx_vector.append(pd.Index(idx[n]))
                # Append the current Y-pool and O-pool outputs to the Y_vector and O_vector
                Y_vector.append(c_output[0][0])
                O_vector.append(c_output[0][1]) 
                # Append the current input fraction to the fraction_vector
                fraction_vector.append([x] * len(c_output[1]))
    if verbose:
        print('generated vectors for Y and O pools')

    # generate an index for each output year
    output_idx = list()
    idx_names = input_df.index.names
    for m, i in enumerate(idx_vector):
        for n in range(len(output_year_vector[m])):
            output_idx.append(i)
    new_idx = pd.MultiIndex.rename(pd.MultiIndex.from_tuples(output_idx), idx_names)
    if verbose:
        print('generated an index for each output year')

    # turn the output timeseries into a vector to be able to represent as column data
    input_area = np.concatenate(input_area_vector)
    output_year = np.concatenate(output_year_vector)
    y_pool = np.concatenate(Y_vector)
    o_pool = np.concatenate(O_vector)
    if not(historic):
        fraction = np.concatenate(fraction_vector)
    if verbose:
        print('turned the timeseries output into a vector')
    
    # store the data and names of the output dataframe in variables
    if historic:
        cols = [input_area, output_year, y_pool, o_pool]
        cols_names = ['input_area', 'output_year', 'y_pool', 'o_pool']   
    else:
        cols = [input_area, output_year, fraction, y_pool, o_pool]
        cols_names = ['input_area', 'output_year', 'fraction', 'y_pool', 'o_pool']
    if verbose:
        print('stored data and names as variables')

    # build a dict with the dtypes of the input columns to set the correct dtypes in the output_df
    cols_dtypes = dict()
    for n, i in enumerate(cols):
        cols_dtypes[cols_names[n]] = type(i[0])
    if verbose:
        print('built a dict of input column dtypes')

    
    # create a dataframe with the soc time series
    soc_df = pd.DataFrame(np.transpose(cols),
                          index=new_idx,
                          columns=cols_names
                         )
    if verbose:
        print('generated dataframe with soc timeseries')

    # Set the correct dtypes on the output_df
    soc_df = soc_df.astype(cols_dtypes)
    if verbose:
        print('adjusted output_df dtypes')
    
    # turn the years in the output_year column into datetime64 format
    soc_df.output_year = pd.to_datetime(soc_df.output_year, format='%Y')
    if verbose:
        print('set output_year to datetime64')
    
    # create a new multiindex with output_year and fraction included
    soc_idx = list(idx_names)
    soc_idx.append('output_year')
    if not(historic):
        soc_idx.append('fraction')
    soc_df.reset_index(inplace=True)
    soc_df.set_index(soc_idx, inplace=True)
    if verbose:
        print('created a new multiindex')

    if verbose:
        print('---input_df_to_soc_df() executed succesfully---')

    return soc_df

def set_df_and_name(input_data: Union[pd.DataFrame, dict], session: cm.Session) -> (pd.DataFrame, str):
    try:
        # if input is a dataframe
        scn_name = str(input_data.index.get_level_values('scn').unique()[0])
        soil_input_df = input_data
    except AttributeError:
        # if inut is not a dataframe, try handling as a dict
        scenarios = {}
        for scn in session.scenarios:
            df = input[scn]
            scenarios[scn] = df
        scn_name = list(scenarios.keys())[0]
        soil_input_df = scenarios[scn_name]
    except Exception as e:
        print('Unexpected problem.')
        # Print a statement indicating the problem
        raise ValueError('input_data  must be a pd.DataFrame or a dict') from e
    return (soil_input_df, scn_name)