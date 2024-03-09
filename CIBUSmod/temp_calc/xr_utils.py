#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created on Mon Nov  20 13:46:01 2023
#
# @author: niceri

import xarray as xr
import pandas as pd
import numpy as np

from typing import Union, List, Tuple


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
    

def xr_calculate_GWP(xr_array: xr.DataArray,
                     reference_year: str = '2020',
                     end_year: str = '2050',
                     year_coord: str = 'output_year', 
                     char_fact: str = 'gwp100',
                     ghg: str = 'co2',
                     ass_rep: str = 'ipcc2013'
                    ) -> int:
    """
    Calculate the Global Warming Potential (GWP) based on a given xarray array.

    Parameters:
    -----------
    xr_array : xarray.DataArray
        The input xarray array representing the variable of interest.

    reference_year : str, optional
        The reference year for the baseline calculation. Default is '2020'.

    end_year : str, optional
        The end year for the calculation. Default is '2050'.

    year_coord : str, optional
        The name of the coordinate representing the years. Default is 'output_year'.

    char_fact : str, optional
        The characterization factor type for GWP calculation. Default is 'gwp100'.

    ghg : str, optional
        The greenhouse gas for GWP calculation. Default is 'co2'.

    ass_rep : str, optional
        The assessment report for GWP calculation. Default is 'ipcc2013'.

    Returns:
    --------
    int
        The rounded Global Warming Potential (GWP) value.

    Notes:
    ------
    - This function calculates GWP based on the given xarray array, reference year, end year, and other parameters.
    - The GWP calculation involves selecting specific years, retrieving characteristic factors, and calculating emissions.
    - The resulting GWP value is rounded to the nearest integer.

    Example:
    ---------
    # Calculate GWP for a specific xarray array
    gwp_value = xr_calculate_GWP(xr_data, reference_year='2015', end_year='2030')
    """
    # Retrieve characterization factor factor
    char_factor: float = global_params.GWP[char_fact][ass_rep][ghg]

    # Calculate emissions based on selected years
    emission: float = xr_array.sel(**{year_coord:end_year}).item() - xr_array.sel(**{year_coord:reference_year}).item()

    # Calculate GWP
    GWP: float = char_factor * emission

    # Display information about GWP calculation
    print(f'SOC in year {reference_year}: {round(xr_array.sel(**{year_coord:reference_year}).item())}')
    print(f'SOC in year {end_year}: {round(xr_array.sel(**{year_coord:end_year}).item())}')
    
    return round(GWP)



def ghg_to_temp(ghg: str ='co2',
                time_horizon: int = 100
               )-> np.ndarray:
    """
    Calculate the absolute temperature response in degrees Celsius resulting from a 1 kg emission
    of the specified greenhouse gas at time t=0.

    Parameters:
    - ghg (str, optional): The abbreviation of the greenhouse gas ('co2', 'ch4', or 'n2o'). Default is 'co2'.
    - time_horizon (int, optional): The time horizon for the evaluation. Default is 100.

    Returns:
    - dt_ghg (array): Absolute temperature response over time.

    Raises:
    - ValueError: If the provided greenhouse gas abbreviation is not 'co2', 'ch4', or 'n2o'.
  
    """
    
    n: int = time_horizon
    t: np.ndarray = np.linspace(0, n, n)

    # Constants
    c1: float = 0.631
    c2: float = 0.429
    d1: float = 8.4
    d2: float = 409.5
    gamma: int = 1
    tso2: float = 0.011
    tbc: float = 0.020
    f1: float = 0.5
    f2: float = 0.15
    M0: int = 1803
    M: int = 1804
    N0: int = 324
    N: int = 325
    C0: int = 391
    C: int = 392
    Ma: float = 28.97
    Tm: float = 5.1352e+18

    # Helper functions for temp calc
    f_mn0: float = 0.47 * np.log(1 + 2.01e-5 * (M * N0) ** 0.75 + 5.31e-15 * M * (M * N0) ** 1.52)
    f_m0n0: float = 0.47 * np.log(1 + 2.01e-5 * (M0 * N0) ** 0.75 + 5.31e-15 * M0 * (M0 * N0) ** 1.52)
    f_m0n: float = 0.47 * np.log(1 + 2.01e-5 * (M0 * N) ** 0.75 + 5.31e-15 * M0 * (M0 * N) ** 1.52)

    # Variables used in the IPCC AR5 calculations
    if ghg == 'co2':
        a0: float = 0.2173
        a1: float = 0.2240
        a2: float = 0.2824
        a3: float = 0.2763
        t1: float = 394.4
        t2: float = 36.54
        t3: float = 4.304
        AlphaC: float = 5.35
        MxC: float = 44.0098
        re_v: float = AlphaC * np.log(C / C0)
        f: float = Tm / 1000000 / Ma * MxC * (C - C0)
        re_m: float = re_v / f
        # temp calcs
        dt_ghg: np.ndarray = re_m * (a0 * c1 * (1 - np.exp(-t / d1)) + a1 * t1 * c1 / (t1 - d1) *
                           (np.exp(-t / t1) - np.exp(-t / d1)) + a2 * t2 * c1 / (t2 - d1) *
                           (np.exp(-t / t2) - np.exp(-t / d1)) + a3 * t3 * c1 / (t3 - d1) *
                           (np.exp(-t / t3) - np.exp(-t / d1)) + a0 * c2 * (1 - np.exp(-t / d2)) +
                           a1 * t1 * c2 / (t1 - d2) * (np.exp(-t / t1) - np.exp(-t / d2)) +
                           a2 * t2 * c2 / (t2 - d2) * (np.exp(-t / t2) - np.exp(-t / d2)) +
                           a3 * t3 * c2 / (t3 - d2) * (np.exp(-t / t3) - np.exp(-t / d2)))
    elif ghg == 'ch4': # Check the parameters and formulas here. This returns a value which is way too big.
        t1: float = 12.4
        AlphaM: float = 0.036
        MxM: float = 16.04276
        re_v: float = AlphaM * (np.sqrt(M) - np.sqrt(M0)) - (f_mn0 - f_m0n0)
        f: float = Tm / 1000000000 / Ma * MxM * (M - M0)
        re_m: float = re_v * (1 + f1 + f2) / f
        # temp calcs
        dt_ghg: np.ndarray = re_m * (t1 * c1 / (t1 - d1) * (np.exp(-t / t1) - np.exp(-t / d1)) + t1 * c2 / (t1 - d2) * (np.exp(-t / t1) - np.exp(-t / d2)))
    elif ghg == 'n2o':
        t1: float = 12.4
        t2: float = 121.0
        AlphaN: float = 0.12
        MxN: float = 44.01288
        re_v: float = AlphaN * (np.sqrt(N) - np.sqrt(N0)) - (f_m0n - f_m0n0)
        f: float = Tm / 1000000000 / Ma * MxN * (N - N0)
        re_m: float = re_v / f
        # temp calcs
        dt_ghg: np.ndarray = re_m * (t2 * c1 / (t2 - d1) * (np.exp(-t / t2) - np.exp(-t / d1)) +
                           t1 * c2 / (t2 - d2) * (np.exp(-t / t2) - np.exp(-t / d2)))
    else:
        raise ValueError("Invalid greenhouse gas. Options are 'co2', 'ch4', and 'n2o'.")
        
    return dt_ghg


def calc_temps(ghg: str,
               ghg_flux: float,              
               input_year: int,
               base_year: int = 2020,
               time_horizon: int = 100
              ) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Calculate the temperature response over time due to a greenhouse gas emission.

    Parameters:
    - ghg_flux (float): Greenhouse gas flux in kilograms.
    - input_year (int): The specific year of the greenhouse gas emission.
    - base_year (int): The base year for temperature calculation. Default is 2020.
    - time_horizon (int): Time horizon for the temperature response calculation. Default is 100 years.

    Returns:
    - temp_series (pd.Series): Temperature response series.
    - time_index (pd.DatetimeIndex): Time index corresponding to the temperature response.

    Note:
    - The function uses a predefined temperature curve for CO2 (tempcurve_co2) and multiplies it by the greenhouse gas flux.
    - The resulting temperature response series is limited to the specified time horizon.
    """

   
    # Calculate the end year based on the time horizon and input year
    end_year = base_year + time_horizon
    
    # Generate a time index starting from the input year to the end year with annual frequency
    time_index = pd.date_range(start=f'{input_year}-01-01',
                               end=f'{end_year - 1}-01-01',
                               freq='YS', name='temp_time')
    vector_len = len(time_index)

    # Extract the temperature curve for the ghg and multiply it by the greenhouse gas flux
    if not pd.isna(ghg_flux):
        tempcurve = ghg_to_temp(ghg)
        temp_series = tempcurve[0:vector_len] * ghg_flux
    else:
        temp_series = np.empty(vector_len)
        temp_series[:] = np.nan

    

    return temp_series, time_index

def add_temp_response(input_data: Union[xr.Dataset, xr.DataArray],
                      ghg: str = 'co2',
                      data_array_name = 'co2_flux',
                      output_label: str = 'temp_response'
                     ) -> xr.Dataset:
    """
    Add temperature response to an xarray Dataset or DataArray.

    Parameters:
    - input_data (xr.Dataset or xr.DataArray): The input data containing greenhouse gas fluxes.
    - ghg (str, optional): The greenhouse gas variable name. Default is 'co2'.
    - output_label (str, optional): The label for the temperature response variable. Default is 'temp_response'.
    - inplace (bool, optional): If True, modify the input_data in place. If False, return a new Dataset.
                                Default is True.

    Returns:
    - xr.Dataset: Returns a new Dataset.

    Raises:
    - ValueError: If the input_data is neither a Dataset nor a DataArray.
    """
    print('Starting temp calc')
    # Check the type of the input (Dataset or DataArray)
    if isinstance(input_data, xr.Dataset):
        # If it's a Dataset, use it as is
        dataset = input_data
    elif isinstance(input_data, xr.DataArray):
        # If it's a DataArray, create a new Dataset and add the DataArray to it
        dataset = xr.Dataset()
        dataset[input_data.name] = input_data
    else:
        # Raise an error if the input is neither a Dataset nor a DataArray
        raise ValueError("Input must be an xarray Dataset or DataArray.")

    # Get the name of the time dimension for the ghg fluxes
    year_dim = None
    dimensions = dataset.dims
    for dim in dimensions:
        if dataset[dim].dtype == 'datetime64[ns]':
            year_dim = dim

    print('Starting stacking operation')
    # Stack the dimensions of the dataset and save the multiindex
    stacked_ds = dataset.stack(stacked=[...])

    idx_in = stacked_ds.indexes['stacked']

    # create empty lists used to build the new output dataset
    temp_list = []
    rows_list = []

    print('Starting temp_calc inner loop')   
    for n, i in enumerate(stacked_ds[data_array_name].data):
        # Calculate temperature response iteratively for each emission and collect lists for multiindex and temperature
        temp_series, times = calc_temps(ghg, i,  idx_in.get_level_values(year_dim)[n].year)      

        for m in range(len(temp_series)):
            idx1 = idx_in[n] + (times[m],)
            rows_list.append(idx1)
            temp_list.append(temp_series[m])
    print('Done with temp_calc inner loop')
    # Build the new multiindex
    index_names = list(idx_in.names)
    index_names.append(times.name)  
    index = pd.MultiIndex.from_tuples(rows_list, names=index_names)

    # create a pd.dataseries and use to create a dataarray of the generated output
    data = pd.Series(temp_list, index=index, name=output_label)
    ds = xr.DataArray.from_series(data)

    # Merge the output dataarray with original dataset
    dataset = dataset.merge(ds) 
    print(f'Dataset created. {dataset}')

    print('dataset returned by function')
    return dataset