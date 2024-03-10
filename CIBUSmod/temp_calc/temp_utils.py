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
                time_horizon: int = 100,
                re_basis: str = 'mass'
               )-> np.ndarray:
    """
    Calculate the temperature response over a specified time horizon due to a 1 kg emission of a selected
    greenhouse gas (GHG) at time t=0, based on a chosen radiative efficiency basis (either mass or volume).

    This function computes the absolute temperature response in degrees Celsius using predefined temperature
    curves for CO2, CH4, and N2O. It accounts for the differences in radiative efficiency and atmospheric
    lifetimes of these gases.

    Parameters:
    - ghg (str, optional): The abbreviation of the greenhouse gas. Valid options are 'co2', 'ch4', or 'n2o'.
                           Default is 'co2'.
    - time_horizon (int, optional): The time horizon, in years, over which the temperature response is evaluated.
                                    Default is 100 years.
    - re_basis (str, optional): The basis for radiative efficiency calculation. Valid options are 'mass' for
                                per kilogram basis or 'vol' for per volume basis. Default is 'mass'.

    Returns:
    - np.ndarray: An array of temperature response values over the specified time horizon.

    Raises:
    - ValueError: If the provided greenhouse gas abbreviation is not among the valid options ('co2', 'ch4', 'n2o').
    - ValueError: If an invalid option is provided for the radiative efficiency basis (not 'mass' or 'vol').

    Note:
    The function utilizes predefined constants and parameters specific to each GHG to compute the temperature
    response. These include radiative efficiency, atmospheric lifetime, and other factors that influence the
    GHG's impact on global temperature.
    """
    t: np.ndarray = np.linspace(0, time_horizon, (time_horizon + 1))

    # Constants used across multiple gases
    c1, c2 = 0.631, 0.429
    d1, d2 = 8.4, 409.5
    Tm, Ma = 5.1352e+18, 28.

    # Gas-specific parameters and calculations
    parameters = {
        'co2': {
            'a': [0.2173, 0.2240, 0.2824, 0.2763],
            'tau': [394.4, 36.54, 4.304],
            'Alpha': 5.35,
            'Mx': 44.0098,
            'C': [391, 392]
        },
        'ch4': {
            'tau': 12.4,
            'Alpha': 0.036,
            'Mx': 16.04276,
            'C': [1803, 1804],  # M values for CH4
            'factors': [0.5, 0.15]  # f1, f2 for CH4
        },
        'n2o': {
            'tau': [121.0, 12.4],
            'Alpha': 0.12,
            'Mx': 44.01288,
            'C': [324, 325],  # N values for N2O
        }
    }

    if ghg not in parameters:
        raise ValueError("Invalid greenhouse gas. Options are 'co2', 'ch4', and 'n2o'.")

    if ghg == 'co2':
        param = parameters[ghg]
        a = param['a']
        tau = param['tau']
        Alpha = param['Alpha']
        Mx = param['Mx']
        C0, C = param['C']

        # Calculate radiative efficiency per volume and per mass
        re_v = Alpha * np.log(C / C0)
        if re_basis == 'mass':
            f = Tm / 1e6 / Ma * Mx * (C - C0)
            re_m = re_v / f
        elif re_basis == 'vol':
            re_m = 1
        else:
            print("Valid options for the radiative efficiency basis keyword (re_basis) are 'mass' or 'vol'")
            print(f'{re_basis} was entered')
            return

        # AGTP calculation for CO2
        dt_ghg = np.zeros_like(t)  # Initialize the array to store temperature changes
        for i in range(len(a)):
            dt_ghg += re_m * (a[i] * (c1 * (1 - np.exp(-t / d1)) + c2 * (1 - np.exp(-t / d2))) if i == 0 else \
                a[i] * tau[i - 1] * (c1 / (tau[i - 1] - d1) * (np.exp(-t / tau[i - 1]) - np.exp(-t / d1)) + \
                                     c2 / (tau[i - 1] - d2) * (np.exp(-t / tau[i - 1]) - np.exp(-t / d2))))

    elif ghg == 'ch4':
        param = parameters[ghg]
        t1 = param['tau']
        M0, M = param['C']
        N0, N = parameters['n2o']['C']
        AlphaM = param['Alpha']
        MxM = param['Mx']
        f1, f2 = param['factors']

        # Helper functions calculations
        f_mn0 = 0.47 * np.log(1 + 2.01e-5 * (M * N0) ** 0.75 + 5.31e-15 * M * (M * N0) ** 1.52)
        f_m0n0 = 0.47 * np.log(1 + 2.01e-5 * (M0 * N0) ** 0.75 + 5.31e-15 * M0 * (M0 * N0) ** 1.52)
        f_m0n = 0.47 * np.log(1 + 2.01e-5 * (M0 * N) ** 0.75 + 5.31e-15 * M0 * (M0 * N) ** 1.52)

        # Radiative efficiency calculations
        re_v = AlphaM * (np.sqrt(M) - np.sqrt(M0)) - (f_mn0 - f_m0n0)
        if re_basis == 'mass':
            f = Tm / 1e9 / Ma * MxM * (M - M0)
            re_m = re_v * (1 + +f1 + f2) / f
        elif re_basis == 'vol':
            re_m = 1
        else:
            print("Valid options for the radiative efficiency basis keyword (re_basis) are 'mass' or 'vol'")
            print(f'{re_basis} was entered')
            return


        # Temperature response time series calculation
        dt_ghg = re_m * (t1 * c1 / (t1 - d1) * (np.exp(-t / t1) - np.exp(-t / d1)) +
                         t1 * c2 / (t1 - d2) * (np.exp(-t / t1) - np.exp(-t / d2)))

    elif ghg == 'n2o':
        param = parameters[ghg]
        t1, t2 = param['tau']
        M0, M = parameters['ch4']['C']
        N0, N = param['C']
        AlphaN = param['Alpha']
        MxN = param['Mx']

        # Helper functions calculations
        f_m0n0 = 0.47 * np.log(1 + 2.01e-5 * (M0 * N0) ** 0.75 + 5.31e-15 * M0 * (M0 * N0) ** 1.52)
        f_m0n = 0.47 * np.log(1 + 2.01e-5 * (M0 * N) ** 0.75 + 5.31e-15 * M0 * (M0 * N) ** 1.52)

        # Radiative efficiency calculations
        re_v = AlphaN * (np.sqrt(M) - np.sqrt(M0)) - (f_m0n - f_m0n0)
        if re_basis == 'mass':
            f = Tm / 1e9 / Ma * MxN * (N - N0)
            re_m = re_v / f
        elif re_basis == 'vol':
            re_m = 1
        else:
            print("Valid options for the radiative efficiency basis keyword (re_basis) are 'mass' or 'vol'")
            print(f'{re_basis} was entered')
            return

        # Temperature response time series calculation
        dt_ghg = re_m * (t2 * c1 / (t2 - d1) * (np.exp(-t / t2) - np.exp(-t / d1)) +
                                     t1 * c2 / (t2 - d2) * (np.exp(-t / t2) - np.exp(-t / d2)))

    else:
            raise ValueError("Invalid greenhouse gas. Options are 'co2', 'ch4', and 'n2o'.")
        
    return dt_ghg


def add_temp_response(input_data: Union[xr.Dataset, xr.DataArray],
                      ghg: str = 'co2',
                      data_array_name = 'co2_flux',
                      output_label: str = 'temp_response',
                      re_basis: str = 'mass',
                      base_year: int = 2020,
                      time_horizon: int = 100
                     ) -> xr.Dataset:
    """
    Add temperature response to an xarray Dataset or DataArray based on greenhouse gas emissions.

    This function calculates the temperature response for a given greenhouse gas (GHG) emission dataset over a specified time horizon and base year.
    It precomputes temperature response curves for CO2, CH4, and N2O based on the specified re_basis ('mass' by default) and then applies these curves to the GHG flux data contained within the input dataset to calculate the temperature response.
    The resulting temperature responses are added to the input dataset as a new data variable.

    Parameters:
    - input_data (xr.Dataset or xr.DataArray): Input data containing greenhouse gas fluxes.
    - ghg (str, optional): The greenhouse gas for which the temperature response is calculated. Default is 'co2'.
    - data_array_name (str, optional): The name of the data array within input_data that contains GHG emissions. Default is 'co2_flux'.
    - output_label (str, optional): The label for the new temperature response data variable to be added to the dataset. Default is 'temp_response'.
    - re_basis (str, optional): The basis for temperature response calculation ('mass' or other specified basis). Default is 'mass'.
    - base_year (int): The base year from which the time horizon for temperature response calculation starts. Default is 2020.
    - time_horizon (int): The time horizon (in years) over which to calculate the temperature response. Default is 100 years.

    Returns:
    - xr.Dataset: A new Dataset containing the original data along with the added temperature response data variable.

    Raises:
    - ValueError: If the input_data is neither an xarray Dataset nor an xarray DataArray, or if an unsupported GHG is specified.
    """
    print('Starting temp calc')
    # Check the type of the input (Dataset or DataArray)
    if isinstance(input_data, xr.Dataset):
        dataset = input_data
    elif isinstance(input_data, xr.DataArray):

        dataset = xr.Dataset({input_data.name: input_data})
    else:
        raise ValueError("Input must be an xarray Dataset or DataArray.")

    # Precompute temperature response curves
    tempcurve_co2 = ghg_to_temp('co2', time_horizon=time_horizon, re_basis=re_basis)
    tempcurve_ch4 = ghg_to_temp('ch4', time_horizon=time_horizon, re_basis=re_basis)
    tempcurve_n2o = ghg_to_temp('n2o', time_horizon=time_horizon, re_basis=re_basis)

    # Select the appropriate temperature curve based on the specified GHG
    if ghg == 'co2':
        base_tempcurve = tempcurve_co2
    elif ghg == 'ch4':
        base_tempcurve = tempcurve_ch4
    elif ghg == 'n2o':
        base_tempcurve = tempcurve_n2o
    else:
        raise ValueError(f"Unsupported GHG: {ghg}")

    # Get the name of the time dimension for the ghg fluxes
    year_dim = [dim for dim in dataset.dims if dataset[dim].dtype == 'datetime64[ns]'][0]

    print('Starting stacking operation')
    all_dims = list(dataset.dims)
    stacked_ds = dataset.stack(stacked=all_dims)

    idx_in = stacked_ds.indexes['stacked']

    temp_list = []
    rows_list = []

    print('Starting temp_calc inner loop')


    for n, ghg_flux in enumerate(stacked_ds[data_array_name].data):
        input_year = idx_in.get_level_values(year_dim)[n].year
        offset = time_horizon - (input_year - base_year)
        time_index = pd.date_range(start=pd.Timestamp(year=input_year, month=1, day=1),
                                   periods=offset,
                                   freq='YS', name='temp_time')

        if np.isnan(ghg_flux):
            temp_series = pd.Series(np.nan, index=time_index)
        else:
            temp_series = pd.Series(base_tempcurve[:len(time_index)] * ghg_flux, index=time_index)

        for m, temp in enumerate(temp_series):
            idx1 = idx_in[n] + (time_index[m],)
            rows_list.append(idx1)
            temp_list.append(temp)

    print('Done with temp_calc inner loop')
    # Build the new multiindex
    index_names = list(idx_in.names) + ['temp_time']
    index = pd.MultiIndex.from_tuples(rows_list, names=index_names)

    # create a pd.dataseries and use to create a dataarray of the generated output
    data = pd.Series(temp_list, index=index, name=output_label)
    ds = xr.DataArray.from_series(data)

    # Merge the output dataarray with original dataset
    dataset = dataset.merge(ds)
    print(f'Dataset created. {dataset}')

    print('dataset returned by function')
    return dataset