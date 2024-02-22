#!/usr/bin/env python

"""
Utility functions for the CIBUSmod_soil program

"""

import csv
import json
import os

# from ipyfilechooser import FileChooser
from IPython import display
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Any, Optional
import xarray as xr

import CIBUSmod.soil.soil_params as soil_params

from ..soil import input_path, export_path
from ..soil import temp_path
from ..soil import root

def colored_rule(color: str,
                 height: int = 1) -> None:
    '''
    Add a colored horizontal rule after the execution of a jupyter cell.

    Parameters
    ----------
    color : str
        Color of the horizontal rule.
    height : int, optional
        Height of the horizontal rule, default is 1.

    Returns
    -------
    None
    '''
    
    rule_html = f'<hr style="border: 0; height: {height}px; background-color: {color};">'
    display.display(display.HTML(rule_html))


### Commented function: FileChoose presently unused in project
# def select_file(filt: str = '*.csv',
#                 only_dirs: bool = False):
#     '''
#     Opens a file chooser dialog for selecting a file.
#
#     Parameters:
#     -----------
#         filt (str, optional): File filter pattern to restrict selectable file types. Defaults to '*.csv'.
#         only_dirs (bool, optional): If True, only directories can be selected. Defaults to False.
#
#     Returns:
#     --------
#         FileChooser: A FileChooser object representing the selected file.
#
#     Behaviour:
#         - Creates a FileChooser (fc) object.
#         - Sets the filter pattern for file selection based on the provided or default filter.
#         - Displays the file chooser dialog.
#         - Returns the FileChooser object, which can be used to access information about the selected file.
#     '''
#
#     fc = FileChooser()
#     fc.default_path = f'{root}/data/soil'
#     fc.show_hidden = False
#     fc.filter_pattern = filt
#     fc.reset(path=f'{root}/data/soil')
#     fc.show_only_dirs = only_dirs
#
#     return fc


def set_scn_name(input, default):
    '''
    Set scenario name.

    Parameters
    ----------
    input : str
        Input scenario name.
    default : str
        Default scenario name to use if input is an empty string.

    Returns
    -------
    str
        Scenario name. If input is an empty string, returns the default scenario name.

    Examples
    --------
    >>> set_scn_name('Scenario1', 'DefaultScenario')
    'Scenario1'

    >>> set_scn_name('', 'DefaultScenario')
    'DefaultScenario'
    '''
    if input == '':
        return default
    else:
        return input
        


def make_df_lower(dataframe: pd.DataFrame,
                  mode: Tuple[int, int, int, int] = (1, 2, 3, 4),
                  index_sorted: bool = True) -> pd.DataFrame:
    '''
    Transform the data, index, and column strings of a pandas DataFrame to lowercase, without altering names.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be case-sanitized.
    mode : tuple, optional
        List of switches that control which elements of the DataFrame to operate on:
        - 1: index labels
        - 2: column labels only
        - 3: index values
        - 4: data strings
        All four are selected as default. (default: (1, 2, 3, 4))
    index_sorted : bool, optional
        Apply sort_index to make operations on the generated DataFrame more efficient. (default: True)

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with all index and column names in lowercase.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['X', 'Y', 'Z']}, index=['one', 'Two', 'ThReE'])
    >>> make_df_lower(df)
          a  b
    one   1  X
    two   2  Y
    three 3  Z

    >>> make_df_lower(df, mode=(2, 4))
          a  b
    one   1  x
    two   2  y
    three 3  z
    '''
    
    df = dataframe.copy()
    df.sort_index(inplace=True)
    
    if 1 in mode: # turn index labels lower case
        if df.index.names[0] == None:
            print('Warning, first index label is None, conversion of index label not possible')
        else:
            dx = df.index
            labels = df.index.names
            df.reset_index(inplace=True)
            old_names = list(labels)
            new_names = list()
            if len(labels) == 1:
                if isinstance(labels[0], str):
                    new_names.append(labels[0].lower())
                    names_dict = dict(zip(old_names, new_names))
                else:
                    new_names.append(labels[0])
                    names_dict = dict(zip(old_names, new_names))
            else:
                for n, i in enumerate(old_names):
                    if isinstance(labels[n], str):
                        new_names.append(labels[n].lower())
                        names_dict = dict(zip(old_names, new_names)) 
                    else:
                        new_names.append(labels[n])
                        names_dict = dict(zip(old_names, new_names))
            df.rename(columns=names_dict, inplace = True)
            df.set_index(new_names, inplace=True)

    if 2 in mode: # turn column labels lower case
        old_names = list(df.columns)
        new_names = list(df.columns.str.lower())
        names_dict = dict(zip(old_names, new_names)) 
        df.rename(columns=names_dict, inplace = True)
                
    if  3 in mode: # turn index values lower
        if df.index.names[0] == None:
            print('Warning, index label is None. No conversion of index values (should be auto-generated ints)')
        else:
            idx = df.index
            labels = df.index.names
            df.reset_index(inplace=True)
            for n, i  in enumerate(labels):
                for l, j in enumerate(idx):
                    if isinstance(df.iloc[l, n], str):
                        df.iloc[l, n] =(df.iloc[l, n]).lower()
            df.set_index(labels, inplace=True)             

    if 4 in mode:  # make data strings lower case
        if isinstance(df.index, pd.core.indexes.range.RangeIndex):
            for i in df.columns:
                if isinstance(df[i][0], str):
                    df[i] = df[i].str.lower()
        else:
            for i in df.columns:
                for j in df.index:
                    if isinstance(df.loc[j, i], str):
                        df.loc[j, i] = df.loc[j, i].lower()

    if index_sorted:
        return df.sort_index()
    else:
        return df


def map_cin_h_to_dataframe(match_col: str,
                           input_df: pd.DataFrame,
                           mapping_df: pd.DataFrame,
                           output_col_name: str = 'h_value',
                           index_sorted: bool = True) -> pd.DataFrame:
    """
    Calculates a multiindex dataframe containing a mapping from crop to individual carbon input fractions used in icbm
    
    Parameters:
    -----------
    match_col : str
        The name of the column in the 'input_df' that is used to match the first index in 'mapping_df'
    input_df : pd.DataFrame
        Dataframe with original crop names as index and name for matching h-values in a column (match_col)
    mapping_df : pd.DataFrame
        Multiindex dataframe with h-values. First index values correspond to match_col, second index to fractions
    output_col_name : str
        Column name for the h-values (Default='h_value')
    index_sorted : bool
        Sort_index applied to make operation on generated dataframe more efficient (default: True)
    
    Returns:
    --------
    pd.DataFrame:
        A multiindex dataframe with index:(input, fraction) and column value: h-value
    """
    
    input_idx = input_df.index
    output_idx = pd.MultiIndex(levels=[[], []],
                               codes=[[], []],
                               names=[u'input', 'fraction'])
    output_df = pd.DataFrame(index=output_idx, columns=[output_col_name])
    for i, x in enumerate(input_df[match_col]):
        crop = input_df.index[i]
        for y in mapping_df.loc[x].index:
            value = mapping_df.loc[(x, y),:].item()
            output_df.loc[(crop, y), :] = value
            
    if index_sorted:
        return output_df.sort_index()
    else:
        return output_df


def filter_namelist(name: Union[str, List[str]],
                    iterable: List[str]) -> List[str]:
    """
    Get a list of variable names containing the specified substring from an iterable.

    This function searches through the elements of the provided iterable and compiles a list of variable names that
    contain the specified 'name' as a substring. It is particularly useful for filtering and extracting specific
    variables based on naming conventions within an iterable.

    Parameters:
    -----------
    name : Union[str, List[str]]
        The substring or list of substrings to search for in variable names.
    iterable : List[str]
        The iterable to search within, typically containing variable names.

    Returns:
    --------
    List[str]
        A list of variable names from the provided iterable that contain the specified substring 'name'.


    Example use case:
    --------------
    If 'name' is 'manure_' and 'iterable' is a list of variable names, the function will return a list of variable names
    that contain 'manure_' in their names from the provided iterable.
    """
    
    x = []
    if isinstance(name, str):
        for i in iterable:
            if name in i:
                x.append(i)
    else:
        for j in name:
            for i in iterable:
                if j in i:
                    x.append(i)
    return x


def get_filtered_namelist(filterlist: str,
                          namelist: Union[str, List[str]],
                          iterable: pd.DataFrame) -> List[str]:
    """
    Apply a filter to a list of column names in a DataFrame and return the filtered result.

    Parameters:
    -----------
    filterlist : str
        The substring to filter the column names by.
    namelist : Union[str, List[str]]
        The column names or list of substrings to search within.
    iterable : pd.DataFrame
        The DataFrame to search within.

    Returns:
    --------
    List of str
        A filtered list of column names from the provided DataFrame that match the specified filter criteria.

    Example use case:
    ----------------
    If 'filter' is 'temperature' and 'namelist' is ['air_', 'soil_'], and 'dataframe' is 'measurement_data_df', the
    function will return a list of column names in 'measurement_data_df' that contain 'temperature' and match either 'air_'
    or 'soil_' from 'namelist'.
    """
    
    new_iter = filter_namelist(namelist, iterable)
    output = filter_namelist(filterlist, new_iter)
    
    return output


def make_idx_continuous(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the index of a DataFrame continuous and reset the index.

    This function takes a DataFrame and replaces its index with a continuous integer index starting from 0.
    It can be useful for creating a new, clean index for a DataFrame.

    Parameters:
    -----------
    input_df : pd.DataFrame
        The input DataFrame with the existing index.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with a continuous integer index starting from 0.

    Example:
    ---------
    # Create an input DataFrame 'original_df'
    >>> original_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # Call the function to make the index continuous
    >>> new_df = make_idx_continuous(original_df)
    # 'new_df' will have a continuous integer index starting from 0.
    """
    
    new_idx = pd.Index(range(len(input_df.index.values)))
    output_df = input_df.set_index(new_idx)
    
    return output_df


def calculate_c_inputs(input_df: pd.DataFrame,
                       mappings: Tuple[Dict[str, Any], ...],
                       sources: Tuple[str, ...],
                       c_allo_df: Tuple[pd.DataFrame, ...],
                       crop_colname: str = 'crop',
                       colprefix: Optional[Dict[str, str]] = None,
                       straw_removed: bool = False,
                       verbose=False) -> pd.DataFrame:
    """
    Calculate carbon allocation factors 'i_ag' and 'i_bg' for crops based on input data and update the input DataFrame.

    This function iterates through the crops in the input DataFrame and, for each crop, checks if carbon allocation data is available for it in the sources provided. 
    If data is found, it calculates 'i_ag' and 'i_bg' values and stores them in the input DataFrame.
    If no data is found, 'i_ag' and 'i_bg' are set to NaN. The 'alloc_source_crop' column is also updated with the data source.

    If there are several matching sources for allocation factors, the first hit in the mappings tuple takes precedence.
    Organize the input order accordingly. The order of sources and c_allo_df should also be the same as that in mappings for the correct operation of this function.

    Note: This function is currently hard-coded to work with the sources Andren2004, Jacobs2020 and Hanna.
          If other sources are used, the function code first needs to be manually adjusted to know how to handle the new sources.

    Parameters:
    -----------
    input_df : pd.DataFrame
        The input DataFrame containing crop and area yield data.
    mappings : Tuple[Dict[str, Any], ...]
        Tuple of dictionaries mapping crops to relevant data.
    sources : Tuple[str, ...]
        Tuple of data sources corresponding to each mapping.
    c_allo_df : Tuple[pd.DataFrame, ...]
        Tuple of DataFrames containing carbon allocation data for each source.
    crop_colname : str, optional
        Name of the crop column. Defaults to 'crop'.
    straw_removed : bool, optional
        Whether to consider straw removal. Defaults to False.

    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with 'i_ag', 'i_bg', and 'alloc_source_crop' columns.
        The original dataframe is not changed
    """

    if verbose:
        print('---Executing calculate_c_inputs()---')
    # set mutable default args if no args given
    if colprefix is None:
        colprefix = {'ag': 'i_ag', 'bg': 'i_bg'}

    output_df = input_df.copy()
    ag_ha_colname = f"{colprefix['ag']}_{crop_colname}_ha"
    bg_ha_colname = f"{colprefix['bg']}_{crop_colname}_ha"
    ag_tot_colname = f"{colprefix['ag']}_{crop_colname}_kgc"
    bg_tot_colname = f"{colprefix['bg']}_{crop_colname}_kgc"
          
    for n, i in enumerate(input_df[crop_colname]):
        for m in range(len(mappings)):
            if mappings[m].get(i, 'missing') == 'missing':
                continue
            if sources[m] == 'Andren2004': # if allometric functions for the crop can be found in Andren et al. 2004
                ag, bg = allom_asH(input_df.areayield[n], mappings[m][i], c_allo_df[m], straw_removed=straw_removed)
                output_df.at[n, ag_ha_colname] = ag
                output_df.at[n, bg_ha_colname] = bg
                output_df.at[n, f'alloc_source_{crop_colname}'] = sources[m]
            elif sources[m] == 'Jacobs2020': # use Jacobs
                if straw_removed:
                    ag, bg = alloc_input(input_df.areayield[n], mappings[m][i], c_allo_df[m])
                    output_df.at[n, ag_ha_colname] = ag - input_df.areayield_residues[n]
                    output_df.at[n, bg_ha_colname] = bg
                    output_df.at[n, f'alloc_source_{crop_colname}'] = sources[m]
                else:
                    ag, bg = alloc_input(input_df.areayield[n], mappings[m][i], c_allo_df[m])
                    output_df.at[n, ag_ha_colname] = ag
                    output_df.at[n, bg_ha_colname] = bg
                    output_df.at[n, f'alloc_source_{crop_colname}'] = sources[m]
            else: # use sources provided by Hanna
                if straw_removed:
                    ag, bg = alloc_input(input_df.areayield[n], mappings[m][i], c_allo_df[m])
                    output_df.at[n, ag_ha_colname] = ag - input_df.areayield_residues[n]
                    output_df.at[n, bg_ha_colname] = bg
                    output_df.at[n, f'alloc_source_{crop_colname}'] = c_allo_df[m].loc[mappings[m][i], 'source']
                else:
                    ag, bg = alloc_input(input_df.areayield[n], mappings[m][i], c_allo_df[m])
                    output_df.at[n, ag_ha_colname] = ag
                    output_df.at[n, bg_ha_colname] = bg
                    output_df.at[n, f'alloc_source_{crop_colname}'] = c_allo_df[m].loc[mappings[m][i], 'source']
            break
        else:
            # if there are no C allocation factors or allometric functions present
            output_df.at[n, ag_ha_colname] = np.nan
            output_df.at[n, bg_ha_colname] = np.nan
            output_df.at[n, f'alloc_source_{crop_colname}'] = 'none'

    # Insert columns with i_ag and i_bg per sko, base on values per ha
    output_df[ag_tot_colname] = output_df[ag_ha_colname] * output_df['area_ha']
    output_df[bg_tot_colname] = output_df[bg_ha_colname] * output_df['area_ha']

    if verbose:
        print('---Leaving calculate_c_inputs()---')

    return output_df


def allom_asH(H: float,
              crop: str,
              param_df: pd.DataFrame,
              straw_removed: bool = False) -> Tuple[float, float]:
    """
    Calculates the above and below ground input for 'crop' using the allometric function C = a + sH
    and parametrisation found in Andren et al(2004)

    H is harvested yield.
    C is the amount of carbon allocated to the other parts of the crop.
    a and s are empirically determined.

    The above ground fractions are divided into straw and stubble (residues in c_allom_andren2004).
    Straw can be removed, but stubble always remain in the field.
    The below ground fraction is just one fraction called roots

    The names of the columns in 'param_df' can be arbitrary,
    but the order of the columns in 'param_df' need to be:
    [0] parameter 'a' for straw
    [1] parameter 's' for straw
    [2] parameter 'a' for stubble
    [3] parameter 's' for stubble
    [4] parameter 'a' for roots
    [5] parameter 's' for roots

    Parameters:
    -----------
    H : float
        harvested yield for 'crop'
    crop : str
        name of the 'crop'
    param_df : pd.DataFrame
        dataframe with the allometric function parameters
    straw_removed : bool, optional
        If True, carbon input in straw is not included in the returned i_ag (default=False)

    Returns:
    --------
    Tuple[float, float]
        a tuple with above and below ground carbon input
    """
    
    if straw_removed:
        i_ag = param_df.iloc[:,2][crop] + param_df.iloc[:,3][crop] * H
    else:
        i_ag_1 = (param_df.iloc[:,0][crop] + param_df.iloc[:,1][crop] * H) 
        i_ag_2 = + (param_df.iloc[:,2][crop] + param_df.iloc[:,3][crop] * H)
        i_ag = i_ag_1 + i_ag_2
    i_bg = + (param_df.iloc[:,4][crop] + param_df.iloc[:,5][crop] * H)
    return i_ag, i_bg


def alloc_input(H: float,
                crop: str,
                input_df: pd.DataFrame,
                straw_removed: bool = False) -> Tuple[float, float]:
    """
    Calculates the above and below ground input based on the allocation factors from the input dataframe
    and harvested yield of a given crop

    The names of the columns in 'input_df' can be arbitrary,
    but the order of the columns in 'input_df' need to be:
    [0] above ground input allocation factor
    [1] below ground input allocation factor

    Parameters:
    -----------
    H : float
        harvested yield for 'crop'
    crop : str
        name of the 'crop'
    input_df : pd.DataFrame
        dataframe with the allometric function parameters
    straw_removed : bool, optional
        Setting to True has no effect in current version of function (default=False)

    Returns:
    --------
    Tuple[float, float]
        a tuple with above and below ground carbon input
    """

    i_ag = input_df.iloc[:,0][crop] * H
    i_bg = input_df.iloc[:,1][crop] * H
    
    return i_ag, i_bg


def to_csv_preserved(dataframe: pd.DataFrame,
                     save_as: str,
                     set_savedir: str = 'support_csv',
                     save_path: str | bool = False,
                     save_type: str = 'none',
                     **kwarg: Any) -> None:
    '''
    Save a DataFrame to a CSV file with preserved data types and index information.

    This function saves a DataFrame to a CSV file while preserving data types and index settings.
    Supporting information is stored in a separate CSV file (with "_help" suffix) in the specified directory.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The DataFrame to be saved.
    save_as : str
        The name the file is saved as
    set_savedir : str, optional
        Directory for saving supporting information. Defaults to 'support_csv'.
    save_path: str | bool
        Optional variable to enable manually setting save_path to other than presets.
    save_type : str
        determines where to save file. 
        Can be either 'none'(default), 'temp' or 'result'
        if 'none', save_path is set to current working dir
        if 'temp',  save_path is set to 'temp_path'
        if 'result',  save_path is set to 'export_path'
    **kwarg : Any
        Additional keyword arguments for DataFrame.to_csv().

    Returns:
    --------
    None
    '''
  
    # Set the directory according to save_type
    if save_path:
        save_path = save_path
    elif save_type == 'none':
        # Save the current working directory
        save_path = os.getcwd()
    elif save_type == 'temp':
        save_path = temp_path
    elif save_type == 'result':
        save_path = export_path
    else:
        print('the save_type parameter is not set correctly. Please revise')

    # Save the DataFrame to a CSV file (all index info will be lost)
    dataframe.to_csv(f'{save_path}/{save_as}.csv', **kwarg)

    # Create a list of all the index levels that should be datetime formatted when reloaded
    idx = dataframe.index
    idx_names_list = list(idx.names)
    if isinstance(idx, pd.RangeIndex):
        datetime_levels = []
    else:
        datetime_levels = [name for name, dtype in zip(idx_names_list, idx.dtypes) if np.issubdtype(dtype, np.datetime64)]
    

    # Create a list of all the column names that should be datetime formatted when reloaded
    datetime_cols = [col for col in dataframe.columns if np.issubdtype(dataframe[col].dtypes, np.datetime64)]

    # Save the supporting info in a dictionary
    if datetime_levels:
        data = {'index_names': idx_names_list, 'datetime_levels': datetime_levels, 'datetime_cols': datetime_cols}
    else:
        data = {'index_names': idx_names_list, 'datetime_cols': datetime_cols}

    # Serialize data: Convert lists to JSON strings before saving
    for key in data.keys():
        data[key] = json.dumps(data[key])

    # Create save_dir if it does not exist
    
    current_dir = os.curdir
    os.chdir(save_path)
    if not os.path.isdir(set_savedir):
        os.mkdir(set_savedir)
    os.chdir(current_dir)
    

    # Save the dictionary to a CSV file, with the same name as the DataFrame
    file_name = f'{save_as}_help'
        
    with open(f'{save_path}/{set_savedir}/{file_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        writer.writerow(data.values())

    return


def read_csv_preserved(filepath,
                       set_helpdir='support_csv',
                       **kwargs):
    """
    Read a CSV file with preserved data types and index, using supporting information from a separate file.

    Args:
    filepath (str): Path to the CSV file.
    set_helpdir (str, optional): Directory containing supporting information files. Defaults to 'support_csv'.
    **kwargs: Additional keyword arguments.

    Returns:
    pd.DataFrame: DataFrame with preserved data types and index.

    This function reads a CSV file while preserving data types and index settings.
    The supporting information is stored in a separate file (CSV with "_help" suffix) in the specified directory.

    """

    current_dir = os.getcwd()

    # Extract file information
    filename = os.path.basename(filepath)
    df_dir = os.path.dirname(filepath)
    name, extension = os.path.splitext(filename)

    # Initialize help_dict
    help_dict = {}

    # Open the CSV-help file for reading
    with open(f'{df_dir}/{set_helpdir}/{name}_help.csv', mode='r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        # Read the first row (keys)
        keys = next(csv_reader)
        # Read the second row (values)
        values = next(csv_reader)

    # Convert JSON strings back to Python objects
    help_dict = {key: json.loads(value) for key, value in zip(keys, values)}

    # Extract datetime_cols, index_cols, and datetime_levels
    datetime_cols = help_dict.get('datetime_cols', None)
    index_cols = help_dict.get('index_names', None)
    datetime_levels = help_dict.get('datetime_levels', None)

    if datetime_levels is None:
        if len(index_cols) > 1:
            # if there are no datetime values in the index, create the df and recreate the index
            output_df = pd.read_csv(filepath, parse_dates=datetime_cols)
            output_df = output_df.set_index(index_cols)
        else:
            # If column index was a range index, recreate the CSV file using row 0 as index
            # to avoid duplicating the index. Also set dtype of datetime index levels (columns) back to datetime format
            output_df = pd.read_csv(filepath, index_col=0, parse_dates=datetime_cols)

    else:
        # Set dtype of datetime index levels and columns back to datetime format
        output_df = pd.read_csv(filepath, parse_dates=datetime_cols)
        for i in datetime_levels:
            output_df[i] = pd.to_datetime(output_df[i])
        # Recreate the multiindex
        output_df = output_df.set_index(index_cols)

    return output_df


def add_prefix(name: str,
               prefix: str,
               input_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add a prefix to variable names in the column headers of a DataFrame.
    
    This function takes a DataFrame and modifies the names of columns in the DataFrame by adding a specified 'prefix' to
    the variable names that contain the 'name' substring. It is useful for renaming and grouping variables in the
    DataFrame based on a common naming convention.

    Parameters:
    -----------
    name : str
        The substring to identify variable names in the DataFrame.
    prefix : str
        The prefix to add to the variable names.
    input_df : pd.DataFrame
        The DataFrame containing the variables to be renamed.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the variable names in the column headers modified by adding the specified prefix.


    Example use case:
    -----------------
    If 'name' is 'manure_' and 'prefix' is 'new', the function will add the 'new_' prefix to all variable names in the
    DataFrame 'input_df' that contain 'manure_' in their names, and return the modified DataFrame.
    '''
    
    # Generate a list of names containing 'name' in the column headers of 'input_df'
    namelist = filter_namelist(name, input_df.columns)
    # Prepend a prefix and '_'to each name in 'namelist'
    prefixed_namelist = [prefix + '_' + x for x in namelist]
    # Create a dictionary to use as name mapping between namelist and input_namelist
    mapping_dict = dict(zip(namelist, prefixed_namelist))
    # Change the names of the 'input_df' columns using mapping_dict
    output_df = input_df.rename(mapper=mapping_dict, axis=1)
    
    return output_df


def crop_map_helper(input_df=False,
                    re_col_name='re',
                    verbose=False
                    ):
    """
    Map CIBUS crop types to corresponding re-crop types using a provided or default dataframe.

    Parameters:
    - input_df (pd.DataFrame, optional): DataFrame containing mapping of CIBUS crop types to re-crop types.
      If not provided, default values are loaded from '<input_path>/crop_carbon_map.csv'.
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
            print(f"Creating 'input_df' from 'crop_carbon_map.csv'in {input_path}")
        input_df = make_df_lower(pd.read_csv(f'{input_path}/crop_carbon_map.csv', index_col="CIBUS"))
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

    if verbose:
        print('---Leaving crop_map_helper()---')

    return crop_in_df, crop_re_dict
def h_map_helper(h_in_df=False,
                 crop_in_df=False,
                 amnd_in_df=False,
                 map_col='h_value_type',
                 output_col_name='h_value',
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
            print(f"Creating h_map_df from 'h_values.csv' in {input_path}")
        h_map_df = make_df_lower(pd.read_csv(f'{input_path}/h_values.csv', index_col=["h_value_type", "h_frac"]))
    else:
        # Make sure all text is lower case
        if verbose:
            print("An h-value mapping dataframe exists.")
            print("Assigning existing df to 'h_map_df'")
        h_map_df = make_df_lower(h_in_df)

    if not isinstance(crop_in_df, pd.DataFrame):
        # create a df from default file input
        if verbose:
            print("CIBUS crop mapping dataframe does not exist")
            print("> Calling 'crop_map_helper()'")
        crop_in_df, crop_re_dict = crop_map_helper(verbose=verbose)
        if verbose:
            print("'crop_in_df' and 'crop_re_dict' created")
    else:
        pass

    if not isinstance(amnd_in_df, pd.DataFrame):
        # create an input dataframe for amendments from default import
        if verbose:
            print("An amendment mapping dataframe does not exist")
            print(f"Creating 'amnd_map_df' from 'amnd_map.csv' in {input_path}")
        amnd_map_df = make_df_lower(pd.read_csv(f'{input_path}/amnd_map.csv', index_col="CIBUS"))
    else:
        # Make sure all text is lower case
        if verbose:
            print("An amendment mapping dataframe exists.")
            print("Assigning existing df to 'amnd_map_df'")
        amnd_map_df = make_df_lower(amnd_in_df)

    #### calculate the h-value dataframe by combiningcrops and amendments
    h_value_df_crops = map_cin_h_to_dataframe(map_col,
                                                    crop_in_df,
                                                    h_map_df,
                                                    output_col_name)
    h_value_df_amendments = map_cin_h_to_dataframe(map_col,
                                                         amnd_map_df,
                                                         h_map_df,
                                                         output_col_name)

    h_value_df = pd.concat([h_value_df_crops, h_value_df_amendments])

    # Create a dict from h_value_df
    h_value_dict = h_value_df.to_dict('dict')[h_value_df.columns[0]]

    if verbose:
        print("'h_value_df' and 'h_value_dict' created")

    if verbose:
        print("---Leaving h_map_helper()---")

    return h_value_df, h_value_dict


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
        name_map_all_df = make_df_lower(pd.read_csv(f'{input_path}/name_map_all.csv')).dropna(subset='crop')
    else:
        name_map_all_df = make_df_lower(name_df).dropna(subset='crop')

    # Create dictionaries between cibus names (crop) and the different sources
    crop_andren2004_map = dict(zip(name_map_all_df.crop, name_map_all_df.andren2004))
    crop_jacobs_map = dict(zip(name_map_all_df.crop, name_map_all_df.jacobs))
    crop_hanna_map = dict(zip(name_map_all_df.crop, name_map_all_df.hanna))
    crop_cpc_map = dict(zip(name_map_all_df.crop, name_map_all_df.cpc))
    crop_crop_group_map = dict(zip(name_map_all_df.crop, name_map_all_df.crop_group))

    # Create dfs with the allometric function parameters or allocation factors
    if isinstance(allo1_df, pd.DataFrame):
        c_allo1__df = make_df_lower(allo1_df)
    else:
        c_allo1_df = make_df_lower(pd.read_csv(f'{input_path}/c_allom_andren2004.csv', index_col=0, decimal=','))

    if isinstance(allo2_df, pd.DataFrame):
        c_allo2_df = make_df_lower(allo2_df)
    else:
        c_allo2_df = make_df_lower(
            pd.read_csv(f'{input_path}/c_alloc_jacobs2020.csv', index_col=0, usecols=['Crop', 'i_ag', 'i_bg'],
                        decimal=','))

    if isinstance(allo3_df, pd.DataFrame):
        c_allo3_df = make_df_lower(allo3_df)
    else:
        c_allo3_df = make_df_lower(
            pd.read_csv(f'{input_path}/c_alloc_hanna.csv', index_col=0, usecols=['Crop', 'i_ag', 'i_bg', 'source'],
                        decimal=','))

    out_dict = dict(zip(('c_allom_andren_df', 'c_alloc_jacobs_df', 'c_alloc_hanna_df', 'crop_andren2004_map',
                         'crop_jacobs_map', 'crop_hanna_map', 'crop_cpc_map', 'crop_crop_group_map'), (
                        c_allo1_df, c_allo2_df, c_allo3_df, crop_andren2004_map, crop_jacobs_map, crop_hanna_map,
                        crop_cpc_map, crop_crop_group_map)))

    if verbose:
        print('---Leaving alloc_helper()---')

    return out_dict


#### Functions below are used with xarray ####

  
def xr_calculate_GWP(xr_array: xr.DataArray,
                     reference_year: str = '2020',
                     end_year: str = '2050',
                     year_coord: str = 'output_year', 
                     char_fact: str = 'gwp100',
                     ghg: str = 'co2',
                     ass_rep: str = 'ipcc2013') -> int:
    '''
    Calculate the Global Warming Potential (GWP) based on a given xarray array.

    - This function calculates GWP based on the given xarray array, reference year, end year, and other parameters.
    - The GWP calculation involves selecting specific years, retrieving characteristic factors, and calculating emissions.
    - The resulting GWP value is rounded to the nearest integer.

    Parameters:
    -----------
    xr_array : xr.DataArray
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



    Example use case:
    ---------
    # Calculate GWP for a specific xarray array
    >>> gwp_value = xr_calculate_GWP(xr_data, reference_year='2015', end_year='2030')
    '''
    
    # Retrieve characterization factor factor
    char_factor: float = soil_params.GWP[char_fact][ass_rep][ghg]

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
    -----------
    ghg : str, optional
        The abbreviation of the greenhouse gas ('co2', 'ch4', or 'n2o'). Default is 'co2'.
        
    time_horizon : int, optional
        The time horizon for the evaluation. Default is 100.

    Returns:
    --------
    dt_ghg : array
        Absolute temperature response over time.

    Raises:
    --------
    ValueError
        If the provided greenhouse gas abbreviation is not 'co2', 'ch4', or 'n2o'.
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
        print("\033[1;31m!!!! Please note that 'ghg_to_temp' was applied to 'CH4' input data. This formula needs proofing. Please verify it's correct implementation in the code before using this output data !!!!\033[0m")
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
    -----------
    ghg : str
        Greenhouse gas abbreviation ('co2', 'ch4', or 'n2o').
    ghg_flux : float
        Greenhouse gas flux in kilograms.
    input_year : int
        The specific year of the greenhouse gas emission.
    base_year : int, optional
        The base year for temperature calculation. Defaults to 2020.
    time_horizon : int, optional
        Time horizon for the temperature response calculation. Defaults to 100 years.

    Returns:
    --------
    Tuple[pd.Series, pd.DatetimeIndex]
        Temperature response series and corresponding time index.

    Notes:
    ------
    - The function uses a predefined temperature curve for CO2 (tempcurve_co2)
      and multiplies it by the greenhouse gas flux.
    - The resulting temperature response series is limited to the specified time horizon.

    Examples:
    ---------
    # Calculate temperature response for a specific greenhouse gas emission
    >>> temp_response, time_index = calc_temps('co2', 1000, 2025, base_year=2020, time_horizon=50)
    """

   
    # Calculate the end year based on the time horizon and input year
    end_year = base_year + time_horizon
    
    # Generate a time index starting from the input year to the end year with annual frequency
    time_index = pd.date_range(start=f'{input_year}-01-01',
                               end=f'{end_year - 1}-01-01',
                               freq='AS', name='temp_time')
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
                      output_label: str = 'temp_response'
                     ) -> xr.Dataset:
    """
    Add temperature response to an xarray Dataset or DataArray.

    - This function adds a temperature response variable to the input xarray Dataset or DataArray.
    - The temperature response is calculated based on greenhouse gas fluxes using the 'calc_temps' function.

    Parameters:
    -----------
    input_data : Union[xr.Dataset, xr.DataArray]
        The input data containing greenhouse gas fluxes.
    ghg : str, optional
        The greenhouse gas variable name. Default is 'co2'.
    output_label : str, optional
        The label for the temperature response variable. Default is 'temp_response'.

    Returns:
    --------
    xr.Dataset
        Returns a new Dataset.

    Raises:
    -------
    ValueError
        If the input_data is neither a Dataset nor a DataArray.

    Examples:
    ---------
    # Add temperature response to an existing xarray Dataset
    >>> result_dataset = add_temp_response(existing_dataset, ghg='ch4', output_label='temperature_change')
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
    
    # Create a dictionary of the input dataset coordinates
    coord_dict = dataset.coords
    coord_dict = {coord:dataset[coord].data for coord in coord_dict}
    
    # Get the name of the time dimension for the ghg fluxes
    year_dim = None
    dimensions = dataset.dims
    for dim in dimensions:
        if dataset[dim].dtype == 'datetime64[ns]':
            year_dim = dim
  
    # Save the dimension names to recreate the index of the pandas_df        
    coords = list(dataset.dims)

    print('Starting stacking operation')
    # Stack the dimensions of the dataset and save the multiindex
    stacked_ds = dataset.stack(stacked=[...])

    idx_in = stacked_ds.indexes['stacked']
    idx_list = stacked_ds.stacked.data

    # create empty lists used to build the new output dataset
    temp_list = []
    rows_list = []

    print('Starting temp_calc inner loop')   
    for n, i in enumerate(stacked_ds['co2'].data):
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


def check_attributes_status(instance, access='public'):
    """
    Checks the status of all attributes in the given class instance.

    Parameters:
    - instance: An instance of a class.
    - access: A string that selects whether to show 'public' or 'private' attributes.

    Returns:
    A dictionary where keys are attribute names and values are tuples containing
    the current type of the attribute and a boolean indicating if it is set (not None).
    """
    attrs_status = {}
    for attribute in dir(instance):
        # Filter out magic methods and attributes
        if not attribute.startswith('__'):
            if access == 'public':
                if not attribute.startswith('_'):
                    attr_value = getattr(instance, attribute)
                    # Check if the attribute is set (not None)
                    is_set = attr_value is not None
                    # Store the type of the attribute and its set status
                    attrs_status[attribute] = (type(attr_value).__name__, is_set)
            if access == 'private':
                if attribute.startswith('_'):
                    attr_value = getattr(instance, attribute)
                    # Check if the attribute is set (not None)
                    is_set = attr_value is not None
                    # Store the type of the attribute and its set status
                    attrs_status[attribute] = (type(attr_value).__name__, is_set)
    return attrs_status


def assign_tot_soc(xarray_input: Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]],
                   totsoc_label: str = 'tot_soc'
                   ) -> Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]]:
    """
    Calculate and assign total soil organic carbon (tot_soc) based on young (y_pool) and old (o_pool) pools directly
    within the provided xarray Dataset or DataArray. If a list of Datasets or DataArrays is provided, the operation
    is applied to each element in the list.

    This function modifies the input xarray object(s) in-place by adding a 'tot_soc' variable that represents the
    sum of 'y_pool' and 'o_pool'.

    Parameters:
    -----------
    xarray_input : Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]]
        The input xarray Dataset or DataArray, or a list of xarray Datasets or DataArrays. Each element must contain
        'y_pool' and 'o_pool' variables from which 'tot_soc' is calculated.

    totsoc_label : str, optional
        The label for the total soil organic carbon variable to be added to the input. Defaults to 'tot_soc'.

    Returns:
    --------
    None
        The function does not return any value. It modifies the provided xarray_input object(s) in-place by adding
        a 'tot_soc' variable.

    Notes:
    ------
    - The 'tot_soc' variable is calculated as the sum of 'y_pool' and 'o_pool'.
    - This function directly modifies the input xarray object(s). Ensure to pass a copy if the original data should
      remain unchanged.

    Example:
    --------
    # For a single xarray Dataset
    assign_tot_soc(ds)

    # For a list of xarray Datasets
    datasets = [ds1, ds2, ds3]
    assign_tot_soc(datasets)

    # For a single xarray DataArray
    assign_tot_soc(da)

    # For a list of xarray DataArrays
    dataarrays = [da1, da2, da3]
    assign_tot_soc(dataarrays)
    """

    def calculate_and_assign(ds_or_da):
        if 'y_pool' in ds_or_da and 'o_pool' in ds_or_da:
            tot_soc = ds_or_da.y_pool + ds_or_da.o_pool
            if isinstance(ds_or_da, xr.DataArray):
                ds_or_da = ds_or_da.to_dataset(name='var')
                ds_or_da[totsoc_label] = tot_soc
                ds_or_da = ds_or_da.to_array()
            elif isinstance(ds_or_da, xr.Dataset):
                ds_or_da[totsoc_label] = tot_soc
        else:
            raise ValueError("Input must contain 'y_pool' and 'o_pool'.")
        return ds_or_da

    if isinstance(xarray_input, list):
        return [calculate_and_assign(item) for item in xarray_input]
    else:
        return calculate_and_assign(xarray_input)


def assign_co2_flux(xarray_input,
                    totsoc_label='tot_soc',
                    co2flux_label='co2_flux',
                    year_coord='output_year'
                   ) -> None:
    """
    Calculates the CO2 flux between consecutive years based on the 'tot_soc' values in the input xarray objects.
    The calculated CO2 flux is added to the input xarray Dataset or DataArray as a new variable.

    Parameters:
    -----------
    xarray_input : Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]]
        The input xarray Dataset or DataArray, or a list thereof, containing 'tot_soc' values.

    totsoc_label : str, optional
        The label for the total soil organic carbon variable. Defaults to 'tot_soc'.

    co2flux_label : str, optional
        The label for the calculated CO2 flux variable. Defaults to 'co2_flux'.

    year_coord : str, optional
        The name of the coordinate representing the temporal dimension over which the CO2 flux is calculated.
        Defaults to 'output_year'.

    Returns:
    --------
    None:
        Modifies the provided xarray_input object(s) in-place, adding the 'co2_flux' variable.

    Notes:
    ------
    - The 'co2_flux' is calculated as the negative difference in 'tot_soc' between consecutive years, multiplied
      by a factor of 3.6, to account for the conversion between carbon and CO2 flux.
    - This function modifies the input xarray object(s) directly; if the original data needs to be preserved,
      consider working on a copy of the data.
    """

    if isinstance(xarray_input, list):
        for item in xarray_input:
            assign_co2_flux(item, totsoc_label, co2flux_label, year_coord)
    else:
        co2_flux = xarray_input[totsoc_label].diff(dim=year_coord) * -3.6
        if isinstance(xarray_input, xr.DataArray):
            raise NotImplementedError("CO2 flux calculation for xr.DataArray is not supported in this implementation.")
        elif isinstance(xarray_input, xr.Dataset):
            xarray_input[co2flux_label] = co2_flux
