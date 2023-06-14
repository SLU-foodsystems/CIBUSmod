import os
import weakref
import warnings
import pandas as pd
import numpy as np
import itertools
import numbers

EMPTY = float("nan")

class ParameterRetriever:
    '''Class that retrieves parameters from an Excel file based on a flexible filtering approach.
    The Excel file needs to include a sheet named 'default' with columns named 'parameter' and 'vlaue'
    including parameter names and corresponding values. Any filter columns should be named 'f_<name>'.
    Any other columns are not used by the ParameterRetriever. All rows without a value in the 'parameter'
    column are also ignored allowing e.g. headings in the parameter sheets.

    Scenrios are defined by creating new sheets named 'scn_<scenario name>'. The scenario sheets are
    structured in a similar way as the 'default' sheet but with values for each year stored in columns
    named 'y_<year>' (e.g. y_2030). The scenario sheets also needs to include a column named 'type'
    that defines how the new parameters are defined:
        'rel'     : relative to default value | new par = def par * scn par
        'rel_chg' : relative change from default value | new par = def par * (1 + scn par)
        'abs'     : absolute value | new par = scn par
        'abs_chg' : absolute change from default value | new par = def par + scn par

    
    Parameters
    ----------
    file : str
        A path to the Excel file storing parameter values relative to the model 'data' folder.
        Use '/' as file path separator.'''

    instances = weakref.WeakSet() # WeakSet of all ParameterRetriever instances

    @classmethod
    def update_all_parameter_values(cls,scenario,year):
        for pr in cls.instances:
            pr.update_parameter_values(scenario,year)

    def __init__(self, file, **kwargs):
        
        ParameterRetriever.instances.add(self)
        
        # Read parameter dataframe with all columns as str except the 'value' column
        path = ''
        for word in file.split('/'):
            path = os.path.join(path, word)
        self.path = os.path.join(path)

        self.data = _read_xl(self.path,'default')
        try:
            self.rel = _read_xl(self.path,'relations')
        except:
            self.rel = None

        self.filters = []
        
        self.set(**kwargs)
        
    def __repr__(self):

        str1 = "\n".join([
            f"{f} : {str(getattr(self,f)[0]) if len(getattr(self,f))==1 else 'List of ' + str(len(getattr(self,f)))}"
            for f in  self.filters
        ])
        unique_pars = self.data.index.get_level_values("parameter").unique()
        str2 = ", ".join(unique_pars[:min(len(unique_pars),10)]) + (" ..." if len(unique_pars) > 10 else "")

        return f"""
ParameterRetriever object with {len(self.data)} parameters in total.

Filters
-------
{str1}

Parameters
----------
{str2}
"""

    def __len__(self):
        return self.max_filter_length
        
    def set(self, **kwargs):
        '''Method to set filter values. Filters are supplied as keyword arguments and applies to columns in the Excel sheet
        named 'f_<key>'. If filters not present in the Excel sheet columns are supplied those are ignored and a warning is
        printed.
        
        Parameters
        ----------
        **kwargs : str or list of str
            If lists are supplied these need to have equal lengths (also to previously supplied filters) or length 1
            
        Returns
        -------
        Nothing. Updates the ParameterRetriever filters.'''
        
        # Set filter values supplied
        for key, value in kwargs.items():
            # Only care about filters that are in the filter columns 
            if ('f_'+key in self.data.index.names):
                if key not in self.filters: self.filters.append(key)
                value = np.array([value]) if isinstance(value, str) else np.array(value)
                setattr(self, key, value)
        
        # Get length of filter values
        l = [len(getattr(self, f)) for f in self.filters]
        self.max_filter_length = max(l) if len(self.filters) > 0 else 0

        if any([(i>1) & (i<max(l)) for i in l]):
            # Raise error if any filter array is longer than 1 and shorter than the maximum filetr array length
            self.remove(list(kwargs.keys()))
            raise ValueError('Lists of differing lengths supplied as filters')

        # Create selection
        selection_dict = {f"f_{f}": list(getattr(self,f)) for f in self.filters}
        self.selection = _build_selection_index(selection_dict)

    def get(self, parameter, **kwargs):
        '''Method to get values of a parameter under the set filters.
        
        Parameters
        ----------
        parameter : str
            Name of the parameter to be retrieved
        **kwargs
            Keyword arguments to be passed on as filters to ParameterRetriever.set()
            
        Returns
        -------
        numpy.ndarray with length equal to the length of filter values. containing the parameter values for the defined filters '''
        self.set(**kwargs)

        result = _get_parameter_values(self.data, self.selection, parameter)

        # If NaNs are return print warning and some useful information
        if np.isnan(result).any():
            if self.selection is None:
                warnings.warn(f"NaN returned! No filters supplied and could not find a default value for '{parameter}' in {self.path}.")
            elif np.isnan(result).all():
                str1 = f"NaNs returned! No value for '{parameter}' found in '{self.path}' matching any of the supplied filters (n={len(self.selection)}): \n----------\n"
                n = min([len(self.selection),5])
                str2 = "\n----------\n".join([
                    "\n".join([
                        key + " = " + val
                        for key,val in zip(self.selection.names, sel)
                    ])
                    for sel in self.selection[0:n]
                ]) + ("\n ..." if n<len(self.selection) else "")
                warnings.warn(str1+str2)
            else:
                nan_sel = self.selection[np.isnan(result)]
                str1 = f"NaNs returned! No value for '{parameter}' found in '{self.path}' for some of the supplied filters (n={len(nan_sel)}): \n----------\n"
                n = min([len(nan_sel),5])
                str2 = "\n----------\n".join([
                    "\n".join([
                        key + " = " + val
                        for key,val in zip(nan_sel.names, sel)
                    ])
                    for sel in nan_sel[0:n]
                ]) + ("\n ..." if n<len(nan_sel) else "")
                warnings.warn(str1+str2)
            
        return result

    def get_from_frame(self,parameter,df,**kwargs):
        '''Get parametervalues based on index and columns in supplied pandas.DataFrame'''

        if min(df.shape)<1:
            raise ValueError('DataFrame must have at least one row and one column')

        row_names = df.index.names
        col_names = df.columns.names

        filters = pd.merge(
            df.index.to_frame().reset_index(drop=True),
            df.columns.to_frame().reset_index(drop=True),
            how='cross'
        ).to_dict('list')

        result = pd.DataFrame(filters)

        try:
            result['value'] = self.get(parameter,**filters,**kwargs)
        except ValueError:
            # remove all filters with lengt > 1 and try again
            for f in self.filters:
                if len(getattr(self,f))>1:
                    self.remove(f)
            result['value'] = self.get(parameter,**filters,**kwargs)

        result = result.pivot(index=row_names,columns=col_names,values='value')

        return result.align(df, join='right')[0]

    def update_parameter_values(self,scenario,year):
        '''Method to update parameter values in ParameterRetriever according to specified scenario and year.
        New parameter values need to be stored in a separate sheet of the parameter Excel file named 'scn_<scenario name>'.
        In the scenario sheet new values are defined in year columns with column names on the format 'y_<year>'.
        These are defined in relation to the default parameter values (i.e. a value of 0.9 implies a 10% reduction
        from the default value). New parameter values can be defined in the Excel sheet for arbitrary years and the
        method linearly interpolates values between defined years.
        
        Parameters
        ----------
        scenario : str or list of str
            Name(s) of scenarios to update parameter values according to. Scenarios later in the list
            will override earlier scenarios if the same parameter value is changed in multiple scnearios.
        year : str or int
            Year to update parameter values to
            
        Returns
        -------
        Nothing. Updates ParameterRetriever parameter values.'''

        year = int(year)
        if isinstance(scenario,str):
            scenario = [scenario]

        # Read default parameter values
        data = _read_xl(self.path,'default')
        # Create pd.Series for updated parameter values
        updated_data = data.copy()

        # Go through all scenarios in consucutive orderd.
        # If the same parameter is updated in multiple scenarios
        # only the scenario that is latest in the list will have
        # an effect.
        for scn in scenario:

            # Read scenario parameter values
            try:
                scn_data = _read_xl(self.path,'scn_'+scn)
            except:
                # If scenario sheet not pressent do not update anything.
                # Should a warning be printed here?
                # warnings.warn(f"No sheet named 'scn_{scn}' found in '{self.path}'. No parameter values were updated according to this scenario.")
                continue
                
            # If sheet was found but contained no parameters, move to next scenario
            if len(scn_data) == 0:
                continue

            # Interpolate scenario parameter values for 'year' by selecting columns
            # Drop 'y_' prefix in columns
            scn_data.columns = scn_data.columns.str.replace('y_','')
            # Add missing years
            first_year = np.array(scn_data.columns).astype(int).min()
            last_year = np.array(scn_data.columns).astype(int).max()
            cols_to_add = list(set(np.array(range(first_year,last_year)).astype(str)) - set(scn_data.columns))
            scn_data[cols_to_add] = np.nan
            scn_data = scn_data.reindex(sorted(scn_data.columns), axis=1)

            # Interpolate values for intermediate years and if neededpropagate first/last value of a
            # parameter backward/forward.
            scn_data = scn_data.interpolate(axis=1, limit_direction='both')

            # Interpolation example:
            # ---------------------        ----------------------------------------------------------------------
            # par  2000  2005  2010  --->  par   2000  2001  2002  2003  2004  2005  2006  2007  2008  2009  2010
            # A    nan   1.0   1.5   --->  A     1.0   1.0   1.0   1.0   1.0   1.0   1.1   1.2   1.3   1.4   1.5
            # B    1.0   1.5   nan   --->  B     1.0   1.1   1.2   1.3   1.4   1.5   1.5   1.5   1.5   1.5   1.5
            # C    1.0   0.5   1.0   --->  C     1.0   0.9   0.8   0.7   0.6   0.5   0.6   0.7   0.8   0.9   1.0
            # ---------------------        ----------------------------------------------------------------------

            # Select year
            scn_data["value"] = scn_data[str(year)]
            scn_data = scn_data["value"]

            # Go through parameters in scenario and update values
            for parameter in scn_data.index.get_level_values('parameter').unique():

                # Create selection
                selection = self.data.xs(parameter,level='parameter', drop_level=False).index
                scn_selection = selection.droplevel('parameter')
                
                # If no filter columns in scenarios sheet (i.e. values to update parameters apply universaly)
                # then make sure that only one value is found and update accordingly
                if len(scn_data.index.names)==1:
                    if len(scn_data==1):
                        values = scn_data.values
                    else:
                        values = [EMPTY]
                else:
                    # Drop selection levels not in scenario filter columns
                    for lvl in (set(selection.names) - set(scn_data.index.names)):
                        scn_selection = scn_selection.droplevel(lvl)

                    # Get scenario values
                    values = _get_parameter_values(scn_data, scn_selection, parameter)

                # Update values
                updated_data.loc[selection] = data.loc[selection] * np.nan_to_num(values,1)
            
        self.data = updated_data

                
    def get_unique(self,filter,qry=None):
        '''Get unique values for specified filter(s) in parameter Excel sheet
        
        Parameters
        ----------
        filter : str or list of str
            filter(s) to get unique values for
        qry : str
            An optional query string filter the parameter sheet before finding
            unique filter values

        Returns
        -------
        numpy.array of unique values for filter if a str is supplied or
        pandas.DataFrame of unique combinations of filter (non-NaN) values with filter names as columns
        '''
        if qry is not None:
            df = self.data.reset_index().query(qry)
        else:
            df = self.data.reset_index()

        if isinstance(filter, list):
            res = df[['f_'+f for f in filter]].dropna(how='any').drop_duplicates()
            res.columns = filter
            return res
        else:
            res = df['f_'+filter].unique() 
            return res[~pd.isna(res)]

    def get_rel(self,from_col='',to_col=''):
        '''Returns a dict with values in 'from_col' as keys and 'to_col' as values'''
        if len(set(self.rel[from_col])) < len(set(self.rel[to_col])):
            raise ValueError(f'Only one-to-one or many-to-one relations are allowed. Did you mean from_col={to_col}, to_col={from_col}?')
        return self.rel[[from_col,to_col]].set_index(from_col).to_dict()[to_col]
    
    def clear(self):
        for f in self.filters:
            delattr(self,f)
        self.filters = []
        self.set()

    def remove(self,item):
        if not isinstance(item, list):
            item = [item]
        for f in item:
            try:
                delattr(self,f)
            except:
                pass
            else:
                self.filters.remove(f)
        
        self.set()

def _read_csv(path,parameter):
    df = pd.read_csv(path, dtype=str)

    if parameter not in df.columns:
        raise ValueError(f"'{parameter}' not found in '{path}'")

    f_cols = [c for c in df.columns if c.startswith("f_")]

    df = df.rename({parameter:'value'}, axis=1)
    df['parameter'] = parameter

    return df.loc[:,f_cols+['parameter','value']]
        
def _read_xl(path,sheet):
        idx = pd.IndexSlice
        # Read xl and set value columns to type float
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)

        # Get parameters from any referenced csv-files
        if sheet=='default':

            # Get rows with reference to csv-files
            to_read_csv = df.value.str.contains('.csv', na=False)
            csvs_to_read = df.loc[to_read_csv,idx['parameter','value']]

            # Drop rows refering to csv-files from df
            df = df.loc[~to_read_csv]

            # Read csvs and append to df
            for parameter,csv_file in csvs_to_read.values:

                csv_path = os.path.join(os.path.dirname(path),csv_file)
                df_csv = _read_csv(csv_path,parameter)

                df = pd.concat([df,df_csv])

        if (sheet=='default') | sheet.startswith('scn_'):
            # Only retain filter column(s), parameter column and value column(s)
            index_cols = [c for c in df.columns if c.startswith("f_")]
            value_cols = "value" if "value" in df.columns else [c for c in df.columns if c.startswith("y_")]

            df = (
                df.loc[lambda d: d["parameter"].notnull()]
                #.loc[lambda d: d["value"].notnull()]
                .set_index(index_cols + ["parameter"])[value_cols]
                .astype(float)
            )

            # Raise error if duplicates found and print some usefull info
            if df.index.duplicated().any():
                dup = df.index[df.index.duplicated()].get_level_values("parameter")
                n = min(len(dup),5)
                str1 = f"One or more parameter(s) in '{path}' have identical filter columns (n={len(dup)}): "
                str2 = ", ".join(["'"+d+"'" for d in dup]) + (", ..." if n<len(dup) else "")
                raise ValueError(str1+str2)
                

        return df

def _build_selection_index(selection):
    if len(selection) == 0:
        return None

    selection_lens = {col: len(labels) for col, labels in selection.items()}
    distinct_selection_lens = set(selection_lens.values())

    # Broadcast length-1 selections
    selection = {
        col: (labels * max(distinct_selection_lens) if len(labels) == 1 else labels)
        for col, labels in selection.items()
    }

    selection_index = pd.MultiIndex.from_frame(pd.DataFrame(selection))
    return selection_index

def _get_problem_data(data, index_cols, parameter):
    if not isinstance(data, pd.Series):
        raise ValueError(f"data should be a pandas.Series")
    if unknown_columns := set(index_cols) - set(data.index.names):
        raise ValueError(f"did not find index columns {unknown_columns} in data index")
    
    problem_data = data.xs(parameter, level="parameter")
    
    # Only chose rows where levels not filtered for are empty
    # If this is not possible return None
    null_cols = set(problem_data.index.names) - set(index_cols)
    for col in null_cols:
        if EMPTY in problem_data.index.get_level_values(col):
            problem_data = problem_data.xs(EMPTY, level=col)
        else:
            return None
        
    # Drop levels that are not filtered for
    for lvl in problem_data.index.names:
        if (problem_data.index.nlevels > 1) & problem_data.index.get_level_values(lvl).isna().all():
            problem_data = problem_data.droplevel(lvl)
        
    if len(problem_data.index.names) > 1:
        problem_data = problem_data.reorder_levels([c for c in index_cols if c in problem_data.index.names]) # NEW!!! [c for c in index_cols if c in problem_data.index.names]

    return problem_data


def _select_with_defaults(data, index, columns_to_take_default):
    partial_data = data
    partial_index = index

    for col in columns_to_take_default:
        try:
            partial_data = (
                partial_data.xs(EMPTY, level=col)
                if isinstance(partial_data.index, pd.MultiIndex)
                else partial_data[EMPTY]
            )
        except KeyError:
            # There is no default/empty key in this index column, which means
            # that ignoring the column is not possible.
            return partial_data[[]]
        try:
            partial_index = partial_index.droplevel(col)
        except ValueError:
            # Can't drop last level so don't
            partial_data = pd.Series(partial_data, index=partial_index.get_level_values(0))

    if len(partial_index.names) == 1:
        partial_index = partial_index.get_level_values(0) # pd.MultiIndex --> pd.Index

    partial_result = partial_data.reindex(partial_index).set_axis(index)
    
    return partial_result


def _select_allowing_any_k_defaults(data, index, k):
    # Generate all the (n choose k) ways to have k default columns
    results = pd.concat(
        [
            _select_with_defaults(data, index, default_cols)
            for default_cols in itertools.combinations(index.names, k)
        ],
        axis=1,
    )
    exactly_one_result = results.notnull().sum(axis=1) == 1
        
    # Get elements with exactly one result
    results = results[exactly_one_result].sum(axis=1)
    # Drop duplicate indexes to be able to merge back (these will have returned the same value anyway)
    results = results[~results.index.duplicated(keep='first')]
    
    return results

def _get_parameter_values(data, selection, parameter):

    # If no filters supplied check if only one value can be returned
    # else return NaN
    if selection is None:
        result = data.xs(parameter, level="parameter")
        # Get rows with only NaNs in filter columns
        result = result.loc[result.index.to_frame().isna().all(axis=1)]
        if len(result) == 1:
            return result.values
        else:
            return np.array(EMPTY)
        
    selection = selection.copy()

    # Get the data subset for the parameter in question,
    # and use the default for each dimension not specified in the selection.
    problem_data = _get_problem_data(data, selection.names, parameter)
    if problem_data is None:
        return np.array([EMPTY]*len(selection))

    # Drop filters not in data
    for lvl in set(selection.names)-set(problem_data.index.names):
        selection = selection.droplevel(lvl)
    
    assert problem_data.index.names == selection.names

    # Start with an empty result
    result = pd.Series(data=EMPTY, index=selection)

    # Fill in the blanks by successively using k = 0, ..., n default values,
    # where n is the number of index columns in the full problem..
    for k in range(len(selection.names) + 1):
        index_remainder = result[result.isnull()].index
        result = result.fillna(
            _select_allowing_any_k_defaults(problem_data, index_remainder, k)
        )

    return result.values