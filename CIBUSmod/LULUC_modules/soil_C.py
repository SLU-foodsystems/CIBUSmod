from __future__ import annotations
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.verbose_print import verbose_init
from ..utils.retriever import ParameterRetriever
from ..utils.data_attr import DataAttr
from ..utils.session_db import Session
from ..utils.misc import index_to_multi, multiindex_product

def icbm_steady_state_vec(
    C_in: float | np.ndarray | pd.Series,
    h: float | np.ndarray | pd.Series,
    re: float | np.ndarray | pd.Series = 1.0,
    ky: float | np.ndarray | pd.Series = 0.26,
    ko: float | np.ndarray | pd.Series = 9.7e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute steady-state young and old carbon pools for the discrete ICBM model.

    This function evaluates the analytical steady-state solution for the canonical
    discrete ICBM formulation assuming constant annual carbon input and constant
    environmental modifier.

    Parameters
    ----------
    C_in
        Annual carbon input. Can be a scalar or array-like.
    h
        Humification coefficient. Can be a scalar or array-like.
    re
        Environmental decomposition modifier. Can be a scalar or array-like.
        Defaults to ``1.0``.
    ky
        Decomposition rate constant for the young pool. Can be a scalar or
        array-like. Defaults to ``0.26``.
    ko
        Decomposition rate constant for the old pool. Can be a scalar or
        array-like. Defaults to ``9.7e-3``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple ``(Y, O)`` with the steady-state values of the young and old
        carbon pools. Scalar inputs are returned as 0-d NumPy arrays.

    Notes
    -----
    The equations are derived from the step version of ICBM as presented in
    Menichetti et al. (2024), under the assumption of constant ``C_in`` and ``re``.

    References
    ----------
    Menichetti et al. (2024). Bayesian calibration of the ICBM/3 soil organic
    carbon model constrained by data from long-term experiments and uncertainties
    of C inputs. https://doi.org/10.1080/17583004.2024.2304749
    """
    C_in = np.asarray(C_in, dtype=float)
    h = np.asarray(h, dtype=float)
    re = np.asarray(re, dtype=float)
    ky = np.asarray(ky, dtype=float)
    ko = np.asarray(ko, dtype=float)

    a = np.exp(-ky * re)
    b = np.exp(-ko * re)

    Y = (C_in * a) / (1.0 - a)
    O = h * ((ky*C_in) / ((ko - ky) * (1.0 - a))) * ((a - b) / (1.0 - b))

    return Y, O


def icbm_vec(
    C_in: pd.Series | pd.DataFrame,
    h: float | pd.Series,
    re: float | pd.Series | pd.DataFrame = 1.0,
    ky: float | pd.Series = 0.26,
    ko: float | pd.Series = 9.7e-3,
    Y0: float | pd.Series = 0.0,
    O0: float | pd.Series = 0.0,
    extend: int = 0,
    spinup: bool = False,
) -> tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the discrete ICBM model for one or more inputs over time.

    The implementation is vectorized over columns and iterates only over the
    time dimension.

    Parameters
    ----------
    C_in
        Carbon input time series. If a Series is provided, it is treated as a
        single input. If a DataFrame is provided, columns represent separate
        inputs or strata.
    h
        Humification coefficient. Can be a scalar or a Series indexed like
        ``C_in.columns``.
    re
        Environmental decomposition modifier.

        Supported forms are:

        - scalar: same modifier for all times and columns
        - Series indexed like ``C_in.index``: time-varying, shared across columns
        - Series indexed like ``C_in.columns``: column-specific, constant over time
        - DataFrame with same shape/index/columns as ``C_in``

    ky
        Young-pool decomposition rate constant. Scalar or Series indexed like
        ``C_in.columns``.
    ko
        Old-pool decomposition rate constant. Scalar or Series indexed like
        ``C_in.columns``.
    Y0
        Initial young-pool stock. Scalar or Series indexed like ``C_in.columns``.
    O0
        Initial old-pool stock. Scalar or Series indexed like ``C_in.columns``.
    extend
        Number of additional years to extend beyond the last index value using
        forward-filled values for ``C_in`` and ``re``. Defaults to ``0``.
    spinup
        If ``True``, initialize ``Y0`` and ``O0`` from steady-state values based
        on the first row of ``C_in`` and ``re``. Defaults to ``False``.

    Returns
    -------
    tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]
        A tuple ``(Y, O)`` with young- and old-pool stocks over time. If
        ``C_in`` is a Series, Series are returned. If ``C_in`` is a DataFrame,
        DataFrames are returned.

    Notes
    -----
    The equations follow the step version of ICBM as presented in
    Menichetti et al. (2024), Appendix A.

    References
    ----------
    Menichetti et al. (2024). Bayesian calibration of the ICBM/3 soil organic
    carbon model constrained by data from long-term experiments and uncertainties
    of C inputs. https://doi.org/10.1080/17583004.2024.2304749
    """

    if isinstance(C_in, pd.Series):
        name = C_in.name if C_in.name is not None else "input"
        C_df = C_in.to_frame(name=name)
        return_series = True
    elif isinstance(C_in, pd.DataFrame):
        C_df = C_in.copy()
        return_series = False
    else:
        raise TypeError("C_in must be Series or DataFrame")

    cols = C_df.columns
    idx = C_df.index
    n_t = len(idx)
    n_c = len(cols)

    def col_param(p, name):
        if np.isscalar(p):
            return pd.Series(float(p), index=cols, dtype=float)
        if isinstance(p, pd.Series):
            if not p.index.equals(cols):
                raise ValueError(f"Index of {name} must match columns of C_in")
            return p.astype(float)
        raise TypeError(f"{name} must be scalar or Series indexed as C_in columns")

    # Align re to DataFrame shape
    if np.isscalar(re):
        re_df = pd.DataFrame(float(re), index=idx, columns=cols, dtype=float)

    elif isinstance(re, pd.DataFrame):
        if not (re.index.equals(idx) and re.columns.equals(cols)):
            raise ValueError("Index and columns of re must match C_in")
        re_df = re.astype(float)

    elif isinstance(re, pd.Series):
        if re.index.equals(idx):
            re_df = pd.DataFrame(
                np.broadcast_to(re.to_numpy(dtype=float)[:, None], (n_t, n_c)),
                index=idx,
                columns=cols,
                dtype=float,
            )
        elif re.index.equals(cols):
            re_df = pd.DataFrame(
                np.broadcast_to(re.to_numpy(dtype=float)[None, :], (n_t, n_c)),
                index=idx,
                columns=cols,
                dtype=float,
            )
        else:
            raise ValueError(
                "If re is a Series, its index must match either C_in.index or C_in.columns"
            )
    else:
        raise TypeError("Invalid type for re")

    if extend:
        idx_ext = pd.Index(
            range(int(C_df.index[0]), int(C_df.index[-1]) + int(extend) + 1)
        ).astype(str)
    
        C_df = C_df.reindex(idx_ext).ffill()
        re_df = re_df.reindex(idx_ext).ffill()
    
        idx = C_df.index
        n_t = len(idx)

    h_s = col_param(h, "h")
    ky_s = col_param(ky, "ky")
    ko_s = col_param(ko, "ko")
    Y0_s = col_param(Y0, "Y0")
    O0_s = col_param(O0, "O0")

    C_arr = C_df.to_numpy(dtype=float)
    re_arr = re_df.to_numpy(dtype=float)
    h_arr = h_s.to_numpy(dtype=float)
    ky_arr = ky_s.to_numpy(dtype=float)
    ko_arr = ko_s.to_numpy(dtype=float)

    if spinup:
        first = 0
        Y_prev, O_prev = icbm_steady_state_vec(
            C_in=C_arr[first, :],
            h=h_arr,
            re=re_arr[first, :],
            ky=ky_arr,
            ko=ko_arr,
        )
    else:
        Y_prev = Y0_s.to_numpy(dtype=float)
        O_prev = O0_s.to_numpy(dtype=float)

    Y_arr = np.empty((n_t, n_c), dtype=float)
    O_arr = np.empty((n_t, n_c), dtype=float)

    for i in range(n_t):
        a = np.exp(-ky_arr * re_arr[i, :])
        b = np.exp(-ko_arr * re_arr[i, :])
        Fi = h_arr * ((ky_arr*(Y_prev + C_arr[i, :])) / (ko_arr - ky_arr))

        Y_t = (Y_prev + C_arr[i, :]) * a
        O_t = (O_prev - Fi) * b + Fi * a

        Y_arr[i, :] = Y_t
        O_arr[i, :] = O_t

        Y_prev = Y_t
        O_prev = O_t

    Y = pd.DataFrame(Y_arr, index=idx, columns=cols)
    O = pd.DataFrame(O_arr, index=idx, columns=cols)

    if return_series:
        col = cols[0]
        return Y[col].rename(C_in.name), O[col].rename(C_in.name)

    return Y, O

class SoilC:
    """
    Module that estimates soil organic carbon stock changes by collecting
    soil carbon inputs from a Session object, calculating the environmental
    modifier (re) and humification coefficients (h), and running the ICBM
    model for each input source.
    It also provides utilities to retrieve aggregated soil carbon stocks
    and fluxes.

    Parameters
    ----------
    par : ParameterRetriever
    session : Session

    Attributes
    ----------
    inputs : dict[str, str]
        Mapping from user-facing input names to session attribute keys.
    agg_cols : list[str]
        Column levels used as the aggregation dimensions.
        Default is ["region"], and should likely not be changed.
    lu : str
        Land-use category used throughout the class.
        Default is "cropland". May be changed if calculations are to
        be performed for another land-use category.
    data_attr : DataAttr
        Container for derived model inputs and outputs.
    par : ParameterRetriever
        Parameter access helper.
    session : Session
        Session/database interface for retrieving source data.
    crop_areas : pd.DataFrame
        Cropland area by aggregation dimensions and crop. Created by
        `.make_input()`.
    total_area : pd.DataFrame
        Total cropland area aggregated over regions. Created by
        `.make_input()`.
    """

    inputs = {
        'crop residues'       : 'fertiliser.crop_residues_C',
        'cover crop residues' : 'fertiliser.cover_crop_residues_C',
        'manure'              : 'fertiliser.manure_C',
        'organic fertiliser'  : 'fertiliser.organic_C'
    }

    agg_cols = ['region'] # <-- This should not be changed
    lu = 'cropland'
    
    def __init__(
        self,
        par: ParameterRetriever,
        session: Session
    ) -> None:

        self.data_attr = DataAttr(self)
        self.par = par
        self.session = session

    def crop_residues_from_Bolinder(
        self,
        input_name: str = 'crop residues'
    ) -> pd.DataFrame:
        """
        Estimate crop residue carbon inputs using the Bolinder et al (2008) approach.

        The method derives above-ground residues, below-ground residues, and
        rhizodeposition from harvested dry matter and parameter relationships.

        Parameters
        ----------
        input_name
            Included for API consistency. Currently only meaningful for
            ``"crop residues"``.

        Returns
        -------
        pd.DataFrame
            Carbon input data with a ``residue`` column level containing
            ``"above ground"``, ``"below ground"``, and ``"rhizodeposition"``.

        References
        ----------
        Bolinder et al. (2007) An approach for estimating net primary productivity and
        annual carbon inputs to soil for common agricultural crops in Canada.
        https://doi.org/10.1016/j.agee.2006.05.013
        """
        self.par.clear()
        gf = self.par.get_from_frame
        
        harvest_DM = (
            self.session.get_attr('C','harvest_DM',{k:None for k in self.agg_cols} | {'crop':['land_use',None],'prod_system':None}, interpolate=True)
            .xs(self.lu,level='land_use',axis=1)
            .reindex(self.crop_areas.columns, axis=1, fill_value=0)
        )
        
        CR_harvest_DM = (
            self.session.get_attr('C','crop_residues_harvest',{k:None for k in self.agg_cols} | {'crop':['land_use',None],'prod_system':None}, interpolate=True)
            .xs(self.lu,level='land_use',axis=1)
            .reindex(self.crop_areas.columns, axis=1, fill_value=0)
        )
        
        R_P = gf('R_P', harvest_DM)
        R_S = gf('R_S', harvest_DM)
        R_R = gf('R_R', harvest_DM)
        R_E = gf('R_E', harvest_DM)
        
        S_P = gf('S_P', harvest_DM)
        S_S = gf('S_S', harvest_DM)
        S_R = gf('S_R', harvest_DM)
        S_E = gf('S_E', harvest_DM)
        
        
        C_P = harvest_DM * gf('C_frac', harvest_DM) * gf('yield_to_biomass', harvest_DM)
        C_S = (R_S/R_P) * C_P
        C_R = (R_R/R_P) * C_P
        C_E = (R_E/R_P) * C_P
        
        Ci_P = C_P * S_P
        Ci_S = (C_S - CR_harvest_DM*gf('C_frac', CR_harvest_DM)) * S_S
        Ci_R = C_R * S_R
        Ci_E = C_E * S_E
        
        above_ground = Ci_P + Ci_S
        below_ground = Ci_R
        rhizodeposition = Ci_E
        
        above_ground.columns = multiindex_product([above_ground.columns,pd.Index(['above ground'], name='residue')])
        below_ground.columns = multiindex_product([below_ground.columns,pd.Index(['below ground'], name='residue')])
        rhizodeposition.columns = multiindex_product([rhizodeposition.columns,pd.Index(['rhizodeposition'], name='residue')])
        
        final_df = pd.concat([above_ground, below_ground, rhizodeposition], axis=1)

        return final_df

    def crop_residues_from_CIBUSmod(
        self,
        input_name: Literal["crop residues", "cover crop residues"] = "crop residues",
    ) -> pd.DataFrame:
        """
        Retrieve crop residue carbon inputs from CIBUSmod data output.

        For ordinary crop residues, rhizodeposition is derived from below-ground
        biomass and added as a separate residue category. For cover crops,
        rhizodeposition is derived from the below-ground residue pool.

        Parameters
        ----------
        input_name
            Residue input type. Must be either ``"crop residues"`` or
            ``"cover crop residues"``.

        Returns
        -------
        pd.DataFrame
            Carbon input data with residue categories in the column MultiIndex.
        """
        self.par.clear()
        self.par.set(input=input_name)
        
        agg_cols = {k:None for k in self.agg_cols}
        if input_name == 'crop residues':
            tab_cols = {'crop':['land_use',None],'prod_system':None,'residue':None}
        elif input_name == 'cover crop residues':
            tab_cols = {'crop':['land_use',None],'prod_system':None,'cover_crop':None,'residue':None}

        # Get base DataFrame
        df = self.session.get_attr('C',self.inputs[input_name], agg_cols | tab_cols, interpolate=True).xs(self.lu,level='land_use',axis=1)
        if input_name == 'crop residues':
            # Get below gound biomass C and add 'residue' columns level
            rhizodep = self.session.get_attr('C','below_ground_biomass_C', {'crop':['land_use',None]} | agg_cols, interpolate=True).xs(self.lu,level='land_use',axis=1)
            # Select only cropland
            rhizodep = pd.concat({'rhizodeposition': rhizodep}, names=['residue'], axis=1)
            # Calculate rhizodeposition
            f_rhizodep = self.par.get_from_frame('rhizodeposition', rhizodep)
            rhizodep = rhizodep * f_rhizodep
            # Add rhizodeposition to below ground residues in main DataFrame
            rhizodep = rhizodep.reorder_levels(df.columns.names, axis=1)
            df = pd.concat([df,rhizodep], axis=1)
        else:
            # For cover crops below ground biomass C equals below ground residue C
            rhizodep = df.xs('below ground', level='residue', axis=1, drop_level=False).rename({'below ground':'rhizodeposition'}, axis=1)
        # Calculate rhizodeposition
        f_rhizodep = self.par.get_from_frame('rhizodeposition', rhizodep)
        rhizodep = rhizodep * f_rhizodep
        # Add rhizodeposition to below ground residues in main DataFrame
        rhizodep = rhizodep.reorder_levels(df.columns.names, axis=1)
        df = pd.concat([df,rhizodep], axis=1)

        return df

    def make_input(
        self,
        crop_residue_method: Literal["Bolinder", "CIBUSmod"] = "Bolinder",
    ) -> None:
        """
        Prepare ICBM input data in terms of re-values, C input and humification coefficients
        for all input sources.

        This method computes:

        - crop areas and total land-use area
        - aggregated environmental modifiers ``re``
        - annual carbon inputs per hectare for each configured input source
        - humification coefficients ``h`` for each input source

        All generated datasets are stored in ``self.data_attr``.

        Parameters
        ----------
        crop_residue_method
            Method used to derive crop residue inputs. Must be either
            ``"Bolinder"`` or ``"CIBUSmod"``.

        Returns
        -------
        None
        """
        vprint = verbose_init(True, id_str='SoilC')
        
        agg_cols = {k:None for k in self.agg_cols}
        tab_cols = {'crop':['land_use',None],'prod_system':None}
        
        self.crop_areas = self.session.get_attr('C','area', agg_cols | tab_cols, interpolate=True).xs(self.lu,level='land_use',axis=1)
        self.total_area = self.crop_areas.T.groupby(self.agg_cols).sum().T
        
        # Calculate re per agg_cols
        vprint("Calculating re ...")
        self.par.clear()
        re_crop = (
            (self.crop_areas * self.par.get_from_frame('re_crop',self.crop_areas))
            .T.groupby(self.agg_cols).sum().T
            .div(self.total_area)
        )
        re_clim = self.par.get_from_frame('re_clim',re_crop)
        re = re_crop * re_clim

        self.data_attr.add(
            re,
            name = f're',
            unit = '-',
            orig = 'SoilCarbon',
            desc = f're-values for {self.lu} aggregated over ({", ".join(agg_cols)})'
        )
        
        for input_name in self.inputs:
            
            vprint(f"Getting {input_name} C input and h-values ...")

            if input_name == 'crop residues':
                if crop_residue_method == 'Bolinder':
                    df = self.crop_residues_from_Bolinder(input_name)
                elif crop_residue_method == 'CIBUSmod':
                    df = self.crop_residues_from_CIBUSmod(input_name)
                else:
                    raise ValueError("crop_residue_method must be one of 'Bolinder' or 'CIBUSmod'")
            elif input_name == 'cover crop residues':
                df = self.crop_residues_from_CIBUSmod(input_name)
            else:
                if input_name == 'manure':
                    tab_cols = {'crop':'land_use','prod_system':None,'species':None,'MMS':None}
                elif input_name == 'organic fertiliser':
                    tab_cols = {'crop':'land_use','prod_system':None,'treatment':None}
                # Get C input data from Session
                df = self.session.get_attr('C',self.inputs[input_name], agg_cols | tab_cols, interpolate=True).xs(self.lu,level='land_use',axis=1)

            # Express C in as kg C/ha total land use aggregated over agg_cols
            C_in = df / index_to_multi(self.total_area, axis=1).reindex(df.columns, axis=1)

            self.data_attr.add(
                C_in,
                name = f'C_in ({input_name})',
                unit = 'kg/ha/year',
                orig = 'SoilCarbon',
                desc = f'Carbon inputs per hectare {self.lu} from {input_name} aggregated over ({", ".join(agg_cols)})'
            )

            # Get h-values
            self.par.clear()
            self.par.set(input=input_name)
            h = pd.Series(
                self.par.get('h', **df.columns.to_frame().to_dict('list')),
                index = df.columns
            ).rename('h')
            
            self.data_attr.add(
                h,
                name = f'h ({input_name})',
                unit = '-',
                orig = 'SoilCarbon',
                desc = f'h-values for {input_name}'
            )

        # Update areas to exclude organic soils as these should not be
        # included in flux calculations.
        org_soils = self.session.get_attr(
            'C',
            'organic_soil_area',
            {'crop':'land_use','region':None},
            interpolate=True
        ).loc[:,self.lu]
        share_mineral_soil = (self.total_area - org_soils) / self.total_area
        self.total_area *= share_mineral_soil
        self.crop_areas *= index_to_multi(share_mineral_soil, axis=1).reindex(self.crop_areas.columns, axis=1)

        vprint(type='end')

    def run_ICBM(self, extend: int = 0) -> None:
        """
        Run the ICBM model for all configured carbon input sources.

        For each input source and scenario, this method simulates the young pool
        ``Y``, old pool ``O``, and total stock ``C = Y + O``. Results are stored
        in ``.data_attr`` and can be retrieved with the methods ``.get_data()``
        and ``.get_flux()``.

        Parameters
        ----------
        extend
            Number of additional years to append to the simulation using
            forward-filled input data. Defaults to ``0``.

        Returns
        -------
        None
        """
        vprint = verbose_init(True, id_str='SoilC')
        vprint('Running ICBM ...')
        
        for input_name in self.inputs:
            Y_dfs = []
            O_dfs = []
            C_dfs = []
            for scn in self.data_attr.get("re").index.unique('scn'):
                
                re = self.data_attr.get("re").loc[scn]
                C_in = self.data_attr.get(f"C_in ({input_name})").loc[scn]
                h = self.data_attr.get(f"h ({input_name})")
                
                Y,O = icbm_vec(
                    C_in = C_in,
                    h = h,
                    re = index_to_multi(re, axis=1).reindex(C_in.columns, axis=1),
                    ky = self.par.get('ky')[0],
                    ko = self.par.get('ko')[0],
                    extend = extend,
                    spinup=True
                )
                
                C=Y+O

                Y_dfs += [pd.concat({scn: Y}, names=['scn'])]
                O_dfs += [pd.concat({scn: O}, names=['scn'])]
                C_dfs += [pd.concat({scn: C}, names=['scn'])]

            Y_final = pd.concat(Y_dfs).rename_axis(['scn','year'])
            O_final = pd.concat(O_dfs).rename_axis(['scn','year'])
            C_final = pd.concat(C_dfs).rename_axis(['scn','year'])
            
            self.data_attr.add(
                Y_final,
                name = f'Y ({input_name})',
                unit = 'kg C/ha',
                orig = 'SoilCarbon',
                desc = f'Young carbon pool for inputs from {input_name} aggregated over ({", ".join(self.agg_cols)})'
            )
            self.data_attr.add(
                O_final,
                name = f'O ({input_name})',
                unit = 'kg C/ha',
                orig = 'SoilCarbon',
                desc = f'Old carbon pool for inputs from {input_name} aggregated over ({", ".join(self.agg_cols)})'
            )
            self.data_attr.add(
                C_final,
                name = f'C ({input_name})',
                unit = 'kg C/ha',
                orig = 'SoilCarbon',
                desc = f'Total carbon pool for inputs from {input_name} aggregated over ({", ".join(self.agg_cols)})'
            )

        vprint(type='end')

    def _normalize_groupby(
        self,
        groupby: str | list[str] | dict[str, str | None] | None,
    ) -> tuple[list[str], dict[str, str]]:

        if groupby == "none" or groupby is None:
            return [], {}
        if isinstance(groupby, str):
            return [groupby], {}
        if isinstance(groupby, list):
            return groupby, {}
        if isinstance(groupby, dict):
            aggregate = {k: v for k, v in groupby.items() if v is not None}
            return list(groupby.keys()), aggregate
        raise TypeError("groupby must be 'none', a string, a list, or a dict")

    def get_data(
        self,
        pool: Literal["Y", "O", "C", "C_in"] = "C",
        input_name: str = "all",
        groupby: str | list[str] | dict[str, str | None] | None = "none"
    ) -> pd.Series | pd.DataFrame:
        """
        Return per-hectare (kg C/ha) soil carbon stocks or inputs.
        
        If ``"region"`` is supplied to ``groupby``, stocks or inputs are expressed
        per hectare of total land use in each region. If region is not used to
        group, stocks or inputs represent the area-weighted national value. As such,
        the area-weighted national stock can increase even if all regional stocks
        decrease if land use in regions with high stocks increase, and vice versa.
        If other ``groupby`` levels are used, those represent the fraction of regional
        or national stocks/inputs derived from that source.


        Parameters
        ----------
        pool
            Carbon pool to retrieve. Must be one of ``"Y"``, ``"O"``, ``"C"``,
            or ``"C_in"`` to retrieve carbon inputs to soils
        input_name
            Input source to retrieve. Use ``"all"`` to aggregate across all
            configured input sources, or pass one of ``self.inputs``.
        groupby
            Grouping specification for output aggregation.

            Supported forms are:

            - ``"none"`` or ``None``: return a total area-weighted Series
            - ``str``: group by one column level
            - ``list[str]``: group by multiple column levels
            - ``dict[str, str | None]``: group and optionally remap levels
              before aggregation; values of ``None`` mean no remapping

        Returns
        -------
        pd.Series | pd.DataFrame
            Per-hectare Soil carbon stocks or inputs. A Series is returned
            when no grouping is requested; otherwise a DataFrame is returned.
            Results are expressed as kg C/ha of total land-use.
        """
        
        groupby, aggregate = self._normalize_groupby(groupby)
            
        if not (input_name in self.inputs or input_name == 'all'):
            raise ValueError(f"input must be 'all' or one of {list(self.inputs)}")

        if input_name == 'all':
            dfs = []
            # Get column levels shared across all DataFrames
            shared_lvls = list(set.intersection(
                *[
                    set(self.data_attr.get(f'{pool} ({inp})').columns.names)
                    for inp in self.inputs
                ]
            ) - set(self.agg_cols))
            for inp in self.inputs:
                part_df = (
                    self.data_attr.get(f'{pool} ({inp})')
                    .T.groupby(self.agg_cols + shared_lvls).sum().T
                )
                part_df.columns = multiindex_product([part_df.columns,pd.Index([inp], name='input')])
                dfs += [part_df]
            df = pd.concat(dfs, axis=1)
        else:
            df = self.data_attr.get(f'{pool} ({input_name})')

        total_area = self.total_area.reindex(df.index, axis=0).ffill()

        # Only the groupby keys based on agg_cols should be used for area aggregation
        groupby_area = [g for g in groupby if g in self.agg_cols]

        C_series = len(groupby) == 0
        area_series = len(groupby_area) == 0

        if C_series:
            final_df = (
                (df * index_to_multi(total_area, axis=1).reindex(df.columns, axis=1))
                .sum(axis=1)
                /
                total_area.sum(axis=1)
            )
        else:
            weighted_df = (
                (df * index_to_multi(total_area, axis=1).reindex(df.columns, axis=1))
                .T.groupby(groupby).sum().T
            )
            if set(aggregate) & set(weighted_df.columns.names):
                for src, tgt in aggregate.items():
                    if src in weighted_df.columns.names:
                        weighted_df = (
                            weighted_df
                            .rename(self.par.get_rel(src,tgt), level=src, axis=1)
                            .rename_axis(columns={src:tgt})
                        )
                weighted_df = weighted_df.T.groupby(weighted_df.columns.names).sum().T
            if area_series:
                final_df = weighted_df.div(total_area.sum(axis=1), axis=0)
            else:
                area_grouped = total_area.T.groupby(groupby_area).sum().T
                if set(aggregate) & set(area_grouped.columns.names):
                    for src, tgt in aggregate.items():
                        if src in area_grouped.columns.names:
                            area_grouped = (
                                area_grouped
                                .rename(self.par.get_rel(src,tgt), level=src, axis=1)
                                .rename_axis(columns={src:tgt})
                            )
                    area_grouped = area_grouped.T.groupby(area_grouped.columns.names).sum().T
                if isinstance(weighted_df.columns, pd.MultiIndex):
                    final_df = (
                        weighted_df /
                        index_to_multi(area_grouped, axis=1).reindex(weighted_df.columns, axis=1)
                    )
                else:
                    final_df = (
                        weighted_df / area_grouped
                    )

        # Fix
        final_df.index.names = df.index.names

        return final_df
    
    def get_flux(
        self,
        pool: Literal["Y", "O", "C"] = "C",
        input_name: str = "all",
        groupby: str | list[str] | dict[str, str | None] | None = "none",
        as_CO2: bool = True,
    ) -> pd.Series | pd.DataFrame:
        """
        Return flux derived from year-to-year changes in per-hectare stocks.

        Flux is defined as the year-to-year change in stock density (kg C/ha)
        within each region multiplied by the total land-use area in the same
        region.

        This means that pure changes in area do not create flux when per-hectare
        stocks are constant.

        Parameters
        ----------
        pool
            Carbon pool to use. Must be one of ``"Y"``, ``"O"``, or ``"C"``.
        input_name
            Input source to retrieve. Use ``"all"`` to aggregate across all
            configured input sources, or pass one of ``.inputs``.
        groupby
            Grouping specification, interpreted in the same way as in
            `.get_data()`.

            Important:
            - flux is always computed at least at ``region`` level
            - if ``region`` is not included in the requested output grouping,
              regional fluxes are summed
        as_CO2
            If ``True`` convert from kg C/year to kg CO2/year using ``44/12``.

        Returns
        -------
        pd.Series | pd.DataFrame
            Flux time series. A Series is returned when no grouping is
            requested; otherwise a DataFrame is returned.

            Units are:
            - kg C/year if ``as_CO2=False``
            - kg CO2/year if ``as_CO2=True``
        """
        groupby_list, _ = self._normalize_groupby(groupby)

        if not (input_name in self.inputs or input_name == "all"):
            raise ValueError(f"input_name must be 'all' or one of {list(self.inputs)}")

        # Always keep region in the intermediate representation so that flux is
        # computed as delta(stock_per_ha)_region * area_region.
        if isinstance(groupby, dict):
            detail_groupby: str | list[str] | dict[str, str | None] | None = (
                {"region": None} | groupby
                if "region" not in groupby
                else groupby
            )
        else:
            detail_levels = ["region"] + [g for g in groupby_list if g != "region"]
            detail_groupby = detail_levels if len(detail_levels) > 0 else "region"

        # Per-hectare stocks, but always resolved at region level (and any
        # requested extra dimensions).
        stock_ha = self.get_data(
            pool=pool,
            input_name=input_name,
            groupby=detail_groupby,
        )

        # Ensure DataFrame for uniform handling
        return_series = isinstance(stock_ha, pd.Series)
        if return_series:
            stock_ha = stock_ha.to_frame(name=pool)

        # Year-to-year change in per-hectare stocks within each scenario
        ha_flux = -stock_ha.groupby("scn").diff()

        # Region area aligned to the same rows (applies if extend > 0 was used in run_ICBM)
        total_area = self.total_area.reindex(stock_ha.index, axis=0).ffill()

        # Multiply by regional area. If extra dimensions are present, broadcast
        # region area across them.
        if isinstance(stock_ha.columns, pd.MultiIndex):
            area_aligned = index_to_multi(total_area, axis=1).reindex(
                stock_ha.columns,
                axis=1,
            )
        else:
            area_aligned = total_area.reindex(stock_ha.columns, axis=1)

        flux = ha_flux * area_aligned

        if as_CO2:
            flux = flux * (44.0 / 12.0)

        # If region is not requested in final output, sum over region after flux
        # has been computed.
        if "region" not in groupby_list:
            if len(groupby_list) == 0:
                result: pd.Series | pd.DataFrame = flux.sum(axis=1)
            else:
                result = flux.T.groupby(groupby_list).sum().T
        else:
            result = flux

        # Return Series when no grouping is requested
        if len(groupby_list) == 0 and isinstance(result, pd.DataFrame):
            return result.iloc[:, 0] if result.shape[1] == 1 else result.sum(axis=1)

        return result

    def get_flux_to_deltaT(self):
        """
        Convenience method to return SOC flux suitable to pass to
        impact.get_deltaT()
        """
        flux = self.get_flux(groupby=['prod_system','region'])
        flux.columns = multiindex_product([
            flux.columns,
            pd.MultiIndex.from_tuples(
                [('SOC flux','SOC flux','mineral soils','CO2')],
                names=['process','sub-process','item','compound']
            )
        ]).reorder_levels(
            ['process','sub-process','prod_system','item','region','compound']
        )
        return flux