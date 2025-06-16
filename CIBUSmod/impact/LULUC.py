import os
import pandas as pd
import numpy as np

from functools import reduce

from . import IMPACT_DATA_PATH
from ..utils.session_db import Session

class LUParcel():
    def __init__(
        self,
        t0 : int,
        A0 : float,
        C0 : dict,
        pars : dict
    ):
        self.pars = pars

        # Set initial time, area and carbon stocks
        self.t = t0
        self.At = A0
        self.Ct = C0

        # Create lists for time, area and carbon stocks time series
        self.T = []
        self.A = []
        self.C = {s : [] for s in self.pool_names}
    
    def spin_up(self, n=1_000):
        for _ in range(n+1):
            self.time_step()
        # Set t to t0
        self.t = self.T[0]
        # Set carbon stocks to those of last iteration
        for s in self.pool_names:
            self.Ct[s] = self.C[s][-1]
        # Reset lists
        self.T = []
        self.A = []
        self.C = {s : [] for s in self.pool_names}

    def stock_time_series(self) -> pd.Series:
        '''Get time series of carbon stocks [kg C]'''
        return pd.DataFrame(
            np.array([c for c in self.C.values()]).T * np.atleast_2d(np.array(self.A)).T,
            index = pd.Index(self.T, name='year'),
            columns = pd.Index([s for s in self.pool_names], name='pool')
        )

class LUParcel1(LUParcel):
    '''Class to keep track of areas and carbon stocks in land use parcels. Carbon stocks
    are assumed to change over time according to a first-order growth/decay function
    given by:
    
    Ct+1 = Ct + α(Cmax - Ct), where
    α = 1 - exp(-ln(2)/T50),
    
    Ct is the carbon stock (kg C/ha) at year t, Cmax is the maximum carbon stock,
    and T50 is the number of years it takes to reach exactly half way from Ct to Cmax
    (i.e. the half-time).
    
    Parameters
    ----------
    t0 : int
        Start year for land use parcel [year]
    A0 : float
        Area of land use parcel at time t=(t0-1) [ha]
    C0 : dict(
        C : float
            Carbon stock at t=(t0-1) [kg C / ha]
    )
    pars : dict(
        Cmax : float
            Maximum carbon stock in land use parcel [kg C/ha]
        T50 : float
            Half-time for first-order growth/decay function [years]
    )
    '''

    # Carbon pools
    pool_names = ['C']

    def time_step(self):
        '''Do one time-step'''
        alpha = 1 - np.exp(-np.log(2)/self.pars['T50'])
        self.Ct['C'] = self.Ct['C'] + alpha * (self.pars['Cmax'] - self.Ct['C'])

        self.A += [self.At]
        self.T += [self.t]
        self.C['C'] += [self.Ct['C']]

        self.t += 1

class LUParcel2(LUParcel):
    '''Class to keep track of areas and carbon stocks in a land use parcel. Carbon stocks
    are modelled with the simple forest carbon cycle model presented by Harmon (2001). The
    model consists of three carbon pools: live biomass (LC), detritus (DC) and soil (SC).

    CL(t+1) = Lmax*(1 - (1 - (CL(t)/Lmax)^(1/B2)) * np.exp(-B1))^B2
    CD(t+1) = CD(t)*(1 - k - sf) + CL(t)*m
    CS(t+1) = CS(t)*(1 - sk) + CD(t)*sf

    The equation for CL has been reformulated from the one presented by Harmon (2001) to express CL(t+1) as a function
    of CL(t) instead of t.

    Mark E. Harmon. 2001. Carbon cycling in forests: simple simulation models. H. J. Andrews Research Report Number 2.
    https://andrewsforest.oregonstate.edu/sites/default/files/lter/pubs/webdocs/reports/ccycleforest/ccycleforest.pdf 
    https://andrewsforest.oregonstate.edu/sites/default/files/lter/pubs/webdocs/reports/~1st-ccycleforest2.htm 
    
    Parameters
    ----------
    t0 : int
        Start year for land use parcel [year]
    A0 : float
        Area of land use parcel at time t=(t0-1) [ha]
    C0 : dict(
        CL : float
            Carbon stock in live biomass at t=(t0-1) [kg C / ha]
        CD : float
            Carbon stock in detritus at t=(t0-1) [kg C / ha]
        CS : float
            Carbon stock in soil at t=(t0-1) [kg C / ha]
    )
    pars : dict(
        Lmax : float
            Maximum live biomass [kg C / ha]
        B1 : float
            Rate of live biomass increase [-]
        B2 : float
            Growth lag [-]
        m : float
            Mortality rate [-]
        k : float
            Decomposition rate [-]
        sf : float
            Soil formation rate [-]
        sk : float
            Soil decomposition rate [-]
    )
    '''

    # Carbon pools
    pool_names = ['CL','CD','CS']

    def time_step(self):
        '''Do one time-step'''

        # The equation for CLt was reformulated to express CL(t+1) as a
        # function of CL(t) instead of t.
        self.Ct['CL'] = self.pars['Lmax']*(1 - (1 - (self.Ct['CL']/self.pars['Lmax'])**(1/self.pars['B2'])) * np.exp(-self.pars['B1']))**self.pars['B2']
        self.Ct['CS'] = self.Ct['CS']*(1 - self.pars['sk']) + self.Ct['CD']*self.pars['sf']
        self.Ct['CD'] = self.Ct['CD']*(1 - self.pars['k'] - self.pars['sf']) + self.Ct['CL']*self.pars['m']
        
        self.T += [self.t]
        self.A += [self.At]
        for s in self.pool_names:
            self.C[s] += [self.Ct[s]]

        self.t += 1

class LUCTimeSeries():
    '''Class that handles land use changes between a 'target' land  use and an 'alternative'
    land use. Land use changes are always assumed to affect the most recently converted land
    use parcels.

    Parameters
    ----------
    Ach : pd.Series
        Time-series of land use changes between 'target' and 'aternative' land use (negative
        values should represent reduced areas of the target land use). Index should represent
        years and have int dtype.
    parcel_class : LUCParcel object
        The LUCParcel object to use
    taget_lu_pars : dict
        Dict with parameters for 'target land use' to pass to LUCParcel
    alt_lu_pars : dict
        Dict with parameters for 'alternative land use' to pass to LUCParcel
    extend : int
        Number of years to extend time-series by. No land use changes are assumed after
        the final year in Ach time-series but carbon stocks are assumed to continue to 
        develop during the extended years.
    '''
    def __init__(
        self,
        Ach : pd.Series,
        parcel_class : LUParcel,
        target_lu_pars : dict,
        alt_lu_pars : dict,
        extend : int = 0
    ):
        # Make sure index is integer
        Ach.index = Ach.index.astype(int)
        # Make sure all years are included
        if list(range(Ach.index[0],Ach.index[-1]+1)) != list(Ach.index):
            raise ValueError('Ach.index must include all years from first to last year.')
        
        if extend > 0:
            Ach = Ach.reindex(
                pd.Index(
                    list(range(Ach.index[0],Ach.index[-1]+extend+1)),
                    name = Ach.index.name
                ),
                fill_value = 0
            )
            
        self.Ach = Ach
        self.LUCParcel = parcel_class
        self.pars = {
            'target' : target_lu_pars,
            'alt' : alt_lu_pars
        }
        

    def calculate_parcels(self) -> None:
        ''''''
        init_target_parcel = self.LUCParcel(
            t0 = self.Ach.index[0],
            A0 = -self.Ach.where(self.Ach<0).sum()+1,
            C0 = {s:0 for s in self.LUCParcel.pool_names},
            pars = self.pars['target']
        )
        init_alt_parcel = self.LUCParcel(
            t0 = self.Ach.index[0],
            A0 = self.Ach.where(self.Ach>0).sum()+1,
            C0 = {s:0 for s in self.LUCParcel.pool_names},
            pars = self.pars['alt']
        )
        # Do spin-up
        init_target_parcel.spin_up()
        init_alt_parcel.spin_up()
        
        self.target_LU_parcels = [init_target_parcel]
        self.alt_LU_parcels = [init_alt_parcel]
        self.inactive_LU_parcels = []

        for t, Ach in zip(self.Ach.index, self.Ach):
            
            if Ach < 0:
                from_parcels = self.target_LU_parcels
                to_parcels = self.alt_LU_parcels
                pars = self.pars['alt']
            elif Ach > 0:
                from_parcels = self.alt_LU_parcels
                to_parcels = self.target_LU_parcels
                pars = self.pars['target']
            else:
                from_parcels = None
                to_parcels = None
                pars = None

            Ach_abs = abs(Ach)

            if from_parcels is not None:
                while True:
                    last_from_parcel = from_parcels.pop()
                    if last_from_parcel.At > Ach_abs:
                        last_from_parcel.At -= Ach_abs
                        from_parcels.append(last_from_parcel)
                        to_parcels.append(
                            self.LUCParcel(
                                t0 = t,
                                A0 = Ach_abs,
                                C0 = last_from_parcel.Ct.copy(),
                                pars = pars
                            )
                        )
                        break
                    else:
                        to_parcels.append(
                            self.LUCParcel(
                                t0 = t,
                                A0 = last_from_parcel.At,
                                C0 = last_from_parcel.Ct.copy(),
                                pars = pars
                            )
                        )
                        Ach_abs -= last_from_parcel.At
                        last_from_parcel.At = 0
                        self.inactive_LU_parcels.append(last_from_parcel)                      
                
            for p in self.target_LU_parcels + self.alt_LU_parcels:
                p.time_step()

        return None

    def get_stocks(self) -> pd.Series:
        '''Get yearly carbon stocks [kg C]. Stocks allways start at zero in the first year'''
        stocks = []
        for p in self.target_LU_parcels + self.alt_LU_parcels + self.inactive_LU_parcels:
            stocks += [p.stock_time_series()]
        df = reduce(lambda x, y: x.add(y, fill_value=0), stocks)
        # Subtract initial carbon stocks
        df = df.sub(df.iloc[0], axis=1)
        return df
    
    def get_CO2(self) -> pd.Series:
        '''Get yearly CO2 emissions [kg CO2] due to carbon stock changes'''
        return -self.get_stocks().diff().fillna(0) * 3.67 # C -> CO2

def get_rewetting_emissions(
        session : Session,
        year0 : str = '2020',
        CO2eq : str|None = 'GWP100 AR4',
        interpolate : bool = False,
        return_area : bool = False,
        EF_CO2 : float = 0.5*(44/12)*1000, # kg CO2/ha
        EF_CH4 : float = 123 # kg CH4/ha
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    '''Function to calculate emissions of CO2 and CH4 from rewetted organic soils. Any
    reduction in the area of organic soils in year0 is assumed to result in an equivalent
    area of rewetted wetlands.

    This will likely be replaced by a more comprehensive framework for handling land use
    change and associated emissions in the future.
    
    Parameters
    ----------
    session : Session object
    year0 : str
    CO2eq : str or None, default 'GWP100 AR4'
        Method for translating GHGs to CO2-eq, if None emissions are not translated to CO2-eq
    interpolate : Bool, default False
        Interpolate between defined years
    return_area : Bool, default False
        If True, returns area tuple of (rewetted area, emissions)
    EF_CO2 : float, default from Lindgren & Lundblad (2014)
        Emission factor for CO2 emissions in kg CO2/ha
    EF_CH4 : float, default from Lindgren & Lundblad (2014)
        Emission factor for CH4 emissions in kg CH4/ha

    Returns
    -------
    pandas.DataFrame
    of the same structure as returned by impact.get_GHG()
    '''

    # Get area of organic soils
    area_org_soil = session.get_attr('c', 'organic_soil_area', 'region', interpolate=interpolate)

    # Get scenarios in data 
    scns = area_org_soil.index.unique('scn')

    # Calculate area of rewetted organic soils per year
    part_dfs = []
    for scn in scns:
        df_scn = area_org_soil.loc[[scn],:]
        part_dfs.append(
            -area_org_soil.loc[[scn],:].sub(
                area_org_soil.loc[(scn,year0),:],
                axis=1
            # If org_soils @ year <= org_soils @ year0 --> No wetlands
            # There are many other ways to think here...
            ).clip(upper=0)
        )
    # Combine areas for all scenarios
    area_rewetted = pd.concat(part_dfs)

    # Calculate CO2 and CH4 emissions
    CO2_rewetted = area_rewetted * EF_CO2
    CH4_rewetted = area_rewetted * EF_CH4
    # Add compound to column index
    CO2_rewetted = pd.concat({'CO2': CO2_rewetted}, names=['compound'], axis=1)
    CH4_rewetted = pd.concat({'CH4bio': CH4_rewetted}, names=['compound'], axis=1)

    if CO2eq:
        # Convert to CO2eq
        CF_CH4 = pd.read_csv(os.path.join(IMPACT_DATA_PATH, 'ghg_to_CO2eq.csv'), index_col=['ghg','method'])['factor'].loc[('CH4bio',CO2eq)]
        CH4_rewetted *= CF_CH4

    # Combine CO2 and CH4 emissions
    rewetting_emissions = pd.concat([CO2_rewetted, CH4_rewetted], axis=1)

    # Fix column index to match df returned by impact.get_GHG()
    rewetting_emissions = pd.concat({'rewetting': rewetting_emissions}, names=['process'], axis=1)
    rewetting_emissions = pd.concat({'rewetting': rewetting_emissions}, names=['sub-process'], axis=1)
    rewetting_emissions = pd.concat({'n/a': rewetting_emissions}, names=['prod_system'], axis=1)
    rewetting_emissions = pd.concat({'wetlands': rewetting_emissions}, names=['item'], axis=1)
    rewetting_emissions = rewetting_emissions.reorder_levels(['process', 'sub-process', 'prod_system', 'item', 'region', 'compound'], axis=1)

    if return_area:
        return (area_rewetted, rewetting_emissions)
    else:
        return rewetting_emissions