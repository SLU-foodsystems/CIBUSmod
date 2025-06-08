import os
import pandas as pd
import numpy as np

from . import IMPACT_DATA_PATH
from ..utils.session_db import Session

class ForestParcel():
    def __init__(self, t, At, Ct, Cmax, T50):
        self.Cmax = Cmax
        self.T50 = T50

        self.t = t
        self.At = At
        self.Ct = Ct
        
        self.C = [Ct]
        self.A = [At]
        self.T = [t]

    def time_step(self):
        self.t += 1
        alpha = 1 - np.exp(-np.log(2)/self.T50)
        self.Ct = self.Ct + alpha * (self.Cmax - self.Ct)

        self.C += [self.Ct]
        self.A += [self.At]
        self.T += [self.t]

    def stock_time_series(self):
        return pd.Series(
            np.array(self.C) * np.array(self.A),
            index = self.T
        )

    def CO2_emissions(self):
        stock = self.stock_time_series()
        return -stock.diff().fillna(0) * 3.67
    
if False:
    Ct = 0 # kg C/ha, initial forest carbon stock
    At = 0 # ha, Area where forest is regenerating
    Cin = 2 # kg C/ha, carbon stock in land converted to forest
    Cmax = 20 # kg C/ha, maximum carbon stock of forest
    T50 = 20 # years, time to reach half Cmax

    forest_parcels = []
    inactive_parcels = []
    old_forest_parcels = []

    At = 0
    A = []

    for t in range(200):
        if t < 100:
            if t < 20:
                Ain = 5 + np.random.randn()*10
            elif t < 50:
                Ain = -5 + np.random.randn()*10
            else:
                Ain = np.random.randn()*10
        else:
            Ain = 0

        A += [At]
        At += -Ain

        if Ain > 0:
            forest_parcels.append(ForestParcel(t, Ain, Cin, Cmax, T50))
        elif Ain < 0:
            while True:
                try:
                    last_parcel = forest_parcels.pop()
                except IndexError:
                    old_forest_parcel = ForestParcel(t-1, -Ain, Cmax, Cmax, T50)
                    old_forest_parcel.At += Ain
                    old_forest_parcel.time_step()
                    old_forest_parcels.append(old_forest_parcel)
                    break
                if last_parcel.At > -Ain:
                    last_parcel.At += Ain
                    break
                else:
                    Ain += last_parcel.At
                    last_parcel.At = 0
                    inactive_parcels += [last_parcel]
                    
            forest_parcels.append(last_parcel)

        for p in forest_parcels:
            p.time_step()

    stocks = []
    CO2 = []
    for p in forest_parcels + inactive_parcels + old_forest_parcels:
        stocks += [p.stock_time_series()]
        CO2 += [p.CO2_emissions()]

    d0 = pd.Series(A)
    d0.sort_index().plot(ylabel='Agri. area change')
    plt.show()

    d1 = pd.concat(stocks,axis=1).fillna(0).sum(axis=1)
    d1.sort_index().plot(ylabel='Regrowth forest carbon stocks')
    plt.show()

    d2 = pd.concat(CO2,axis=1).fillna(0).sum(axis=1)
    d2.sort_index().plot(ylabel='CO2 emissions')
    plt.show()

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