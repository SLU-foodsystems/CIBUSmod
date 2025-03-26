import numpy as np
import pandas as pd

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