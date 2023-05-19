import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr, multiply_aligned
from ..utils.misc import Container

class FertiliserMgmt():
    '''Class that that calculates ammount of fertiliser applications needed for crop production
    and balances this with manure generation, etc.
    
    Parameters
    ----------
    crops : CropProduction object
    herds : (pandas.Series of) AnimalHerd object(s)
    par : ParameterRetriever object
    '''

    def __init__(self,crops,herds,par):

        self.par = par

        if isinstance(herds, pd.Series):
            self.herds = herds
        else:
             self.herds = pd.Series(
                data=herds,
                index=pd.MultiIndex.from_tuples(
                    [(herds.species,herds.breed,herds.prod_system,herds.sub_system)],
                    names=['species','breed','prod_system','sub_system']
                )
            )

        self.check_index()

        self.index = list(self.herds)[0].index

    def check_index(self):
        if len(self.herds)>0:
            for n in range(len(self.herds)-1):
                if (self.herds[n].index != self.herds[n+1].index).any():
                    raise Exception('Indexes does not match across herds!')
                
    