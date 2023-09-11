import warnings
import pandas as pd

from ..production_systems.animal_herd import AnimalHerd
from ..production_systems.animal_herd import StaticAnimalHerd
from ..production_systems.feed_mgmt import Feed
from ..production_systems.manure_mgmt import Manure

from ..utils.misc import rgetattr,rsetattr

def concat_herds(herds):
    '''Combines multiple AnimalHerd objects
    
    Parameters
    ----------
    herds : itterable of AnimalHerd objects
    
    Returns
    -------
    StaticAnimalHerd object'''
    res_herd = StaticAnimalHerd()

    for attr in AnimalHerd.id_attr:
        setattr(res_herd,attr,'aggregated')

    res_herd.feed = Feed()
    res_herd.manure = Manure()

    # Check presence of data attributes in AnimalHerd objects
    # Only attributes present in all AnimalHerd objects are 
    # retained in the combined StaticAnimalHerd object
    data_attr_union = set.union(*[set(h.data_attr) for h in herds])
    data_attr_in_all = set.intersection(*[set(h.data_attr) for h in herds])
    data_attr_in_some = data_attr_union - data_attr_in_all
    if len(data_attr_in_some) > 0:
        pass
        # Should a warning be printed here?
        # warnings.warn(f'Data attributes {data_attr_in_some} not pressent in all AnimalHerds and therfore not retained.')

    # Go through data attributes and concatenate
    for attr in data_attr_in_all:

        df = pd.concat(
            [
                pd.concat({herd.species : 
                    pd.concat({herd.breed :
                        pd.concat({herd.sub_system : rgetattr(herd,attr)},
                            names=['sub_system'],axis=1)},
                        names=['breed'],axis=1)},
                    names=['species'],axis=1)
                if rgetattr(herd,attr) is not None else None for herd in herds
            ],
            axis=1
        )

        # Group and sum columns to avoid duplicates
        df = df.groupby(df.columns.names, axis=1).sum()

        rsetattr(res_herd,attr,df)

    return res_herd
