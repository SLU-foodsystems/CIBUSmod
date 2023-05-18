import pandas as pd

from ..production_systems.animal_herd import AnimalHerd
from ..production_systems.animal_herd import StaticAnimalHerd
from ..production_systems.feed_mgmt import Feed
from ..production_systems.manure_mgmt import Manure

from ..utils.misc import rgetattr,rsetattr

def concat_herds(herds):
    res_herd = StaticAnimalHerd()

    for attr in AnimalHerd.id_attr:
        setattr(res_herd,attr,'aggregated')

    res_herd.feed = Feed()
    res_herd.manure = Manure()

    for attr in AnimalHerd.data_attr:
        rsetattr(
            res_herd,attr,
            pd.concat(
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
        )

    return res_herd