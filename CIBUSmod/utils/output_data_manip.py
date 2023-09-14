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

    res_herd.id_attr = AnimalHerd.id_attr
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
    
    res_herd.data_attr = data_attr_in_all

    return res_herd

def get_GHG(output, CO2eq=True):
    
    # Conversion factors --------->
    to_GHG = {
        'CH4' : 1,
        'N2O-N' : (44/28),
        'NH3-N' : 0.01 * (44/28),

    }
    # -----
    to_GHG_names = {
        'CH4' : 'CH4',
        'N2O-N' : 'N2O',
        'NH3-N' : 'N2Oind'
    }
    # -----
    to_CO2eq = {
        'CO2' : 1,
        'CH4' : 25,
        'N2O' : 298,
        'N2Oind' : 298
    }
    # <--------------------------
    
    for scn, year in output.index:
        # ENTERIC METHANE
        enteric = (
            output.loc[(scn,year),'ani'].enteric_methane
            .groupby(['prod_system','species','breed'], axis=1)
            .sum()
        )
        enteric.columns = pd.MultiIndex.from_tuples(
            [(ps,f'{sp}, {br}','CH4') for ps,sp,br in enteric.columns],
            names = ['prod_system', 'item', 'compound']
        )
        enteric = pd.concat([enteric], keys=['enteric fermentation'], names=['process'], axis=1)

        # MANURE MANAGEMENT
        manure = (
            pd.concat([
                # N losses
                output.loc[(scn,year),'ani']
                .manure.N_loss
                .groupby(['prod_system','species','breed','compound'], axis=1)
                .sum(),
                # VS losses
                output.loc[(scn,year),'ani']
                .manure.VS_loss
                .groupby(['prod_system','species','breed','compound'], axis=1)
                .sum()
            ], axis=1)
        )
        manure.columns = pd.MultiIndex.from_tuples(
            [(ps,f'{sp}, {br}',cp) for ps,sp,br,cp in manure.columns],
            names = ['prod_system', 'item', 'compound']
        )
        manure = pd.concat([manure], keys=['manure management'], names=['process'], axis=1)

        # AGRICULTURAL SOILS
        soils = (
            pd.concat([
                pd.concat([
                    getattr(output.loc[(scn,year),'crp'].fertiliser, attr)
                    .unstack('prod_system')
                    .groupby(['prod_system','compound'], axis=1).sum()
                    .groupby('region').sum()
                ], keys=[attr.split('_N_')[0].replace('_',' ')], names=['item'], axis=1)
                for attr in
                ['manure_N_application_loss','manure_N_soil_loss',
                'mineral_N_application_loss','mineral_N_soil_loss',
                'crop_residues_N_soil_loss']
            ], axis=1)
            .rename(columns={'mineral':'mineral fertiliser'})
            .reorder_levels(['prod_system','item','compound'], axis=1)
        )
        soils = pd.concat([soils], keys=['agricultural soils'], names=['process'], axis=1)

        res = (
            pd.concat([enteric,manure,soils], axis=1)
            .groupby(['process','prod_system','item','compound'], axis=1)
            .sum()
        )

        # Get only GHGs and compunds with indirect GHG emissions
        res = res.loc[:,res.columns.isin(to_GHG, level='compound')]

        # Convert to GHG emissions
        res = (
            res
            .mul([to_GHG[cp] for cp in res.columns.get_level_values('compound')], axis=1)
            .rename(to_GHG_names, axis=1)
        )

        res = pd.concat([res], keys=[year], names=['year'], axis=1)
        res = pd.concat([res], keys=[scn], names=['scn'], axis=1)

        if CO2eq:
            # Calculate CO2 equivalents
            res = (
                res
                .mul([to_CO2eq[cp] for cp in res.columns.get_level_values('compound')], axis=1)
            )

        return res