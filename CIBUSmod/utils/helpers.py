'''This module contains some helper functions for acheiving specific behaviour in model runs
'''

import pandas as pd
import numpy as np
import warnings

def check_constraints(geodist):
    '''Produces boxplots to check constraint violation'''

    import matplotlib.pyplot as plt
    from .plot.utils import wrapText
    
    x = geodist.problem.variables()[0].value

    plot_dfs = []
    for str, cons in geodist.constraints.items():
        left = cons['left']
        right = cons['right']
        rel = cons['rel']
        pars = cons['pars']
    
        M_rows = [m.rows for m in pars.values() if hasattr(m, 'rows')][0]
        try:
            M_rows = np.concatenate(list(M_rows.values()))
        except:
            pass
        res = pd.DataFrame(
            index = M_rows
        )
        
        res['left'] = left(x, **pars)
        res['right'] = right(**pars)
        res['left - right'] = res['left'] - res['right']
        res['(left - right) / right'] = res['left - right'] / res['right'].where(res['right']>0, np.nan)

        plot_dfs.append(res['left - right'].rename(str))

    nrow = int(np.ceil(len(plot_dfs)/3))
    fig,axs = plt.subplots(nrow, 3, figsize=(3*3,3*nrow))
    for i,df in enumerate(plot_dfs):
        ax=axs.flatten()[i]
        ax.axhline(0, c='black', linewidth=0.5, linestyle='--')
        ax.boxplot(df, flierprops=dict(marker='.', markerfacecolor='red', markeredgecolor='red', markersize=2))

        if '==' in df.name:
            ax.text(0.05, 0.95, f'max: {df.max():.1e}', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.05, 0.05, f'min: {df.min():.1e}', verticalalignment='center', transform=ax.transAxes)
        elif '>=' in df.name:
            ax.text(0.05, 0.95, f'min: {df.min():.1e}', verticalalignment='center', transform=ax.transAxes)
        elif '<=' in df.name:
            ax.text(0.05, 0.05, f'max: {df.max():.1e}', verticalalignment='center', transform=ax.transAxes)
            
        
        ax.set_ylabel('left - right')
        ax.set_xticks([])
        wrapText(ax.set_title(df.name, size=9))
        

    plt.tight_layout()
    
    return plot_dfs

def induce_beef_exports(demand, herds, beef_food_name = 'Bovine meat and products'):

    '''Induces beef exports in DemandAndConversions if beef production from dairy
    systems under given demand for milk products exceeds total beef demand.
    This is to avoid not finding any solution when running the GeoDistributor.

    The following data attributes in the DemandAndConversions object are modified:
    'export_demand'
    'animal_prod_demand'
    'animal_by_products' are modified

    This function should be run after `DemandAndConversion` and `AnimalHerd`s
    have been calculated but before running `GeoDistributor`
    
    Parameters
    ----------
    demand : DemandAndConversions object
    herds : pd.Series of AnimalHerd objects
    beef_food_name : str, default 'Bovine meat and products'
        Name of food item representing beef
    
    Returns
    -------
    None
    '''  

    # Get cattle milk and meat demand
    milk_demand = demand.animal_prod_demand.xs(('cattle', 'milk'), level=('species', 'animal_prod')).sum(axis=1)
    meat_demand = demand.animal_prod_demand.xs(('cattle', 'meat'), level=('species', 'animal_prod')).sum(axis=1)

    # Calculate meat/milk for all dairy herds
    meat_per_milk = pd.DataFrame()
    for herd in herds:
        if herd.species == 'cattle' and herd.breed == 'dairy':
            ps = herd.prod_system
            prod = herd.production.groupby(['prod_system','animal_prod'], axis=1).sum()
            milk = prod.xs('milk', axis=1, level='animal_prod').loc[:,ps]
            meat = prod.xs('meat', axis=1, level='animal_prod')
            # Calculate meat per milk add herd production system and append to DF
            meat_per_milk = pd.concat([
                meat_per_milk,
                pd.concat({ps: meat.div(milk, axis=0)}, names=['herd_prod_system'], axis=1)
            ], axis=1)

    # Check that meat/milk is equal across herds and take mean
    # otherwise take median and warn
    if meat_per_milk.groupby(['herd_prod_system', 'prod_system'], axis=1).transform(lambda x: abs(x-x.mean()) < 1e-6).all(axis=1).all():
        meat_per_milk = meat_per_milk.groupby(['herd_prod_system', 'prod_system'], axis=1).mean()
    else:
        warnings.warn('meat/milk from dairy herds not equal across all sub_systems. median(meat/milk) is used but this is likely to induce more beef exports than strictly needed')
        meat_per_milk = meat_per_milk.groupby(['herd_prod_system', 'prod_system'], axis=1).median()
    
    # Check that meat/milk is equal across regions and take mean
    # otherwise take median and warn
    if meat_per_milk.transform(lambda x: abs(x-x.mean()) < 1e-6).all().all():
        meat_per_milk = meat_per_milk.mean()
    else:
        warnings.warn('meat/milk from dairy herds not equal across all regions. median(meat/milk) is used but this is likely to induce more beef exports than strictly needed')
        meat_per_milk = meat_per_milk.median()

    # Calculate ammount of meat from dairy systems
    meat_from_dairy = milk_demand * 0
    for ps in milk_demand.index:
        meat_from_dairy = meat_from_dairy.add(milk_demand.loc[ps] * meat_per_milk.loc[ps], fill_value=0)
    
    # Calculate beef exports needed
    induced_beef_exports = meat_from_dairy-meat_demand
    induced_beef_exports = induced_beef_exports.where(induced_beef_exports > 0, 0)

    if (induced_beef_exports > 0).any():
        # Add food and food_group index levels
        induced_beef_exports = pd.concat([induced_beef_exports], keys=['export'], names=['food_group'])
        induced_beef_exports = pd.concat([induced_beef_exports], keys=[beef_food_name], names=['food'])

        # Apply conversion factor CW --> meat as consumed
        demand.par.clear()
        induced_beef_exports = (
            induced_beef_exports *
            demand.par.get(
                'conv_factor_main',
                species='cattle',
                animal_prod='meat',
                **induced_beef_exports.index.to_frame().to_dict('list')
            )/100
        )
        
        # Add induced beef exports to export demand
        demand.export_demand = demand.export_demand.add(
            induced_beef_exports,
            fill_value = 0
        )
        # Recalculate animal product demand and by-products
        demand.calculate_animal_product_demand()

        print(f"Induced export of {round(induced_beef_exports.sum()/1000):,} tonnes '{beef_food_name}'")

    return None

def drop_from_objective(geodist, which, key, level=0):
    '''Drops items from objective function by removing corresponding elements
    from the P1 matrix, x0 and scale_f.

    This function must be run after running GeoDistributor.make() and before
    running GeoDistributor.solve()
    
    Parameters
    ----------
    geodist : GeoDistributor object
    which : str {'crp','ani'}
        Whether to operate on crops ('crp') or animals ('ani')
    key : label or sequence of labels
    level : int/level name or list thereof, optional

    Returns
    -------
    pandas.MultiIndex corresponding to the dropped items
    '''
    
    try:
        idx = geodist.x0_idx[which]
    except KeyError:
        raise ValueError("which must be one of 'ani' and 'crp'")
    
    assert (geodist.P1.rows[which] == idx).all()
    assert (geodist.x0[which].index == idx).all()
    assert (geodist.scale_f[which].index == idx).all()
    
    def _to_bool_array(locs, n):
        bool_array = np.zeros(n, dtype=bool)
        bool_array[np.r_[locs]] = True
        return bool_array
    
    def _get_locs(idx, key, level):
        drop_locs, drop_idx = idx.get_loc_level(key, level=level, drop_level=False)
        if not isinstance(drop_locs, np.ndarray):
            drop_locs = _to_bool_array(drop_locs, len(idx))
        locs = np.invert(drop_locs)
        return locs, drop_idx
    
    part_locs, drop_idx = _get_locs(idx, key, level)
    new_idx = idx[part_locs]
    
    if which == 'ani':
        locs = np.concatenate([part_locs, np.ones(len(geodist.x0_idx['crp']), dtype=bool)])
    else:
        locs = np.concatenate([np.ones(len(geodist.x0_idx['ani']), dtype=bool), part_locs])
    
    # Drop items from P1 and x0 and scale_f
    geodist.P1.M = geodist.P1.M[locs, :]
    geodist.P1.rows[which] = new_idx
    geodist.x0[which] = geodist.x0[which].loc[new_idx]
    geodist.x0_idx[which] = new_idx
    geodist.scale_f[which] = geodist.scale_f[which].loc[new_idx]
    
    return drop_idx