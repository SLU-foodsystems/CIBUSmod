'''This module contains helper functions for mapping GeoDistributor/FeedDistributor
solution vectors (x_ani, x_crp, x_fds) to dietary nutrient supply, and for using that
mapping to change a GeoDistributor/FeedDistributor's optimisation objective from the
default (minimise deviation of crop areas/animal numbers from x0) to instead maximise
the supply of a selected nutrient from a chosen set of crop and/or animal products.

See 'change_objective()' for the main entry point (to be called after 'dist.make()'
and before 'dist.solve()'), 'update_demand()' for the helper to be run after
'dist.solve()'/'dist.apply_solution()' to keep 'DemandAndConversions' consistent with
the optimised production, and 'nutrient_supply_from_x()' for the underlying per-unit
nutrient mapper (which can also be used on its own, e.g. for reporting "how much of
nutrient X came from crop Y" in an already-solved model).
'''

import warnings

import cvxpy
import numpy as np
import pandas as pd
import scipy.sparse

from ..optimisation.indexed_matrix import IndexedMatrix
from ..optimisation.utils import make_cvxpy_constraint

# Constraint keys as built by FeedDistributor.make_C10()/make_C15() -- shared
# between _split_byprod_supply_for_selected_products() and its two helpers so they
# can't drift out of sync with each other
_C10_KEY = 'C10: A10 @ x >= b10'
_C15_KEY = 'C15: A15 @ x >= 0'

def _empty_byprod_series():
    '''Empty (prod_system, by_prod)-indexed float Series, i.e. the identity element
    for the '.add(..., fill_value=0)' chains combining by-product supply/generation
    Series throughout this module. Must carry the right index NAMES (not just be
    empty) -- pandas' MultiIndex '.add()' raises "cannot join with no overlapping
    index names" when aligning an empty, unnamed-index Series against a named one,
    rather than treating it as a no-op.
    '''
    return pd.Series(
        dtype=float,
        index=pd.MultiIndex.from_tuples([], names=['prod_system', 'by_prod']),
    )

def change_objective(
        dist,
        nutrient,
        crop_prods=None,
        animal_prods=None,
        lambda_reg=1e-6,
        lambda_ridge=None,
        keep_linear=False,
    ):
    '''Changes a GeoDistributor/FeedDistributor's optimisation problem (in place) from
    the default objective (minimising deviation of crop areas/animal numbers from x0)
    to instead MAXIMISE the total domestic, edible supply of 'nutrient' from a
    selected set of crop products ('crop_prods') and/or animal products
    ('animal_prods'). All other (non-selected) products remain constrained to exactly
    meet their baseline demand, as in the default model.

    Must be called after 'dist.make(...)' and before 'dist.solve()' -- see notebook
    'notebooks/no_test/Minimise Feed Imports.ipynb' for the general pattern this is
    based on (there, 'make_import_opt_goal()'/'make_opt_goal()' swap in a custom
    objective for a FeedDistributor in the same way). After solving, call
    'update_demand()' to keep 'dist.demand' consistent with the optimised production.

    What this does, step by step
    -----------------------------
    1. Relaxes constraint C1 for the selected products only (see
       '_relax_C1_for_selected_products()'). C1 normally reads
       '[production] - [feed use] == [demand]' (where [demand] = food + non-food +
       export demand) for every crop/animal product. For the selected crop_prods/
       animal_prods this is changed to
       '[production] - [feed use] >= [non-food demand] + [export demand]', i.e. only
       the FOOD portion of demand is relaxed: production must still (at minimum)
       cover whatever of it is used as feed plus the original non-food and export
       demand, but is otherwise free to move -- up or down -- away from the original
       FOOD demand figure, so that the optimiser can trade off between selected
       products purely based on which combination maximises the objective. Every
       other (non-selected) product keeps its original equality constraint, i.e.
       keeps exactly meeting demand (food + non-food + export) as before.
    2. Rebuilds constraints 'C10'/'C15' (if present -- FeedDistributor only; see
       '_split_byprod_supply_for_selected_products()') so that the by-product
       supply they credit to a selected product is split the same way: a
       food-route part that moves with the now-relaxed production, and a fixed
       non-food+export-route part frozen at the baseline floor. This is needed
       because C10/C15 apply 'DemandAndConversions.by_prod_per_crop_prod'/
       'by_prod_per_animal_prod' -- ratios BLENDED across a product's demand
       routes -- directly to x_ani/x_crp, not to a fixed demand figure like C1; see
       that function for why the unmodified constraints would otherwise let
       production growth credit by-products (and hence feed availability) that the
       extra production was never actually going to generate.
    3. Builds a nutrient-supply mapper restricted to the selected products (see
       'nutrient_supply_from_x()').
    4. Replaces 'dist.problem' with a new cvxpy.Problem that maximises the mapped
       nutrient supply (implemented as minimising its negative) subject to the same
       constraints built by 'dist.make()' (with C1, C10 and C15 modified per steps
       1-2), plus a small regularisation term on variables that are not touched by
       the objective, purely to keep the LP numerically well posed (mirrors
       'make_opt_goal()' in 'notebooks/no_test/help_funs.py'), and a much smaller
       "ridge" term on EVERY x_ani/x_crp (selected or not) -- see 'lambda_ridge'
       below and '_set_maximise_objective()' for why this second term is needed.

    Other constraints were reviewed for interference with allowing selected products
    to exceed their baseline demand (requirement to check this is why this list
    exists -- see the module/notebook this is based on):
    - C10/C15 (by-product balance): DOES need adjusting -- see step 2 above -- since
      it applies a demand-route-blended ratio directly to x_ani/x_crp rather than to
      a fixed demand figure, unlike every other constraint below.
    - C2 (regional feed demand share), C16 (crop residue balance): inequalities
      relative to actual production with no equivalent blended-ratio dependency, so
      extra production only relaxes them further, never breaks them.
    - C3 (max land use), C5 (max crop share of a crop_prod), C6 (rotation shares):
      real physical limits on x_crp itself, correctly still enforced -- e.g. C3
      still caps how much *more* of a selected crop can physically be grown.
    - C4 (animal sub-system shares), C11-C14 (feed rations/imports): relate to
      x_ani/x_fds directly and are unaffected by relaxing C1's link between crop/
      animal PRODUCTION and demand.
    Only C1, C10 and C15 tie production to a fixed demand figure or a demand-route-
    blended ratio and so needed adjusting.

    Note on interpretation: with C1 relaxed, any production beyond baseline FOOD
    demand for the selected products has no explicit downstream disposition in the
    model (i.e. it is not literally allocated to extra consumption, export, etc. by
    other CIBUSmod modules) until 'update_demand()' is run after solving -- see that
    function. Non-food and export demand, by contrast, is still guaranteed to be at
    least met (it forms the floor of the relaxed constraint), so it does not need
    this treatment.

    Parameters
    ----------
    dist : GeoDistributor or FeedDistributor object
        Must already have been through 'dist.make(...)' (including constraint 1).
        Modified in place: 'dist.constraints' (C1 replaced) and 'dist.problem'
        (replaced with the new objective).
    nutrient : str
        Nutrient to maximise supply of, see 'nutrient_supply_from_x()'.
    crop_prods : list of str, default None
        crop_prod(s) to relax demand for and reward in the objective. At least one of
        'crop_prods'/'animal_prods' must be given.
    animal_prods : list of (species, animal_prod) tuples, default None
        Analogous to 'crop_prods' but for animal products.
    lambda_reg : float, default 1e-6
        Weight of the regularisation term applied to x not touched by the objective
        (see step 3 above).
    lambda_ridge : float, default None
        Weight of the additional ridge term applied to EVERY x_ani/x_crp regardless
        of selection (see '_set_maximise_objective()'). If None (default), uses
        'lambda_reg * 1e-3'. Only increase this if 'dist.solve()' still fails
        numerically after the default; a value too large will noticeably discourage
        growth of the selected products too.
    keep_linear : bool, default False
        If True, use a linear (L1) instead of quadratic (L2) regularisation term.

    Returns
    -------
    dict of pandas.Series
        The nutrient-supply mapper actually used in the objective (i.e. already
        restricted to 'crop_prods'/'animal_prods'), with keys 'ani', 'crp' and (for a
        FeedDistributor) 'fds', indexed like 'dist.x_idx[...]'.
    '''

    if not crop_prods and not animal_prods:
        raise ValueError(
            "At least one of 'crop_prods' or 'animal_prods' must be given."
        )
    if not hasattr(dist, 'x_idx_short'):
        raise ValueError(
            "dist.x_idx_short is not defined. Run dist.make(...) first "
            "(including constraint 1)."
        )

    crop_prods = list(crop_prods) if crop_prods else []
    animal_prods = list(animal_prods) if animal_prods else []

    # 1) Relax C1 for the selected products only
    _relax_C1_for_selected_products(dist, crop_prods, animal_prods)

    # 2) Split C10/C15's by-product supply matrices (if present) so growing a
    #    selected product only credits food-route by-products for the excess over
    #    its baseline non-food+export floor
    _split_byprod_supply_for_selected_products(dist, crop_prods, animal_prods)

    # 3) Build a nutrient-supply mapper restricted to the selected products. Note:
    #    pass the (possibly empty) lists as-is, NOT 'crop_prods or None' -- an empty
    #    list here must mean "reward no crop_prod", whereas None to
    #    nutrient_supply_from_x means "reward every crop_prod", which would credit
    #    (and, combined with the C1 relaxation below, implicitly allow to grow)
    #    products the caller never selected.
    mapper = nutrient_supply_from_x(
        dist,
        nutrient,
        crop_prods=crop_prods,
        animal_prods=animal_prods,
    )

    # 4) Replace dist.problem with one that maximises the mapped nutrient supply
    _set_maximise_objective(
        dist, mapper, lambda_reg=lambda_reg, lambda_ridge=lambda_ridge, keep_linear=keep_linear
    )

    return mapper

def update_demand(dist):
    '''After 'dist.solve()' (and 'dist.apply_solution()', the default when calling
    '.solve()'), updates 'dist.demand' (a DemandAndConversions object) so that its
    'crop_prod_demand'/'animal_prod_demand' -- and 'by_products', which is derived
    from them -- reflect the ACTUAL, optimised production of whichever crop_prods/
    animal_prods 'change_objective()' relaxed constraint C1 for, rather than the
    stale, pre-optimisation baseline demand.

    Why this is needed: 'change_objective()' relaxes C1 for the selected products
    from '[production] - [feed use] == [demand]' to
    '[production] - [feed use] >= [non-food demand] + [export demand]' (only the
    FOOD portion of demand is relaxed away -- see 'change_objective()'), so the
    solved production for those products can end up above (or below) their original
    FOOD demand. Nothing else in the model is told about that difference --
    'dist.demand.data_attr' still holds the OLD demand figures, and the surplus (or
    deficit) production has no explicit disposition. Any module that runs after
    solving and reads from 'dist.demand.data_attr' would otherwise silently work off
    stale figures. Chiefly this matters for 'ByProductMgmt', which reads 'by_products'
    (generated by-products) and 'by_prod_demand' to balance by-product supply and
    demand -- see its 'calculate()'.

    What this does
    --------------
    1. Re-evaluates the SAME 'C1 (relaxed)' constraint matrix built by
       'change_objective()' at the solved 'dist.x', giving the actual
       '[production] - [feed use]' achieved for each relaxed crop_prod/animal_prod
       row (i.e. exactly the crop_prods/animal_prods selected in 'change_objective()'
       -- there is no need to pass them again here, they are read back off the
       constraint itself).
    2. Adds the difference between that and the original demand total to the
       'export' column of 'crop_prod_demand'/'animal_prod_demand' (mirroring how
       'CIBUSmod.utils.helpers.induce_beef_exports()' handles a similar
       production-exceeds-demand situation elsewhere in the model): the extra (or
       missing) production is treated as extra (or reduced) export, leaving domestic
       food consumption ('food_demand', 'nutrient_supply', etc.) untouched, since
       nothing about what people actually eat has changed. If actual production ends
       up BELOW the original demand for a relaxed row (possible since C1 was relaxed
       to '>= 0', not '>= demand' -- see 'change_objective()'), this can push
       'export' negative.
    3. Recomputes 'by_products' (total generated by-products, prod_system x
       by_prod). For crop_prods/animal_prods that were NOT relaxed, this is
       unchanged from before: the blended 'by_prod_per_crop_prod'/
       'by_prod_per_animal_prod' ratio applied to their (unchanged) baseline
       total. For the relaxed rows, the SAME food/non-food+export split used by
       'change_objective()' for constraints C10/C15 (see
       '_split_byprod_supply_for_selected_products()') is applied here too: the
       food-attributable share of actual production (i.e. actual minus the
       non-food+export floor from step 1, clipped at 0 defensively -- the relaxed
       C1 constraint itself guarantees actual >= floor) is credited via
       'by_prod_per_crop_prod_food'/'by_prod_per_animal_prod_food', and the frozen
       floor itself via 'by_prod_per_crop_prod_nonfood_export'/
       'by_prod_per_animal_prod_nonfood_export'. Using the blended ratio on the
       new total instead (as a naive re-application of the old logic would) would
       misattribute the food-driven production change across whichever non-food/
       export by-product routes happened to exist at baseline -- see
       '_split_byprod_supply_for_selected_products()' docstring for a concrete
       example (Swedish wheat/ethanol). All by-products are non-negative by
       construction here: the blended-ratio part from an unchanged, non-negative
       baseline total, and the split part from a non-negative food-attributable
       share and a non-negative floor.
    4. Re-adds 'crop_prod_demand'/'animal_prod_demand' with 'allow_neg=True' on their
       metadata (preserving their other metadata as-is), since step 2 can
       legitimately make 'export' (and so their total) negative -- without this,
       storing them to a session database (see 'CIBUSmod.utils.session_db') would
       raise spurious negative-value warnings for a value that is expected here, not
       an anomaly. 'by_products' keeps its normal allow_neg=False, since step 3's
       clip guarantees it is never negative.

    Deliberately NOT recalculated:
    - 'food_demand', 'non_food_demand', 'export_demand': the food-item-level sources
      that 'crop_prod_demand'/'animal_prod_demand' are normally derived FROM. Updating
      those instead would mean reversing 'conv_factor_main' and recipe shares back to
      specific food items, and arbitrarily choosing one where several foods share a
      crop_prod -- 'crop_prod_demand'/'animal_prod_demand' are the natural level to
      apply the correction at instead, since they already live at the crop_prod/
      animal_prod level with an explicit 'export' column to absorb it.
    - 'by_prod_demand': demand FOR by-products is driven by food/non-food/export/feed
      demand for the by-product itself, independent of how much of some other
      crop_prod/animal_prod was over/under-produced.
    - 'crop_resid_demand': likewise an independent demand category.
    - Anything in CropResidueMgmt, PlantNutrientMgmt, MachineryAndEnergyMgmt, etc.:
      these read actual production directly from 'crops'/'herds' data attributes,
      which 'dist.apply_solution()' already updates correctly, rather than from
      'dist.demand', so they need no adjustment here.

    Must be called after 'dist.solve()' and 'dist.apply_solution()', and before any
    module that consumes 'dist.demand.data_attr' is calculated -- in particular
    before 'ByProductMgmt.calculate()', which aborts (with a warning) if run more
    than once per 'DemandAndConversions.calculate()' -- see its docstring.

    Parameters
    ----------
    dist : GeoDistributor or FeedDistributor object
        Must have been through 'change_objective(dist, ...)' followed by
        'dist.solve()'.

    Returns
    -------
    None. Modifies 'dist.demand.data_attr' in place ('crop_prod_demand',
    'animal_prod_demand' and 'by_products').
    '''

    if dist.x is None:
        raise ValueError("dist.x is not defined. Run dist.solve() first.")

    key = 'C1 (relaxed): A1 @ x >= non-food + export demand'
    if key not in dist.constraints:
        raise ValueError(
            f"Constraint '{key}' not found in dist.constraints. Run change_objective() "
            "and dist.solve() before calling update_demand()."
        )

    demand = dist.demand
    A1_sel = dist.constraints[key]['pars']['A1']

    def _update_data_attr(name, data, allow_neg):
        '''Like demand.data_attr.update(), but also sets 'allow_neg' on the
        attribute's metadata (preserving unit/orig/desc/scalable as-is). Used with
        allow_neg=True for 'crop_prod_demand'/'animal_prod_demand', since
        update_demand() can legitimately produce a negative 'export' entry there (see
        step 2 in the docstring above) if actual production ends up below the
        original demand for a relaxed row; without allow_neg=True, storing the
        resulting data to a session database would raise spurious negative-value
        warnings (see CIBUSmod.utils.session_db) for a value that is expected here,
        not an anomaly. 'by_products' does not need this -- see '_by_products_from()'
        -- so is left with its normal allow_neg=False.
        '''
        meta = demand.data_attr.metadata[name]
        demand.data_attr.add(
            data,
            name=name,
            unit=meta['unit'],
            orig=meta['orig'],
            desc=meta['desc'],
            scalable=meta['scalable'],
            allow_neg=allow_neg,
        )

    # Evaluate [production] - [feed use] for the relaxed rows at the solved x,
    # reusing the exact matrix change_objective() built (so this automatically stays
    # correct for whatever GeoDistributor/FeedDistributor-specific structure went
    # into C1)
    x_short = np.concatenate([
        dist.x[k].reindex(dist.x_idx_short[k], fill_value=0).to_numpy()
        for k in dist.x_idx_short.keys()
    ])
    actual = A1_sel.M @ x_short

    n_ani_sel = len(A1_sel.rows['ani'])
    actual_ani = pd.Series(actual[:n_ani_sel], index=A1_sel.rows['ani'])
    actual_crp = pd.Series(actual[n_ani_sel:], index=A1_sel.rows['crp'])

    # Non-food+export "floor" for the relaxed rows, i.e. the same floor
    # _relax_C1_for_selected_products() used to relax C1 -- captured from the
    # ORIGINAL (pre-update) demand, before _apply_surplus_to_export() below can
    # touch 'export', and used to split by-product generation further down
    ani_floor = (
        demand.data_attr.get('animal_prod_demand')
        .reindex(actual_ani.index, fill_value=0)[['non-food', 'export']].sum(axis=1)
    )
    crp_floor = (
        demand.data_attr.get('crop_prod_demand')
        .reindex(actual_crp.index, fill_value=0)[['non-food', 'export']].sum(axis=1)
    )
    food_attrib_ani = (actual_ani - ani_floor).clip(lower=0)
    food_attrib_crp = (actual_crp - crp_floor).clip(lower=0)

    def _apply_surplus_to_export(attr_name, actual_by_row):
        if len(actual_by_row) == 0:
            return
        attr = demand.data_attr.get(attr_name).copy()
        # Extend with any relaxed rows not already present (e.g. a crop_prod that
        # previously had no food/non-food/export demand at all, only feed use)
        attr = attr.reindex(attr.index.union(actual_by_row.index), fill_value=0)
        original_total = attr.loc[actual_by_row.index].sum(axis=1)
        surplus = actual_by_row - original_total
        attr.loc[surplus.index, 'export'] += surplus
        _update_data_attr(attr_name, attr, allow_neg=True)

    _apply_surplus_to_export('crop_prod_demand', actual_crp)
    _apply_surplus_to_export('animal_prod_demand', actual_ani)

    # Recompute 'by_products': blended ratio on the (unchanged) baseline total for
    # rows that weren't relaxed, food/non-food+export split (mirroring
    # _split_byprod_supply_for_selected_products(), which applies the same split to
    # C10/C15 during the solve) for the relaxed rows
    def _apply_ratio(ratio, qty_by_row, levels):
        if len(ratio) == 0 or len(qty_by_row) == 0:
            return _empty_byprod_series()
        merged = (
            ratio.rename('ratio').reset_index()
            .merge(qty_by_row.rename('qty').reset_index(), on=levels, how='inner')
        )
        merged['value'] = merged['ratio'] * merged['qty']
        return merged.groupby(['prod_system', 'by_prod'])['value'].sum()

    def _by_products_from(ratio_attr, demand_attr, levels, actual_by_row, floor_by_row, food_attrib_by_row):
        # Rows that were NOT relaxed: ordinary blended ratio on their (unchanged,
        # non-negative) baseline total
        total = demand.data_attr.get(demand_attr).sum(axis=1).clip(lower=0)
        non_relaxed_total = total.drop(index=actual_by_row.index, errors='ignore')
        result = _apply_ratio(demand.data_attr.get(ratio_attr), non_relaxed_total, levels)

        # Relaxed rows: food-attributable share via the food-only ratio, frozen
        # floor via the non-food+export-only ratio -- both non-negative by
        # construction (see docstring)
        result = result.add(
            _apply_ratio(demand.data_attr.get(ratio_attr + '_food'), food_attrib_by_row, levels),
            fill_value=0,
        )
        result = result.add(
            _apply_ratio(demand.data_attr.get(ratio_attr + '_nonfood_export'), floor_by_row, levels),
            fill_value=0,
        )
        return result

    new_by_products = (
        _by_products_from(
            'by_prod_per_crop_prod', 'crop_prod_demand', ['prod_system', 'crop_prod'],
            actual_crp, crp_floor, food_attrib_crp,
        )
        .add(
            _by_products_from(
                'by_prod_per_animal_prod', 'animal_prod_demand',
                ['prod_system', 'species', 'animal_prod'],
                actual_ani, ani_floor, food_attrib_ani,
            ),
            fill_value=0,
        )
    )
    _update_data_attr('by_products', new_by_products, allow_neg=False)

    return None

def _relax_C1_for_selected_products(dist, crop_prods, animal_prods):
    '''Splits constraint C1 ('C1: A1 @ x == b1', i.e.
    '[production] - [feed use] == [demand]', where [demand] = food + non-food +
    export demand) into two constraints:
    - For rows matching a selected crop_prod (in 'crop_prods') or (species,
      animal_prod) (in 'animal_prods'): relaxed to an inequality
      '[production] - [feed use] >= [non-food demand] + [export demand]', i.e. only
      the FOOD portion of demand is relaxed -- production must still (at minimum)
      cover feed use plus the original non-food and export demand, but is otherwise
      free to move -- up or down -- away from the original FOOD demand figure.
    - For all other rows: kept exactly as before, i.e. '== [demand]' (food + non-food
      + export).

    This is done by row-slicing the existing 'A1'/'b1' matrices built by
    'dist.make()' rather than rebuilding them from scratch, so it automatically picks
    up whatever GeoDistributor/FeedDistributor-specific logic went into constructing
    C1 in the first place (e.g. FeedDistributor's extra x_fds columns). The
    non-food/export floor for the relaxed rows is read directly from
    'dist.demand.data_attr' ('crop_prod_demand'/'animal_prod_demand', BEFORE
    'update_demand()' has a chance to touch them), not from 'b1' itself, since 'b1'
    only holds the food+non-food+export TOTAL with no way to recover the split.

    Modifies 'dist.constraints' in place: the original 'C1: A1 @ x == b1' entry is
    replaced by up to two new entries, 'C1 (fixed): ...' and 'C1 (relaxed): ...'.

    Parameters
    ----------
    dist : GeoDistributor or FeedDistributor object
    crop_prods : list of str
        crop_prod(s) to relax
    animal_prods : list of (species, animal_prod) tuples
        Animal product(s) to relax

    Returns
    -------
    None
    '''

    key = 'C1: A1 @ x == b1'
    if key not in dist.constraints:
        raise ValueError(
            f"Constraint '{key}' not found in dist.constraints. Make sure "
            "constraint 1 was included when calling dist.make(use_cons=...)."
        )
    c1 = dist.constraints.pop(key)
    A1 = c1['pars']['A1']
    b1 = c1['pars']['b1']

    # Row index of A1 is a dict {'ani': (prod_system, species, animal_prod),
    # 'crp': (prod_system, crop_prod)} with rows stacked in that order (matching b1)
    ani_idx = A1.rows['ani']
    crp_idx = A1.rows['crp']

    sel_ani = np.array([(sp, ap) in animal_prods for _, sp, ap in ani_idx])
    sel_crp = np.array([cp in crop_prods for _, cp in crp_idx])
    sel = np.concatenate([sel_ani, sel_crp])

    # Warn about any requested crop_prods/animal_prods that don't correspond to a
    # row of C1 at all (checked against the FULL row index, not just the selected
    # rows, so this also catches e.g. a typo'd crop_prod name or an animal_prod with
    # no food item mapped to it via 'conv_factor_main' -- in which case it is
    # neither relaxed nor rewarded in the objective, silently, unless warned about
    # here). This is separate from (and a superset of) the "nothing matched at all"
    # case.
    available_crp = set(crp_idx.get_level_values('crop_prod'))
    available_ani = set(zip(ani_idx.get_level_values('species'), ani_idx.get_level_values('animal_prod')))
    unmatched_crp = [cp for cp in crop_prods if cp not in available_crp]
    unmatched_ani = [ap for ap in animal_prods if ap not in available_ani]
    if unmatched_crp or unmatched_ani:
        warnings.warn(
            "Some of the given crop_prods/animal_prods do not correspond to any row "
            "of constraint C1 (i.e. no food item maps to them via 'conv_factor_main' "
            "in DemandAndConversions, so they have no demand and hence no C1 row to "
            "relax) and will be silently ignored: neither relaxed nor rewarded in "
            "the objective. If the product's animals/crops are still grown (e.g. as "
            "a joint product of a herd whose main product WAS selected), that will "
            "show up in CropProduction/AnimalHerd's own production data, but "
            "DemandAndConversions has nothing to reconcile it against.\n"
            f"Unmatched crop_prods: {unmatched_crp}\n"
            f"Unmatched animal_prods: {unmatched_ani}"
        )

    # non-food + export demand for the relaxed rows only -- this is the floor that
    # replaces the full (food + non-food + export) demand figure for those rows
    ani_floor = (
        dist.demand.data_attr.get('animal_prod_demand')[['non-food', 'export']].sum(axis=1)
        .reindex(ani_idx[sel_ani], fill_value=0)
    )
    crp_floor = (
        dist.demand.data_attr.get('crop_prod_demand')[['non-food', 'export']].sum(axis=1)
        .reindex(crp_idx[sel_crp], fill_value=0)
    )
    b1_floor = np.concatenate([ani_floor.to_numpy(), crp_floor.to_numpy()])

    M_csr = A1.M.tocsr()

    def _sub_matrix(row_mask, ani_mask, crp_mask):
        return IndexedMatrix(
            M_csr[row_mask, :].tocsc(),
            row_idx={'ani': ani_idx[ani_mask], 'crp': crp_idx[crp_mask]},
            col_idx=A1.cols,
        )

    new_constraints = {}

    A1_rest = _sub_matrix(~sel, ~sel_ani, ~sel_crp)
    if A1_rest.shape[0] > 0:
        new_constraints['C1 (fixed): A1 @ x == b1'] = {
            'left': lambda x, A1, b1: A1.M @ x,
            'right': lambda A1, b1: b1,
            'rel': '==',
            'pars': {'A1': A1_rest, 'b1': b1[~sel]},
        }

    A1_sel = _sub_matrix(sel, sel_ani, sel_crp)
    if A1_sel.shape[0] > 0:
        new_constraints['C1 (relaxed): A1 @ x >= non-food + export demand'] = {
            'left': lambda x, A1, b1: A1.M @ x,
            'right': lambda A1, b1: b1,
            'rel': '>=',
            # Relaxed rows are no longer tied to the original FOOD demand figure, but
            # must still (at minimum) cover feed use plus non-food and export demand
            'pars': {'A1': A1_sel, 'b1': b1_floor},
        }

    dist.constraints.update(new_constraints)

    return None

def _nonfood_export_floor(dist, crop_prods, animal_prods):
    '''Returns (ani_floor, crp_floor): baseline non-food+export demand for the
    given animal_prods/crop_prods -- the same "floor" production level
    '_relax_C1_for_selected_products()' uses for the relaxed C1 rows. Recomputed
    here directly from 'dist.demand.data_attr' (rather than threaded through from
    that function) since nothing runs between '_relax_C1_for_selected_products()'
    and '_split_byprod_supply_for_selected_products()' that would change it, so
    both always agree on what counts as "floor" for a given selected product
    without needing to share state.

    Parameters
    ----------
    dist : GeoDistributor or FeedDistributor object
    crop_prods : list of str
    animal_prods : list of (species, animal_prod) tuples

    Returns
    -------
    (ani_floor, crp_floor) : tuple of pandas.Series
        Indexed (prod_system, species, animal_prod) and (prod_system, crop_prod)
        respectively, restricted to rows matching 'animal_prods'/'crop_prods'.
    '''

    animal_prod_demand = dist.demand.data_attr.get('animal_prod_demand')
    crop_prod_demand = dist.demand.data_attr.get('crop_prod_demand')

    if animal_prods:
        ani_sel = animal_prod_demand.index.droplevel('prod_system').isin(animal_prods)
    else:
        ani_sel = np.zeros(len(animal_prod_demand), dtype=bool)
    if crop_prods:
        crp_sel = crop_prod_demand.index.get_level_values('crop_prod').isin(crop_prods)
    else:
        crp_sel = np.zeros(len(crop_prod_demand), dtype=bool)

    ani_floor = animal_prod_demand.loc[ani_sel, ['non-food', 'export']].sum(axis=1)
    crp_floor = crop_prod_demand.loc[crp_sel, ['non-food', 'export']].sum(axis=1)

    return ani_floor, crp_floor

def _split_byprod_supply_for_selected_products(dist, crop_prods, animal_prods):
    '''Rebuilds constraints 'C10: A10 @ x >= b10' and 'C15: A15 @ x >= 0' (if either
    is present in 'dist.constraints' -- both FeedDistributor-only), splitting the
    by-product supply credited to a selected crop_prod/animal_prod into a FOOD-route
    part (that moves with the now-relaxed production) and a fixed non-food+export-
    route part (frozen at the baseline non-food+export "floor" -- see
    '_relax_C1_for_selected_products()').

    Why this is needed: C10/C15 are built by 'FeedDistributor.make_A10_ani()'/
    'make_A10_crp()', which multiply 'DemandAndConversions.by_prod_per_animal_prod'/
    'by_prod_per_crop_prod' -- a single ratio BLENDED across a product's food/
    non-food/export demand routes -- directly against x_ani/x_crp. This is exact
    under the default objective, where C1 pins x_ani/x_crp to that same blended
    demand total, so "blended ratio times total production" and "blended ratio
    times total demand" are identical. Once 'change_objective()' relaxes C1 for a
    selected product, that identity breaks: production can now exceed the blended
    baseline total, but the unmodified constraint keeps applying the SAME blended
    mix to the excess, crediting by-products from routes (e.g. an ethanol/non-food
    processing route) that the extra, food-attributed production was never destined
    for. Since C10/C15 gate how much of that by-product is available as feed, this
    overstated supply can loosen the by-product balance enough to change which
    animal products the optimiser favours -- not merely a reporting artefact (see
    'update_demand()', which has the analogous problem for the 'by_products' data
    attribute reported AFTER solving, and is fixed the same way).

    How the split works: for a selected crop_prod/animal_prod with baseline
    non-food+export demand ("floor") f,
    - the excess over f (whatever the now-relaxed C1 lets production move by) is
      credited using 'by_prod_per_crop_prod_food'/'by_prod_per_animal_prod_food'
      (the food-route-only ratio), so growing this product for the objective only
      credits by-products the food processing route actually generates,
    - the frozen floor f itself is credited using
      'by_prod_per_crop_prod_nonfood_export'/'by_prod_per_animal_prod_nonfood_export'
      (the non-food+export-route-only ratio); since f never moves this is a known
      constant, so rather than a variable term on the supply (left-hand) side it is
      folded in as a SUBTRACTION from the right-hand side (moving a constant across
      '>=' flips its sign: '[supply] + const >= [demand]' rearranges to
      '[supply] >= [demand] - const').
    Non-selected products are untouched: their C1 equality still pins x_ani/x_crp to
    their blended baseline total, so the original blended ratio remains exact for
    them.

    Implementation: temporarily substitutes a modified 'by_prod_per_crop_prod'/
    'by_prod_per_animal_prod' (selected items' entries replaced by the food-only
    ratio) into 'dist.demand.data_attr', calls
    'dist.make_A10_ani()'/'make_A10_crp()'/'make_A10_fds_by_prod_corr()' (and, for
    C15, 'make_A15_fds()'/'make_A15_fds_gen()') again so the rebuilt matrices pick
    up the substitution automatically, then restores the original attributes. This
    reuses FeedDistributor's own matrix-building methods (and so stays correct for
    whatever FeedDistributor-specific structure feeds into C10/C15) rather than
    reimplementing their merge logic here.

    Important limitation for C15 (regional): the floor f is only defined
    NATIONALLY in DemandAndConversions (crop_prod_demand/animal_prod_demand have no
    regional breakdown), so there is no principled way to know which region(s) a
    selected product's frozen non-food+export production is located in. Rather than
    inventing a regional allocation assumption the rest of the model doesn't make
    (e.g. pro-rata by baseline regional production share), C15 only gets the
    food-route-ratio substitution for the excess part -- the fixed floor
    contribution is NOT subtracted from its right-hand side. This means C15 may
    slightly UNDER-state regional by-product supply for a selected product's frozen
    floor portion, which can at most make the model lean slightly MORE on imported
    feed for that by-product regionally -- the safe direction, unlike the original
    (unfixed) behaviour, which could overstate supply. C10 (national) has no such
    ambiguity and gets the fix including the floor contribution. "Slightly" is not
    guaranteed, though: if the missing floor credit concerns a by-product that
    ALSO needs a genuinely REGIONAL share of feed demand met (i.e. has its own C15
    row, meaning 'share_domestic>0' and 'share_regional>0' for it somewhere in
    FeedMgmt), C15's regional-sourcing guarantee cannot be relied on for that
    by-product at all -- see '_warn_c15_regional_gap()', which checks for and
    warns about exactly this case.

    Parameters
    ----------
    dist : GeoDistributor or FeedDistributor object
        Should be called after '_relax_C1_for_selected_products()' (i.e. from
        within 'change_objective()'). No-ops if neither 'C10: A10 @ x >= b10' nor
        'C15: A15 @ x >= 0' is present in 'dist.constraints' (e.g. a GeoDistributor,
        or a FeedDistributor whose 'use_cons' didn't include 10/15).
    crop_prods : list of str
    animal_prods : list of (species, animal_prod) tuples

    Returns
    -------
    None. Modifies 'dist.constraints' in place.
    '''

    if _C10_KEY not in dist.constraints and _C15_KEY not in dist.constraints:
        return None

    demand = dist.demand

    ani_floor, crp_floor = _nonfood_export_floor(dist, crop_prods, animal_prods)

    def _food_only_for_selected(ratio_name, selected, key_is_crop):
        '''Copy of demand.data_attr.get(ratio_name) with entries for 'selected'
        crop_prods/(species, animal_prod)s replaced by the corresponding
        '<ratio_name>_food' entry (0 if absent, i.e. the food route generates none
        of that by-product); everything else is left unchanged.'''
        blended = demand.data_attr.get(ratio_name)
        if len(blended) == 0 or not selected:
            return blended
        if key_is_crop:
            sel_mask = blended.index.get_level_values('crop_prod').isin(selected)
        else:
            sel_mask = blended.index.droplevel('prod_system').isin(selected)
        if not sel_mask.any():
            return blended
        food_ratio = demand.data_attr.get(ratio_name + '_food')
        modified = blended.copy()
        sel_idx = blended.index[sel_mask]
        modified.loc[sel_idx] = food_ratio.reindex(sel_idx, fill_value=0)
        return modified

    def _floor_contribution(ratio_name, floor, levels):
        '''Fixed (prod_system, by_prod) supply contribution from 'floor'
        production, using the non-food+export-only ratio.'''
        if len(floor) == 0:
            return _empty_byprod_series()
        ratio = demand.data_attr.get(ratio_name + '_nonfood_export')
        if len(ratio) == 0:
            return _empty_byprod_series()
        merged = (
            ratio.rename('ratio').reset_index()
            .merge(floor.rename('floor').reset_index(), on=levels, how='inner')
        )
        merged['value'] = merged['ratio'] * merged['floor']
        return merged.groupby(['prod_system', 'by_prod'])['value'].sum()

    floor_supply_national = (
        _floor_contribution('by_prod_per_crop_prod', crp_floor, ['prod_system', 'crop_prod'])
        .add(
            _floor_contribution(
                'by_prod_per_animal_prod', ani_floor,
                ['prod_system', 'species', 'animal_prod'],
            ),
            fill_value=0,
        )
    )

    if _C15_KEY in dist.constraints:
        _warn_c15_regional_gap(dist, crp_floor, ani_floor)

    modified_crp = _food_only_for_selected('by_prod_per_crop_prod', crop_prods, key_is_crop=True)
    modified_ani = _food_only_for_selected('by_prod_per_animal_prod', animal_prods, key_is_crop=False)

    orig_crp = demand.data_attr.get('by_prod_per_crop_prod')
    orig_ani = demand.data_attr.get('by_prod_per_animal_prod')
    demand.data_attr.update('by_prod_per_crop_prod', modified_crp)
    demand.data_attr.update('by_prod_per_animal_prod', modified_ani)
    try:
        if _C10_KEY in dist.constraints:
            _rebuild_C10(dist, floor_supply_national)
        if _C15_KEY in dist.constraints:
            _rebuild_C15(dist)
    finally:
        demand.data_attr.update('by_prod_per_crop_prod', orig_crp)
        demand.data_attr.update('by_prod_per_animal_prod', orig_ani)

    return None

def _warn_c15_regional_gap(dist, crp_floor, ani_floor):
    '''Warns if C15 ('C15: A15 @ x >= 0') requires a REGIONAL share of feed demand
    (i.e. has a row for it, meaning some feed use of it has 'share_domestic>0' AND
    'share_regional>0' in FeedMgmt -- see 'FeedDistributor.make_C15()') for a
    by-product that a selected crop_prod's/animal_prod's frozen non-food+export
    floor also generates.

    Why this matters: as explained in
    '_split_byprod_supply_for_selected_products()' (see its "Important limitation
    for C15" paragraph), the floor's by-product supply is NOT credited anywhere in
    C15 -- there is no regional breakdown of the (national-only) floor to place it
    in. For a by-product nobody needs sourced regionally (no C15 row at all, e.g.
    'share_regional' is 0 everywhere it's used as feed), this is harmless: C15
    simply doesn't constrain it. But for a by-product that DOES need a regional
    share, C15 can no longer verify that requirement is met the way it was
    designed to: the model may end up relying on regionally imported feed more
    than physically necessary, or pushing the selected product's own production
    higher than the nutrient objective alone would call for, purely to plug this
    accounting gap -- not a real physical shortfall. In other words, C15's
    regional-sourcing guarantee cannot be relied upon for these specific
    by-products once the crop_prod/animal_prod generating them (via its non-food/
    export route) is selected in 'change_objective()'.

    Parameters
    ----------
    dist : FeedDistributor object
        Must have 'C15: A15 @ x >= 0' in 'dist.constraints' (checked by the
        caller, '_split_byprod_supply_for_selected_products()', before calling
        this).
    crp_floor, ani_floor : pandas.Series
        As returned by '_nonfood_export_floor()'.

    Returns
    -------
    None. Issues a single combined 'warnings.warn()' if any affected (crop_prod/
    animal_prod, by_prod) combination is found; otherwise does nothing.
    '''

    demand = dist.demand
    row_idx = dist.constraints[_C15_KEY]['pars']['A15'].rows
    regional_byprods = set(row_idx.get_level_values('by_prod'))
    if not regional_byprods:
        return None

    def _affected(ratio_name, floor, levels):
        floor_nonzero = floor[floor > 0]
        if len(floor_nonzero) == 0:
            return pd.DataFrame()
        ratio = demand.data_attr.get(ratio_name + '_nonfood_export')
        if len(ratio) == 0:
            return pd.DataFrame()
        merged = (
            ratio.rename('ratio').reset_index()
            .merge(floor_nonzero.rename('floor').reset_index(), on=levels, how='inner')
        )
        return merged.loc[
            (merged['ratio'] != 0) & merged['by_prod'].isin(regional_byprods),
            levels + ['by_prod'],
        ]

    affected_crp = _affected('by_prod_per_crop_prod', crp_floor, ['prod_system', 'crop_prod'])
    affected_ani = _affected(
        'by_prod_per_animal_prod', ani_floor, ['prod_system', 'species', 'animal_prod'],
    )

    if len(affected_crp) == 0 and len(affected_ani) == 0:
        return None

    lines = []
    if len(affected_crp) > 0:
        lines.append(
            "crop_prod -> by_prod: " + ", ".join(
                f"{r.crop_prod!r} -> {r.by_prod!r}" for r in affected_crp.itertuples()
            )
        )
    if len(affected_ani) > 0:
        lines.append(
            "(species, animal_prod) -> by_prod: " + ", ".join(
                f"({r.species!r}, {r.animal_prod!r}) -> {r.by_prod!r}"
                for r in affected_ani.itertuples()
            )
        )

    warnings.warn(
        "change_objective(): constraint C15 requires a REGIONAL share of feed "
        "demand (share_domestic>0 and share_regional>0 in FeedMgmt) for at least "
        "one by-product that a selected crop_prod's/animal_prod's frozen non-food+"
        "export floor also generates. That floor's by-product supply is not "
        "credited anywhere in C15 (see '_split_byprod_supply_for_selected_products()' "
        "docstring), so C15's regional-sourcing guarantee cannot be relied on for "
        "these by-products: the model may lean on regionally imported feed more "
        "than physically necessary, or push the selected product's production "
        "higher than the objective alone would call for, to cover this accounting "
        "gap rather than a real shortfall.\n" + "\n".join(lines)
    )

    return None

def _prune_to_x_idx_short(dist, mat):
    '''Slices 'mat's columns down to 'dist.x_idx_short', mirroring
    'FeedDistributor.make_C7()''s own column-dropping exactly (same 'isel'
    construction). This is needed because C7 -- which drops crop-region
    combinations below 'min_GDD5' (and so indirectly the animals/feeds that
    depend on them) from every constraint matrix built so far -- runs ONCE, near
    the end of 'dist.make()', over whatever matrices exist in 'dist.constraints'
    AT THAT TIME (see its docstring: "This must be run after all other
    constraints have been defined!"). '_rebuild_C10()'/'_rebuild_C15()' build
    fresh A10/A15 matrices AFTER 'dist.make()' has already finished (and so after
    C7 already ran once), using 'dist.x_idx' (the FULL, un-dropped index, since
    that is all 'dist.make_A10_ani()' etc. know how to build against) -- so
    without reapplying C7's slice here, these rebuilt matrices come out WIDER
    than the 'x' variable '_set_maximise_objective()' builds (which is sized to
    'dist.x_idx_short'), causing a cvxpy dimension-mismatch error the moment the
    constraint is used.

    No-ops if 'mat' is already the short width (e.g. 'crp' happened to have
    nothing dropped for this dataset).

    Parameters
    ----------
    dist : FeedDistributor object
    mat : IndexedMatrix
        Must have a column dict with 'ani'/'crp'/'fds' keys stacked in that
        order (as built by '_rebuild_C10()'/'_rebuild_C15()').

    Returns
    -------
    IndexedMatrix
        'mat', modified in place (also returned for convenience).
    '''

    ani_idx = dist.x_idx['ani']
    crp_idx = dist.x_idx['crp']
    fds_idx = dist.x_idx['fds']
    sel_crp = dist.x_idx_short['crp']

    n_ani = len(ani_idx)
    n_crp = len(crp_idx)
    isel = (
        list(range(0, n_ani))
        + [crp_idx.get_loc(s) + n_ani for s in sel_crp]
        + list(range(n_ani + n_crp, n_ani + n_crp + len(fds_idx)))
    )

    if mat.M.shape[1] <= len(isel):
        return mat

    mat.M = mat.M[:, isel]
    mat.cols['ani'] = ani_idx.copy()
    mat.cols['crp'] = sel_crp.copy()
    mat.cols['fds'] = fds_idx.copy()
    return mat

def _rebuild_C10(dist, floor_supply):
    '''Rebuilds 'C10: A10 @ x >= b10' using whatever 'by_prod_per_crop_prod'/
    'by_prod_per_animal_prod' is CURRENTLY set on 'dist.demand.data_attr' (the
    caller, '_split_byprod_supply_for_selected_products()', temporarily substitutes
    a modified version before calling this), and subtracts 'floor_supply' (the fixed
    non-food+export contribution for selected products) from the right-hand side --
    i.e. how much of 'b10' is left for x to cover once the frozen floor's own
    by-product generation is credited. Mirrors 'FeedDistributor.make_C10()' exactly,
    just re-run after the
    substitution -- see '_split_byprod_supply_for_selected_products()' for the full
    rationale.

    Parameters
    ----------
    dist : FeedDistributor object
    floor_supply : pandas.Series
        Indexed (prod_system, by_prod), as returned by '_floor_contribution()'
        inside '_split_byprod_supply_for_selected_products()'.

    Returns
    -------
    None. Replaces 'dist.constraints["C10: A10 @ x >= b10"]' in place.
    '''

    c10 = dist.constraints[_C10_KEY]
    row_idx = c10['pars']['A10'].rows
    b10 = c10['pars']['b10']

    A10_ani = dist.make_A10_ani(row_idx)
    A10_crp = dist.make_A10_crp(row_idx)
    A10_fds_demand = dist.make_A1_3(row_idx=row_idx, prod_type='by_prod')
    A10_fds_gen = dist.make_A1_3_gen(row_idx=row_idx)
    A10_fds_crop_corr = dist.make_A10_fds_by_prod_corr(row_idx)
    A10_fds_net = IndexedMatrix(
        A10_fds_demand.M + A10_fds_gen.M + A10_fds_crop_corr.M,
        row_idx,
        {'fds': dist.x_idx['fds']},
    )
    A10 = IndexedMatrix(
        scipy.sparse.hstack([A10_ani.M, A10_crp.M, A10_fds_net.M], format='csc'),
        row_idx,
        {
            'ani': dist.x_idx['ani'],
            'crp': dist.x_idx['crp'],
            'fds': dist.x_idx['fds'],
        },
    )
    A10 = _prune_to_x_idx_short(dist, A10)

    # C10 reads '[supply] >= [demand]'. The frozen floor's by-product generation is
    # real SUPPLY that the substituted (food-only) A10_ani/A10_crp no longer credit
    # -- adding it back in is equivalent to reducing how much of 'b10' still needs
    # to come from x, i.e. SUBTRACTING it from the right-hand side (moving a known
    # constant from the left of '>=' to the right flips its sign), not adding to it.
    b10_new = b10 - floor_supply.reindex(row_idx, fill_value=0).to_numpy()

    dist.constraints[_C10_KEY] = {
        'left': lambda x, A10, b10: A10.M @ x,
        'right': lambda A10, b10: b10,
        'rel': '>=',
        'pars': {'A10': A10, 'b10': b10_new},
    }

    return None

def _rebuild_C15(dist):
    '''Rebuilds 'C15: A15 @ x >= 0' using whatever 'by_prod_per_crop_prod'/
    'by_prod_per_animal_prod' is CURRENTLY set on 'dist.demand.data_attr' (see
    '_rebuild_C10()'/'_split_byprod_supply_for_selected_products()' for why).
    Mirrors 'FeedDistributor.make_C15()' exactly, just re-run after the
    substitution. Unlike '_rebuild_C10()', the right-hand side is left unchanged
    (still 0) -- see '_split_byprod_supply_for_selected_products()' docstring for
    why the fixed floor contribution cannot be added back in regionally.

    Parameters
    ----------
    dist : FeedDistributor object

    Returns
    -------
    None. Replaces 'dist.constraints["C15: A15 @ x >= 0"]' in place.
    '''

    c15 = dist.constraints[_C15_KEY]
    row_idx = c15['pars']['A15'].rows

    A15_ani = dist.make_A10_ani(row_idx)
    A15_crp = dist.make_A10_crp(row_idx)
    A15_fds = dist.make_A15_fds(row_idx)
    A15_fds_gen = dist.make_A15_fds_gen(row_idx)
    A15_fds_crop_corr = dist.make_A10_fds_by_prod_corr(row_idx)

    A15 = IndexedMatrix(
        scipy.sparse.hstack(
            [A15_ani.M, A15_crp.M, A15_fds.M + A15_fds_gen.M + A15_fds_crop_corr.M]
        ),
        row_idx=row_idx,
        col_idx={
            'ani': dist.x_idx['ani'],
            'crp': dist.x_idx['crp'],
            'fds': dist.x_idx['fds'],
        },
    )
    A15 = _prune_to_x_idx_short(dist, A15)

    dist.constraints[_C15_KEY] = {
        'left': lambda x, A15: A15.M @ x,
        'right': lambda A15: 0,
        'rel': '>=',
        'pars': {'A15': A15},
    }

    return None

def _set_maximise_objective(dist, mapper, lambda_reg=1e-6, lambda_ridge=None, keep_linear=False):
    '''Replaces 'dist.problem' with a cvxpy.Problem that maximises
    sum(mapper['crp'] * x['crp']) + sum(mapper['ani'] * x['ani'])
    + sum(mapper['fds'] * x['fds']) (implemented as minimising its negative, since
    'mapper['fds']' is already negative-signed -- see 'nutrient_supply_from_x()'),
    subject to the constraints currently in 'dist.constraints'.

    A small quadratic (or, if 'keep_linear', linear) regularisation term is added on
    the DEVIATION FROM 'dist.x0' of the x-elements that are not part of the objective
    (i.e. have a 0 coefficient in the mapper), purely to keep the LP numerically well
    posed. Crucially this regularises towards 'dist.x0' rather than towards 0 (unlike
    'make_opt_goal()' in 'notebooks/no_test/help_funs.py', which this otherwise
    mirrors): although non-selected crop_prod/animal_prod remain pinned to their
    baseline TOTAL via the (fixed, equality) part of C1, that total is often not
    unique -- e.g. a crop_prod can be produced by more than one crop, in more than one
    region, and a fixed animal_prod total still leaves other animal_prod from the same
    herd (and hence its x_ani) otherwise free. Regularising raw x toward 0 would then
    bias the solution toward whichever mix produces that fixed total with the least
    x, silently reshuffling non-selected crops/animals/regions (e.g. shrinking
    non-feed crops, greenhouse area, or even the selected animals' own herds, if that
    happens to reduce OTHER unrelated x more than it reduces the reward) even though
    nothing was actually asked to change for them. Regularising toward 'dist.x0'
    keeps everything not explicitly selected as close as possible to the baseline
    solution instead, matching the default GeoDistributor/FeedDistributor objective's
    own philosophy (minimise deviation from x0) for anything the caller didn't ask to
    optimise. x_fds ('dist.x0["fds"]' is 0 by construction, see
    FeedDistributor.get_x0()) is excluded from this regularisation entirely rather
    than regularised towards that 0 -- see below.

    The deviation is additionally weighted by 'dist.scale_f' (the same per-variable
    scaling factors the default objective uses, see
    'GeoDistributor.calculate_scaling_factors()'), because x_ani/x_crp mix wildly
    different absolute scales (e.g. animal head counts vs. crop hectares, or a small
    specialised crop vs. a large-area feed crop like grazing land). Without this, an
    unweighted sum_squares(x - x0) is dominated by whichever variables happen to have
    the largest absolute x0 (typically large-area feed crops), so the solver would
    rather shrink the OBJECTIVE's own animals -- to avoid needing more of that
    large-area feed -- than grow them, even though growing them was the entire point.

    x_fds (FeedDistributor only) is excluded from regularisation altogether, whether
    or not it has a nonzero mapper coefficient (i.e. even when only 'animal_prods',
    not 'crop_prods', is selected). If it were regularised towards its x0 of 0 like
    everything else, growing a selected animal_prod -- which mechanically requires
    growing its herd's feed intake too, via constraints C11-C13 tying x_fds to x_ani
    -- would be fighting a penalty on every one of those feed variables for moving
    away from 0, directly opposing the very growth the objective is trying to
    achieve. x_fds does not need this regularisation for numerical stability either:
    C11-C13 already tie it tightly (as ration shares) to x_ani, which is itself
    either regularised towards x0 (if not selected) or driven directly by the
    objective (if selected), so it cannot float independently regardless.

    A second, much smaller ridge term of the same form is ALSO applied to every
    x_ani/x_crp regardless of selection (i.e. 'not_in_obj' is replaced by an
    all-ones mask, still excluding x_fds for the same reason as above). This is
    needed for numerical reasons distinct from the main regularisation above: the
    selected products get a purely LINEAR reward in 'obj_fun' with zero quadratic
    curvature, while everything else has the (quadratic) main regularisation term.
    Mixing zero-curvature and positive-curvature blocks in one QP, combined with
    this model's constraint matrices already spanning an extreme coefficient range
    (independent of anything this function does -- the same range shows up solving
    the unmodified default objective, which tolerates it fine because it is purely
    quadratic everywhere), can make cvxpy/GUROBI's barrier method fail outright with
    a generic "SolverError"/"No solution found" rather than reporting infeasibility
    -- observed in practice once enough crop_prods/animal_prods are selected at once
    for the zero-curvature block to become large. Giving every variable a little
    curvature restores numerical stability without perceptibly changing the
    solution, since 'lambda_ridge' defaults to a small fraction of 'lambda_reg'.

    Parameters
    ----------
    dist : GeoDistributor or FeedDistributor object
    mapper : dict of pandas.Series
        Per-unit nutrient mapper as returned by 'nutrient_supply_from_x()'
    lambda_reg : float, default 1e-6
    lambda_ridge : float, default None
        If None, uses 'lambda_reg * 1e-3'.
    keep_linear : bool, default False

    Returns
    -------
    None
    '''

    if lambda_ridge is None:
        lambda_ridge = lambda_reg * 1e-3

    # Variable order must match dist.x_idx_short's key order ('ani', 'crp', [,'fds']),
    # since GeoDistributor.solve() splits the solved x back up in that same order
    keys = list(dist.x_idx_short.keys())
    sizes = {k: len(dist.x_idx_short[k]) for k in keys}
    n = sum(sizes.values())
    x = cvxpy.Variable(n, nonneg=True)

    # Coefficients: negative of the nutrient mapper, so that MINIMISING this sum is
    # equivalent to MAXIMISING nutrient supply. Reindexed to dist.x_idx_short in case
    # constraint C7 (dropping crops/animals that can't be grown/kept in a region) was
    # used, which shrinks the variable set relative to dist.x_idx.
    full_map = np.concatenate([
        (
            -mapper[k].reindex(dist.x_idx_short[k], fill_value=0).to_numpy()
            if k in mapper else np.zeros(sizes[k])
        )
        for k in keys
    ])

    # Nutrient-supply coefficients are typically very large (e.g. kcal per hectare
    # can be in the millions), which on its own leads to severe numerical
    # ill-conditioning in the solver (huge objective values relative to the O(1)
    # regularisation and constraint right-hand sides). Since minimising c^T x and
    # minimising (c/k)^T x for any positive constant k has the same optimal x,
    # rescaling full_map to a maximum absolute value of 1 fixes this without
    # affecting the result.
    max_abs = np.abs(full_map).max()
    if max_abs > 0:
        full_map = full_map / max_abs

    obj_fun = cvxpy.sum(cvxpy.multiply(x, full_map))

    # Regularisation target and weights: dist.x0 (baseline areas/animal numbers/0 for
    # feeds) and dist.scale_f (per-variable scaling factors), both reindexed to
    # dist.x_idx_short. For 'ani' this can't be a plain .reindex(): dist.x0['ani']/
    # dist.scale_f['ani'] lack the 'sub_system' level (all of a species/breed/
    # prod_system/region's sub_systems share one x0/scale_f value), so reorder levels
    # to put 'sub_system' last first -- this is the exact same reindex
    # GeoDistributor.define_cvx_problem() does to build its own 'sf'/'x0s', and is
    # required for pandas to broadcast the shared value across sub_systems instead of
    # matching nothing at all (see that method for the reference implementation).
    def _reindex_like_x0(series, k):
        if k == 'ani':
            lvls = ['species', 'breed', 'prod_system', 'region', 'sub_system']
            return (
                series
                .reindex(dist.x_idx['ani'].reorder_levels(lvls))
                .reindex(dist.x_idx_short['ani'].reorder_levels(lvls))
                .set_axis(dist.x_idx_short['ani'])
                .to_numpy()
            )
        else:
            return series.reindex(dist.x_idx_short[k], fill_value=0).to_numpy()

    x0_vec = np.concatenate([_reindex_like_x0(dist.x0[k], k) for k in keys])
    scale_f_vec = np.concatenate([_reindex_like_x0(dist.scale_f[k], k) for k in keys])

    in_obj = (np.abs(full_map) > 0).astype(float)
    not_in_obj = 1 - in_obj

    # x_fds is excluded from regularisation entirely, regardless of whether it has a
    # nonzero mapper coefficient. dist.x0['fds'] is always 0 by construction (it is
    # not a meaningful baseline -- FeedDistributor doesn't track a "baseline" feed
    # use), so regularising x_fds towards it directly fights any attempt to grow a
    # SELECTED animal_prod: growing selected animals requires proportionally more of
    # their feed (via constraints C11-C13), and every one of those feed variables
    # would otherwise be penalised for moving away from 0. x_fds does not need its
    # own regularisation anyway: C11-C13 already tie it tightly (as ration shares) to
    # x_ani, which is itself regularised (if not selected) or objective-driven (if
    # selected), so it cannot float independently.
    if 'fds' in dist.x_idx_short:
        not_in_obj[-sizes['fds']:] = 0

    reg_weight = not_in_obj * scale_f_vec
    if keep_linear:
        reg_fun = lambda_reg * cvxpy.norm1(cvxpy.multiply(reg_weight, x - x0_vec))
    else:
        reg_fun = lambda_reg * cvxpy.sum_squares(cvxpy.multiply(reg_weight, x - x0_vec))

    # Ridge term: same shape as reg_fun above, but covering ALL of x_ani/x_crp (not
    # just the non-selected part) with a much smaller weight -- see docstring for why
    # this is needed even for the selected (in_obj) variables.
    ridge_mask = np.ones(n)
    if 'fds' in dist.x_idx_short:
        ridge_mask[-sizes['fds']:] = 0
    ridge_weight = ridge_mask * scale_f_vec
    ridge_fun = lambda_ridge * cvxpy.sum_squares(cvxpy.multiply(ridge_weight, x - x0_vec))

    objective = cvxpy.Minimize(obj_fun + reg_fun + ridge_fun)

    constraints = [
        make_cvxpy_constraint(cons, x) for cons in dist.constraints.values()
    ]

    dist.problem = cvxpy.Problem(objective=objective, constraints=constraints)

    return None

def nutrient_supply_from_x(geodist, nutrient, crop_prods=None, animal_prods=None):
    '''Builds a mapper from x_ani, x_crp and (if available) x_fds to the dietary
    supply of a given 'nutrient', based on the 'composition', 'conv_factor_main' and
    'waste_share' (and, where used, 'recipe') parameters in 'DemandAndConversions'.

    This returns PER-UNIT coefficients (not the nutrient supply for any particular
    x) -- i.e. how much 'nutrient' one unit of a given crop area/animal number/feed
    amount is worth. To get the nutrient supply for a given x, take
    sum(mapper['crp'] * x['crp']) + sum(mapper['ani'] * x['ani'])
    + sum(mapper['fds'] * x['fds']) (the last term already carries a negative sign,
    see below).

    Crop areas (x_crp) and animal numbers (x_ani) are mapped to the crop and animal
    products they produce (via 'CropProduction.production' and 'AnimalHerd.production')
    and get POSITIVE coefficients, representing crop/animal production that ends up
    as food. Feed amounts (x_fds), if available, are mapped to the crop products they
    consume (via the 'feed_to_prod' parameter in the 'FeedMgmt' module) and get
    NEGATIVE coefficients, representing crop production diverted to feed rather than
    food.

    The per-unit nutrient content used for x_crp/x_ani/x_fds is the crop/animal
    product's POTENTIAL food-nutrient density -- i.e. based ONLY on
    DemandAndConversions.crop_prod_demand/animal_prod_demand's 'food' column, ignoring
    'non-food'/'export' entirely (see '_nutrient_supply_factors()'). This means the
    mapper deliberately does NOT net out to DemandAndConversions.nutrient_supply when
    summed over a feasible x in general -- it only does so for a crop_prod/animal_prod
    with no non-food/export demand at baseline. Where a crop_prod/animal_prod DOES
    have non-food/export demand, x_crp/x_ani's production credits the FULL
    food-nutrient density to the ENTIRE quantity produced (including the portion that,
    at baseline, actually goes to non-food/export), which is what makes the mapper
    suitable for 'change_objective()': it answers "how much MORE food-nutrient supply
    is achievable from this product", not "how much does the baseline mix actually
    deliver on average". x_fds is still subtracted using the same per-unit figure, so
    feed use is still correctly netted out of whatever x_crp/x_ani credited -- this
    exclusion only concerns the non-food/export split, not the feed split.

    Must be run after GeoDistributor.make() (so that geodist.x_idx is defined).
    CropProduction.production/AnimalHerd.production are re-scaled in place by
    GeoDistributor.apply_solution() (the default when calling .solve()). The crop
    part of the mapper is computed directly from CropProduction's parameters and so
    is unaffected by this either way. The animal part is derived from AnimalHerd's
    current data (there being no similarly simple parameter-only route to herd
    production), which is safe UNLESS a prior .solve() has already scaled some herd
    to 0 animals in a given region -- in that case the corresponding per-animal
    production rate is unrecoverable and a warning is raised. To avoid this, call
    this function right after .make(), before .solve().

    Parameters
    ----------
    geodist : GeoDistributor or FeedDistributor object
    nutrient : str
        Nutrient to build the mapper for, e.g. one of 'DM', 'N', 'protein', 'fat',
        'fibre' or 'kcal' (see the 'nutrient' filter values for parameter
        'composition' in DemandAndConversions).
    crop_prods : list of str, default None
        If given, restrict the mapper to only credit these crop_prod(s) (i.e. all
        other crop_prod get a 0 coefficient in 'mapper['crp']'/'mapper['fds']'). If
        None (default), all crop_prod are included.
    animal_prods : list of (species, animal_prod) tuples, default None
        Analogous to 'crop_prods' but for animal products (restricts
        'mapper['ani']'). If None (default), all animal_prod are included.

    Returns
    -------
    dict of pandas.Series
        Keys 'ani', 'crp' (and 'fds' if 'geodist' is a FeedDistributor). Each Series
        is indexed like the corresponding entry in geodist.x_idx and gives the
        nutrient supply (same units as DemandAndConversions.nutrient_supply, i.e. kg
        or kcal/year) generated per unit of that variable.
    '''

    if not hasattr(geodist, 'x_idx'):
        raise ValueError("geodist.x_idx is not defined. Run GeoDistributor.make() first.")

    demand = geodist.demand
    crops = geodist.crops
    herds = geodist.herds

    nutrient_per_crop_prod, nutrient_per_animal_prod = _nutrient_supply_factors(demand, nutrient)

    if crop_prods or animal_prods:
        _warn_zero_nutrient_credit(
            demand, nutrient, crop_prods, animal_prods,
            nutrient_per_crop_prod, nutrient_per_animal_prod,
        )

    if crop_prods is not None:
        nutrient_per_crop_prod = nutrient_per_crop_prod[
            nutrient_per_crop_prod.index.get_level_values('crop_prod').isin(crop_prods)
        ]
    if animal_prods is not None:
        nutrient_per_animal_prod = nutrient_per_animal_prod[
            nutrient_per_animal_prod.index.droplevel('prod_system').isin(animal_prods)
        ]

    mapper = {}

    # --- x_crp: crop areas --> crop product production --> nutrient supply ---
    # Computed directly from CropProduction's parameters (not its data_attr, which
    # GeoDistributor.apply_solution() rescales in place via crops.scale() -- and
    # which loses the underlying per-area rate entirely wherever the solution has
    # set area to exactly 0 for some crop/region).
    net_production_per_area = _crop_net_production_rate(crops)
    crp_long = (
        net_production_per_area
        .stack('crop_prod')
        .rename('rate')
        .reset_index()
    )
    merged = crp_long.merge(
        nutrient_per_crop_prod.rename('nutrient_per_unit').reset_index(),
        on=['prod_system', 'crop_prod'],
        how='inner',
    )
    merged['value'] = merged['rate'] * merged['nutrient_per_unit']
    mapper['crp'] = (
        merged
        .groupby(['crop', 'prod_system', 'region'])['value']
        .sum()
        .reindex(geodist.x_idx['crp'], fill_value=0)
    )

    # --- x_ani: animal numbers --> animal product production --> nutrient supply ---
    # 'production' is stored relative to AnimalHerd's current basis for its
    # "defining animal" (herd.x_is), which is 1 on a freshly calculated AnimalHerd
    # object but reflects the solved x_ani after GeoDistributor.apply_solution() has
    # called herd.scale(). Dividing by that same basis (mirroring the 'old_x' lookup
    # in AnimalHerd.scale()) gives a per-defining-animal rate that is invariant to
    # any such scaling -- UNLESS the basis has been scaled to exactly 0 for some
    # region (i.e. the solution has 0 animals there), which destroys the underlying
    # rate; see warning below.
    ani_rows = []
    zero_basis_herds = []
    for herd in herds:
        sp, br, ps, ss = herd.species, herd.breed, herd.prod_system, herd.sub_system
        prod = herd.data_attr.get('production')
        x_basis = _get_x_is_basis(herd)
        if (x_basis == 0).any():
            zero_basis_herds.append((sp, br, ps, ss))
        for ap in prod.columns.unique('animal_prod'):
            for ops in prod.columns.unique('prod_system'):
                per_head = (prod.loc[:, (ops, slice(None), ap)].sum(axis=1) / x_basis).fillna(0)
                if (per_head == 0).all():
                    continue
                ani_rows.append(pd.DataFrame({
                    'species': sp, 'breed': br, 'prod_system': ps, 'sub_system': ss,
                    'region': per_head.index, 'ops': ops, 'animal_prod': ap,
                    'per_head': per_head.values,
                }))

    if zero_basis_herds:
        warnings.warn(
            "nutrient_supply_from_x: for some AnimalHerd(s) the current per-defining-"
            "animal basis (herd.x_is) is 0 in one or more regions -- likely because "
            "GeoDistributor.apply_solution() already scaled the herd to a solution with "
            "0 animals there, which destroys the underlying per-animal production rate. "
            "The mapper for x_ani will be 0 for those entries. To avoid this, call "
            "nutrient_supply_from_x() right after GeoDistributor.make(), before .solve(). "
            f"Affected herds (species, breed, prod_system, sub_system): {zero_basis_herds}"
        )

    if len(ani_rows) > 0:
        ani_long = pd.concat(ani_rows, ignore_index=True)
        merged = ani_long.merge(
            nutrient_per_animal_prod.rename('nutrient_per_unit').reset_index().rename(columns={'prod_system': 'ops'}),
            on=['ops', 'species', 'animal_prod'],
            how='inner',
        )
        merged['value'] = merged['per_head'] * merged['nutrient_per_unit']
        mapper['ani'] = (
            merged
            .groupby(['species', 'breed', 'prod_system', 'sub_system', 'region'])['value']
            .sum()
            .reindex(geodist.x_idx['ani'], fill_value=0)
        )
    else:
        mapper['ani'] = pd.Series(0.0, index=geodist.x_idx['ani'])

    # --- x_fds: feed amounts --> crop product usage --> nutrient supply (negative) ---
    if 'fds' in geodist.x_idx:
        factors = geodist._get_feed_to_prod_factors('crop_prod', drop_region=True)
        merged = factors.merge(
            nutrient_per_crop_prod.rename('nutrient_per_unit').reset_index(),
            on=['prod_system', 'crop_prod'],
            how='inner',
        )
        merged['value'] = -1 * merged['feed_to_prod'] * merged['share_domestic'] * merged['nutrient_per_unit']
        fds_no_region = (
            merged
            .groupby(['feed', 'animal', 'species', 'breed', 'prod_system', 'sub_system'])['value']
            .sum()
        )
        # 'feed_to_prod' factors don't vary by region, so broadcast across all
        # regions in geodist.x_idx['fds'] by reindexing on the region-less index
        # (which preserves row order) before restoring the full index
        mapper['fds'] = (
            fds_no_region
            .reindex(geodist.x_idx['fds'].droplevel('region'), fill_value=0)
            .set_axis(geodist.x_idx['fds'])
        )

    return mapper

def _warn_zero_nutrient_credit(
        demand, nutrient, crop_prods, animal_prods,
        nutrient_per_crop_prod, nutrient_per_animal_prod,
    ):
    '''Warns if a requested crop_prod/animal_prod is "matched" -- i.e. has at least
    one row in 'crop_prod_demand'/'animal_prod_demand', meaning some food item maps
    to it via 'conv_factor_main' and so it has a legitimate constraint C1 row (so
    'change_objective()' WILL relax C1 for it and the optimiser CAN grow it) -- but
    still ends up with a ZERO coefficient in the nutrient mapper built by
    '_nutrient_supply_factors()'. This is distinct from
    '_relax_C1_for_selected_products()''s "unmatched" warning, which covers a
    product with NO row (and hence no C1 row) at all -- here the product IS a
    valid, relaxed row, it just earns nothing towards the objective for growing,
    which is easy to miss (the optimiser will simply leave it near baseline,
    indistinguishable from "nothing better was available" unless this is pointed
    out).

    The most common cause: EVERY food item mapped to the product via
    'conv_factor_main' is 100% imported, i.e. 'food_demand_to_processing' has 0
    domestic demand for all of them. Since 'crop_prod_demand'/'animal_prod_demand'
    are themselves computed purely from DOMESTIC demand (see
    'DemandAndConversions._get_demand()'), this case shows up as an explicit
    all-ZERO 'crop_prod_demand'/'animal_prod_demand' row (still "matched", not
    "unmatched" -- see above) rather than a missing one. Separately,
    '_nutrient_supply_factors()' also only credits DOMESTIC consumption when
    building the mapper (it filters out food items with 0 domestic demand before
    computing anything), so there is no domestic food-nutrient density left to
    attribute to the crop_prod/animal_prod either way. This is checked explicitly
    here and named in the warning when it applies; otherwise a more generic message
    is given (e.g. the mapped food item(s) could have a 0 entry for 'nutrient' in
    'composition', or a 100% waste_share).

    Parameters
    ----------
    demand : DemandAndConversions object
    nutrient : str
        Only used to word the warning message.
    crop_prods : list of str or None
    animal_prods : list of (species, animal_prod) tuples or None
    nutrient_per_crop_prod, nutrient_per_animal_prod : pandas.Series
        UNFILTERED outputs of '_nutrient_supply_factors()' (i.e. before restricting
        to 'crop_prods'/'animal_prods'), indexed (prod_system, crop_prod) and
        (prod_system, species, animal_prod) respectively.

    Returns
    -------
    None. Issues a single combined 'warnings.warn()' if any affected crop_prod/
    animal_prod is found; otherwise does nothing.
    '''

    par = demand.par

    def _fully_imported_by(of):
        '''Series indexed by 'of' (['crop_prod'] or ['species','animal_prod']),
        True where EVERY food item mapped to that crop_prod/animal_prod via
        'conv_factor_main' has 0 domestic demand in 'food_demand_to_processing'.'''
        par.clear()
        try:
            prs = par.get_unique(['food'] + of, 'parameter == "conv_factor_main"').set_index('food')
        except KeyError:
            return pd.Series(dtype=bool)
        food_dom = demand.data_attr.get('food_demand_to_processing')['domestic']
        prs = prs.assign(fully_imported=(food_dom.reindex(prs.index, fill_value=0) <= 0))
        return prs.groupby(of)['fully_imported'].all()

    def _reason(fully_imported, key):
        if fully_imported.get(key, False):
            return "all mapped food item(s) are 100% imported (share_imported=100%)"
        return "unknown -- check the mapped food item(s)' 'composition'/'waste_share'"

    # "Matched" here means the crop_prod/animal_prod has AT LEAST ONE row in
    # crop_prod_demand/animal_prod_demand -- i.e. some food item maps to it via
    # 'conv_factor_main' -- regardless of whether that row's VALUES are 0. This
    # matters because crop_prod_demand/animal_prod_demand are themselves computed
    # purely from DOMESTIC demand (see DemandAndConversions._get_demand()): a
    # product whose only mapped food item(s) are 100% imported gets an explicit
    # all-ZERO row here, not a missing one -- it is still "matched" (distinct from
    # '_relax_C1_for_selected_products()''s "unmatched" case, where the crop_prod/
    # animal_prod has no row at all), it just has nothing to credit.
    zero_crp = []
    if crop_prods:
        cpd = demand.data_attr.get('crop_prod_demand')
        matched = cpd.index[cpd.index.get_level_values('crop_prod').isin(crop_prods)]
        if len(matched) > 0:
            credit = nutrient_per_crop_prod.reindex(matched, fill_value=0)
            zero_rows = matched[credit == 0]
            if len(zero_rows) > 0:
                fully_imported = _fully_imported_by(['crop_prod'])
                zero_crp = [
                    (ps, cp, _reason(fully_imported, cp)) for ps, cp in zero_rows
                ]

    zero_ani = []
    if animal_prods:
        apd = demand.data_attr.get('animal_prod_demand')
        matched = apd.index[apd.index.droplevel('prod_system').isin(animal_prods)]
        if len(matched) > 0:
            credit = nutrient_per_animal_prod.reindex(matched, fill_value=0)
            zero_rows = matched[credit == 0]
            if len(zero_rows) > 0:
                fully_imported = _fully_imported_by(['species', 'animal_prod'])
                zero_ani = [
                    (ps, sp, ap, _reason(fully_imported, (sp, ap)))
                    for ps, sp, ap in zero_rows
                ]

    if not zero_crp and not zero_ani:
        return None

    lines = []
    if zero_crp:
        lines.append("crop_prods: " + "; ".join(
            f"{cp!r} ({ps}) -- {reason}" for ps, cp, reason in zero_crp
        ))
    if zero_ani:
        lines.append("animal_prods: " + "; ".join(
            f"({sp!r}, {ap!r}) ({ps}) -- {reason}" for ps, sp, ap, reason in zero_ani
        ))

    warnings.warn(
        "nutrient_supply_from_x(): the following selected crop_prod(s)/animal_prod(s) "
        "are 'matched' (some food item maps to them via 'conv_factor_main', so they "
        "have a valid constraint C1 row -- change_objective() WILL relax C1 for "
        f"them) but have a ZERO '{nutrient}' coefficient in the mapper, so growing "
        "them earns nothing towards the objective (they will simply stay near "
        "baseline). Reduce 'share_imported' below 100% for at least one mapped "
        "food item if domestic production of these should be rewarded.\n" + "\n".join(lines)
    )

    return None

def _crop_net_production_rate(crops):
    '''Returns net production (production minus seed demand) of each crop_prod per
    unit area of each crop/prod_system/region, computed directly from
    CropProduction's parameters ('crop_to_prod', 'yield' and 'seed'), mirroring
    CropProduction.calculate_production()/calculate_seed_demand().

    This is independent of CropProduction's current data_attr state. GeoDistributor.
    apply_solution() rescales 'production'/'seed_demand'/'area' in place via
    crops.scale(), and wherever the solution sets area to exactly 0 for some
    crop/region the underlying per-area rate is destroyed (0 * rate = 0, so the rate
    can no longer be recovered by dividing production by area). Recomputing from
    parameters avoids that entirely.

    Parameters
    ----------
    crops : CropProduction object

    Returns
    -------
    pandas.DataFrame indexed like crops.index, columns 'crop_prod'
    '''

    par = crops.par
    idx = crops.index

    par.clear()
    cps = par.get_unique('crop_prod')
    production_rate = par.get_from_frame(
        'crop_to_prod',
        pd.DataFrame(index=idx, columns=pd.Index(cps, name='crop_prod')),
    )

    par.clear()
    par.set(**idx.to_frame().to_dict('list'))
    production_rate = production_rate.mul(pd.Series(par.get('yield'), index=idx), axis=0)

    par.clear()
    seed_cps = par.get_unique('crop_prod', qry='parameter == "seed"')
    seed_rate = par.get_from_frame(
        'seed',
        pd.DataFrame(index=idx, columns=pd.Index(seed_cps, name='crop_prod')),
    )

    return production_rate.sub(seed_rate, fill_value=0)

def _get_x_is_basis(herd):
    '''Returns the same "current basis" quantity that AnimalHerd.scale() uses as
    'old_x' when scaling data attributes (i.e. the current value, per region, of
    whatever 'herd.x_is' represents). Since all of AnimalHerd's scalable data
    attributes (including 'production') are scaled by the same 'new_x/old_x' factor,
    dividing 'production' by this basis gives a per-defining-animal rate that is
    invariant to whether/how the herd has already been scaled.

    Parameters
    ----------
    herd : AnimalHerd object

    Returns
    -------
    pandas.Series indexed by region
    '''

    x_is = herd.x_is
    if x_is == 'sows+gilts':
        x_is = 'sows'

    if x_is in ['milk', 'meat', 'fish']:
        return herd.data_attr.get('production').loc[:, (slice(None), slice(None), x_is)].sum(axis=1)
    elif x_is == 'total horses':
        return herd.data_attr.get('heads').sum(axis=1)
    elif x_is == 'ewes+rams':
        return herd.data_attr.get('heads').loc[:, (slice(None), ['ewes', 'rams'])].sum(axis=1)
    elif x_is == 'total hens':
        return herd.data_attr.get('heads').drop('laying chicks', level='animal', axis=1).sum(axis=1)
    else:
        return herd.data_attr.get('heads').loc[:, (herd.prod_system, x_is)]

def _nutrient_supply_factors(demand, nutrient):
    '''Calculates the dietary nutrient supply generated per unit of domestically
    produced crop_prod / animal_prod, based on DemandAndConversions' 'composition',
    'conv_factor_main' and 'waste_share' parameters (and 'recipe' where compound
    foods are used).

    Since a given crop_prod/animal_prod can be shared by several food items, the
    nutrient content of each food item is weighted by its own share of the total
    food-driven production requirement (DemandAndConversions.crop_prod_demand /
    animal_prod_demand, 'food' column ONLY) to give a single food-nutrient-density
    figure per unit of crop_prod/animal_prod.

    Deliberately excludes the 'non-food' and 'export' columns of crop_prod_demand /
    animal_prod_demand from the denominator (even though a crop_prod/animal_prod's
    TOTAL production, and hence x_crp/x_ani, covers all three uses): the result is
    meant to represent the POTENTIAL food-nutrient supply obtainable from a unit of
    that product, not the average diluted by whatever share of it happens to go to
    non-food/export at baseline. E.g. if a crop_prod is 50% food and 50% export at
    baseline, this still returns the full food-nutrient density (as if the export
    share didn't exist), rather than half of it -- appropriate for
    'change_objective()', which explores how much MORE food-nutrient supply is
    achievable, not how much the baseline mix happens to deliver on average.

    Parameters
    ----------
    demand : DemandAndConversions object
    nutrient : str
        Nutrient to calculate the per-unit supply for (a value of the 'nutrient'
        filter for parameter 'composition' in DemandAndConversions)

    Returns
    -------
    (nutrient_per_crop_prod, nutrient_per_animal_prod) : tuple of pandas.Series
        Indexed (prod_system, crop_prod) and (prod_system, species, animal_prod)
        respectively. Units match DemandAndConversions.nutrient_supply (kg or
        kcal/year) per kg of crop_prod / animal_prod produced.
    '''

    par = demand.par

    par.clear()
    nuts = par.get_unique('nutrient', qry='parameter == "composition"')
    if nutrient not in nuts:
        raise ValueError(
            f"'{nutrient}' is not a valid nutrient. Available: {', '.join(nuts)}"
        )

    # Amount of food (as defined in the diet) entering processing (i.e. before
    # household/retail/processing waste). Only the domestically produced share is
    # relevant here, since x_ani/x_crp/x_fds concern domestic production.
    food_dom = demand.data_attr.get('food_demand_to_processing')['domestic']
    food_dom = food_dom[food_dom > 0]

    # Share of food surviving household/retail/processing waste
    par.clear()
    waste_shares = par.get_from_frame(
        'waste_share',
        pd.DataFrame(
            index=food_dom.index,
            columns=pd.Index(['household', 'retail', 'processing'], name='waste_level'),
        ),
    ) / 100
    retention = (1 - waste_shares).prod(axis=1)

    # Nutrient composition per food item
    par.clear()
    composition = par.get_from_frame(
        'composition',
        pd.DataFrame(1, index=food_dom.index, columns=pd.Index([nutrient], name='nutrient')),
    )[nutrient]

    # Nutrient supply (domestic) per food item, i.e. as actually consumed (after waste).
    # The columns axis must be named (not the default None) so that
    # _split_by_recipe()'s call to par.get_from_frame() below can build its filters
    # correctly -- get_from_frame() cross-joins df.index and df.columns level names,
    # and an unnamed columns axis produces a column literally named 0 (int) in that
    # cross-join while still looking up a filter key of None, causing a KeyError.
    nutrient_flow = (composition * food_dom * retention).rename('value').to_frame()
    nutrient_flow.columns.name = 'metric'

    # Split nutrient flow of compound foods (recipes) across their food_ingr
    # components, analogous to DemandAndConversions.resolve_recipes()
    nutrient_flow = _split_by_recipe(demand, nutrient_flow)

    def _per_unit_production(of, demand_attr):
        par.clear()
        try:
            prs = par.get_unique(['food'] + of, 'parameter == "conv_factor_main"').set_index('food')
        except KeyError:
            return pd.Series(dtype=float)

        sel = nutrient_flow.index.get_level_values('food').isin(prs.index)
        if not sel.any():
            return pd.Series(dtype=float)

        gen = nutrient_flow.loc[sel].join(prs).set_index(of, append=True)
        gen = gen.groupby(['prod_system'] + of)['value'].sum()

        # Only the 'food' column -- NOT 'non-food'/'export' -- so that the result is
        # the nutrient content per unit of crop_prod/animal_prod WHEN USED FOR FOOD,
        # i.e. the potential food-nutrient density of that product, undiluted by
        # whatever share of it happens to go to non-food/export uses at baseline. See
        # '_nutrient_supply_factors()' docstring for the reasoning.
        food_demand = demand.data_attr.get(demand_attr)['food']
        food_demand = food_demand.reindex(gen.index).replace({0: np.nan})

        return gen.div(food_demand).fillna(0)

    nutrient_per_crop_prod = _per_unit_production(['crop_prod'], 'crop_prod_demand')
    nutrient_per_animal_prod = _per_unit_production(['species', 'animal_prod'], 'animal_prod_demand')

    return nutrient_per_crop_prod, nutrient_per_animal_prod

def _split_by_recipe(demand, df):
    '''Splits the values for compound 'food' items (i.e. those defined via the
    'recipe' parameter) in 'df' into their 'food_ingr' components, analogous to
    DemandAndConversions.resolve_recipes(), but generalised to arbitrary value
    columns (rather than being restricted to 'domestic'/'imported' origin columns).

    Parameters
    ----------
    demand : DemandAndConversions object
    df : pandas.DataFrame
        Indexed (food, food_group, prod_system) with arbitrary value columns

    Returns
    -------
    pandas.DataFrame with the same index names and columns as 'df', with values for
    compound foods redistributed to their food_ingr components (renamed to 'food')
    '''

    par = demand.par
    par.clear()
    try:
        fis = par.get_unique(['food', 'food_ingr'], 'parameter == "recipe"').set_index(['food', 'food_ingr'])
    except KeyError:
        return df

    is_compound = df.index.get_level_values('food').isin(fis.index.get_level_values('food'))
    if not is_compound.any():
        return df

    resolved = df.loc[is_compound].join(fis)
    par.clear()
    # get_from_frame() needs a named columns axis to build its filters (an unnamed
    # axis, e.g. a plain single 'value' column, otherwise raises KeyError: None) --
    # give it a placeholder name here regardless of what the caller passed
    orig_columns = resolved.columns
    if resolved.columns.name is None and not isinstance(resolved.columns, pd.MultiIndex):
        resolved.columns = resolved.columns.rename('metric')
    resolved = resolved * (par.get_from_frame('recipe', resolved) / 100)
    resolved.columns = orig_columns

    resolved = (
        resolved
        .groupby(['food_ingr', 'food_group', 'prod_system'])
        .sum()
        .rename_axis(index={'food_ingr': 'food'})
    )

    return (
        pd.concat([df.loc[~is_compound], resolved])
        .groupby(['food', 'food_group', 'prod_system'])
        .sum()
    )
