# Maximise nutrient supply

The `maximise_nutrient_supply` module contains helper functions for changing the optimisation objective of a [`GeoDistributor`](geo_dist.md) or [`FeedDistributor`](feed_dist.md) from the default (minimise deviation of crop areas and animal numbers from `x0`) to instead **maximise the domestic supply of a selected nutrient** (e.g. protein) from a chosen set of crop and/or animal products. All non-selected products remain constrained to exactly meet their baseline demand.

The functions are accessed via `cm.maximise_nutrient_supply` and are called in a specific order relative to the `.make()`, `.solve()` and `.apply_solution()` methods of the distributor, as shown in the flowchart below.

```mermaid
{{ mermaid_init() }}

flowchart TD
  A["<b>FeedDistributor</b> / <b>GeoDistributor</b>"]:::mod_opt

  A --> M[".make()"]:::method
  M --> CO["change_objective()"]:::helper
  CO --> CN["constrain_nutrient()
  (optional, repeatable)"]:::helper
  CN --> S[".solve(apply_solution=False)"]:::method
  S --> UD["update_demand()"]:::helper
  UD --> DAC["<b>DemandAndConversions</b>.data_attr
  re-made to match optimised production"]:::mod_main
  UD --> AS[".apply_solution()
  (or re-make and re-solve with default objective)"]:::method
  AS --> SOC["store_objective_contributions()
  (optional)"]:::helper
  SOC --> OUT["<b>FeedDistributor.data_attr.</b>
  nutrient_contrib_animals, nutrient_contrib_crops"]:::data

  CO -. "uses" .-> NS["nutrient_supply_from_x()"]:::helper

{{ mermaid_style() }}
```

A typical use looks like this:

```python
feeddist.make()

cm.maximise_nutrient_supply.change_objective(
    feeddist,
    nutrient='protein',
    crop_prods=['food wheat', 'peas', 'broad beans'],
    animal_prods=[('cattle', 'milk'), ('cattle', 'meat')]
)
# Optionally require at least 0.5 kg fat per kg protein from the selected products
cm.maximise_nutrient_supply.constrain_nutrient(feeddist, 'fat', 0.5, '>=')

feeddist.solve(apply_solution=False)
cm.maximise_nutrient_supply.update_demand(feeddist, mode='trade')
feeddist.apply_solution()

cm.maximise_nutrient_supply.store_objective_contributions(feeddist)
```

!!! note
    `update_demand()` should be run **before** `.apply_solution()`, since `.apply_solution()` allocates crop production between uses based on `crop_prod_demand`, which would otherwise still reflect the pre-optimisation demand. It must also be run before `ByProductMgmt.calculate()`.

## Changing the objective

### `maximise_nutrient_supply.change_objective()`

`change_objective()` modifies the distributor in place after `.make()` has built all constraints. It does four things:

1. **Relaxes constraint C1 for the selected products.** C1 normally requires that production minus feed use exactly equals demand (food + non-food + export) for every crop and animal product. For the selected products this is changed to an inequality where only the *food* part of demand is relaxed:

    $$
    \text{production} - \text{feed use} \geq \text{non-food demand} + \text{export demand}
    $$

    Production of the selected products can therefore move freely (up or down) away from the original food demand, while non-food and export demand are still guaranteed to be met. All non-selected products keep their original equality constraint.

2. **Rebuilds constraints C10 and C15** (by-product balances, `FeedDistributor` only). These constraints credit by-products generated per unit of production using ratios that are averaged over all demand routes (food, non-food and export). Without adjustment, extra production of a selected product would credit by-products (and hence feed availability) that the extra production would not actually generate. The by-product supply is therefore split into a fixed part for the baseline non-food and export demand and a part for the food route that scales with the relaxed production.

3. **Builds a nutrient mapper** restricted to the selected products using [`nutrient_supply_from_x()`](#maximise_nutrient_supplynutrient_supply_from_x).

4. **Replaces the objective** with one that maximises the mapped nutrient supply, subject to the same constraints as before (with C1, C10 and C15 modified as above):

    $$
    \max \; \frac{\sum m_{crp} \, x_{crp} + \sum m_{ani} \, x_{ani} + \sum m_{fds} \, x_{fds}}{M} \;-\; \text{regularisation}
    $$

    where $m$ are the per-unit nutrient coefficients from the mapper ($m_{fds}$ is negative, see below), and $M$ is the maximum absolute coefficient across the active variables (or 1 if all coefficients are zero). Scaling the nutrient reward improves numerical conditioning; the ridge weight acts relative to this scaled reward, rather than directly in kg or kcal.

    The single regularisation term is a quadratic ridge penalty with weight `lambda_ridge` (default `1e-9`):

    $$
    \text{regularisation} = \lambda_{ridge} \sum_{i \in \{\text{crops, animals}\}} \left[s_i (x_i - x_{0,i})\right]^2
    $$

    Here $s_i$ is the corresponding `dist.scale_f` weight. The penalty applies to **all** crop areas and animal numbers, whether or not they contribute to the nutrient objective. It favours the initial distribution `x0` and adds curvature that can improve numerical stability. Feed quantities (`x_fds`) are excluded because their `x0` values are zero placeholders rather than an initial ration.

    Set `lambda_ridge=0` to disable regularisation. Larger weights favour the initial distribution more strongly and may reduce nutrient supply; their influence on total supply and the solution mix should be checked for the model being solved.

    If the solver still has numerical problems, try `scale_constraints=True` (rescales each constraint row without changing the feasible region) before increasing `lambda_ridge`.

Other constraints (land use, rotations, sub-system shares, feed rations etc.) are left unchanged. These either limit `x` directly or are only relaxed further by increased production.

```mermaid
{{ mermaid_init() }}

flowchart TD
  CO["change_objective()"]:::helper

  CO --> R1["_relax_C1_for_selected_products()"]:::helper
  CO --> R2["_split_byprod_supply_for_selected_products()"]:::helper
  CO --> NS["nutrient_supply_from_x()"]:::helper
  CO --> SO["_set_maximise_objective()"]:::helper

  R1 --> C1["<u>Constraint C1</u>
  Non-selected products: A1 @ x == b1
  Selected products: A1 @ x >= non-food + export demand"]:::const
  R2 --> C10["<u>Constraints C10, C15</u>
  By-product supply split into fixed baseline
  non-food/export part and food-route part"]:::const
  NS --> MAP["<b>mapper:</b> nutrient per unit of
  x_ani, x_crp, x_fds"]:::matrix
  MAP --> SO
  C1 --> SO
  C10 --> SO
  SO --> P["<b>dist.problem</b>
  max nutrient supply − regularisation"]:::data

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.utils.maximise_nutrient_supply.change_objective", "CIBUSmod/utils/maximise_nutrient_supply.py") }}

### `maximise_nutrient_supply.constrain_nutrient()`

Maximising a single nutrient often gives a solution that is unbalanced in other respects, e.g. high in protein but with very little energy. `constrain_nutrient()` adds a constraint that the supply of another nutrient from the *same selected products* must be at least (`'>='`) or at most (`'<='`) a given `factor` times the supply of the maximised nutrient:

$$
\text{supply}(\text{nutrient}) \;\; \geq \text{ or } \leq \;\; \text{factor} \times \text{supply}(\text{maximised nutrient})
$$

Both supplies are calculated with the same type of mapper as the objective. `factor` is expressed in units of the constrained nutrient per unit of the maximised nutrient (kg/kg, or kcal/kg for energy). For example, after maximising protein:

```python
cm.maximise_nutrient_supply.constrain_nutrient(feeddist, 'fat', 0.5, '>=')     # at least 0.5 kg fat per kg protein
cm.maximise_nutrient_supply.constrain_nutrient(feeddist, 'kcal', 20_000, '<=') # at most 20,000 kcal per kg protein
```

The constraint is implemented as a single row $(m_{\text{nutrient}} - \text{factor} \cdot m_{\text{obj}}) \, x \; \gtrless \; 0$, scaled to a maximum absolute coefficient of 1. The function can be called several times to constrain several nutrients, or to set both a minimum and a maximum for the same nutrient. Calling it again with the same `nutrient` and `rel` replaces the earlier constraint. The objective is rebuilt with the same settings as in `change_objective()`. All added constraints are removed when `.make()` is run again.

{{ docstring("CIBUSmod.utils.maximise_nutrient_supply.constrain_nutrient", "CIBUSmod/utils/maximise_nutrient_supply.py") }}

## After solving

### `maximise_nutrient_supply.update_demand()`

After solving, production of the selected products will usually differ from the demand stored in `DemandAndConversions`, and the difference has no explicit use (e.g. consumption or export) in the model. `update_demand()` re-makes the demand so that it is consistent with the optimised production, which is needed by modules that are calculated after solving (e.g. `ByProductMgmt`).

1. The actual production minus feed use of each selected product is read off the relaxed C1 constraint at the solved `x`. This is the target that the re-made `crop_prod_demand`/`animal_prod_demand` must match.
2. The gap between the target and current demand is allocated to food items whose main product (via `conv_factor_main`) is the selected product. How the gap is allocated depends on `mode`:

    | `mode` | Surplus production | Production deficit |
    | --- | --- | --- |
    | `'trade'` (default) | Consumption unchanged. Imports are first replaced by domestic production, then exports are added. | Consumption unchanged. Domestic production is replaced by imports. |
    | `'consumption'` | Domestic consumption is scaled up. Imported consumption is unchanged, so the import share decreases. | Domestic consumption is scaled down. Imported consumption is unchanged. |

    Food items made from a by-product of a selected product (e.g. butter from cream, a by-product of milk) are adjusted in proportion to how much of the by-product is generated. This is consistent with how their nutrients are credited in the objective.

3. Waste, nutrient supply, product demand and by-products are re-calculated in `DemandAndConversions` from the new `food_demand`/`export_demand`.
4. Steps 2–3 are repeated (up to `max_iter` times) until the re-made totals match the target within `rtol`/`atol`. Iteration is needed because the relation between food items and products is not fully linear, e.g. because of the dairy cream balance.

Demand is re-made starting from the *current* `food_demand` and `export_demand`, so changes made to these after `DemandAndConversions.calculate()` (e.g. by `induce_beef_exports()`) are kept. A warning is issued if a gap cannot be fully allocated (e.g. a selected product with no directly mapped food item) or if demand for non-selected products changes as a side effect.

{{ docstring("CIBUSmod.utils.maximise_nutrient_supply.update_demand", "CIBUSmod/utils/maximise_nutrient_supply.py") }}

### `maximise_nutrient_supply.store_objective_contributions()`

`store_objective_contributions()` calculates how much each animal and crop contributes to the objective (nutrient coefficient × `x`) at the current solution. The results are stored as data attributes on the distributor, so that they are saved with the scenario when `session.store()` is called:

- `nutrient_contrib_animals`: $m_{ani} \cdot x_{ani}$, indexed like `x_animals`.
- `nutrient_contrib_crops`: $m_{crp} \cdot x_{crp}$, indexed like `x_crops`, net of feed use. Feed use is only known nationally per crop product, so the (negative) feed term is allocated to the crops and regions producing that crop product in proportion to their contribution.

The sum of both attributes equals the unscaled nutrient supply from the mapper, in kg or kcal, before regularisation. The solver minimises the negative of this supply divided by $M$, plus the regularisation penalties, so its objective value has a different sign and scale. The mapper from the most recent `change_objective()` call is used, but contributions are evaluated at whatever `dist.x` currently holds. The function can therefore be called either directly after solving or after `update_demand()` followed by a re-make and re-solve with the default objective.

!!! warning
    `dist.data_attr` persists across scenarios. If a later scenario does not call `store_objective_contributions()`, the attributes from the previous scenario are stored again unless removed, e.g. with `dist.data_attr.remove('nutrient_contrib_animals')`.

{{ docstring("CIBUSmod.utils.maximise_nutrient_supply.store_objective_contributions", "CIBUSmod/utils/maximise_nutrient_supply.py") }}

## Nutrient mapping

### `maximise_nutrient_supply.nutrient_supply_from_x()`

`nutrient_supply_from_x()` is the underlying function that translates the optimisation variables into dietary nutrient supply. It returns *per-unit* coefficients, i.e. how much of the nutrient one hectare of a crop, one animal or one unit of feed is worth, as a dict of Series with keys `'crp'`, `'ani'` and (for `FeedDistributor`) `'fds'`. The nutrient supply for a given solution is then

$$
\sum m_{crp} \, x_{crp} + \sum m_{ani} \, x_{ani} + \sum m_{fds} \, x_{fds}
$$

The coefficients are calculated as follows:

- **Crops and animals** (`x_crp`, `x_ani`) are mapped to the products they produce (via `CropProduction.production` and `AnimalHerd.production`) and get **positive** coefficients.
- **Feeds** (`x_fds`) are mapped to the crop products they use (via `FeedMgmt.par.feed_to_prod`) and get **negative** coefficients, representing crop production used as feed instead of food.
- The nutrient content per kg of product is based on `composition`, `conv_factor_main`, `waste_share` and `recipe` in `DemandAndConversions`. When a product is used for several food items, these are weighted by their share of the product's *food* demand.
- Non-food and export demand are deliberately ignored. A product's whole production is credited with its full food-nutrient density, so the mapper represents the *potential* food-nutrient supply from more production, not the baseline average. As a result, the mapper does not sum to `DemandAndConversions.nutrient_supply` for products that have non-food or export demand.
- Nutrients from food items made from by-products (e.g. butter from cream) are credited to the products generating the by-product. Only the share of the by-product used for food at baseline is credited.

The function can also be used on its own, e.g. for reporting how much of a nutrient comes from each crop or animal in a solved model. It should be called right after `.make()` and before `.solve()`: if a previous solution has scaled a herd to zero animals in a region, the per-animal production cannot be recovered and a warning is raised.

{{ docstring("CIBUSmod.utils.maximise_nutrient_supply.nutrient_supply_from_x", "CIBUSmod/utils/maximise_nutrient_supply.py") }}
