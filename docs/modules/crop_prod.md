# `CropProduction`

The `CropProduction` module calculates mass of harvested crop products (`crop_prod`) generated from cultivating a crop on a given area. Each production unit is specified by a `crop`, production system (`prod_system`; e.g. conventional or organic), and `region`. The model allows for parametrising multiple crops producing the same product (e.g. winter and spring wheat) as well as a single crop producing multiple products (i.e. to represent intercropping). The main parameter in this module is crop yields, but it also takes parameters for seeding density, above and below ground crop residues, etc.

The `.scale()` method scales all `CropProduction`´s data attributes with the `scalable=True` flag based on new crop areas.

```mermaid
{{ mermaid_init() }}

flowchart TD

  %% -------------------------
  %% Main
  %% -------------------------

  A["<b>CropProduction</b>"]:::mod_main
  A --> C["<b>.calculate()</b>"]:::method
  A --> SC["<b>.scale()</b>"]:::method

  R["<b>Regions</b>"]:::mod_main --> A

  %% -------------------------
  %% Calculate() sequence
  %% -------------------------

  P_Y["<b>CropProduction.par.</b>
yield, crop_dm"]:::param --> C

  C --> DA_C["<b>CropProduction.data_attr.</b>
area, harvest, harvest_dm"]:::data

  CP["<b>.calculate_production()</b>"]:::method
  CR["<b>.calculate_crop_residues()</b>"]:::method
  SD["<b>.calculate_seed_demand()</b>"]:::method

  %% -------------------------
  %% Production (crop -> crop_prod mapping)
  %% -------------------------

  P_C2P["<b>CropProduction.par.</b>
crop_to_prod"]:::param --> CP
  DA_C ---> CP
  CP --> DA_P["<b>CropProduction.data_attr.</b>
production"]:::data

  %% -------------------------
  %% Crop residues (above/below ground)
  %% -------------------------

  P_RES["<b>CropProduction.par.</b>
ag_resid, bg_resid, frac_renew, crop_dm"]:::param --> CR
  DA_C ---> CR
  CR --> DA_CR["<b>CropProduction.data_attr.</b>
crop_residues, below_ground_biomass_C"]:::data

  %% -------------------------
  %% Seed demand
  %% -------------------------

  P_SEED["<b>CropProduction.par.</b>
seed"]:::param --> SD
  DA_C ---> SD
  SD --> DA_SD["<b>CropProduction.data_attr.</b>
seed_demand"]:::data

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.main_modules.crop_prod.CropProduction", "CIBUSmod/main_modules/crop_prod.py") }}