# `CropProduction`

The `CropProduction` module calculates mass of harvested crop products (`crop_prod`) generated from cultivating a crop on a given area. Each production unit is specified by a `crop`, production system (`prod_system`; e.g. conventional or organic), and `region`. The model allows for parametrising multiple crops producing the same product (e.g. winter and spring wheat) as well as a single crop producing multiple products (i.e. to represent intercropping). The main parameter in this module is crop yields, but it also takes parameters for seeding density, above and below ground crop residues, etc.

The `.scale()` method scales all `CropProduction`´s data attributes with the `scalable=True` flag based on new crop areas.

```mermaid
{{ mermaid_init() }}

flowchart TD

  %% -------------------------
  %% Main
  %% -------------------------

  A["**CropProduction**"]:::mod_main
  A --> C["**.calculate()**"]:::method
  A --> SC["**.scale()**"]:::method

  R["**Regions**"]:::mod_main --> A

  %% -------------------------
  %% Calculate() sequence
  %% -------------------------

  P_Y["**CropProduction.par.**
yield, crop_dm"]:::param --> C

  C --> DA_C["**CropProduction.data_attr.**
area, harvest, harvest_dm"]:::data

  CP["**.calculate_production()**"]:::method
  CR["**.calculate_crop_residues()**"]:::method
  SD["**.calculate_seed_demand()**"]:::method

  %% -------------------------
  %% Production (crop -> crop_prod mapping)
  %% -------------------------

  P_C2P["**CropProduction.par.**
crop_to_prod"]:::param --> CP
  DA_C ---> CP
  CP --> DA_P["**CropProduction.data_attr.**
production"]:::data

  %% -------------------------
  %% Crop residues (above/below ground)
  %% -------------------------

  P_RES["**CropProduction.par.**
ag_resid, bg_resid, frac_renew, crop_dm"]:::param --> CR
  DA_C ---> CR
  CR --> DA_CR["**CropProduction.data_attr.**
crop_residues"]:::data

  %% -------------------------
  %% Seed demand
  %% -------------------------

  P_SEED["**CropProduction.par.**
seed"]:::param --> SD
  DA_C ---> SD
  SD --> DA_SD["**CropProduction.data_attr.**
seed_demand"]:::data

{{ mermaid_style() }}
```