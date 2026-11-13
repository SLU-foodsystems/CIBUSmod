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

## Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `ag_resid` | kg DM/kg harvested DM | `prod_system`, `region`, `crop` | Ratio of above-ground residue dry matter to harvested crop dry matter. |
| `ag_resid_C` | kg C/kg DM | `prod_system`, `region`, `crop` | Carbon content of above-ground crop residues; read by CropResidueMgmt. |
| `ag_resid_N` | kg N/kg DM | `prod_system`, `region`, `crop` | Nitrogen content of above-ground crop residues; read by CropResidueMgmt. |
| `bg_resid` | kg DM/kg above-ground DM | `prod_system`, `region`, `crop` | Ratio of below-ground residue dry matter to total above-ground biomass dry matter. |
| `bg_resid_C` | kg C/kg DM | `prod_system`, `region`, `crop` | Carbon content of below-ground biomass; also read by CropResidueMgmt. |
| `bg_resid_N` | kg N/kg DM | `prod_system`, `region`, `crop` | Nitrogen content of below-ground crop residues; read by CropResidueMgmt. |
| `crop_dm` | kg DM/kg | `prod_system`, `region`, `crop` | Dry-matter fraction of harvested crop mass; also used by MachineryAndEnergyMgmt for drying. |
| `crop_to_prod` | kg/kg | `prod_system`, `region`, `crop`, `crop_prod` | Mass of each crop product obtained per unit harvested crop mass. |
| `frac_renew` | fraction/year | `prod_system`, `region`, `crop` | Fraction of crop biomass renewed annually; applied to both above- and below-ground residues in the code. |
| `max_in_rot` | % | `prod_system`, `region`, `crop_group` | Maximum share of a crop group in the rotation; read by the distributors. |
| `min_GDD5` | degree days above 5 degrees C | `prod_system`, `region`, `crop` | Minimum growing degree days for crop suitability; read by the distributors. |
| `seed` | kg/ha (kg/m2 for greenhouse crops) | `prod_system`, `region`, `crop`, `crop_prod` | Seed requirement per cultivated area, expressed as demand for a crop product. |
| `yield` | kg/ha (kg/m2 for greenhouse crops) | `prod_system`, `region`, `crop` | Harvested crop mass per unit cultivated area at the specified crop water content. |
