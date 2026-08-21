# `Regions`

The `Regions` module handles specification of the initial state for crop areas (`x0_crops`) and animal numbers (`x0_animals`), which are used in the default optimisation goal function in the `GeoDistributor` and `FeedDistributor` modules. It also handles parameters for climate and soil properties and the maximum allowable land use per land use category (`land_use`; e.g. cropland).

```mermaid
{{ mermaid_init() }}

flowchart TD

  %% -------------------------
  %% Main
  %% -------------------------

  A["<b>Regions</b>"]:::mod_main

  A --> C["<b>.calculate()</b>"]:::method

  S["
  <b><u>Settings</u></b>
  max_land_use_from_x0 = True
  max_land_use_from_scenario_x0 = False
  "]:::settings --> A

  A --> GX0_1["<b>.get_x0()</b>"]:::method
  C --> GX0_2["<b>.get_x0()</b>"]:::method
  C --> CL["<b>.get_climate()</b>"]:::method
  C --> SL["<b>.get_soil()</b>"]:::method
  SL_T["<b>.classify_soil_texture()</b>"]:::method
  SL_PK["<b>.classify_soil_PK()</b>"]:::method
  MLU["<b>.calculate_max_land_use()</b>"]:::method

  %% -------------------------
  %% Get x0
  %% -------------------------
  
  P_X0_1["<b>Regions.par.</b>
x0_crops, x0_animals"]:::param --> GX0_1
  GX0_1 --> DA_X0_1["<b>Regions.data_attr.</b>
x0_crops_init, x0_animals_init"]:::data

  P_X0_2["<b>Regions.par.</b>
x0_crops, x0_animals"]:::param --> GX0_2
  GX0_2 --> DA_X0_2["<b>Regions.data_attr.</b>
x0_crops, x0_animals"]:::data

  %% -------------------------
  %% Climate
  %% -------------------------

  P_CL["<b>Regions.par.</b>
GDD5"]:::param --> CL
  CL --> DA_CL["<b>Regions.data_attr.</b>
climate"]:::data

  %% -------------------------
  %% Soil
  %% -------------------------

  P_SL["<b>Regions.par.</b>
soil_clay, soil_silt, soil_sand, soil_OM, soil_pH, soil_P_AL, soil_K_AL"]:::param --> SL
  SL --> DA_SL["<b>Regions.data_attr.</b>
soil"]:::data

  DA_SL --> SL_T
  DA_SL --> SL_PK

  SL_T --> DA_SL_T["<b>Regions.data_attr.</b>
soil_texture"]:::data

  SL_PK --> D_SL_PK["<b>Regions.data_attr.</b>
soil_P_class, soil_K_class"]:::data

  %% -------------------------
  %% Max land use
  %% -------------------------

  P_MLU["<b>Regions.par.</b>
max_land_use, max_land_use_factor"]:::param --> MLU

  DA_X0_1 --> MLU
  DA_X0_2 --> MLU

  MLU --> DA_MLU["<b>Regions.data_attr.</b>
max_land_use"]:::data

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.main_modules.regions.Regions", "CIBUSmod/main_modules/regions.py") }}