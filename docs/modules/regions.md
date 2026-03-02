# `Regions`

The `Regions` module handles specification of the initial state for crop areas (`x0_crops`) and animal numbers (`x0_animals`), which are used in the default optimisation goal function in the `GeoDistributor` and `FeedDistributor` modules. It also handles parameters for climate and soil properties and the maximum allowable land use per land use category (`land_use`; e.g. cropland).

```mermaid
{{ mermaid_init() }}

flowchart TD

  %% -------------------------
  %% Main
  %% -------------------------

  A["**Regions**"]:::mod_main

  A --> C["**.calculate()**"]:::method

  S["
  **<u>Settings</u>**
  max_land_use_from_x0 = True
  max_land_use_from_scenario_x0 = False
  "]:::settings --> A

  A --> GX0_1["**.get_x0()**"]:::method
  C --> GX0_2["**.get_x0()**"]:::method
  C --> CL["**.get_climate()**"]:::method
  C --> SL["**.get_soil()**"]:::method
  SL_T["**.classify_soil_texture()**"]:::method
  SL_PK["**.classify_soil_PK()**"]:::method
  MLU["**.calculate_max_land_use()**"]:::method

  %% -------------------------
  %% Get x0
  %% -------------------------
  
  P_X0_1["**Regions.par.**
x0_crops, x0_animals"]:::param --> GX0_1
  GX0_1 --> DA_X0_1["**Regions.data_attr.**
x0_crops_init, x0_animals_init"]:::data

  P_X0_2["**Regions.par.**
x0_crops, x0_animals"]:::param --> GX0_2
  GX0_2 --> DA_X0_2["**Regions.data_attr.**
x0_crops, x0_animals"]:::data

  %% -------------------------
  %% Climate
  %% -------------------------

  P_CL["**Regions.par.**
GDD5"]:::param --> CL
  CL --> DA_CL["**Regions.data_attr.**
climate"]:::data

  %% -------------------------
  %% Soil
  %% -------------------------

  P_SL["**Regions.par.**
soil_clay, soil_silt, soil_sand, soil_OM, soil_pH, soil_P_AL, soil_K_AL"]:::param --> SL
  SL --> DA_SL["**Regions.data_attr.**
soil"]:::data

  DA_SL --> SL_T
  DA_SL --> SL_PK

  SL_T --> DA_SL_T["**Regions.data_attr.**
soil_texture"]:::data

  SL_PK --> D_SL_PK["**Regions.data_attr.**
soil_P_class, soil_K_class"]:::data

  %% -------------------------
  %% Max land use
  %% -------------------------

  P_MLU["**Regions.par.**
max_land_use, max_land_use_factor"]:::param --> MLU

  DA_X0_1 --> MLU
  DA_X0_2 --> MLU

  MLU --> DA_MLU["**Regions.data_attr.**
max_land_use"]:::data

{{ mermaid_style() }}
```