# `PlantNutrientMgmt`

The `PlantNutrientMgmt` module calculates crop plant available nitrogen (N), phosphorus (P) and potassium (K) requirements, distributes manure and other organic fertilisers to meet these requirements, and calculates any remaining mineral fertiliser requirements. Total ammoniacal nitrogen (TAN) assumed to equal plant available N for manure and other organic fertilisers. It also calculates NH<sub>3</sub> losses from fertiliser and manure application, N losses from soil processes, losses from organic soils, N leaching, and lime requirements together with the resulting CO<sub>2</sub> emissions.

The `.calculate()` method runs all calculations in sequence. First, `.calculate_TAN_req()` and `.calculate_PK_req(element)` (called once each for `P` and `K`) calculate crop nutrient requirements. Generated manure and other organic fertilisers are then distributed to crop areas with `.distribute_manure()` and `.distribute_organic_fertilisers()` to cover as much of the N requirement as possible (see *Distributing manure* and *Distributing other organic fertilisers* below). `.calculate_mineral_NPK_application(element)` (called once each for `N`, `P` and `K`) then calculates any remaining mineral fertiliser requirement, and `.calculate_manure_application_area()` estimates the field area receiving manure (used to estimate energy use for manure application in the `MachineryAndEnergyMgmt` module). Ammonia losses from fertiliser and manure application are calculated with `.calculate_N_application_losses(of)`, and other soil N losses with `.calculate_N_soil_losses(of)`, both called once for each fertiliser/manure/residue source (`of`). Finally, `.calculate_organic_soil_losses()`, `.calculate_leaching_N()`, `.calculate_lime_application()` and `.calculate_liming_emissions()` calculate losses from organic soils, N leaching, and the lime required to counteract the acidifying effect of nutrient removal and fertiliser/manure application, together with the resulting CO<sub>2</sub> emissions.

If a `CoverCropsMgmt` module is supplied, residual N from cover crop residues is included in `.calculate_TAN_req()`, `.calculate_N_soil_losses()` also accounts for soil N losses from cover crop residues and applies a cover-crop adjustment factor to all soil N losses, and `.calculate_leaching_N()` accounts for N in cover crop residues and applies a cover-crop adjustment factor to N leaching.

## Distributing manure

Manure available to spread (`manure.<TAN/N/P/K/C>_to_spread`, calculated by `ManureMgmt`) is allocated to crop areas by `.distribute_manure()` in the following order, primarily based on plant available N requirements (`fertiliser.TAN_req`):

1. Manure deposited while grazing is distributed to the "grazing crops" grazed by the depositing herd, in proportion to the share of grazed biomass produced by each grazing crop and used by that herd.
2. Manure originating from organic production is distributed to organic crop areas, up to 100% of their TAN requirement.
3. Remaining manure (conventional, and any organic manure remaining after step 2) is distributed to organic crop areas, up to a share of their TAN requirement set by the `manure_TAN_max` parameter.
4. Remaining manure is distributed per animal herd to the crop areas used to produce feed for that herd, based on remaining TAN requirements.
5. Any manure still remaining is distributed to conventional crop areas based on remaining TAN requirements.

At each step (2-5), manure not used to meet TAN requirements is carried forward to the next step, and any regional shortages that arise (i.e. more TAN is required in a region than the manure allocated to it) are resolved by redistributing the shortage nationally in proportion to remaining manure supply. Before distribution, TAN available to spread is increased by a share (set by the parameter `N_resid_manure`) of the non-TAN N in manure, to account for N that becomes plant available in the longer term. Once the distribution of TAN is resolved, N, P, K and C in manure are allocated to crop areas in the same proportions as TAN.

## Distributing other organic fertilisers

Non-manure organic fertilisers (`organic_fertiliser_<TAN/N/P/K/C>`, calculated by `WasteAndCircularity`) are distributed by `.distribute_organic_fertilisers()` in a similar way, based on TAN requirements remaining after manure application (including its long-term plant-available N, see above):

1. Organic fertiliser generated in a region is distributed to organic crops in that region, based on regional TAN requirements.
2. Organic fertiliser remaining after step 1 is distributed to organic crops nationally, in proportion to remaining TAN requirements.
3. Organic fertiliser remaining in a region after steps 1-2 is distributed to conventional crops in that region, based on regional TAN requirements.
4. Organic fertiliser remaining after step 3 is distributed to conventional crops nationally, in proportion to remaining TAN requirements.

Steps 1-2 are skipped if there is no organic production. As for manure, TAN available to spread is first increased by a share (set by `N_resid_organic`) of the non-TAN N in the organic fertiliser, and N, P, K and C are allocated in the same proportions as TAN once the TAN distribution is resolved.

```mermaid
---
title: Part 1
---

{{ mermaid_init() }}

graph TD
  subgraph I[" "]
    I1["<b>DemandAndConversions</b>"]:::mod_main
    I2["<b>Regions</b>"]:::mod_main
    I3["<b>CropProduction</b>"]:::mod_main
    I4["pd.Series(<b>AnimalHerd</b>)"]:::mod_main
    I5["<b>WasteAndCircularity</b>"]:::mod_main
    I6["<b>CoverCropsMgmt</b> (optional)"]:::mod_mgmt
  end

  I --> A

  A["<b>PlantNutrientMgmt</b>"]:::mod_mgmt

  A --> C[".calculate()"]:::method

  %% -------------------------
  %% calculate_TAN_req
  %% -------------------------
  C ---> TANR[".calculate_TAN_req()"]:::method

  TANR_DI["<b>CropProduction.data_attr.</b>
  area, harvest, production_per_use"]:::data --> TANR
  TANR_P["<b>PlantNutrientMgmt.par.</b>
  N_rec_a, N_rec_b, N_rec_m, N_resid_crop"]:::param --> TANR
  TANR_CC["<b>CoverCropsMgmt</b> (optional)
  .get_residual_N()"]:::mod_mgmt --> TANR

  TANR ----> TANR_DO["<b>CropProduction.data_attr.</b>
  fertiliser.TAN_req"]:::data

  %% -------------------------
  %% calculate_PK_req (sequence only, no data dependency on TANR)
  %% -------------------------

  TANR -..-> PKR[".calculate_PK_req(element)"]:::method

  PKR_DI1["<b>CropProduction.data_attr.</b>
  area, harvest"]:::data --> PKR
  PKR_DI2["<b>Regions.data_attr.</b>
  soil_P/K_class"]:::data --> PKR
  PKR_P["<b>PlantNutrientMgmt.par.</b>
  P/K_rec_a, P/K_rec_m, P/K_adj"]:::param --> PKR

  PKR --> PKR_DO["<b>CropProduction.data_attr.</b>
  fertiliser.P/K_req"]:::data

  %% -------------------------
  %% distribute_manure
  %% -------------------------
  
  TANR_DO --> DM[".distribute_manure()"]:::method

  DM_DI1["<b>AnimalHerd.data_attr.</b>
  manure.TAN/N/P/K/C_to_spread"]:::data --> DM

  DM_DI2["<b>CropProduction.data_attr.</b>
  area, production_per_use"]:::data --> DM
  
  DM_P["<b>PlantNutrientMgmt.par.</b>
  N_resid_manure, manure_TAN_max"]:::param --> DM

  DM --> DM_DO["<b>CropProduction.data_attr.</b>
  fertiliser.manure_TAN/N/P/K/C"]:::data

  %% -------------------------
  %% distribute_organic_fertilisers
  %% -------------------------
  DM_DO --> DOF[".distribute_organic_fertilisers()"]:::method
  TANR_DO2["<b>CropProduction.data_attr.</b>
  fertiliser.TAN_req"]:::data --> DOF

  DOF_DI1["<b>WasteAndCircularity.data_attr.</b>
  organic_fertiliser_TAN/N/P/K/C"]:::data --> DOF
  DOF_P["<b>PlantNutrientMgmt.par.</b>
  N_resid_manure, N_resid_organic"]:::param --> DOF

  DOF --> DOF_DO["<b>CropProduction.data_attr.</b>
  fertiliser.organic_TAN/N/P/K/C"]:::data

{{ mermaid_style() }}
```

```mermaid
---
title: Part 2
---

{{ mermaid_init() }}

graph TD

  %% -------------------------
  %% calculate_mineral_NPK_application (input from Part 1)
  %% -------------------------
  CP_DO["<b>CropProduction.data_attr.</b>
  fertiliser.TAN/P/K_req
  fertiliser.manure_TAN/N/P/K/C
  fertiliser.organic_TAN/N/P/K/C"]:::data --> NPKAP[".calculate_mineral_NPK_application(element)"]:::method
  NPKAP_P["<b>PlantNutrientMgmt.par.</b>
  mineral_N/P/K_fertiliser_share,
  N_resid_manure, N_resid_organic"]:::param --> NPKAP

  NPKAP --> NPKAP_DO["<b>CropProduction.data_attr.</b>
  fertiliser.mineral_N/P/K"]:::data

  %% -------------------------
  %% calculate_manure_application_area
  %% -------------------------
  NPKAP_DO --> MAA[".calculate_manure_application_area()"]:::method
  DM_DO2["<b>CropProduction.data_attr.</b>
  area
  fertiliser.manure_TAN/N/P/K/C
  fertiliser.organic_TAN/N/P/K/C"]:::data --> MAA
  MAA_P["<b>PlantNutrientMgmt.par.</b>
  min_share_manure_N_where_applied"]:::param --> MAA

  MAA --> MAA_DO["<b>CropProduction.data_attr.</b>
  fertiliser.manure_application_area"]:::data

  %% -------------------------
  %% calculate_N_application_losses (sequence only, no data dependency on MAA)
  %% -------------------------
  
  NPKAP_DO2["<b>CropProduction.data_attr.</b>
  fertiliser.mineral_N
  fertiliser.manure_TAN
  fertiliser.organic_TAN
  "]:::data --> NAL[".calculate_N_application_losses(of)"]:::method
  MAA -...-> NAL
  NAL_P["<b>PlantNutrientMgmt.par.</b>
  application_losses"]:::param --> NAL

  NAL --> NAL_DO["<b>CropProduction.data_attr.</b>
  fertiliser.mineral/manure/organic_N_application_loss"]:::data

  %% -------------------------
  %% calculate_N_soil_losses (sequence only, no data dependency on NAL)
  %% -------------------------
  
  NPKAP_DO3["<b>CropProduction.data_attr.</b>
  fertiliser.mineral_N
  fertiliser.manure_N
  fertiliser.organic_N"]:::data --> NSL[".calculate_N_soil_losses(of)"]:::method
  NAL -...-> NSL
  NSL_DI["<b>CropProduction.data_attr.</b>
  fertiliser.crop_residues_N,
  fertiliser.cover_crop_residues_N (optional)"]:::data --> NSL
  NSL_P["<b>PlantNutrientMgmt.par.</b>
  soil_losses_mineral/manure/organic/
  crop_residues/cover_crop_residues"]:::param --> NSL
  NSL_CC["<b>CoverCropsMgmt</b> (optional)
  .get_soil_loss_adjust()"]:::mod_mgmt --> NSL

  NSL --> NSL_DO["<b>CropProduction.data_attr.</b>
  fertiliser.mineral/manure/organic/crop_residues/cover_crop_residues_N_soil_loss"]:::data

{{ mermaid_style() }}
```

```mermaid
---
title: Part 3
---

{{ mermaid_init() }}

graph TD

  %% -------------------------
  %% calculate_organic_soil_losses
  %% -------------------------
  OSL_DI["<b>CropProduction.data_attr.</b>
  area"]:::data --> OSL[".calculate_organic_soil_losses()"]:::method
  OSL_REL["<b>Rel:</b> crop → land_use"]:::param --> OSL
  OSL_P1["<b>Regions.par.</b>
  share_org_soil"]:::param --> OSL
  OSL_P2["<b>PlantNutrientMgmt.par.</b>
  soil_losses_organic_soils"]:::param --> OSL

  OSL --> OSL_DO["<b>CropProduction.data_attr.</b>
  organic_soil_losses, organic_soil_area"]:::data

  %% -------------------------
  %% calculate_leaching_N (sequence only, no data dependency on OSL)
  %% -------------------------
  NPKAP_DO["<b>CropProduction.data_attr.</b>
  fertiliser.mineral_N
  fertiliser.manure_N
  fertiliser.organic_N"]:::data --> LEACH[".calculate_leaching_N()"]:::method
  OSL -...-> LEACH
  LEACH_DI["<b>CropProduction.data_attr.</b>
  fertiliser.crop_residues_N,
  fertiliser.cover_crop_residues_N (optional)"]:::data --> LEACH
  LEACH_REL["<b>Rel:</b> crop → land_use"]:::param --> LEACH
  LEACH_P["<b>PlantNutrientMgmt.par.</b>
  N_leaching_frac"]:::param --> LEACH
  LEACH_CC["<b>CoverCropsMgmt</b> (optional)
  .get_leach_adjust()"]:::mod_mgmt --> LEACH

  LEACH --> LEACH_DO["<b>CropProduction.data_attr.</b>
  fertiliser.leaching_N"]:::data

  %% -------------------------
  %% calculate_lime_application (sequence only, no data dependency on LEACH)
  %% -------------------------
  NPKAP_DO2["<b>CropProduction.data_attr.</b>
  fertiliser.mineral_N/P/K
  fertiliser.manure_N
  fertiliser.organic_N"]:::data --> LIME[".calculate_lime_application()"]:::method
  LEACH -...-> LIME
  LIME_DI["<b>CropProduction.data_attr.</b>
  harvest_dm, crop_residues_harvest"]:::data --> LIME
  LIME_REL["<b>Rel:</b> crop → land_use"]:::param --> LIME
  LIME_P["<b>PlantNutrientMgmt.par.</b>
  lime_effect_crops, lime_effect_crop_resids,
  lime_effect_manure, lime_effect_organic
  lime_effect_fertiliser_N/P/K,
  liming_agent_share, liming_agent_CaO_value"]:::param --> LIME

  LIME --> LIME_DO["<b>CropProduction.data_attr.</b>
  fertiliser.liming"]:::data

  %% -------------------------
  %% calculate_liming_emissions
  %% -------------------------
  LIME_DO --> LIMEEM[".calculate_liming_emissions()"]:::method
  LIMEEM_P["<b>PlantNutrientMgmt.par.</b>
  liming_emissions"]:::param --> LIMEEM

  LIMEEM --> LIMEEM_DO["<b>CropProduction.data_attr.</b>
  fertiliser.liming_emissions"]:::data

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.mgmt_modules.plant_nutrient_mgmt.PlantNutrientMgmt", "CIBUSmod/mgmt_modules/plant_nutrient_mgmt.py") }}
