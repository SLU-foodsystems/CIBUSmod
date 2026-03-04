# `PlantNutrientMgmt`

```mermaid
{{ mermaid_init() }}

graph TD
  subgraph I[" "]
    I1["**DemandAndConversions**"]:::mod_main
    I2["**Regions**"]:::mod_main
    I3["**CropProduction**"]:::mod_main
    I4["pd.Series(**AnimalHerd**)"]:::mod_main
    I5["**WasteAndCircularity**"]:::mod_main
    I6["**CoverCropsMgmt** (optional)"]:::mod_mgmt
  end

  I --> A

  A["**PlantNutrientMgmt**"]:::mod_mgmt

  A --> C[".calculate()"]:::method

  %% Methods
  TANR[".calculate_TAN_req()"]:::method
  PKR[".calculate_PK_req()"]:::method
  DM[".distribute_manure()"]:::method
  DOF[".distribute_organic_fertilisers()"]:::method
  NPKAP[".calculate_mineral_NPK_application()"]:::method
  MAA[".calculate_manure_application_area()"]:::method
  NAL[".calculate_N_application_losses()"]:::method
  NSL[".calculate_N_soil_losses()"]:::method
  OSL[".calculate_organic_soil_losses()"]:::method
  LEACH[".calculate_leaching_N()"]:::method

  C ---> TANR
  C ---> PKR

  %% calculate_TAN_req
  TANR_DI["**CropProduction.data_attr.**
  area, harvest, production_per_use"]:::data --> TANR

  TANR_P["**PlantNutrientMgmt.par.**
  N_rec_a, N_rec_b, N_rec_m, N_resid_crop"]:::param --> TANR

  TANR --> TANR_DO["**CropProduction.data_attr.**
  fertiliser.TAN_req"]:::data
  
  %% calculate_PK_req
  PKR_DI1["**CropProduction.data_attr.**
  area, harvest, production_per_use"]:::data --> PKR

  PKR_DI2["**Regions.data_attr.**
  soil_P/K_class"]:::data --> PKR

  PKR_P["**PlantNutrientMgmt.par.**
  P/K_rec_a, P/K_rec_m, P/K_rec_adj"]:::param --> PKR

  PKR --> PKR_DO["**CropProduction.data_attr.**
  fertiliser.P/K_req"]:::data
  
  %% distribute_manure

  TANR_DO ---> DM

  DM_DI1["**AnimalHerd.data_attr.**
  manure.TAN/N/P/K/C_to_spread"]:::data --> DM
  DM_DI2["**CropProduction.data_attr.**
  area, production_per_use"]:::data --> DM

  DM_P["**PlantNutrientMgmt.par.**
  N_resid_manure, manure_TAN_max"]:::param --> DM

  DM --> DM_DO["**CropProduction.data_attr.**
  fertiliser.manure_TAN/N/P/K/C"]:::data

  %% distribute_organic_fertilisers
  DM_DO --> DOF
  TANR_DO --> DOF

  DOF_DI1["**WasteAndCircularity.data_attr.**
  organic_fertiliser_TAN/N/P/K/C"]:::data --> DOF

  DOF_P["**PlantNutrientMgmt.par.**
  N_resid_manure, N_resid_organic"]:::param --> DOF

  DOF --> DOF_DO["**CropProduction.data_attr.**
  fertiliser.organic_TAN/N/P/K/C"]:::data
  
  %% calculate_mineral_NPK_application
  TANR_DO --> NPKAP
  PKR_DO --> NPKAP
  %% back arrows???
  NPKAP --- DM_DO
  NPKAP --- DOF_DO 
  %% ???
  NPKAP_P["**PlantNutrientMgmt.par.**
  mineral_N/P/K_fertiliser_share, N_resid_manure"]:::param --> NPKAP
  
  

  NPKAP --> NPKAP_DO["**CropProduction.data_attr.**
  fertiliser.mineral_N/P/K"]:::data

  %% calculate_manure_application_area
  DM_DO --> MAA
  MAA_P["**PlantNutrientMgmt.par.**
  min_share_manure_N_where_applied"]:::param --> MAA
  MAA_DI["**CropProduction.data_attr.**
  area"]:::data --> MAA
  DOF_DO --> MAA
  NPKAP_DO --> MAA
  
  %% calculate_N_application_losses
  
  %% calculate_N_soil_losses
  
  %% calculate_organic_soil_losses
  
  %% calculate_leaching_N

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.mgmt_modules.plant_nutrient_mgmt.PlantNutrientMgmt", "CIBUSmod/mgmt_modules/plant_nutrient_mgmt.py") }}