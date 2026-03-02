# `PlantNutrientMgmt`

```mermaid
{{ mermaid_init() }}

graph TD

  %% -------------------------
  %% Main orchestration
  %% -------------------------
  A["**PlantNutrientMgmt**"]:::method
  C["**.calculate()**"]:::method

  TAN["**.calculate_TAN_req()**"]:::method
  PKP["**.calculate_PK_req(element='P')**"]:::method
  PKK["**.calculate_PK_req(element='K')**"]:::method

  DM["**.distribute_manure()**"]:::method
  DO["**.distribute_organic_fertilisers()**"]:::method

  MN["**.calculate_mineral_NPK_application(element='N')**"]:::method
  MP["**.calculate_mineral_NPK_application(element='P')**"]:::method
  MK["**.calculate_mineral_NPK_application(element='K')**"]:::method

  MAREA["**.calculate_manure_application_area()**"]:::method

  NAPP_MIN["**.calculate_N_application_losses(of='mineral_N')**"]:::method
  NAPP_MAN["**.calculate_N_application_losses(of='manure_TAN')**"]:::method
  NAPP_ORG["**.calculate_N_application_losses(of='organic_TAN')**"]:::method

  NSOIL_MIN["**.calculate_N_soil_losses(of='mineral_N')**"]:::method
  NSOIL_MAN["**.calculate_N_soil_losses(of='manure_N')**"]:::method
  NSOIL_ORG["**.calculate_N_soil_losses(of='organic_N')**"]:::method
  NSOIL_RES["**.calculate_N_soil_losses(of='crop_residues_N')**"]:::method
  NSOIL_COV["**.calculate_N_soil_losses(of='cover_crop_residues_N')**\n(if cover_crops_mgmt)"]:::method

  ORGSOIL["**.calculate_organic_soil_losses()**"]:::method
  LEACH["**.calculate_leaching_N()**"]:::method

  LIME["**.calculate_lime_application()**"]:::method
  LIME_EM["**.calculate_liming_emissions()**"]:::method

  A --> C
  C --> TAN



  %% -------------------------
  %% Inputs grouped by module
  %% -------------------------

  P_PAR_REQ["**PlantNutrientMgmt.par**
  N_rec_a, N_rec_b, N_rec_m,
  N_resid_crop,
  P_rec_a, P_rec_m, P_adj,
  K_rec_a, K_rec_m, K_adj,
  manure_TAN_max,
  N_resid_manure, N_resid_organic
  **Rel:** crop→crop_group"]:::param

  P_PAR_MIN["**Par (PlantNutrientMgmt.par)**
  mineral_N/P/K_fertiliser_share
  (fertiliser_type)"]:::param

  P_PAR_APP["**Par (PlantNutrientMgmt.par)**\napplication_losses\n(compound)"]:::param

  P_PAR_SOIL["**Par (PlantNutrientMgmt.par)**\nsoil_losses_mineral/manure/organic/crop_residues\n(compound)\nsoil_losses_organic_soils\n(compound)"]:::param

  P_PAR_LEACH["**Par (PlantNutrientMgmt.par)**\nN_leaching_frac\n(compound)\ncrop→land_use rel"]:::param

  P_PAR_LIME["**Par (PlantNutrientMgmt.par)**\nlime_effect_crops\nlime_effect_manure\nlime_effect_fertiliser_N/P/K\nliming_agent_share\nliming_agent_CaO_value\nliming_emissions\n(compound)"]:::param


  D_CROPS_REQ["**CropProduction.data_attr.**\narea\nharvest\nproduction_per_use"]:::data
  D_CROPS_LIME["**CropProduction.data_attr.**\nharvest_dm\ncrop_residues_harvest"]:::data

  D_REG_PK["**Regions.data_attr.**\nsoil_P_class\nsoil_K_class"]:::data
  D_REG_ORGSOIL["**Regions.par.**\nshare_org_soil"]:::param

  D_HERDS["**AnimalHerd.data_attr.**\nmanure.N/TAN/P/K/C_to_spread\n(+ MMS='grazing' in TAN_to_spread)"]:::data

  D_WASTE["**WasteAndCircularity.data_attr.**\norganic_fertiliser_N/TAN/P/K/C\n(+ treatment)"]:::data

  D_COVER["**CoverCropsMgmt (optional)**\nget_residual_N()\nget_soil_loss_adjust()\nget_leach_adjust()"]:::helper


  %% -------------------------
  %% TAN requirements
  %% -------------------------
  P_PAR_REQ --> TAN
  D_CROPS_REQ --> TAN
  D_COVER --> TAN

  TAN --> DA_TANREQ["**CropProduction.data_attr.**\nfertiliser.TAN_req"]:::data


  %% -------------------------
  %% P/K requirements (soil class dependent)
  %% -------------------------
  P_PAR_REQ --> PKP
  D_CROPS_REQ --> PKP
  D_REG_PK --> PKP

  P_PAR_REQ --> PKK
  D_CROPS_REQ --> PKK
  D_REG_PK --> PKK

  PKP --> DA_PKREQ["**CropProduction.data_attr.**\nfertiliser.P_req"]:::data
  PKK --> DA_PKREQ2["**CropProduction.data_attr.**\nfertiliser.K_req"]:::data


  %% -------------------------
  %% Manure distribution (writes manure_* to crops)
  %% -------------------------
  DA_TANREQ --> DM
  D_CROPS_REQ --> DM
  D_HERDS --> DM
  P_PAR_REQ --> DM

  DM --> DA_MAN["**CropProduction.data_attr.**\nfertiliser.manure_N/TAN/P/K/C"]:::data


  %% -------------------------
  %% Non-manure organic fertilisers distribution (writes organic_* to crops)
  %% -------------------------
  DA_TANREQ --> DO
  DA_MAN --> DO
  D_WASTE --> DO
  P_PAR_REQ --> DO

  DO --> DA_ORG["**CropProduction.data_attr.**\nfertiliser.organic_N/TAN/P/K/C"]:::data


  %% -------------------------
  %% Mineral fertiliser application (N/P/K) after manure+organic
  %% -------------------------
  P_PAR_MIN --> MN
  P_PAR_REQ --> MN
  DA_TANREQ --> MN
  DA_MAN --> MN
  DA_ORG --> MN

  P_PAR_MIN --> MP
  DA_PKREQ --> MP
  DA_MAN --> MP
  DA_ORG --> MP

  P_PAR_MIN --> MK
  DA_PKREQ2 --> MK
  DA_MAN --> MK
  DA_ORG --> MK

  MN --> DA_MIN["**CropProduction.data_attr.**\nfertiliser.mineral_N"]:::data
  MP --> DA_MIN2["**CropProduction.data_attr.**\nfertiliser.mineral_P"]:::data
  MK --> DA_MIN3["**CropProduction.data_attr.**\nfertiliser.mineral_K"]:::data


  %% -------------------------
  %% Manure application area (share-area based on manure N share)
  %% -------------------------
  P_PAR_REQ --> MAREA
  D_CROPS_REQ --> MAREA
  DA_MAN --> MAREA
  DA_ORG --> MAREA
  DA_MIN --> MAREA

  MAREA --> DA_MAREA["**CropProduction.data_attr.**\nfertiliser.manure_application_area"]:::data


  %% -------------------------
  %% N application losses (NH3 etc.) from TAN
  %% -------------------------
  P_PAR_APP --> NAPP_MIN
  DA_MIN --> NAPP_MIN
  NAPP_MIN --> DA_NAPP_MIN["**CropProduction.data_attr.**\nfertiliser.mineral_N_application_loss"]:::data

  P_PAR_APP --> NAPP_MAN
  DA_MAN --> NAPP_MAN
  NAPP_MAN --> DA_NAPP_MAN["**CropProduction.data_attr.**\nfertiliser.manure_N_application_loss\n(manure_TAN aggregated over species/breed/animal/MMS)"]:::data

  P_PAR_APP --> NAPP_ORG
  DA_ORG --> NAPP_ORG
  NAPP_ORG --> DA_NAPP_ORG["**CropProduction.data_attr.**\nfertiliser.organic_N_application_loss"]:::data


  %% -------------------------
  %% N soil losses (N2O/NOx etc.) from N sources
  %% -------------------------
  P_PAR_SOIL --> NSOIL_MIN
  DA_MIN --> NSOIL_MIN
  D_COVER --> NSOIL_MIN
  NSOIL_MIN --> DA_NSOIL_MIN["**CropProduction.data_attr.**\nfertiliser.mineral_N_soil_loss"]:::data

  P_PAR_SOIL --> NSOIL_MAN
  DA_MAN --> NSOIL_MAN
  D_COVER --> NSOIL_MAN
  NSOIL_MAN --> DA_NSOIL_MAN["**CropProduction.data_attr.**\nfertiliser.manure_N_soil_loss"]:::data

  P_PAR_SOIL --> NSOIL_ORG
  DA_ORG --> NSOIL_ORG
  D_COVER --> NSOIL_ORG
  NSOIL_ORG --> DA_NSOIL_ORG["**CropProduction.data_attr.**\nfertiliser.organic_N_soil_loss"]:::data

  P_PAR_SOIL --> NSOIL_RES
  D_CROPS_REQ --> NSOIL_RES
  D_COVER --> NSOIL_RES
  NSOIL_RES --> DA_NSOIL_RES["**CropProduction.data_attr.**\nfertiliser.crop_residues_N_soil_loss"]:::data

  D_COVER --> NSOIL_COV
  P_PAR_SOIL --> NSOIL_COV
  NSOIL_COV --> DA_NSOIL_COV["**CropProduction.data_attr.**\nfertiliser.cover_crop_residues_N_soil_loss"]:::data


  %% -------------------------
  %% Losses from organic soils + organic soil area
  %% -------------------------
  P_PAR_SOIL --> ORGSOIL
  D_REG_ORGSOIL --> ORGSOIL
  D_CROPS_REQ --> ORGSOIL

  ORGSOIL --> DA_ORGSOIL["**CropProduction.data_attr.**\norganic_soil_losses\norganic_soil_area"]:::data


  %% -------------------------
  %% N leaching
  %% -------------------------
  P_PAR_LEACH --> LEACH
  D_COVER --> LEACH
  DA_MIN --> LEACH
  DA_MAN --> LEACH
  DA_ORG --> LEACH

  LEACH --> DA_LEACH["**CropProduction.data_attr.**\nfertiliser.leaching_N"]:::data


  %% -------------------------
  %% Lime application + liming emissions
  %% -------------------------
  P_PAR_LIME --> LIME
  D_CROPS_LIME --> LIME
  DA_MAN --> LIME
  DA_MIN --> LIME

  LIME --> DA_LIME["**CropProduction.data_attr.**
  fertiliser.liming"]:::data

  P_PAR_LIME --> LIME_EM
  DA_LIME --> LIME_EM
  LIME_EM --> DA_LIME_EM["**CropProduction.data_attr.**
  fertiliser.liming_emissions"]:::data

{{ mermaid_style() }}
```