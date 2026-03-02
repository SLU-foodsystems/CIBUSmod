# `ManureMgmt`

The `ManureMgmt` module estimates manure excretion (VS,N,P,K), losses in stables and storage, and ammount of manure available for field application. It also handles the distribution of manure across manure management systems (`MMS`). VS excretion is estimated through an energy-balance approach, while N,P,K excretion is either estimated through a mass balance or using fixed factors. The mass balance is used if `NPK_excretion_from_balance == True` and only for `AnimalHerd`s that store the `lwg` data attribute.

```mermaid
{{ mermaid_init() }}

graph TD

  %% -------------------------
  %% Main orchestration
  %% -------------------------
  I1["pd.Series(**AnimalHerd**)"]:::mod_main --> A
  I2["**FeedMgmt**"]:::mod_mgmt --> A
  A["**ManureMgmt**"]:::mod_mgmt
  C["**.calculate()**"]:::method

  S["
  **<u>Settings</u>**
  MMS_grazing_from_feed = False
  NPK_excretion_from_balance = True
  "]:::settings

  MMS["**.calculate_MMS_shares()**"]:::method
  BED["**.calculate_bedding_material_use()**"]:::method

  VSX["**.calculate_VS_excretion()**"]:::method
  VSL["**.calculate_VS_losses()**"]:::method

  NPKX_BAL["**.calculate_NPK_excretion(element)**
  Balance calc for AnimalHerds with 'lwg'
  if 'NPK_excretion_from_balance'"]:::method
  NPKX_FACT["**.calculate_NPK_excretion(element)**
  Excretion factors used for AnimalHerds without 'lwg'"]:::method

  NPKL["**.calculate_NPK_losses(element)**"]:::method

  S --> A
  A --> C
  C ---> MMS

  %% -------------------------
  %% MMS shares (manure management systems)
  %% -------------------------
  P_MMS1["**Par:** mms_share, share_feed_on_pasture"]:::param --> MMS
  P_MMS2["**AnimalHerd.par**
  grazing_period, indoors_during_grazing"]:::param --> MMS
  D_MMS["**AnimalHerd.data_attr.**
  feed.consumption (if MMS_grazing_from_feed)"]:::data --> MMS

  MMS --> DA_MMS["**AnimalHerd.data_attr.**
  manure.mms_shares"]:::data

  %% -------------------------
  %% Bedding material use (DM + N/P/K in bedding)
  %% -------------------------
  P_BED1["**Par:** bedding_material_use"]:::param --> BED
  P_BED2["**FeedMgmt.par.**
  feed_composition (feed_par = DM,N,P,K)"]:::param --> BED

  DA_MMS --> BED
  D_BED["**AnimalHerd.data_attr.**
  heads"]:::data --> BED

  BED --> DA_BED["**AnimalHerd.data_attr.**
  bedding_material, bedding_material_N/P/K"]:::data


  %% -------------------------
  %% VS excretion (distributed across MMS)
  %% -------------------------
  P_VSX["**Par:** UE_of_GE"]:::param --> VSX

  D_VSX["**AnimalHerd.data_attr.**
  feed.consumption
  feed.ration_GE/DE/AME(poultry)/ASH"]:::data --> VSX
  DA_MMS ---> VSX

  VSX --> DA_VSX["**AnimalHerd.data_attr.**
  manure.VS_excr"]:::data


  %% -------------------------
  %% VS losses + carbon to spread + to treatment
  %% -------------------------
  P_VSL["**Par:** off-farm_treatment,
  methane_B0, methane_MCF,
  manure_VS_C, C_loss"]:::param --> VSL

  DA_VSX --> VSL

  VSL --> DA_VSL["**AnimalHerd.data_attr.manure.**
  VS_loss, C_to_spread
  VS_to_treatment, B0_to_treatment, C_to_treatment
  "]:::data

  %% -------------------------
  %% N/P/K excretion (two alternative pathways)
  %% -------------------------

  NPKX_BAL --> DA_NPKX["**AnimalHerd.data_attr.manure.**
  N/P/K_excr"]:::data
  NPKX_FACT --> DA_NPKX

  %% Balance
  P_NPKX_BAL["**Par:** manure_excr_N/P/K, N/P/K_in_LW, N/P/K_in_prod"]:::param --> NPKX_BAL
  D_NPKX_BAL["**AnimalHerd.data_attr..**
  heads, production, lwg,
  feed.consumption, feed.ration_N/P/K, feed.feeding_losses
  milk_to_calves (optional)"]:::data --> NPKX_BAL
  DA_BED --> NPKX_BAL
  DA_MMS --> NPKX_BAL
  %% Factors
  P_NPKX_FACT["**Par:** manure_excr_N/P/K"]:::param --> NPKX_FACT
  DA_MMS --> NPKX_FACT
  D_NPKX_FACT["**AnimalHerd.data_attr..**
  heads"]:::data --> NPKX_FACT


  %% -------------------------
  %% N/P/K losses + to spread + to treatment (+ TAN for N)
  %% -------------------------
  P_NPKL["**Par:** loss_stable, loss_storage,
  TAN_share, loss_storage_of_TAN,
  off-farm_treatment"]:::param --> NPKL

%%   DA_MMS --> NPKL
  DA_NPKX --> NPKL

  NPKL --> DA_NPKL["**AnimalHerd.data_attr.manure.**
  N/P/K_loss, P_loss, K_loss,
  N/TAN/P/K_to_spread
  N/TAN/P/K_to_treatment"]:::data

{{ mermaid_style() }}
```