# `ManureMgmt`

The `ManureMgmt` module estimates manure excretion (VS,N,P,K), losses in stables and storage, and ammount of manure available for field application. It also handles the distribution of manure across manure management systems (`MMS`). VS excretion is estimated through an energy-balance approach, while N,P,K excretion is either estimated through a mass balance or using fixed factors. The mass balance is used if `NPK_excretion_from_balance == True` and only for `AnimalHerd`s that store the `lwg` data attribute.

The `.calculate()` method runs all calculations in sequence for each `AnimalHerd`. First, `.calculate_MMS_shares()` distributes manure across `MMS` (e.g. grazing, liquid and solid manure), followed by `.calculate_bedding_material_use()` which estimates the amount and nutrient (N,P,K) content of bedding materials used. VS excretion and losses are then calculated with `.calculate_VS_excretion()` and `.calculate_VS_losses()`, the latter estimating CH<sub>4</sub> and CO<sub>2</sub> losses from stored manure with the IPCC Tier 2 method and the carbon remaining for spreading or off-farm treatment. Finally, `.calculate_NPK_excretion()` and `.calculate_NPK_losses()` are each called once per element (N, P and K) to estimate excretion and the corresponding losses in stables and storage, and the resulting amounts available for spreading or sent to off-farm treatment. For nitrogen, total ammoniacal nitrogen (TAN) to spread and to treatment is also calculated.

Three settings control parts of these calculations. `MMS_grazing_from_feed` (default `False`) determines how the share of manure deposited on pasture (`MMS = 'grazing'`) is estimated; if `True` it is derived from the share of dry matter feed intake from grazing instead of from the `grazing_period` and `indoors_during_grazing` parameters. `NPK_excretion_from_balance` (default `True`) determines whether N,P,K excretion is calculated from a mass balance, as described above, or from the fixed parameter `manure_excr_<N/P/K>`. `MMS_TAN_balance` (default `False`) determines how N losses in stables and storage are calculated; if `True`, TAN in excretion is calculated directly from the `TAN_share` parameter and all losses are assumed to only affect the TAN share of total N (using `loss_stable_of_TAN` and `loss_storage_of_TAN`), while if `False` losses are instead calculated from total N (`loss_stable`) and, in storage, either from total N or TAN (`loss_storage`/`loss_storage_of_TAN`) assuming a constant `TAN_share` throughout storage.

```mermaid
{{ mermaid_init() }}

graph TD

  %% -------------------------
  %% Main orchestration
  %% -------------------------
  I1["pd.Series(<b>AnimalHerd</b>)"]:::mod_main --> A
  I2["<b>FeedMgmt</b>"]:::mod_mgmt --> A
  A["<b>ManureMgmt</b>"]:::mod_mgmt
  C["<b>.calculate()</b>"]:::method

  S["
  <b><u>Settings</u></b>
  MMS_grazing_from_feed = False
  NPK_excretion_from_balance = True
  MMS_TAN_balance = False
  "]:::settings

  MMS["<b>.calculate_MMS_shares()</b>"]:::method
  BED["<b>.calculate_bedding_material_use()</b>"]:::method

  VSX["<b>.calculate_VS_excretion()</b>"]:::method
  VSL["<b>.calculate_VS_losses()</b>"]:::method

  NPKX_BAL["<b>.calculate_NPK_excretion(element)</b>
  Balance calc for AnimalHerds with 'lwg'
  if 'NPK_excretion_from_balance'"]:::method
  NPKX_FACT["<b>.calculate_NPK_excretion(element)</b>
  Excretion factors used for AnimalHerds without 'lwg'"]:::method

  NPKL["<b>.calculate_NPK_losses(element)</b>"]:::method

  S --> A
  A --> C
  C ---> MMS

  %% -------------------------
  %% MMS shares (manure management systems)
  %% -------------------------
  P_MMS1["<b>Par:</b> mms_share, share_feed_on_pasture"]:::param --> MMS
  P_MMS2["<b>AnimalHerd.par</b>
  grazing_period, indoors_during_grazing"]:::param --> MMS
  D_MMS["<b>AnimalHerd.data_attr.</b>
  feed.consumption (if MMS_grazing_from_feed)"]:::data --> MMS

  MMS --> DA_MMS["<b>AnimalHerd.data_attr.</b>
  manure.mms_shares"]:::data

  %% -------------------------
  %% Bedding material use (DM + N/P/K in bedding)
  %% -------------------------
  P_BED1["<b>Par:</b> bedding_material_use"]:::param --> BED
  P_BED2["<b>FeedMgmt.par.</b>
  feed_composition (feed_par = DM,N,P,K)"]:::param --> BED

  DA_MMS --> BED
  D_BED["<b>AnimalHerd.data_attr.</b>
  heads"]:::data --> BED

  BED --> DA_BED["<b>AnimalHerd.data_attr.</b>
  bedding_material, bedding_material_N/P/K"]:::data


  %% -------------------------
  %% VS excretion (distributed across MMS)
  %% -------------------------
  P_VSX["<b>Par:</b> UE_of_GE"]:::param --> VSX

  D_VSX["<b>AnimalHerd.data_attr.</b>
  feed.consumption
  feed.ration_GE/DE/AME(poultry)/ASH"]:::data --> VSX
  DA_MMS ---> VSX

  VSX --> DA_VSX["<b>AnimalHerd.data_attr.</b>
  manure.VS_excr"]:::data


  %% -------------------------
  %% VS losses + carbon to spread + to treatment
  %% -------------------------
  P_VSL["<b>Par:</b> off-farm_treatment,
  methane_B0, methane_MCF,
  manure_VS_C, C_loss"]:::param --> VSL

  DA_VSX --> VSL

  VSL --> DA_VSL["<b>AnimalHerd.data_attr.manure.</b>
  VS_loss, C_to_spread
  VS_to_treatment, B0_to_treatment, C_to_treatment
  "]:::data

  %% -------------------------
  %% N/P/K excretion (two alternative pathways)
  %% -------------------------

  NPKX_BAL --> DA_NPKX["<b>AnimalHerd.data_attr.manure.</b>
  N/P/K_excr"]:::data
  NPKX_FACT --> DA_NPKX

  %% Balance
  P_NPKX_BAL["<b>Par:</b> manure_excr_N/P/K, N/P/K_in_LW, N/P/K_in_prod"]:::param --> NPKX_BAL
  D_NPKX_BAL["<b>AnimalHerd.data_attr.</b>
  heads, production, lwg,
  feed.consumption, feed.ration_N/P/K, feed.feeding_losses
  milk_to_calves (optional)"]:::data --> NPKX_BAL
  DA_BED --> NPKX_BAL
  DA_MMS --> NPKX_BAL
  %% Factors
  P_NPKX_FACT["<b>Par:</b> manure_excr_N/P/K"]:::param --> NPKX_FACT
  DA_MMS --> NPKX_FACT
  D_NPKX_FACT["<b>AnimalHerd.data_attr.</b>
  heads"]:::data --> NPKX_FACT


  %% -------------------------
  %% N/P/K losses + to spread + to treatment (+ TAN for N)
  %% -------------------------
  P_NPKL["<b>Par:</b> loss_stable, loss_storage,
  TAN_share, loss_stable_of_TAN, loss_storage_of_TAN,
  off-farm_treatment"]:::param --> NPKL

%%   DA_MMS --> NPKL
  DA_NPKX --> NPKL

  NPKL --> DA_NPKL["<b>AnimalHerd.data_attr.manure.</b>
  N/P/K_loss, P_loss, K_loss,
  N/TAN/P/K_to_spread
  N/TAN/P/K_to_treatment"]:::data

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.mgmt_modules.manure_mgmt.ManureMgmt", "CIBUSmod/mgmt_modules/manure_mgmt.py") }}