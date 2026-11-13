# `AnimalHerd`

Livestock production is managed through specific `AnimalHerd` modules (i.e. child classes of `AnimalHerd`) for each species and in some cases breed. Each production unit is defined by a `species`, `breed`, production system (`prod_system`), sub-system (`sub_system`), and `region`. The `AnimalHerd` modules calculates herd structure, i.e. the number of different `animal` categories in relation to a defining animal category (e.g. cows for the `CattleHerd` module).

The `.scale()` method scales all `AnimalHerd`´s data attributes with the `scalable=True` flag based on a new number of the defining animal category.

```mermaid
{{ mermaid_init() }}

flowchart TD

  %% -------------------------
  %% Main (module family)
  %% -------------------------

  A["<b>AnimalHerd</b>"]:::mod_main
  A --> C["<b>.calculate()</b>"]:::method
  A --> SC["<b>.scale()</b>"]:::method

  %% Typical upstream source of x0 and region structure
  R["<b>Regions</b>"]:::mod_main --> A


  %% -------------------------
  %% Calculate() – typical sequence
  %% -------------------------

  C --> HS["<b>.calculate_herd()</b>"]:::method
  FEED["<b>.calculate_feed_req()</b>"]:::method
  PROD["<b>.calculate_production()</b>"]:::method

  %% -------------------------
  %% Herd structure
  %% -------------------------
  P_HS["
  <b>AnimalHerd.par.</b>
  Many different parameters depending on animal species (and breed)
  "]:::param
  P_HS --> HS

  HS --> DA_HS["<b>AnimalHerd.data_attr.</b>
heads, lwg, slaughtered_n, lost_n, lost_lw"]:::data

  DA_HS ---> PROD

  %% -------------------------
  %% Feed requirements
  %% -------------------------

  DA_HS ---> FEED
  P_FEED["
  <b>AnimalHerd.par.</b>
  Different parameters depending on animal species (and breed)
  "]:::param --> FEED

  FEED --> DA_FEED["<b>AnimalHerd.data_attr.</b>
feed_req_eq, feed_req_min, feed_req_max, feed_req_of_DM_min, feed_req_of_DM_max"]:::data

  %% -------------------------
  %% Livestock product outputs
  %% -------------------------
  P_PROD["
  <b>AnimalHerd.par.</b>
  slaughter_weight, milk_prod, milk_loss, milk_protein, milk_fat
  "]:::param --> PROD

  PROD --> DA_PROD["<b>AnimalHerd.data_attr.</b>
production"]:::data

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.main_modules.animal_herd.AnimalHerd", "CIBUSmod/main_modules/animal_herd.py") }}

## Parameters

### Shared parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `max_feed_par_of_DM` | feed_par unit/kg DM | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `feed_par` | Maximum concentration of a feed property in total ration dry matter. Used only with FeedDistributor (constraint 13). |
| `max_share_in_ration` | % | `species`, `breed`, `sub_system`, `prod_system`, `animal`, `feed` | Maximum feed inclusion share. Used only by FeedDistributor. |
| `min_feed_par_of_DM` | feed_par unit/kg DM | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `feed_par` | Minimum concentration of a feed property in total ration dry matter. Used only with FeedDistributor (constraint 13). |
| `min_share_in_ration` | % | `species`, `breed`, `sub_system`, `prod_system`, `animal`, `feed` | Minimum feed inclusion share. Used only by FeedDistributor. |
| `share_in_ration` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `feed` | Fixed feed inclusion share. Read by GeoDistFeedMgmt for prescribed rations and by FeedDistributor for fixed-share constraints. |

The two `*_share_in_ration` bounds are retrieved by FeedDistributor without a `region` filter. Its fixed `share_in_ration` query also omits `region`; GeoDistFeedMgmt includes it. The `*_feed_par_of_DM` parameters are optional, apply to any herd with suitable `feed_par` data, and are read by `AnimalHerd.calculate_feed_req()` even though the resulting bounds are enforced only by FeedDistributor. The same applies to species-specific nutrient bounds below. DM = dry matter; LW = live weight; CW = carcass weight; ME = metabolisable energy; NE = net energy; ECM = energy-corrected milk; AAT = amino acids absorbed in the small intestine; PBV = rumen protein balance.

## Specific `AnimalHerd` child modules

### `CattleHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `AAT_factor` | g AAT/MJ ME | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Minimum absorbable amino acids per unit metabolisable energy. Used only with FeedDistributor to constrain feed rations. |
| `AFC` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Age at first calving. |
| `birth_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Live body mass at birth. |
| `breeding_bulls_per_cow` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of breeding bulls kept per cow. |
| `calving_interval` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Time between successive calvings. |
| `energy_adjustment_addend` | MJ ME/head/day | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Daily energy adjustment added before multiplying by the energy adjustment factor. |
| `energy_adjustment_factor` | dimensionless | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Multiplier applied to calculated energy requirements, e.g. for activity or breed. |
| `energy_in_milk_to_calves` | MJ ME/kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy supplied per kg milk fed to calves. |
| `energy_share_before_weaning_from_milk` | % of ME | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of suckling young animals' energy requirements met by milk. |
| `gestation_energy_factor` | MJ ME/kg LW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy coefficient for gestation. |
| `grazing_period` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual grazing duration; read by ManureMgmt to allocate excretion to pasture. |
| `indoors_during_grazing` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of grazing-period time spent indoors; read by ManureMgmt. |
| `lactation_energy_factor` | MJ ME/kg ECM | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy coefficient for milk production/lactation. |
| `live_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Live body mass of the animal category. |
| `live_weight_first_calving` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Heifer live body mass at first calving. |
| `live_weight_slaughter` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Live body mass at slaughter. |
| `maintanance_energy_factor` | MJ ME/(day x kg LW^0.75) | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy coefficient for maintenance per unit metabolic live weight (parameter spelling follows the code). |
| `max_DMI` | kg DM/head/day | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Maximum daily dry-matter intake. Used only with FeedDistributor to constrain feed rations. |
| `max_PBV` | g PBV/head/day | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Additive upper bound for rumen protein balance per head/day. Used only with FeedDistributor. |
| `max_PBV_per_ME` | g PBV/MJ ME | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy-dependent upper bound for rumen protein balance. Used only with FeedDistributor. |
| `milk_fat` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Milk fat concentration, used to calculate energy-corrected milk. |
| `milk_loss` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of produced milk lost before sale. |
| `milk_prod` | kg/cow/year | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual milk production per adult female, including milk fed to young. |
| `milk_protein` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Milk protein concentration, used to calculate energy-corrected milk. |
| `min_PBV` | g PBV/head/day | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Additive lower bound for rumen protein balance per head/day. Used only with FeedDistributor. |
| `min_PBV_per_ME` | g PBV/MJ ME | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy-dependent lower bound for rumen protein balance. Used only with FeedDistributor. |
| `mortality` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `mortality_female_0towean` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of live-born female calves dying before weaning. |
| `mortality_female_0towean_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Mean age at death of females dying before weaning. |
| `mortality_female_weantoslaught` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Share of weaned females dying before slaughter. |
| `mortality_female_weantoslaught_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Mean age at death of females dying between weaning and slaughter. |
| `mortality_male_0towean` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of live-born male calves dying before weaning. |
| `mortality_male_0towean_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Mean age at death of males dying before weaning. |
| `mortality_male_weantoslaught` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Share of weaned males dying before slaughter. |
| `mortality_male_weantoslaught_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Mean age at death of males dying between weaning and slaughter. |
| `ratio_male_calf` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Male share of calves born. |
| `redist_female_calves` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Share of weaned female calves transferred from one production system to another. |
| `redist_male_calves` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Share of weaned male calves transferred from one production system to another. |
| `replacement_rate` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Annual share of adult breeding females replaced. |
| `replacement_rate_bulls` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Annual share of breeding bulls replaced. |
| `slaughter_age` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Age at slaughter. |
| `slaughter_share_as_calf` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Share of young cattle slaughtered as calves. |
| `slaughter_share_male_as_steers` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Share of male cattle reared as steers. |
| `slaughter_weight` | kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Carcass mass per slaughtered animal. |
| `stillborn_calf` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of calves stillborn. |
| `time_lag_replacement` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Delay between cow removal and first calving of its replacement. |
| `twin_birth` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of calvings producing twins. |
| `weaning_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `from_ps`, `to_ps` | Age at weaning. |

### `PigHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `age_at_first_farrowing` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Gilt age at first farrowing. |
| `birth_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass at birth. |
| `boars_per_sows+gilts` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Number of breeding boars per sow plus gilt. |
| `dead_per_litter` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Number of stillborn piglets per litter. |
| `feed_energy_per_growth` | MJ NE/kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Net feed energy required per kg live-weight gain. |
| `finishing_period` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Duration of the finishing stage. |
| `gestation_period` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Duration of pregnancy. |
| `growing_period` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Duration of the growing stage. |
| `litters_per_sow` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual litters produced per sow. |
| `live_per_litter` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Number of live-born piglets per litter. |
| `live_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass of the animal category. |
| `live_weight_delivery` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass when delivered from nursing to growing production. |
| `live_weight_slaughter` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass at slaughter. |
| `live_weight_weaning` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass at weaning. |
| `max_feed_par_per_NE` | feed_par unit/MJ NE | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `feed_par` | Maximum feed-property supply per unit net energy. Used only with FeedDistributor (constraint 12). |
| `min_feed_par_per_NE` | feed_par unit/MJ NE | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal`, `feed_par` | Minimum feed-property supply per unit net energy. Used only with FeedDistributor (constraint 12). |
| `mortality` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `mortality_0towean` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Share of live-born piglets dying before weaning. |
| `mortality_weantodelivery` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Share of weaned piglets dying before delivery. |
| `post_weaning_nursing_period` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Duration of nursing after weaning. |
| `replacement_rate` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Annual share of adult breeding females replaced. |
| `slaughter_weight` | kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Carcass mass per slaughtered animal. |
| `weaning_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at weaning. |

### `LayerHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `average_live_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Average live body mass used to calculate growth and mortality losses. |
| `chicks_per_breeding hen` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of chicks produced per breeding hen (parameter name includes a space). |
| `egg_production` | kg/head/year | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual egg mass produced per head. |
| `feed_per_head` | kg DM/head/year | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual feed dry-matter intake per head. |
| `mortality` | % of inserted | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `rejections_at_slaughter` | % of slaughtered | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of slaughtered animals rejected for food production. |
| `roosters_per_breeding_hen` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of breeding roosters per breeding hen. |
| `slaughter_weight` | kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Carcass mass per slaughtered animal. |
| `time_in_hen_house` | weeks | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Duration of the laying period in the hen house. |
| `weeks_in_stage` | weeks | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Time spent in an animal production stage. |

### `BroilerHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `eggs_per_breeding_hen` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Lifetime number of hatching eggs per breeding hen. |
| `feed_conversion_ratio` | kg/kg LWG | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Feed mass per kg live-weight gain. |
| `feed_per_animal` | kg/animal | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Lifetime feed per inserted breeding hen, including the associated rooster feed. |
| `hens_per_rooster` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Number of breeding hens per rooster. |
| `live_weight_per_CW` | kg LW/kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Ratio of live body mass to carcass mass. |
| `mortality` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `rejections_at_slaughter` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of slaughtered animals rejected for food production. |
| `rounds_per_year` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Number of broiler production cycles per year. |
| `slaughter_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at slaughter. |
| `slaughter_weight` | kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Carcass mass per slaughtered animal. |
| `start_weight` | kg LW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass when entering the production stage. |

### `SheepHerd`

#### Parameters

Feed requirements use either the prescribed DM intake parameters or the optional ME equations when the necessary energy coefficients are present.

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `age_at_first_lambing` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at first lambing. |
| `birth_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass at birth. |
| `energy_share_before_weaning_from_milk` | % of ME | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of suckling young animals' energy requirements met by milk. |
| `feed_per_head` | kg DM/head/year | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual feed dry-matter intake per head. |
| `feed_per_lifetime` | kg DM | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Lifetime feed dry-matter requirement per lamb. |
| `fertility` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Share of adult females giving birth. |
| `gestation_energy_add` | MJ ME/pregnancy | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy requirement per pregnancy. |
| `growth_energy_factor` | MJ ME/(kg LW^0.75 x kg gain) | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy coefficient for live-weight gain. |
| `lactation_energy_factor` | MJ ME/head/day of lactation | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy coefficient for milk production/lactation. |
| `lambs_per_ewe` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of lambs per lambing ewe. |
| `live_weight_per_CW` | kg LW/kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Ratio of live body mass to carcass mass. |
| `maintanance_energy_factor` | MJ ME/(day x kg LW^0.75) | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy coefficient for maintenance per unit metabolic live weight (parameter spelling follows the code). |
| `max_share_sub_system` | % | `species`, `breed`, `sub_system`, `prod_system` | Maximum share of a herd sub-system within its species/breed/production system; read by the distributors. |
| `mortality` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `rams_per_ewe` | nr | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of breeding rams per ewe. |
| `replacement_rate` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Annual share of adult breeding females replaced. |
| `replacement_rate_rams` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Annual share of breeding rams replaced. |
| `slaughter_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at slaughter. |
| `slaughter_weight` | kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Carcass mass per slaughtered animal. |
| `weaning_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at weaning. |

### `GoatHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `age_at_first_kidding` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at first kidding. |
| `birth_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass at birth. |
| `bucks_per_doe` | bucks/doe | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of breeding bucks per doe. |
| `energy_in_milk_to_kids` | MJ ME/kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy supplied per kg milk fed to goat kids. |
| `energy_share_before_weaning_from_milk` | % of ME | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of suckling young animals' energy requirements met by milk. |
| `fertility` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of adult females giving birth. |
| `gestation_energy_add` | MJ ME/pregnancy | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy requirement per pregnancy. |
| `growth_energy_factor` | MJ ME/kg gain | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy coefficient for live-weight gain. |
| `kids_per_doe` | kids/kidding doe | `species`, `breed`, `sub_system`, `prod_system`, `region` | Number of kids per kidding doe. |
| `lactation_energy_factor` | MJ ME/kg ECM | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy coefficient for milk production/lactation. |
| `live_weight_per_CW` | kg LW/kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Ratio of live body mass to carcass mass. |
| `maintanance_energy_factor` | MJ ME/(day x kg LW^0.75) | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy coefficient for maintenance per unit metabolic live weight (parameter spelling follows the code). |
| `milk_fat` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Milk fat concentration, used to calculate energy-corrected milk. |
| `milk_loss` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of produced milk lost before sale. |
| `milk_prod` | kg/doe/year | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual milk production per adult female, including milk fed to young. |
| `milk_protein` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Milk protein concentration, used to calculate energy-corrected milk. |
| `mortality` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `replacement_rate` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Annual share of adult breeding females replaced. |
| `replacement_rate_bucks` | % | `species`, `breed`, `sub_system`, `prod_system`, `region` | Annual share of breeding bucks replaced. |
| `slaughter_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at slaughter. |
| `slaughter_weight` | kg CW | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Carcass mass per slaughtered animal. |
| `surplus_kids_culled` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of surplus kids culled rather than reared. |
| `weaning_age` | days | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Age at weaning. |

### `HorseHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `energy_adjustment_factor` | dimensionless | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Activity energy increment relative to maintenance energy. |
| `foals_per_year` | foals/mare/year | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual foals per breeding mare. |
| `gestation_energy_factor` | fraction of maintenance energy per foal | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy coefficient for gestation. |
| `grazing_period` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Annual grazing duration; read by ManureMgmt to allocate excretion to pasture. |
| `indoors_during_grazing` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of grazing-period time spent indoors; read by ManureMgmt. |
| `lactation_energy_factor` | fraction of maintenance energy per foal | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Energy coefficient for milk production/lactation. |
| `live_weight` | kg | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Live body mass of the animal category. |
| `maintenance_energy_factor` | MJ ME per day/kg LW^0.75 | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Metabolisable energy coefficient for maintenance per unit metabolic live weight. |
| `max_DMI` | kg DM/head/day | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Maximum daily dry-matter intake. Used only with FeedDistributor to constrain feed rations. |
| `share_of_horses` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Share of horses in each animal category. |

### `AquacultureHerd`

#### Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `age_delivery` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Juvenile fish age at delivery. |
| `eFCR` | kg DM/kg fish | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Economic feed conversion ratio: feed dry matter per unit harvested fish mass. |
| `months_production_cycle` | months | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Duration of the adult fish production cycle. |
| `mortality` | % | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Mortality share for the animal category (annual for adult stock, stage/cycle for young stock). |
| `slaughter_weight` | kg round weight/fish | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Round body mass per harvested adult fish. |
| `weight_delivery` | g/fish (feed calculation); see note | `species`, `breed`, `sub_system`, `prod_system`, `region`, `animal` | Juvenile fish body mass at delivery to adult production. |

### `FisheriesHerd`

#### Parameters

No species-specific parameters are queried: fish landings are represented directly by the production variable, and there are no herd-structure or feed requirements. The optional shared interface above remains inherited.

### `ReindeerHerd`

#### Parameters

`ReindeerHerd` is an abstract placeholder without implemented calculations or a species-specific parameter set.

## Helper functions

### `make_herds()`

{{ docstring("CIBUSmod.main_modules.animal_herd.make_herds", "CIBUSmod/main_modules/animal_herd.py") }}

### `concat_herds()`

{{ docstring("CIBUSmod.main_modules.animal_herd.concat_herds", "CIBUSmod/main_modules/animal_herd.py") }}