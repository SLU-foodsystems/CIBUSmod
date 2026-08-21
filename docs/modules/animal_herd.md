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

## Specific `AnimalHerd` child modules

### `CattleHerd`

### `PigHerd`

### `LayerHerd`

### `BroilerHerd`

### `SheepHerd`

### `GoatHerd`

### `HorseHerd`

### `AquacultureHerd`

### `FisheriesHerd`

## Helper functions

### `make_herds()`

{{ docstring("CIBUSmod.main_modules.animal_herd.make_herds", "CIBUSmod/main_modules/animal_herd.py") }}

### `concat_herds()`

{{ docstring("CIBUSmod.main_modules.animal_herd.concat_herds", "CIBUSmod/main_modules/animal_herd.py") }}