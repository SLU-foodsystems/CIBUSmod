# `AnimalHerd`

Livestock production is managed through specific `AnimalHerd` modules (i.e. child classes of `AnimalHerd`) for each species and in some cases breed. Each production unit is defined by a `species`, `breed`, production system (`prod_system`), sub-system (`sub_system`), and `region`. The `AnimalHerd` modules calculates herd structure, i.e. the number of different `animal` categories in relation to a defining animal category (e.g. cows for the `CattleHerd` module).

The `.scale()` method scales all `AnimalHerd`´s data attributes with the `scalable=True` flag based on a new number of the defining animal category.

```mermaid
{{ mermaid_init() }}

flowchart TD

  %% -------------------------
  %% Main (module family)
  %% -------------------------

  A["**AnimalHerd**"]:::mod_main
  A --> C["**.calculate()**"]:::method
  A --> SC["**.scale()**"]:::method

  %% Typical upstream source of x0 and region structure
  R["**Regions**"]:::mod_main --> A


  %% -------------------------
  %% Calculate() – typical sequence
  %% -------------------------

  C --> HS["**.calculate_herd()**"]:::method
  FEED["**.calculate_feed_req()**"]:::method
  PROD["**.calculate_production()**"]:::method

  %% -------------------------
  %% Herd structure
  %% -------------------------
  P_HS["
  **AnimalHerd.par.**
  Many different parameters depending on animal species (and breed)
  "]:::param
  P_HS --> HS

  HS --> DA_HS["**AnimalHerd.data_attr.**
heads, lwg, slaughtered_n, lost_n, lost_lw"]:::data

  DA_HS ---> PROD

  %% -------------------------
  %% Feed requirements
  %% -------------------------

  DA_HS ---> FEED
  P_FEED["
  **AnimalHerd.par.**
  Different parameters depending on animal species (and breed)
  "]:::param --> FEED

  FEED --> DA_FEED["**AnimalHerd.data_attr.**
feed_req_eq, feed_req_min, feed_req_max, feed_req_of_DM_min, feed_req_of_DM_max"]:::data

  %% -------------------------
  %% Livestock product outputs
  %% -------------------------
  P_PROD["
  **AnimalHerd.par.**
  slaughter_weight, milk_prod, milk_loss, milk_protein, milk_fat
  "]:::param --> PROD

  PROD --> DA_PROD["**AnimalHerd.data_attr.**
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