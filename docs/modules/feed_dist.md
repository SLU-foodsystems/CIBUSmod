# `FeedDistributor`

```mermaid
{{ mermaid_init() }}

flowchart TD
  subgraph I[" "]
    I1["**DemandAndConversions**"]:::mod_main
    I2["**Regions**"]:::mod_main
    I3["**CropProduction**"]:::mod_main
    I4["pd.Series(**AnimalHerd**)"]:::mod_main
  end

  A["**FeedDistributor**"]:::mod_opt

  I --> A

  A --> M[".make()"]:::method
  A ----> S[".solve()"]:::method

  %% -------------------------
  %% x0
  %% -------------------------

  M --> GX0[".get_x0()"]:::method

  %% -------------------------
  %% Scaling
  %% -------------------------

  M --> SF[".calculate_scaling_factors()"]:::method

  %% -------------------------
  %% Constraints
  %% -------------------------

  M --> MC[".make_C1() ... C16()"]:::method

  S --> CVX[".define_cvx_problem()"]:::method
  S --> OUT["**FeedDistributor.data_attr.**
  x_crops, x_animals, x_feeds"]:::data
  OUT --> AS[".apply_solution()"]:::method
  AS --> AHS["**AnimalHerd**.scale()"]:::mod_main
  AS --> CPS["**CropProduction**.scale()"]:::mod_main

{{ mermaid_style() }}
```

{{ docstring("CIBUSmod.optimisation.feed_dist.FeedDistributor", "CIBUSmod/optimisation/feed_dist.py") }}

## `.make()`

{{ docstring("CIBUSmod.optimisation.geo_dist.GeoDistributor.make", "CIBUSmod/optimisation/geo_dist.py") }}


## `.solve()`

{{ docstring("CIBUSmod.optimisation.geo_dist.GeoDistributor.solve", "CIBUSmod/optimisation/geo_dist.py") }}

## Constraints

### `.make_C1()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC1[".make_C1()"]:::method

  mGD[".get_demand()"]:::method

  mC1 ---> mA1_1[".make_A1_1()"]:::method
  mC1 ---> mA1_2[".make_A1_2()"]:::method
  mC1 ---> mA1_3[".make_A1_3()"]:::method

  dDAC["DemandAndConversions.data_attr.
  crop_prod_demand, animal_prod_demand"]:::data --> mGD -->
  mb1["**b1:** demand for crop_prod and animal_prod"]:::matrix

  dAH["AnimalHerd.data_attr.
  production"]:::data --> mA1_1
  --> A1_1["**A1<sub>1</sub>:** animal (x_ani) → animal_prod (national level)"]:::matrix

  dCP["CropProduction.data_attr.
  production, seed_demand"]:::data --> mA1_2
  --> A1_2["**A1<sub>2</sub>:** crop (x_crp) → crop_prod (national level)"]:::matrix

  pFM["FeedMgmt.par.
  feed_to_prod, share_domestic"]:::param --> mA1_3
  --> A1_3["**A1<sub>3</sub>:** feed (x_fds) → crop_prod demand (national level; negative)"]:::matrix
  
  C1["<u>Constraint C1</u>
  Main constraint to ensure that production exactly meets demand. Crop and animal products without any demand remain unconstrained.
  A2 @ x == b1
  "]:::const
  mb1 --> C1
  A1_1 --> C1
  A1_2 --> C1
  A1_3 --> C1

{{ mermaid_style() }}
```

### `.make_C2()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC2[".make_C2()"]:::method

  mC2 ----> mA2_1[".make_A2_1()"]:::method
  mC2 ----> mA2_2[".make_A2_2()"]:::method

  dCP["CropProduction.data_attr.
  production"]:::data --> mA2_1
  --> A2_1["**A2<sub>1</sub>:** crops (x_crp) → crop_prod supply (per prod_system, region)"]:::matrix

  pFM["FeedMgmt.par.
  feed_to_prod, share_domestic, share_regional"]:::param --> mA2_2
  --> A2_2["**A2<sub>2</sub>:** feeds (x_fds) → crop_prod demand
  (domestic × regional share; negative)"]:::matrix

  C2["<u>Constraint C2</u>
  Constrain minimum regional share of feed crop-product demand.
  A2 @ x >= 0"]:::const
  A2_1 --> C2
  A2_2 --> C2

{{ mermaid_style() }}
```

### `.make_C3()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC3[".make_C3()"]:::method
  mC3 ---> mA3[".make_A3()"]:::method

  pREL["**Rel:** crop → land_use"]:::param --> mA3
  dMLU["Regions.data_attr.
  max_land_use"]:::data --> mA3
  --> A3["**A3:** crops (x_crp) → land_use totals (per region)"]:::matrix

  dMLU --> b3["**b3:** max_land_use ceilings (per land_use, region)"]:::matrix

  C3["<u>Constraint C3</u>
  Max land_use area per region.
  A3 @ x <= b3"]:::const
  A3 --> C3
  b3 --> C3

{{ mermaid_style() }}
```

### `.make_C4()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC4[".make_C4()"]:::method
  mC4 ---> mA4[".make_A4()"]:::method

  pH["AnimalHerd.par.
  max_share_sub_system"]:::param --> mA4
  --> A4["**A4:** animals (x_ani) → sub_system share constraint
  (national, per sp/br/ps/ss)"]:::matrix

  C4["<u>Constraint C4</u>
  Cap sub_system share of defining animal heads.
  A4 @ x <= 0"]:::const
  A4 --> C4

{{ mermaid_style() }}
```

### `.make_C5()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC5[".make_C5()"]:::method

  mC5 ---> mA5_1[".make_A5_1()"]:::method
  mC5 ---> mA5_2[".make_A5_2()"]:::method

  pREL["**Rel:** crop → crop_group"]:::param --> mA5_1
  dCP["CropProduction.data_attr.
  production"]:::data --> mA5_1
  --> A5_1["**A5<sub>1</sub>:** crops (x_crp) → crop_prod supply
  (per crop_group, region)"]:::matrix

  pFM["FeedMgmt.par.
  max_crop_in_crop_prod,
  feed_to_prod, share_domestic"]:::param --> mA5_2
  --> A5_2["**A5<sub>2</sub>:** feeds (x_fds) → crop_prod demand ceiling
  (max share by crop_group; negative)"]:::matrix

  C5["<u>Constraint C5</u>
  Cap max share of feed demand for a crop_prod
  supplied by a crop_group.
  A5 @ x <= 0"]:::const
  A5_1 --> C5
  A5_2 --> C5

{{ mermaid_style() }}
```

### `.make_C6()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC6[".make_C6()"]:::method
  mC6 ---> mA6min[".make_A6('min')"]:::method
  mC6 ---> mA6max[".make_A6('max')"]:::method

  pCPmin["**CropProduction.par.**
  min_in_rot"]:::param --> mA6min
  pREL["**Rel:** crop → land_use,
  **Rel:** crop → (crop_group)<sup>* </sup>
  <sup> *</sup>Can be defined on different levels"]:::param --> mA6min
  --> A6min["**A6<sub>min</sub>:** crops (x_crp) → crop_group rotation share
  (min)"]:::matrix

  pCPmax["**CropProduction.par.**
  max_in_rot"]:::param --> mA6max
  pREL --> mA6max
  --> A6max["**A6<sub>max</sub>:** crops (x_crp) → crop_group rotation share
  (max)"]:::matrix

  C6min["<u>Constraint C6_min</u>
  Min share in rotation.
  A6 @ x >= 0"]:::const
  C6max["<u>Constraint C6_max</u>
  Max share in rotation.
  A6 @ x <= 0"]:::const
  A6min --> C6min
  A6max --> C6max

{{ mermaid_style() }}
```

### `.make_C7()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC7[".make_C7()"]:::method

  pR["**Regions.par.**
  GDD5"]:::param --> mC7
  pCP["**CropProduction.par.**
  min_GDD5"]:::param --> mC7

  mC7 --> XSHORT["x_idx_short
  (keeps only crop-region combos where GDD5 >= min_GDD5)"]:::helper
  mC7 --> PRUNE["Prunes columns in all stored matrices
  (objective + constraints)
  (mat.M = mat.M[:, isel])"]:::helper

  C7["<u>Action C7</u>
  Not a solver constraint.
  Drops variables to prevent infeasible crop-region combos."]:::const
  XSHORT --> C7
  PRUNE --> C7

{{ mermaid_style() }}
```

### `.make_C8()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC8[".make_C8()"]:::method
  mC8 --> mA8[".make_A8()"]:::method

  pC8["Par: C8_crp, C8_ani, C8_fds,
  C8_rel, C8_tol"]:::param --> mC8
  pC8 --> mA8

  mA8 --> A8["**A8:** select variables by identity rows
  (per-index element constraint)"]:::matrix
  mC8 --> b8["**b8:** concatenated target values
  (from Series values)"]:::matrix

  C8["<u>Constraint C8</u>
  Flexible per-variable constraint(s).
  A8 @ x <rel> b8
  (== implemented as min/max with tol)"]:::const
  A8 --> C8
  b8 --> C8

{{ mermaid_style() }}
```

### `.make_C9()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC9[".make_C9()"]:::method
  mC9 --> mA9[".make_A9()"]:::method

  pC9["Par: C9_crp, C9_ani, C9_fds,
  C9_rel, C9_tol"]:::param --> mC9
  pC9 --> mA9

  mA9 --> A9["**A9:** 1×N selector row
  (sum over specified indices)"]:::matrix
  mC9 --> b9["**b9:** sum(target Series values)"]:::matrix

  C9["<u>Constraint C9</u>
  Flexible sum constraint(s).
  A9 @ x <rel> b9
  (== implemented as min/max with tol)"]:::const
  A9 --> C9
  b9 --> C9

{{ mermaid_style() }}
```

### `.make_C10()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC10[".make_C10()"]:::method
  mC10 ---> mA1_3[".make_A1_3(prod_type='by_prod')"]:::method

  dDC["DemandAndConversions.data_attr.
  by_products, by_prod_demand"]:::data --> mC10
  --> dD["D: net by-products available
  (by_products - food uses)"]:::data

  pFM["FeedMgmt.par.
  feed_to_prod, share_domestic"]:::param --> mA1_3
  --> A10["**A10:** feeds (x_fds) → by_prod demand
  (domestic; negative)"]:::matrix

  C10["<u>Constraint C10</u>
  By-product supply covers feed demand.
  A10 @ x + D >= 0"]:::const
  A10 --> C10
  dD --> C10

{{ mermaid_style() }}
```

### `.make_C11()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC11[".make_C11()"]:::method

  mC11 ---> mA11eq[".make_A11('share_in_ration')"]:::method
  mC11 ---> mA11min[".make_A11('min_share_in_ration')"]:::method
  mC11 ---> mA11max[".make_A11('max_share_in_ration')"]:::method

  pH["AnimalHerd.par.
  share_in_ration,
  min_share_in_ration,
  max_share_in_ration"]:::param --> mA11eq
  pH --> mA11min
  pH --> mA11max

  pFM["FeedMgmt.par.
  storage_losses, feeding_losses
  (used as losses_factor)"]:::param --> mA11eq
  pFM --> mA11min
  pFM --> mA11max

  mA11eq --> A11eq["**A11<sub>eq</sub>:** feeds (x_fds) → deviation from feed ration share equality
  (per feed, animal)
  A11 @ x == 0"]:::matrix
  mA11min --> A11min["**A11<sub>min</sub>:** feeds (x_fds) → deviation from feed ration share minimum
  (per feed, animal)
  A11 @ x >= 0"]:::matrix
  mA11max --> A11max["**A11<sub>max</sub>:** feeds (x_fds) → deviation from feed ration share maximum
  (per feed, animal)
  A11 @ x <= 0"]:::matrix

  C11["<u>Constraint C11</u>
  Ration share constraints (min / eq / max)
  Built as up to three constraints."]:::const
  A11eq --> C11
  A11min --> C11
  A11max --> C11

{{ mermaid_style() }}
```

### `.make_C12()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC12[".make_C12()"]:::method

  mC12 ---> mA12_1min[".make_A12_1(rel='min')"]:::method
  mC12 ---> mA12_1eq[".make_A12_1(rel='eq')"]:::method
  mC12 ---> mA12_1max[".make_A12_1(rel='max')"]:::method
  mC12 ---> mA12_2[".make_A12_2()"]:::method

  dH["AnimalHerd.data_attr.
  feed_req_min, feed_req_eq, feed_req_max"]:::data --> mA12_1min
  dH --> mA12_1eq
  dH --> mA12_1max

  mA12_1min --> A12_1min["**A12<sub>1,min</sub>:** animals (x_ani) → feed_par demand (min)
  (negative)"]:::matrix
  mA12_1eq --> A12_1eq["**A12<sub>1,eq</sub>:** animals (x_ani) → feed_par demand (eq)
  (negative)"]:::matrix
  mA12_1max --> A12_1max["**A12<sub>1,max</sub>:** animals (x_ani) → feed_par demand (max)
  (negative)"]:::matrix

  pFM["FeedMgmt.par.
  feed_composition,
  storage_losses, feeding_losses"]:::param --> mA12_2
  --> A12_2["**A12<sub>2</sub>:** feeds (x_fds) → feed_par supply
  (composition × loss_factor)"]:::matrix

  C12["<u>Constraint C12</u>
  Absolute nutrient/parameter requirements.
  Builds up to 3 constraints:
  A12<sub>min</sub> @ x >= 0
  A12<sub>eq</sub> @ x == 0
  A12<sub>max</sub> @ x <= 0"]:::const

  A12_1min --> C12
  A12_1eq --> C12
  A12_1max --> C12
  A12_2 --> C12

{{ mermaid_style() }}
```

### `.make_C13()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC13[".make_C13()"]:::method
  mC13 ----> mA13min[".make_A13(rel_type='min')"]:::method
  mC13 ----> mA13max[".make_A13(rel_type='max')"]:::method

  dH["AnimalHerd.data_attr.
  feed_req_of_DM_min,
  feed_req_of_DM_max"]:::data --> mA13min
  dH --> mA13max

  pFM["FeedMgmt.par.
  feed_composition,
  storage_losses, feeding_losses"]:::param ---> mA13min
  pFM ---> mA13max

  mA13min --> A13min["**A13<sub>min</sub>:** feeds (x_fds) → (feed_par - req_share_of_DM)
  × losses_factor
  (min)"]:::matrix
  mA13max --> A13max["**A13<sub>max</sub>:** feeds (x_fds) → (feed_par - req_share_of_DM)
  × losses_factor
  (max)"]:::matrix

  C13["<u>Constraint C13</u>
  Feed parameters as % of DM.
  A13<sub>min</sub> @ x >= 0
  A13<sub>max</sub> @ x <= 0"]:::const
  A13min --> C13
  A13max --> C13

{{ mermaid_style() }}
```

### `.make_C14()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC14[".make_C14()"]:::method

  mC14 ----> mb14cp[".make_b14('crop_prod')"]:::method
  mC14 ----> mb14by[".make_b14('by_prod')"]:::method
  mC14 ----> mA1_3cp[".make_A1_3(domestic=False, prod_type='crop_prod')"]:::method
  mC14 ----> mA1_3by[".make_A1_3(domestic=False, prod_type='by_prod')"]:::method

  pFM1["FeedMgmt.par.
  max_total_imported"]:::param --> mb14cp
  pFM1 --> mb14by

  mb14cp --> b14cp["**b14<sub>cp</sub>:** ceilings (tonnes→kg)
  (prod_system, crop_prod)"]:::matrix
  mb14by --> b14by["**b14<sub>by</sub>:** ceilings (tonnes→kg)
  (prod_system, by_prod)"]:::matrix

  pFM2["FeedMgmt.par.
  feed_to_prod, share_domestic
  (used to compute imported share when domestic=False)"]:::param --> mA1_3cp
  pFM2 --> mA1_3by

  mA1_3cp --> A14cp["**A14<sub>cp</sub>:** feeds (x_fds) → imported crop_prod volume"]:::matrix
  mA1_3by --> A14by["**A14<sub>bp</sub>:** feeds (x_fds) → imported by_prod volume"]:::matrix

  C14["<u>Constraint C14</u>
  Total imported feed does not exceed max_total_imported.
  A14 @ x + b14 >= 0"]:::const
  A14cp --> C14
  A14by --> C14
  b14cp --> C14
  b14by --> C14

{{ mermaid_style() }}
```

### `.make_C15()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC15[".make_C15()"]:::method
  mC15 ----> mA15_1[".make_A15_1()"]:::method
  mC15 ----> mA15_2[".make_A15_2()"]:::method
  mC15 ----> mA15_3[".make_A15_3()"]:::method

  dDC1["DemandAndConversions.data_attr.
  by_prod_per_animal_prod"]:::data --> mA15_1
  dAH["AnimalHerd.data_attr.
  production"]:::data --> mA15_1
  --> A15_1["**A15<sub>1</sub>:** animals (x_ani) → regionally generated by-products
  (from animal products)"]:::matrix

  dDC2["DemandAndConversions.data_attr.
  by_prod_per_crop_prod"]:::data --> mA15_2
  dCP["CropProduction.data_attr.
  (production via internal net_production helper)"]:::data --> mA15_2
  --> A15_2["**A15<sub>2</sub>:** crops (x_crp) → regionally generated by-products
  (from crop products)"]:::matrix

  pFM["FeedMgmt.par.
  feed_to_prod, share_domestic, share_regional"]:::param --> mA15_3
  --> A15_3["**A15<sub>3</sub>:** feeds (x_fds) → regional by-product demand
  (domestic × regional; negative)"]:::matrix

  C15["<u>Constraint C15</u>
  Generated by-products cover regional feed demand share.
  A15 @ x >= 0"]:::const
  A15_1 --> C15
  A15_2 --> C15
  A15_3 --> C15

{{ mermaid_style() }}
```

### `.make_C16()`
```mermaid
{{ mermaid_init() }}

flowchart TD

  mC16[".make_C16()"]:::method

  mC16 ----> mA16_1[".make_A16_1()"]:::method
  mC16 ----> mA16_2[".make_A16_2()"]:::method
  mC16 ----> mA1_3[".make_A1_3(prod_type='crop_resid')"]:::method

  dD["DemandAndConversions.data_attr.
  crop_resid_demand"]:::data --> mC16
  mC16 --> b16["**b16:** crop_resid demand for food/energy"]:::matrix

  dAH["AnimalHerd.data_attr.
  heads"]:::data --> mA16_1
  pMM["ManureMgmt.par.
  bedding_material_use"]:::param --> mA16_1
  pFM["FeedMgmt.par.
  feed_to_prod, share_domestic"]:::param ---> mA16_1
  --> A16_1["**A16<sub>1</sub>:** animals (x_ani) → crop_resid demand for bedding
  (negative)"]:::matrix

  dCP["CropProduction.data_attr.
  crop_residues (above ground)"]:::data --> mA16_2
  pCRM["CropResidueMgmt.par.
  crop_resid_harvestable"]:::param --> mA16_2
  --> A16_2["**A16<sub>2</sub>:** crops (x_crp) → harvestable crop residues supply"]:::matrix

  pFM ---> mA1_3
  --> A16_3["**A16<sub>3</sub>:** feeds (x_fds) → crop_resid demand
  (domestic; negative)"]:::matrix

  C16["<u>Constraint C16</u>
  Crop residues supply covers crop_resid_demand
  (incl. feed + bedding use).
  A16 @ x >= b16"]:::const
  A16_1 --> C16
  A16_2 --> C16
  A16_3 --> C16
  b16 --> C16

{{ mermaid_style() }}
```