# `WasteAndCircularity`

{{ docstring("CIBUSmod.main_modules.waste_and_circularity.WasteAndCircularity", "CIBUSmod/main_modules/waste_and_circularity.py") }}

## Parameters

| Parameter name | Unit | Filter levels used for access | Description |
| --- | --- | --- | --- |
| `CH4_density` | kg/Nm3 | None | Methane density used to convert normal gas volume to mass. |
| `CH4_specific_energy` | kWh/kg | None | Energy content of methane. |
| `CO2_density` | kg/Nm3 | None | Carbon dioxide density used to convert normal gas volume to mass. |
| `biogas_CH4_frac` | vol. % | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type`, `energy_prod`, `energy_source` | Methane volume fraction of raw biogas. |
| `biogas_CH4_slip` | % of CH4 prod | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Share of produced methane leaking during biogas production. |
| `biogas_CH4_yield` | % | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Share of feedstock methane potential realised during anaerobic digestion. |
| `biogas_energy_use` | kWh/Nm3 CH4 | `region`, `feedstock`, `treatment`, `energy_source`, `feedstock_group`, `feedstock_type`, `energy_prod` | Energy input per normal cubic metre of methane produced. |
| `biogas_flare_frac` | % of CH4 prod | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Share of produced biogas sent to flaring. |
| `biogas_flare_slip` | % of CH4 to flare | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Share of flared methane escaping combustion. |
| `biogas_use_CH4_slip` | %; % of CH4 | `region`, `feedstock`, `treatment`, `energy_prod`, `feedstock_group`, `feedstock_type` | Methane leakage during each biogas use. |
| `biogas_use_energy_use` | kWh/kg CH4 (code basis) | `region`, `feedstock`, `treatment`, `energy_prod`, `energy_source`, `feedstock_group`, `feedstock_type` | Energy input per kg methane used, by energy source and energy product. |
| `biogas_use_share` | % | `region`, `feedstock`, `treatment`, `energy_prod`, `feedstock_group`, `feedstock_type` | Share of biogas allocated to each energy product/use. |
| `digestate_B0` | Nm3 CH4/kg VS | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type`, `energy_prod`, `energy_source` | Residual methane potential per unit digestate volatile solids. |
| `digestate_MCF` | % of B0 | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type`, `energy_prod`, `energy_source` | Share of residual methane potential released during digestate storage. |
| `digestate_TAN_share` | % of total N |`region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type`, `element`, `compound` | Share of digestate nitrogen present as total ammoniacal nitrogen. |
| `digestate_loss_storage` | % of total N; % of TAN; % of P; % of K | `region`, `feedstock`, `treatment`, `element`, `compound`, `feedstock_group`, `feedstock_type` | Storage losses of each digestate constituent/compound, relative to total nutrient or TAN as appropriate. |
| `digestate_share_as_fertiliser` | % | `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Share of stored digestate returned to land; the remainder goes to landfill. |
| `feedstock_B0` | Nm3 CH4/kg VS | `feedstock`, `feedstock_group`, `feedstock_type` | Maximum methane potential per unit feedstock volatile solids. |
| `feedstock_C` | kg C/kg VS | `feedstock`, `feedstock_group`, `feedstock_type` | Carbon content per unit feedstock volatile solids. |
| `feedstock_DM` | kg DM/kg | `feedstock`, `feedstock_group`, `feedstock_type` | Dry-matter fraction of feedstock fresh mass. |
| `feedstock_K` | kg K/kg DM | `feedstock`, `feedstock_group`, `feedstock_type` | Potassium content per unit feedstock dry matter. |
| `feedstock_N` | kg N/kg DM | `feedstock`, `feedstock_group`, `feedstock_type` | Nitrogen content per unit feedstock dry matter. |
| `feedstock_P` | kg P/kg DM | `feedstock`, `feedstock_group`, `feedstock_type` | Phosphorus content per unit feedstock dry matter. |
| `feedstock_VS` | kg VS/kg DM | `feedstock`, `feedstock_group`, `feedstock_type` | Volatile solids fraction of feedstock dry matter. |
| `incineration_efficiency` | kWh/kWh | `region`, `feedstock`, `treatment`, `energy_prod`, `feedstock_group`, `feedstock_type` | Fraction of incinerated energy recovered as each energy product. |
| `incineration_emissions` | kg/kWh | `region`, `feedstock`, `treatment`, `element`, `compound`, `feedstock_group`, `feedstock_type` | Emission of each compound per unit incinerated energy. |
| `incineration_heating_value` | kWh/kg VS | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Energy content per unit feedstock volatile solids incinerated. |
| `landfill_emissions` | kg/kg VS | `region`, `feedstock`, `treatment`, `element`, `compound`, `feedstock_group`, `feedstock_type` | Emission of each compound per unit feedstock volatile solids landfilled. |
| `treatment_share` | % | `region`, `feedstock`, `treatment`, `feedstock_group`, `feedstock_type` | Share of each waste/feedstock stream sent to each treatment. |
| `wastewater_emissions` | kg/kg N; kg/kg COD | `region`, `element`, `compound`, `energy_source` | Emission of each compound per unit incoming constituent. |
| `wastewater_energy_use` | kWh/kg N | `region`, `element`, `energy_source` | Energy use per unit incoming wastewater constituent, by energy source. |
| `wastewater_incoming_grey` | kg VS/pers/year; kg COD/pers/year; kg N/pers/year; kg P/pers/year; kg K/pers/year | `region`, `element` | Annual per-capita load of each constituent in grey water. |
| `wastewater_incoming_toilet` | kg VS/pers/year; kg COD/pers/year; kg N/pers/year; kg P/pers/year; kg K/pers/year | `region`, `element` | Annual per-capita load of each constituent in toilet water; dietary N, P and K override these loads when available. |
| `wastewater_input_use` | kg input/kg incoming constituent | `region`, `element`, `input` | Material input use per unit incoming wastewater constituent. |
| `wastewater_recovery_sludge` | % of incoming constituent | `region`, `element` | Share of incoming constituent recovered in sewage sludge. |
| `wastewater_share_connected` | % | `region` | Population share connected to wastewater treatment. |

`biogas_use_energy_use` is multiplied by methane mass in the code. Some default rows label it `kWh/kWh`; the table gives the basis used by the calculation. VS = volatile solids; COD = chemical oxygen demand; TAN = total ammoniacal nitrogen; B0 = maximum methane potential; MCF = methane conversion factor; Nm3 = normal cubic metre.
