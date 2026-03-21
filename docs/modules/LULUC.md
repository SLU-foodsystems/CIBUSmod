# LULUC modules

Currently only one module (`SoilC`) is included here, but additional modules are to be added to account for all land use and land use change related emissions associated with different scenarios.

## `SoilC`
**Module for estimating soil organic carbon stocks in mineral soils**

The `SoilC` module uses the Introductory Carbon Balance Model (ICBM; *Kätterer & Andrén, 2001*) to estimate changes in soil organic carbon (SOC) stocks in mineral soils. This module is run after running the main model and data has been stored in the `Session` database.

`SoilC` implements the step-wise ICBM equations for the young ($Y$) and old ($O$) soil carbon pools as presented in *Menichetti et al. (2024)*

$$
Y_{t+1} = (Y_t + I_t) e^{-k_y re_t} 
$$

$$
O_{t+1} = (O_t - h \frac{k_y (Y_t+I_t)}{k_o - k_y}) e^{-k_o re_t} + h \frac{k_y (Y_t+I_t)}{k_o - k_y} e^{-k_y re_t}
$$

where $I_t$ is carbon input to the soil at year $t$, $h$ is the humification coefficient for the present carbon input, $k_y$ and $k_o$ are the decay rates for the young and old carbon pool respectively, and $re_t$ is the environmental modifier at year $t$.

The starting values (i.e. at $t = 0$) for the young and old pool ($Y_0$ and $O_0$) are found by analytically solving the steady states ($Y^*$ and $O^*$) assuming constant $I=I_0$ and $r=re_0$ equal to the the first year in the modelled scenario.

$$
Y^* = \frac{I_0 \times e^{-k_y re_0}}{1 - e^{-k_y * re_0}}
$$

$$
O^* = h \frac{k_y \times I_0}{(k_o - k_y) (1 - e^{-k_y * re_0})} \times \frac{e^{-k_y * re_0} - e^{-k_o re_0}}{1 - e^{-k_o re_0}}
$$

The `SoilC` module is instantiated with a `ParameterRetriever` and a `Session` object containing the scenarios to be modelled. The ICBM calculations are performed by first calling the `.make_input()` method, which gathers all the required inputs from the `Session` object (i.e. cropland areas and carbon inputs), followed by the `.run_ICBM()` method.

```python
soil = cm.SoilC(
    par = cm.ParameterRetriever('SoilC'),
    session = my_session
)
soil.make_input()
soil.run_ICBM()
```

ICBM is run for each region and input separately on a per-hectare basis. Results are retrieved with the `.get_data()` method, where SOC stocks can be retrieved for separate inputs or aggregated to regional or national totals. The module can also calculate the CO<sub>2</sub> flux associated with changes in SOC stocks via the `.get_flux()` method.

{{ docstring("CIBUSmod.LULUC_modules.soil_C.SoilC", "CIBUSmod/LULUC_modules/soil_C.py") }}

!!! abstract "**References**"
    <a href="https://doi.org/10.1016/j.agee.2006.05.013" target="_blank" rel="noopener">
    Bolinder et al. (2007). An approach for estimating net primary productivity and annual carbon inputs to soil for common agricultural crops in Canada. https://doi.org/10.1016/j.agee.2006.05.013
    </a>

    <a href="https://doi.org/10.1016/S0304-3800(00)00420-8" target="_blank" rel="noopener">
    Kätterer & Andrén (2001). The ICBM family of analytically solved models of soil carbon, nitrogen and microbial biomass dynamics — descriptions and application examples. https://doi.org/10.1016/S0304-3800(00)00420-8
    </a>

    <a href="https://doi.org/10.1080/17583004.2024.2304749" target="_blank" rel="noopener">
    Menichetti et al. (2024). Bayesian calibration of the ICBM/3 soil organic carbon model constrained by data from long-term experiments and uncertainties of C inputs. https://doi.org/10.1080/17583004.2024.2304749
    </a>

### `.make_input()`

{{ docstring("CIBUSmod.LULUC_modules.soil_C.SoilC.make_input", "CIBUSmod/LULUC_modules/soil_C.py") }}

### `.run_ICBM()`

{{ docstring("CIBUSmod.LULUC_modules.soil_C.SoilC.run_ICBM", "CIBUSmod/LULUC_modules/soil_C.py") }}

### `.get_data()`

{{ docstring("CIBUSmod.LULUC_modules.soil_C.SoilC.get_data", "CIBUSmod/LULUC_modules/soil_C.py") }}

### `.get_flux()`

The flux ($E$) is calculated based on year-to-year changes in per-hectare SOC stock in each region multiplied by the total (crop)land use in the region to get the total flux of CO<sub>2</sub> to and from the atmosphere

$$
E_{i,t} = -(SOC_{i,t} - SOC_{i,t-1})A_{i,t} \times (\frac{44}{12})
$$

where $E_{i,t}$ is the flux of CO<sub>2</sub> (kg CO<sub>2</sub>) for region $i$ in year $t$, $SOC$ is the per-hectare SOC stock (kg C/ha), and $A$ is the (crop)land area. The total national flux can then be obtain by summing all regional fluxes.

!!! note
    Since the flux is calculated from per-hectare changes in SOC stocks in each region, and only after that multiplied by the area in each region, the flux will always be zero if per-hectare stocks are steady even if the national average SOC stock may change due to changes in regional areas (i.e. if area increase in a region with a high SOC stock, the national SOC stock will increase even if per-hectare stocks are constant across all regions).

{{ docstring("CIBUSmod.LULUC_modules.soil_C.SoilC.get_flux", "CIBUSmod/LULUC_modules/soil_C.py") }}