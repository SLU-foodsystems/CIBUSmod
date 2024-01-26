<img src="figs/logo.png" height="100px">

# Users guide

1. [Introduction](#introduction) 
1. [Data structure](#data-structure)
    - [Setting data folder path](#setting-data-folder-path)
    - [Default data workbooks](#default-data-workbooks)
    - [Scenario data workbooks](#scenario-data-workbooks)
1. [Defining and running scenarios](#defining-and-running-scenarios)
1. Main modules
    - Regions
    - DemandAndConversions
    - CropProduction
    - AnimalHerd
1. Management (Mgmt) modules
    - FeedMgmt
    - ManureMgmt
    - PlantNutrientMgmt
    - MachineryAndEnergyMgmt
    - InputsMgmt
1. GeoDistributor
    - Constraints

# Introduction
Intro here

# Data structure
All data used to run the model and outputs produced are stored in a data folder with the folder structure shown below

```bash
 data
 | 
 ├── default
 │   ├── *.xlsx
 │   └── *.csv
 ├── ecoinvent
 │   └── *.xml
 ├── output
 │   └── *.sqlite
 ├── scenarios
 │   └── *.xlsx
 └── relation_tables.xlsx
```

## Setting data folder path
The path to the datafolder is set by initialising a new `Session` object

```python
import CIBUSmod as cm
my_session = cm.Session(
    name = 'name_of_session',
    data_path = '../path/to/data'
)
```

The `my_session` then connects to a SQLite database in `data/output/name_of_session.sqlite` or creates it if it does not already exist. This database file stores all scenario definitions and generated outputs

> **Note:** *The database files grows quite large in size so it may be a good idea to limit the number of scenarios+years contained in one session. If many scenarios have ben added/removed, running `my_session.clean()` will tidy up the database file and potentially save som space.*

## Default data workbooks
All data used to run the model (refered to as parameters) are stored in Excel wokrbooks in `data/default/`. This folder contains one Excel workbook for each CIBUSmod module. When a module is initialised it is done so with a `ParameterRetriever` object that is responsible for accessing parameters. The name defined for the `ParameterRetriever` object correspons to the name of the Excel workbook where it will access paramters.

Below the `Regions` module is initialised and uses parameters in `data/default/Regions.xlsx`

```python
regions = cm.Regions(
    par = cm.ParameterRetriever('Regions')
)
```

The Excel workbooks should contain one sheet named `default` where all data is stored. All other sheets in the workbook are ignored and may be used freely to e.g. document data collection. 

The first row in the `default` sheet contains column headings. When the `ParamterRetriever` reads the Excel file only columns with the headings `parameter`, `value` and all starting with `f_` are retained. Other columns can again be used for documentation. All rows that do not have anything in the `parameter` column are also skipped, allowing for e.g. headings separating diferent groups of parameters in the Excel sheet (see example below).

The `parameter` columns is used for the parameter names and the `value` columns for the corresponding value. A parameter value can only be a number (equations and references are OK) or the name of an external .csv file containing additional parameter values (more on that later).

All columns starting with `f_` are interpreted as "filter levels". When the model tries to access a parameter value it does so by providing a set of filters. The `ParamterRetriever` then tries to find the single paramter value that most closely matches the supplied filter. If a single value can't be returned, `NaN` is returned and a warning with some additional information is printed. 

For example, when the `CropProduction` module accesses the paramter `seed` (defining the seeding density in kg/ha) it does so with the filter levels `crop`, `prod_system`, `region` and `crop_prod` (e.g. `crop='Wheat, winter'`, `prod_system='conventional'`, `region='111'` and `crop_prod='wheat'`). Assuming that the parameter sheet looks like below the value `210` would be returned as `crop='Wheat, winter'`, `prod_system='conventional'` and `crop_prod='wheat'` represents a unique match as no other value is equally well defined (i.e. with the same number of matching filters).  As the `f_region` column was left blank for the rows representing 'Wheat, winter' the `region` filter level could be ignored.

<img src="figs/manual/default_data_example1.png"> \
*Example of default Excel data sheet*

However, trying to access the `seed` parameter for `crop='Rye'`,`prod_system='organic'`, `region='111'` and `crop_prod='rye'` would yield `NaN` and a warning since there are two equally well defined matches on either `crop`, `prod_system` and `crop_prod` or `crop`, `crop_prod` and `region`. This represents an error in the parameter Excel sheet and would need to be corrected there.

> **Tip:** *The filter levels used in the model when accessing different parameters are stored in the `.qry_log` attribute of each `PrameterRetriever` object.*

### Using external .csv-files
The Excel workbooks for default parameters can be extended with .csv-files. This is done by writing the file name of a .csv-file in the default data workbook under the `value` column instead of a parameter value. The csv files needs to be located in `data/default` (see example below). Filter values specified for that row are ignored, instead these need to be specified in the .csv-file. 

<img src="figs/manual/default_data_ref_csv.png"> \
*Example of how to read data from external .csv-file via default workbooks*

The .csv-files can be structured in two ways, either with *parameters as columns* or with *filter values as columns*. In the the first case (i.e. *parameters as columns*) the first row in the .csv-file is interpreted as column headings and any column heading preceded by `f_` is interpreted as a filter level in a similar way as in the default data workbooks. Other column headings are interpreted as parameter names (see example below). Only columns corresponding to the parameter names specified in the default workbook are retrieved (see example above where the same .csv-file is defined for multiple parameters).

<img src="figs/manual/default_data_csv_par.png"> \
*Example of .csv-file with **parameters as columns** structure*

The second structure (i.e. *filter values as columns*) is invoked by writing `cols_as_filter: <filter level>` in the first cell of the .csv-file, where `<filter level>` is a filter level name preceded by `f_`. The second row is then interpreted as column names and any column name preceded by `f_` are assumed to represent filter levels and all other columns are assumed to represent filter values on the level specified in the first cell of the .csv-file (see example below). The paramter name is infered from the parameter name stated in the default data workbook on the row refereing to the .csv-file.

<img src="figs/manual/default_data_csv_filter.png"> \
*Example of .csv-file with **filter values as columns** structure*

## Scenario data workbooks
Scenarios are defined in Excel wrokbooks located in `data/scenarios/`. When defining scenarios only parameters that are to be changed compared to the default values need to be specified. Any parameters not defined in a scenario workbook are retained with their default values. The scenario workbooks should contain one sheet per module where parameters are to be changed in the scenario (see example below). Each sheet needs to include the column headings `paramter` and `val_is` as well as at least one defined year (column headings starting with `y_`). They may also include column headings for filter levels (starting with `f_`).

<img src="figs/manual/scenario_data_example1.png"> \
*Example of scenario Excel data sheet*

Changes in parameter values can be specified in absolute or relative terms by writing `abs` or `rel` in the ` val_is` column, respectively. When specifying parameter values in relative terms a factor to be multiplied by the parameter's default value is specified. So, in the above example the yield of all crops are increased by 25% (factor 1.25) to 2050 compared to their respective default yields, except if `crop='Ley for fodder'` where the yield is increased by 30% (factor 1.3).

Scenario parameter values can be specified for any chosen years and the model will interpolate between specified years when running the scenario.

Parameter values to change in a scenario can be defined in more general terms than default parameters (i.e. applying to several default parameter values, such as in the case of yield above) but never more precise. So, if the default value for the parameter `seed` from the previous example was defined for the filter levels `crop`, `crop_prod` and `prod_system` a scenario can't change this parameter independently on the `region` level without first explicitly specifying this filter level in the default data workbook.

# Defining and running scenarios

To run a scenario defined in one or more scenario Excel workbooks it first needs to be added to the `Session` object. This is done via the method `.add_scenario()`, which takes five parameters; `name`, `years`, `scenario`, `modules` and `pars`.

```python
my_session.add_scenario(
    name = 'my_scenario',
    years = [2020, 2030, 2040, 2050],
    scenario = ['my_scn1', 'my_scn2'],
    modules = 'all',
    pars = 'all'
)
```

The `name` parameter gives the scenario a name which is what will be printed in output tables etc. and the `years` parameter specifies the years to be run. The 'scenario' parameter is the filename(s) (exuding the .xlsx extension) of the scenario Excel workbook(s) to use. If a list of multiple workbooks is given, as in the example above, these are handled in consecutive order. If multiple scenario workbooks change the same parameter only the last one in the list will have an effect. The parameters `modules` and `pars` controls for which modules parameter values should be updated and which parameters to update, respectively. Using the keyword 'all' means that all modules and parameters included in the scenario Excel workbooks will be updated. 
