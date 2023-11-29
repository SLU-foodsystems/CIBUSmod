<img src="figs/logo.png" height="100px">

# Manual

1. [Introduction](#introduction) 
1. [Data structure](#data-structure)
    - [Setting data folder path](#setting-data-folder-path)
    - [Default data sheets](#default-data-sheets)
    - [Scenario data sheets](#scenario-data-sheets)
1. Defining and running scenarios
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

## Default data sheets
All data used to run the model (refered to as parameters) are stored in Excel files in `data/default`. This folder contains one Excel file for each CIBUSmod module. When a module is initialised it is done so with a `ParameterRetriever` object that is responsible for accessing parameters. The name defined for the `ParameterRetriever` object correspons to the name of the Excel sheet where it will access paramters.

Below the `Regions` module is initialised and uses parameters in `data/default/Regions.xlsx`
```python
regions = cm.Regions(
    par = cm.ParameterRetriever('Regions')
)
```
The Excel files should contain one sheet named `default` where all data is stored. All other sheets in the file are ignored and may be used freely to e.g. document data collection. 

The first row in the `default` sheet contains column headings. When the `ParamterRetriever` reads the Excel file only columns with the headings `parameter`, `value` and all starting with `f_` are retained. Other columns can again be used for documentation. All rows that do not have anything in the `parameter` column are also skipped, allowing for e.g. headings separating diferent groups of parameters in the Excel sheet.

The `parameter` columns is used for the parameter names and the `value` columns for the corresponding value. A parameter value can only be a number or the name of an external .csv file containing additional parameter values (more on that later).

All columns starting with `f_` are interpreted as "filter levels". When the model tries to access a parameter value it does so by providing a set of filters. The `ParamterRetriever` then tries to find the single paramter value that most closely matches the supplied filter. If a single value can't be returned, `NaN` is returned and a warning with some additional information is printed. 

For example, when the `CropProduction` module accesses the paramter `seed` (defining the seeding density in kg/ha) it does so with the filter levels `crop`, `prod_system`, `region` and `crop_prod` (e.g. `crop='Wheat, winter', prod_system='conventional'`, `region='111'` and `crop_prod='wheat'`). Assuming that the parameter sheet looks like below the value `210` would be returned as `crop='Wheat, winter'`, `prod_system='conventional'` and `crop_prod='wheat'` represents a unique match as no other value is equally well defined (i.e. with the same number of matching filters).  As the `f_region` column was left blank for the rows representing 'Wheat, winter' the `region` filter level could be ignored ignored.

<img src="figs/manual/default_data_example1.png"> \
*Example of default Excel data sheet*

However, trying to access the `seed` parameter for `crop='Rye'`,`prod_system='organic'`, `region='111'` and `crop_prod='rye'` would yield `NaN` and a warning since there are two equally well defined matches on either `crop`, `prod_system` and `crop_prod` or `crop`, `crop_prod` and `region`. This represents an error in the parameter Excel sheet and would need to be corrected there.

> **Tip:** *The filter levels used in the model when accessing different parameters are stored in the `.qry_log` attribute of each `PrameterRetriever` object.*


## Scenario data sheets


