2023-11-28
----------
- Supressed warning for :code:`Session.apply_solution()`
- Implemented SQLite database for storing outputs via :code:`Session`
- Updated handling of MMS='grazing'
- Implemented :code:`SheepHerd`
- Implemented calculation of crop requirements of P and K (data only for P so far) as well as calculation of P/K in manure and need for mineral P/K
- Added constraint C4 in :code:`GeoDistributor` to constrain the maximum share of animal sub-systems on national level
- Many big/small fixes

2023-11-17
----------
- Updated :code:`GeoDist.allocate_crop_production_per_use()` to account for different shares of semi-natural pasture in grazing for different animals. Assertions fail, need to check later
- Moved AnimalHerd modules to separate files
- Improved docstrings in especially GeoDistributor
- Implemented calculations of demand for and harvest of crop residues (i.e. straw)
- Implemented new :code:`DataAttr` class to keep track of units and other metadata in outputs and revised `Session` print to give usefull info
- Fixed misstake in :code:`data/scenarios/food_as_industry.xlsx` and added :code:`data/scenarios/no_cows.xlsx`
- Various minor fixes
