TODO
====

Validation and default parameters
---------------------------------
- DemandAndConversions: Missing demand for oats and barley
- DemandAndConversions: Check rapeseed
- DemandAndConversions: Check exports
- DemandAndConversions: Demand for Herbs
- DemandAndConversions: Check induced skim milk exports
- CropProduction: Check/add crop residue factors for all crops
- CropProduction: Fix yield 'ley for seed'
- CropProduction: Check if yields are missing for more crops
- CropProduction: Check 'Mixed cereals' -> korn+havre eller havre/korn+ärter
- PlantNutrientMgmt: Add K requirements
- MachineryAndEnergyMgmt: Check energy use in stables. Completely missing for layers and sheep
- MachineryAndEnergyMgmt: Fix operations in organic production
- ManureMgmt: Check N excretion... higher than NIR but seems to align fairly
  well with Dutch method. bulls and sows are high.
- CattleHerd: Check rations (e.g. grazing beef vs dairy steers)

Overarching
-----------
- Implement DataAttr.get() across modules
  
Session
-------
- Possibility to reorder scenarios

ParameterRetriever
------------------

Regions
-------
- (???) Jordarter enl. SMED? https://pub.epsilon.slu.se/30474/1/johnsson-h-et-al-20230323.pdf

DemandAndConversions
--------------------
- Handle demand for by-products (e.g. )
- Processing energy use per food item + energy source
- Transports. A generic factor for Swedish produced. Different factors for imported foods. Elin kollar på detta...

CropProduction
--------------
- Implement way to handle crops that are not harvested 'Fallow', 'Green manure', 'Ley not harvested'
- Implement module to handle cover crops

AnimalHerd
----------

FeedMgmt
--------
- Implement way to balance production of and demand for by-products
    - Adjust feed rations, imports/exports and/or by-products to biogas
- Handle by-products from feed production

ManureMgmt
----------
- Add bedding material to VS excretion (??)
- Handle manure that is not returned to cropland? (mainly horses?)

PlantNutrientMgmt
-----------------
- Implement alternative to use N2O EF from Rochette et al
- Improve leaching calculations (currently IPCC method and only for N)

MachineryAndEnergyMgmt
----------------------

WasteMgmt (new module to be created)
----------------------
- Calculate sewege sludge generation
- Food waste biogas + digestate
- Slaughter waste biogas + digestate
- Handle energy production from biogas incl. from manure
- Possible to implement innovative recycling technologies

GeoDistributor
--------------
- Implement constraint for min share in rotation
- Implement constraint for max oversupply of roughage
- Fix "allocate production per use meathods"
