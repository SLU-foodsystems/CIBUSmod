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
- CattleHerd: Too much grazing adjust feed rations
- FeedMgmt: Check share grazing from semi-natural grasslands
- PlantNutrientMgmt: Add K requirements
- MachineryAndEnergyMgmt: Check energy use in stables. Completely missing for layers and sheep
- MachineryAndEnergyMgmt: Fix operations in organic production
- ManureMgmt: Check N excretion... higher than NIR but seems to align fairly
  well with Dutch method. bulls and sows are high.

Overarching
-----------
- Implement DataAttr.get() across modules
- Session: Fix concat_herds() and .get_attr() for scalable=False. Implement in .get_attr() check scalable and do not aggregate if False
- Session: reorder scenarios

ParameterRetriever
------------------

Regions
-------
- Fix x0 input data and handling
- (???) Jordarter enl. SMED? https://pub.epsilon.slu.se/30474/1/johnsson-h-et-al-20230323.pdf

DemandAndConversions
--------------------
- Processing energy use per food item + energy source
- Transports. A generic factor for Swedish produced. Different factors for imported foods. Elin kollar på detta...

CropProduction
--------------
- Yields semi-natural grasslands. New module: GrasslandMgmt
- Implement way to handle crops that are not harvested 'Fallow', 'Green manure', 'Ley not harvested'
- Implement module to handle cover crops

AnimalHerd
----------

FeedMgmt
--------
- Implement way to balance production of and demand for by-products
    - Adjust feed rations, imports/exports and/or by-products to biogas

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
- Implement method for energy use in grain drying

WasteMgmt (new method)
----------------------
- Calculate sewege sludge generation
- Food waste biogas + digestate
- Slaughter waste biogas + digestate
- Handle energy production from biogas incl. from manure
- Possible to implement innovative recycling technologies

GeoDistributor
--------------
- Implement constraint for min share in rotation
