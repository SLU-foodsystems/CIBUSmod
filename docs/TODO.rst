TODO
====

Validation and default parameters
---------------------------------
- Check rapeseed
- Check induced skim milk exports
- CattleHerd: Too much grazing adjust feed rations
- MachineryAndEnergyMgmt: Check 

ParameterRetriever
------------------
- 

Regions
-------
- Fix x0 input data and handling
- (???) Jordarter enl. SMED? https://pub.epsilon.slu.se/30474/1/johnsson-h-et-al-20230323.pdf

DemandAndConversions
--------------------
- Check exports
- Demand for Herbs
- Processing energy use per food item + energy source
- Transports. A generic factor for Swedish produced. Different factors for imported foods. Elin kollar på detta...

CropProduction
--------------
- Check/add crop residue factors for all crops
- Yields semi-natural grasslands. New module: GrasslandMgmt
- Fix yield 'ley for seed'
- Check if yields are missing for more crops
- Implement way to handle crops that are not harvested 'Fallow', 'Green manure', 'Ley not harvested'
- Implement module to handle cover crops
- Check 'Mixed cereals' -> korn+havre eller havre/korn+ärter

AnimalHerd
----------
- Add SheepHerd based on input data from Hanna
- CattleHerd
    - Use only recruitment or slaughter age for cows
    - Milk for calves?


FeedMgmt
--------
- Implement way to balance production of and demand for by-products
    - Adjust feed rations, imports/exports and/or by-products to biogas

ManureMgmt
----------
- Internaly calculate MMS='grazing' from relative contribution of grazing to total DM feed intake
- Check MMS shares (should add to 100% ?)
- Handle manure that is not returned to cropland? (mainly horses?)
- Handle manure going to biogas digesters

PlantNutrientMgmt
-----------------
- Implement alternative to use N2O EF from Rochette et al
- Improve leaching calculations (currently IPCC method and only for N)
- Pot. update manure distribution to better reflect which crops recieve manure


MachineryAndEnergyMgmt
----------------------
- Filed operations in organic production
- Implement method for energy use in grain drying

WasteMgmt (new method)
----------------------
- sewege sludge generation
- Food waste biogas + digestate
- Slaughter waste biogas + digestate
- Innovative recycling technologies

GeoDistributor
--------------
- Fix problems with index sorting for .apply_solution()
