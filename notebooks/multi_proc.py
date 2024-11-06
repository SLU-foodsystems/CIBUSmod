import sys
import os
import time

from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.join(os.getcwd(),'..'))
import CIBUSmod as cm

def do_run(session, scn_year):

    # Activate session in environment
    session.activate()

    scn, year = scn_year

    # Create log folder if it does not exist
    log_path = os.path.join(session.data_path_output, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, f'{scn}_{year}.log'), 'w') as f,\
        redirect_stdout(f), redirect_stderr(f):

        print(session.data_path)

        tic = time.time()
    
        # Increase timeout to avoid failing to write if multiple processes try to write at the same time
        session.db_timeout = 60
    
        ###############################
        ###   INSTANTIATE MODULES   ###
        ###############################
            
        # Instatiate Regions
        regions = cm.Regions(
            par = cm.ParameterRetriever('Regions')
        )
        
        # Instantiate DemandAndConversions
        demand = cm.DemandAndConversions(
            par = cm.ParameterRetriever('DemandAndConversions')
        )
        
        # Instantiate CropProduction
        crops = cm.CropProduction(
            par = cm.ParameterRetriever('CropProduction'),
            index = regions.data_attr.get('x0_crops').index
        )    
        
        # Instantiate AnimalHerds
        # Each AnimalHerd object is stored in an indexed pandas.Series
        herds = cm.make_herds(regions)
        
        # Instantiate WasteAndCircularity
        waste = cm.WasteAndCircularity(
            demand = demand,
            crops = crops,
            herds = herds,
            par = cm.ParameterRetriever('WasteAndCircularity')
        )
        
        # Instantiate feed management
        feed_mgmt = cm.FeedMgmt(
            herds = herds,
            par = cm.ParameterRetriever('FeedMgmt')
        )
        
        # Instantiate by-product management
        byprod_mgmt = cm.ByProductMgmt(
            demand = demand,
            herds = herds,
            par = cm.ParameterRetriever('ByProductMgmt')
        )
        
        # Instantiate manure management
        manure_mgmt = cm.ManureMgmt(
            herds = herds,
            feed_mgmt = feed_mgmt,
            par = cm.ParameterRetriever('ManureMgmt'),
            settings = {
                'NPK_excretion_from_balance' : True
            }
        )
        
        # Instantiate crop residue managment
        crop_residue_mgmt = cm.CropResidueMgmt(
            demand = demand,
            crops = crops,
            herds = herds,
            par = cm.ParameterRetriever('CropResidueMgmt')
        )
        
        # Instantiate plant nutrient management
        plant_nutrient_mgmt = cm.PlantNutrientMgmt(
            demand = demand,
            regions = regions,
            crops = crops,
            waste = waste,
            herds = herds,
            par = cm.ParameterRetriever('PlantNutrientMgmt')
        )
        
        # Instatiate machinery and energy management
        machinery_and_energy_mgmt  = cm.MachineryAndEnergyMgmt(
            regions = regions,
            crops = crops,
            waste = waste,
            herds = herds,
            par = cm.ParameterRetriever('MachineryAndEnergyMgmt')
        )
        
        # Instatiate inputs management
        inputs = cm.InputsMgmt(
            demand = demand,
            crops = crops,
            waste = waste,
            herds = herds,
            par = cm.ParameterRetriever('InputsMgmt')
        )
        
        # Instantiate geo distributor
        geodist = cm.GeoDistributor(
            regions = regions,
            demand = demand,
            crops = crops,
            herds = herds,
            feed_mgmt = feed_mgmt,
            par = cm.ParameterRetriever('GeoDistributor')
        )
    
        ############################
        ###   RUN CALCULATIONS   ###
        ############################
    
        # Update all parameter values
        cm.ParameterRetriever.update_all_parameter_values(
            **session[scn],
            year = year
        )
        
        # Get region attributes
        regions.calculate(verbose=True)
        
        # Calculate food demand
        demand.calculate(verbose=True)
        
        # Calculate crops
        crops.calculate(verbose=True)
        
        # Calculate herds
        for h in herds:
            h.calculate(verbose=True)
    
        # Induce beef exports in DemandAndConversions if beef production from dairy
        # systems under given demand for milk products exceeds total beef demand.
        # This is to avoid not finding any solution when running the GeoDistributor.
        cm.helpers.induce_beef_exports(
            demand = demand,
            herds = herds
        )
        
        # Calculate feed
        feed_mgmt.calculate(verbose=True)    
        
        # Distribute animals and crops
        # Make optimisation problem
        geodist.make(use_cons=[1,2,3,4,5,6,7])
        # Solve optimisation problem
        geodist.solve(verbose=True)
        
        # Redistribute feeds (not yet implemented) and calculate enteric CH4 emissions
        feed_mgmt.calculate2(verbose=True)
    
        # Balance by-product demand and suply
        byprod_mgmt.calculate(verbose=True)
        
        # Calculate manure
        manure_mgmt.calculate(verbose=True)
        
        # Calculate harvest of crop residues
        crop_residue_mgmt.calculate(verbose=True)
    
        # Calculate treatment of wastes and other feedstocks
        waste.calculate()
        
        # Calculate plant nutrient management
        plant_nutrient_mgmt.calculate(verbose=True)
        
        # Calculate energy requirements
        machinery_and_energy_mgmt.calculate(verbose=True)
        
        # Calculate inputs supply chain emissions
        inputs.calculate(verbose=True)
    
        ########################
        ###   STORE OUTPUT   ###
        ########################
        
        # Store results (try again if first atempt fails)
        try:
            session.store(
                scn, year,
                demand, regions, crops, waste, herds
            )
        except:
            session.store(
                scn, year,
                demand, regions, crops, waste, herds
            )
    
        t = time.time() - tic
        m = int(t/60)
        s = int(round(t - m*60))
        print(f'{scn}, {year} finished successfully in {m}min {s}s')

    return t