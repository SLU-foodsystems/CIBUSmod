import os
import pandas as pd

from ..utils.verbose_print import verbose_init

from ecoinvent_interface import Settings, EcoinventProcess, ProcessFileType
import xml.dom.minidom as minidom
from pathlib import Path

class InputsMgmt(object):
    '''

    Parameters
    ----------
    demand : DemandAndConverions object
    crops : CropProduction object
    herds : (pandas.Series of) AnimalHerd object(s)
    par : ParameterRetriever


    '''

    def __init__(
        self,
        demand,
        crops,
        herds,
        par,
        ecoinvent_settings = {
            'version' : '3.7.1',
            'system_model' : 'cutoff'
        }
        ):
        
        self.par = par
        self.demand = demand
        self.crops = crops
        
        if isinstance(herds, pd.Series):
            self.herds = herds
        else:
            self.herds = pd.Series(
                data=herds,
                index=pd.MultiIndex.from_tuples(
                    [(herds.species,herds.breed,herds.prod_system,herds.sub_system)],
                    names=['species','breed','prod_system','sub_system']
                )
            )
        
        path = os.path.join(self.par.data_path,'ecoinvent')
        self.get_ei_data(ecoinvent_settings, path)

    def get_ei_data(self, ecoinvent_settings, path):
        
        ei_version = ecoinvent_settings['version']
        ei_model = ecoinvent_settings['system_model']
        
        # Get available ecoinvent xml files
        files_available = os.listdir(path)
        
        # Get inputs and ecoinvent activity ids
        inputs_and_ei_ids = self.par.get_unique(['input','ecoinvent_id'])
        
        # Get ecoinvent data files needed
        files_needed = ['-'.join(['ecoinvent',ei_version,ei_model,'lci',ei_id])+'.xml'
                        for ei_id in inputs_and_ei_ids.ecoinvent_id]
        
        # Get activity ids that need to be downloaded
        # and download ecoinvent xml files
        ei_activity_ids_to_download = [ei_id for ei_id,f in
                                       zip(inputs_and_ei_ids.ecoinvent_id,files_needed)
                                       if f not in files_available]
        if len(ei_activity_ids_to_download)>0:
            ep = _ei_connect(ei_version,ei_model)
            _ei_get_files(ep, Path(path), ids=ei_activity_ids_to_download)
            
        d=[]
        for inpt, ei_activity_id, file in zip(inputs_and_ei_ids.input,inputs_and_ei_ids.ecoinvent_id,files_needed):
            # Read activity data from ecoinvent xml files
            activity_name, geography_name, reference_unit, ee = \
                _read_xml(os.path.join(path,file))
            
            # Create dataframe and sum flows of the same compund and compartment
            df = (
                pd.DataFrame(ee)
                .groupby(['ecoinvent_compartment','compound','ecoinvent_unit'])
                .sum()
                .T
                .set_index(pd.MultiIndex.from_tuples(
                    [(inpt,ei_activity_id,geography_name,activity_name,reference_unit)],
                    names = ['input','ecoinvent_id','ecoinvent_geography','ecoinvent_activity','ecoinvent_reference_unit']
                ))
            )
            
            d.append(df)
        
        # concatenate dataframe
        df = pd.concat(d)
        # Scale from ecoinvent unit to 'input' unit (according to parameter Excel sheet)
        df = df * self.par.get_from_frame('input_to_ecoinvent',df)
        
        self.data = df
        
    def calculate(self, verbose=False):
        
        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='InputsMgmt')
        vprint('Calculating supply chain emissions for food processing inputs (NOT IMPLEMENTED) ...')
        self.calculate_demand()
        
        vprint('Calculating supply chain emissions for crop production inputs  ...')
        self.calculate_crops()
        
        vprint('Calculating supply chain emissions for animal herd inputs (NOT IMPLEMENTED)  ...')
        self.calculate_herds()
        
    def calculate_crops(self):
        # Translation from ecoinvent elementary flow names to names
        # used in CIBUSmod.
        # CHANGE TO USE ECOINVENT NAMES IN FUTURE!
        translate_compounds = {
            'Carbon dioxide, fossil' : 'CO2',
            'Methane, non-fossil' : 'CH4bio',
            'Methane, fossil' : 'CH4fos',
            'Dinitrogen monoxide' : 'N2O',
        }

        # Get quantities of inputs used
        input_use = pd.concat([
            self.crops.energy_use.groupby('energy_source', axis=1).sum().rename_axis('input', axis=1),
            self.crops.fertiliser.mineral_N.groupby('fertiliser_type', axis=1).sum().rename_axis('input', axis=1)
        ], axis=1)

        input_use_stacked = input_use.stack()

        input_supply_chain_emissions = (
            (
                self.data
                # Select only compounds used and rename. HANDLE ALL COMPOUNDS IN FUTURE!
                .loc[:,self.data.columns.get_level_values('compound').isin(translate_compounds)]
                .rename(translate_compounds, axis=1)
                # Drop all index levels except 'input'
                .droplevel(self.data.index.names[1:])
                # Group and sum by compund. For compounds used now all emissions
                # are to 'air'. Different compartments will need to ba handled
                # later on
                .groupby('compound', axis=1).sum()
                # Reindex to align with 'input_use_stacked'
                .reindex(input_use_stacked.index.get_level_values('input'))
                .set_index(input_use_stacked.index)
            )
            # Multiply emissions by use volumes
            .mul(input_use_stacked, axis=0)
            .unstack('input')
            .reorder_levels([1,0], axis=1)
            .sort_index(axis=1)
        )

        self.crops.input_supply_chain_emissions = input_supply_chain_emissions
        self.crops.data_attr.update(['input_supply_chain_emissions'])
        
    def calculate_herds(self):
        pass
    
    def calculate_demand(self):
        pass

def _ei_connect(ei_version,ei_model):

    u = input('Ecoinvent user name: ')
    pw = input('Ecoinvent password: ')

    my_settings = Settings(username=u, password=pw)

    ep = EcoinventProcess(my_settings)

    ep.set_release(version=ei_version, system_model=ei_model)
    ep.set_release(version="3.7.1", system_model="cutoff")
    
    return ep
    
def _ei_get_files(ep, path, ids=[]):
    for i in ids:
        ep.select_process(dataset_id=i)
        print('Downloading LCI data for "', ep.get_basic_info()['activity_name'], '"...')
        ep.get_file(file_type=ProcessFileType.lci, directory=path)
        
def _read_xml(file_path):
    # Read xml file
    xml = minidom.parse(file_path)
    
    # Get activity name
    activity_name = xml.getElementsByTagName('activityName')[0].firstChild.nodeValue

    # Get geography
    geography_name = xml.getElementsByTagName('geography')[0] \
    .getElementsByTagName('shortname')[0].firstChild.nodeValue
    
    # Get reference unit
    intermediate_exchange = xml.getElementsByTagName('intermediateExchange')[0]
    reference_unit = ' '.join([
        intermediate_exchange.getAttribute('amount'),
        intermediate_exchange.getElementsByTagName('unitName')[0].firstChild.nodeValue,
        intermediate_exchange.getElementsByTagName('name')[0].firstChild.nodeValue,
    ])
    
    elementary_exchange = xml.getElementsByTagName('elementaryExchange')
    ee = [] # List to store elemantary exchanges
    for element in elementary_exchange:
        # Get 'outputGroup' element
        output_group = element.getElementsByTagName('outputGroup')   
        if len(output_group) == 1:
            # Make sure 'outputGroup' is 4 = flow to environment (??)
            if (int(output_group[0].firstChild.nodeValue)) == 4:
                # Get activity name
                name = element.getElementsByTagName('name')[0].firstChild.nodeValue
                # Get compartment
                compartment = element.getElementsByTagName('compartment')[0]\
                    .getElementsByTagName('compartment')[0].firstChild.nodeValue
                # Get unit of flow
                unit = element.getElementsByTagName('unitName')[0].firstChild.nodeValue
                # Get value
                value = element.getAttribute('amount')
        
        ee.append({
            'ecoinvent_compartment':compartment,
            'compound':name,
            'ecoinvent_unit':unit,
            'value':float(value)
        })
        
    
        
    return (activity_name, geography_name, reference_unit, ee)