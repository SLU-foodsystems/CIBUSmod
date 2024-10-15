import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr

class Regions(object):
    '''Class that handles region attributes such as soil and climate paramters as well as the baseline
    crop areas and animal numbers (x0) and parameters to control maximum land use.

    Parameters
    ----------
    par : ParameterRetriever object
    settings : dict
        Dict with <setting name> : <value>
        Allowed settings are:
            'max_land_use_from_scenario_x0' : bool, default False
                If True, use 'x0_crops' (i.e. baseline crop areas) as updated in scenarios
                to calculate maximum land use per region. Otherwise maximum land use is
                calculated from 'x0_crops' as defined in default parameters irrespective
                of scenario.
    '''

    module_name = 'Regions'

    def __init__(self, par, settings={}):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = DataAttr(self)

        self.par = par

        # Default settings
        self.settings = {
            'max_land_use_from_scenario_x0' : False
        }
        # Update settings if valid input
        for k,v in settings.items():
            if k in self.settings:
                if type(v) is type(self.settings[k]):
                    self.settings.update({k:v})
                else:
                    raise TypeError(f'Expected {type(self.settings[k])} for setting "{k}"')
            else:
                raise ValueError(f'"{k}" is not a valid setting')

        # Get baseline crop areas and animal numbers
        self.get_x0()
        # Set x0_init
        self.data_attr.add(
            self.data_attr.get('x0_crops').copy(),
            name = 'x0_crops_init',
            unit = 'ha or m2',
            orig = 'Regions',
            desc = 'Baseline crop areas in ha (or m2 for greenhouse crops) in default data (i.e. not affected by scenarios)'
        )
        self.data_attr.add(
            self.data_attr.get('x0_animals').copy(),
            name = 'x0_animals_init',
            unit = 'heads',
            orig = 'Regions',
            desc = 'Baseline numbers of defining animals per animal production system in default data (i.e. not affected by scenarios)'
        )

    def calculate(self, verbose=False):

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str='Regions')

        # Get initial crop areas and animal numbers
        vprint('Getting initial crop areas and animal numbers (x0) ...')
        self.get_x0()

        # Get climate and soil attributes
        vprint('Getting climate and soil attributes ...')
        self.get_climate()

        self.get_soil()
        self.classify_soil_texture()
        self.classify_soil_PK('P')
        self.classify_soil_PK('K')

        # Calculate max land use
        vprint('Calculating maximum land use ...')
        self.calculate_max_land_use()

        vprint(type='end')

    def get_x0(self):
        # Get x0_crops
        self.par.clear()
        cps_pss = self.par.get_unique(['crop','prod_system'], qry='parameter == "x0_crops"').values
        res = self.par.get_unique('region', qry='parameter == "x0_crops"')
        idx = pd.MultiIndex.from_tuples(
            [(cp, ps, re) for cp,ps in cps_pss for re in res],
            names = ['crop', 'prod_system', 'region']
        )
        x0_crp = pd.Series(
            self.par.get('x0_crops', **idx.to_frame().to_dict('list')),
            index = idx,
            name = 'area'
        ).sort_index()

        # Get x0_animals
        self.par.clear()
        sps_brs_pss = self.par.get_unique(['species','breed','prod_system'], qry='parameter == "x0_animals"').values
        res = self.par.get_unique('region', qry='parameter == "x0_animals"')
        idx = pd.MultiIndex.from_tuples(
            [(sp, br, ps, re) for sp,br,ps in sps_brs_pss for re in res],
            names = ['species', 'breed', 'prod_system', 'region']
        )
        x0_ani = pd.Series(
            self.par.get('x0_animals', **idx.to_frame().to_dict('list')),
            index = idx,
            name = 'number'
        ).sort_index()

        self.data_attr.add(
            x0_crp,
            name = 'x0_crops',
            unit = 'ha or m2',
            orig = 'Regions',
            desc = 'Baseline crop areas in ha (or m2 for greenhouse crops)'
        )
        self.data_attr.add(
            x0_ani,
            name = 'x0_animals',
            unit = 'heads',
            orig = 'Regions',
            desc = 'Baseline numbers of defining animals per animal production system'
        )


    def get_climate(self):

        # Get index and set filters
        self.par.clear()
        index = self.data_attr.get('x0_crops').index.get_level_values('region').unique()
        p = self.par.get
        self.par.set(**index.to_frame().to_dict('list'))

        # Get climate parameters
        climate = pd.DataFrame(
            np.array([p('GDD5')]).T,
            index = index,
            columns = ['GDD5']
        )

        # Add data attribute
        self.data_attr.add(
            climate,
            name = 'climate',
            unit = 'GDD5 : days*C',
            orig = 'Regions',
            desc = 'GDD5 : Growing degree days',
            scalable = False
        )

    def get_soil(self):

        # Get index and set filters
        self.par.clear()
        index = self.data_attr.get('x0_crops').index.get_level_values('region').unique()
        p = self.par.get
        self.par.set(**index.to_frame().to_dict('list'))

        # Get soil parameters
        soil = pd.DataFrame(
            np.array([
                p('soil_clay'),
                p('soil_silt'),
                p('soil_sand'),
                p('soil_OM'),
                p('soil_pH'),
                p('soil_P_AL'),
                p('soil_K_AL')
            ]).T,
            index = index,
            columns = ['clay','silt','sand','OM','pH','P_AL','K_AL']
        )

        # Add data attribute
        self.data_attr.add(
            soil,
            name = 'soil',
            unit = 'texture/OM:% | pH:n/a | P/K_AL:mg/kg',
            orig = 'Regions',
            desc = 'Soil characteristics',
            scalable = False
        )

    def classify_soil_texture(self):

        soil = self.data_attr.get('soil').copy()

        # Classify soil texture
        # Nodes in the USDA soil texture triangle (x=sand, y=clay)
        c01 = (0.00,0.00); c02 = (0.20,0.00); c03 = (0.50,0.00); c04 = (0.70,0.00)
        c05 = (0.85,0.00); c06 = (1.00,0.00); c07 = (0.00,0.12); c08 = (0.08,0.12)
        c09 = (0.43,0.07); c10 = (0.52,0.07); c11 = (0.85,0.15); c12 = (0.90,0.10)
        c13 = (0.00,0.27); c14 = (0.20,0.27); c15 = (0.23,0.27); c16 = (0.45,0.27)
        c17 = (0.52,0.20); c18 = (0.80,0.20); c19 = (0.00,0.40); c20 = (0.20,0.40)
        c21 = (0.45,0.40); c22 = (0.45,0.35); c23 = (0.65,0.35); c24 = (0.00,0.60)
        c25 = (0.45,0.55); c26 = (0.00,1.00)

        # Polygons for USDA soil texture classes (not used)
        # txt_classes = {
        #     'silt' : np.array([c01, c07, c08, c02]),
        #     'silt_loam' : np.array([c07, c13, c15, c03, c02, c08]),
        #     'loam' : np.array([c15, c16, c17, c10, c09]),
        #     'sandy_loam' : np.array([c09, c10, c17, c18, c11, c04, c03]),
        #     'loamy_sand' : np.array([c04, c11, c12, c05]),
        #     'sand' : np.array([c05, c12, c06]),
        #     'silty_clay_loam' : np.array([c13, c19, c20, c14]),
        #     'clay_loam' : np.array([c14, c20, c21, c16]),
        #     'sandy_clay_loam' : np.array([c16, c22, c23, c18, c17]),
        #     'silty_clay' : np.array([c19, c24, c20]),
        #     'sandy_clay' : np.array([c22, c25, c23]),
        #     'clay' : np.array([c24, c26, c25, c21, c20])
        # }


        # Polygons for aggregated soil texture classes
        txt_classes = {
            'medium' : [c01, c13, c16, c22, c23, c18, c17, c10, c09, c03],
            'coarse' : [c03, c09, c10, c17, c18, c06],
            'fine' : [c13, c26, c23, c22, c16]
        }

        # Get texture class per region
        soil['texture class'] = \
            soil.apply(
                _get_soil_class,
                txt_classes=txt_classes,
                axis=1
            )

        soil_texture = soil['texture class']

        # Add data attribute
        self.data_attr.add(
            soil_texture,
            name = 'soil_texture',
            unit = '-',
            orig = 'Regions',
            desc = 'Soil texture class',
            scalable = False
        )

    def classify_soil_PK(self, element):

        classes_def = {
            # Soil P classes (up to x mg/kg)
            'P' : {
                'I' : 20,
                'II' : 40,
                'III' : 80,
                'IVA' : 120,
                'IVB' : 160,
                'V' : np.inf,
            },
            # Soil K classes (up to x mg/kg)
            'K' : {
                'I' : 40,
                'II' : 80,
                'III' : 160,
                'IV' : 320,
                'V' : np.inf,
            }
        }

        element_to_name = {'P' : 'phosphorous', 'K' : 'potassium'}

        classes = classes_def[element]

        soil = self.data_attr.get('soil').copy()
        soil[f'{element}_class'] = np.nan

        for k,v in classes.items():
            soil[f'{element}_class'] = soil[f'{element}_class'].where((soil[f'{element}_AL']>v) | ~soil[f'{element}_class'].isna(), k)

        soil_class = soil[f'{element}_class']

        # Add data attribute
        self.data_attr.add(
            soil_class,
            name = f'soil_{element}_class',
            unit = '-',
            orig = 'Regions',
            desc = f'Soil {element_to_name[element]} ({element}-AL) class',
            scalable = False
        )

    def calculate_max_land_use(self):
        # Get land uses with a maximum land use
        land_uses = self.par.get_unique(
            'land_use',
            qry='parameter == "max_land_use_factor"'
        )

        if self.settings['max_land_use_from_scenario_x0']:
            x0_crops = self.data_attr.get('x0_crops').copy()
        else:
            x0_crops = self.data_attr.get('x0_crops_init').copy()

        # Calculate land use in x0
        lu = (
            x0_crops
            .rename(self.par.get_rel('crop','land_use'))
            .rename_axis(['land_use','prod_system','region'])
            .groupby(['region','land_use'])
            .sum()
            .unstack()
            .loc[:,land_uses]
        )

        # Calculate maximum land use
        max_land_use = lu * self.par.get_from_frame('max_land_use_factor',lu)

        # Add data attribute
        self.data_attr.add(
            max_land_use,
            name = 'max_land_use',
            unit = 'ha or m2',
            orig = 'Regions',
            desc = 'Maximum allowed land use per land use class in ha (or m2 for greenhouse crops)'
        )

def _get_soil_class(x, txt_classes):
    for txt_class in txt_classes:
        if _point_in_polygon([x['sand']/100,x['clay']/100],txt_classes[txt_class]):
            return txt_class
    return np.nan

def _point_in_polygon(point,polygon):
    # Returns True if point is in, on the edge
    # or on a node of polygon.
    x = point[0]
    y = point[1]
    point = [x,y]
    n = len(polygon)
    inside = False

    if point in polygon:
        # catch point on node
        return True

    p1x,p1y = polygon[0]
    for i in range(n+1):
        p2x,p2y = polygon[i % n]
        if y == p1y and y == p2y and x <= max(p1x,p2x) and x >= min(p1x,p2x):
            # catch pont on horizontal edge
            return True
        if y > min(p1y,p2y) and y <= max(p1y,p2y) and x <= max(p1x,p2x):
            if p1y != p2y:
                xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                if x == xints:
                    # catch point on edge
                    return True
                if p1x == p2x or x <= xints:
                    inside = not inside

        p1x,p1y = p2x,p2y

    return inside

class StaticRegions():
    '''Class used to create static copy of DemandAndConversions object. These stores all attributes except 'par'
    but does not inherit any methods'''

    def __repr__(self):
        return Regions.__repr__(self)
