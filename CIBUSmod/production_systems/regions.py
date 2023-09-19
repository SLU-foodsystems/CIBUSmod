import pandas as pd
import numpy as np
import os

from ..utils.verbose_print import verbose_init
from ..utils.misc import Container, rgetattr,rsetattr

class Regions(object):
    '''Class that handles region attributes

    Parameters
    ----------
    x0 : 
    par : 

    Attributes set on init
    ----------------------
    index : 

    Attributes set by Regions.calculate()
    --------------------------------------------
    climate :
    soil :
    '''

    def __init__(self,par):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = set()
        
        self.par = par

        # Get initial crop areas and animal numbers
        self.get_x0()

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

        vprint(type='end')

    def make_static(self):
        '''Returns a StaticRegions object that retains all data attributes but
        has no methods or ParameterRetriever'''
        
        obj = StaticRegions()

        obj.data_attr = self.data_attr.copy()

        for attr in obj.data_attr:
            if rgetattr(self, attr) is not None:
                rsetattr(obj, attr, rgetattr(self, attr).copy())
            else:
                rsetattr(obj, attr, None)

        return obj

    def get_x0(self):

        self.par.clear()

        # THIS IS A TEMPORY IMPLEMENTATION. X0 WILL LATER BE DEFINED
        # AS PARAMETERS AND RETRIEVED WITH THE PARAMETERRETRIEVER !!!!

        path = os.path.join(self.par.data_path_default,'..','x0')

        # Define x0_crp

        x0_crp = \
            pd.read_csv(os.path.join(path,'x0_crp.csv'), dtype={'region': object})\
            .set_index(['crop','prod_system','region'])['area']

        # Define x0_ani
        x0_ani = pd.read_csv(os.path.join(path,'x0_ani.csv'), dtype={'region': object})

        x0_ani['species'] = np.nan
        x0_ani['species'] = \
        np.where(np.isin(x0_ani['animal'], ['kor för mjölkproduktion', 'kor för uppfödning av kalvar']),'cattle',x0_ani['species'])
        x0_ani['species'] = \
        np.where(x0_ani['animal']=='suggor för avel','pigs',x0_ani['species'])
        x0_ani['species'] = \
        np.where(x0_ani['animal']=='slaktkycklingar','poultry',x0_ani['species'])
        x0_ani['species'] = \
        np.where(x0_ani['animal']=='hästar','horses',x0_ani['species'])

        # Add horse breeds
        x0_ani = pd.concat([
            x0_ani[x0_ani.animal != 'hästar'],
            x0_ani[x0_ani.animal == 'hästar'].replace({'hästar':'ponnyer'}),
            x0_ani[x0_ani.animal == 'hästar'].replace({'hästar':'kallblod'}),
            x0_ani[x0_ani.animal == 'hästar'].replace({'hästar':'ridhästar'}),
            x0_ani[x0_ani.animal == 'hästar'].replace({'hästar':'trav och galopp'})
            ])

        x0_ani['breed'] = np.nan
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='kor för mjölkproduktion','dairy',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='kor för uppfödning av kalvar','beef',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='suggor för avel','none',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='slaktkycklingar','broiler',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='ponnyer','ponies and Icelandic horses',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='kallblod','cold blooded horses',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='ridhästar','riding horses',x0_ani['breed'])
        x0_ani['breed'] = \
        np.where(x0_ani['animal']=='trav och galopp','trotters and racehorses',x0_ani['breed'])

        x0_ani = x0_ani[x0_ani['species']!='nan'][['species','breed','prod_system','region','number']].set_index(['species','breed','prod_system','region'])['number']

        # Numbers per horse breed
        x0_ani.update(x0_ani.xs(('horses','ponies and Icelandic horses'),level=('species','breed'),drop_level=False) * 0.297)  
        x0_ani.update(x0_ani.xs(('horses','cold blooded horses'),level=('species','breed'),drop_level=False) * 0.100)  
        x0_ani.update(x0_ani.xs(('horses','riding horses'),level=('species','breed'),drop_level=False) * 0.343)  
        x0_ani.update(x0_ani.xs(('horses','trotters and racehorses'),level=('species','breed'),drop_level=False) * 0.260)  

        x0_ani = x0_ani.fillna(0)

        self.x0_crops = x0_crp
        self.x0_animals = x0_ani
        self.data_attr.update(['x0_crops','x0_animals'])


    def get_climate(self):

        # Get index and set filters
        self.par.clear()
        index = self.x0_crops.index.get_level_values('region').unique()
        p = self.par.get
        self.par.set(**index.to_frame().to_dict('list'))

        # Get climate parameters
        self.climate = pd.DataFrame(
            np.array([p('GDD5')]).T,
            index = index,
            columns = ['GDD5']
        )
        self.data_attr.update(['climate'])

    def get_soil(self):

        # Get index and set filters
        self.par.clear()
        index = self.x0_crops.index.get_level_values('region').unique()
        p = self.par.get
        self.par.set(**index.to_frame().to_dict('list'))

        # Get soil parameters
        self.soil = pd.DataFrame(
            np.array([
                p('soil_clay'),
                p('soil_silt'),
                p('soil_sand'),
                p('soil_OM'),
                p('soil_pH')
            ]).T,
            index = index,
            columns = ['clay','silt','sand','OM','pH']
        )

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
        self.soil['class'] = \
            self.soil.apply(
                _get_soil_class,
                txt_classes=txt_classes,
                axis=1
            )
        
        self.data_attr.update(['soil'])

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

class StaticRegions(Container):
    '''Class used to create static copy of DemandAndConversions object. These stores all attributes except 'par'
    but does not inherit any methods'''

    def __repr__(self):
        return Regions.__repr__(self)