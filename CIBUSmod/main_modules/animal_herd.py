import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from ..utils.verbose_print import verbose_init
from ..utils.data_attr import DataAttr
from ..utils.retriever import ParameterRetriever

from typing import Literal, Any

class AnimalHerd(ABC):
    '''Class that handels animal herd structure, feed requirements, production etc.

    Parameters
    ----------
    par : ParameterRetriever object
    index : pandas.Index or pandas.MultiIndex
        Index for the rows. This is also passed on to the ParameterRetriever
    **kwargs : str or list
        Keyword arguments to be passed on as filters to the ParameterRetriever, along with
        the index, self.species. Special cases are 'breed', 'prod_system' and 'sub_system'
        which if supplied are also stored as attributes in the AnimalHerd object.

    Attributes set on init
    ----------------------
    index : pandas.Index or padnas.MultiIndex
        Index for rows

    species : str
        Animal species (depends on which AnimalHerd sub-class is used)
    breed : str
        Specifies the breed (e.g. dairy or beef breed)
    prod_system : str
        Specifies the production system which links feed
    sub_system : str

    animals : list
        List of str specifying animal categories in the herd (depends on which AnimalHerd
        sub-class is used)

    Attributes set by AnimalHerd.calculate()
    ----------------------------------------
    heads : pandas.DataFrame
        Yearly average number of heads per animal [heads]
    slaughtered_n : pandas.DataFrame
        Number of slaughtered animals per year [heads]
    lost_n : pandas.DataFrame
        Number of lost animals per year [heads]
    production : pandas.DataFrame
        Production per year for each animal and product  [kg]

    Attributes set by FeedMgmt.calculate()
    --------------------------------------
    feed.energy_req : pandas.DataFrame
        Total energy requirements per year for each animal type [MJ]
        Energy is expressed differently across animal species. E.g. for cattle energy is in
        terms of Metabolizable Energy (ME) and for pigs energy is in terms of Net Energy (NE)
    feed.consumption : pandas.DataFrame
        Feed demand per year for each animal type and feed (in terms of feed consumed) [kg DM]

    Attributes set by ManureMgmt.calculate()
    ----------------------------------------
    manure.<element>_excr : pandas.DataFrame
        Manure excretion by animals [kg <element>]
    manure.<element>_loss : pandas.DataFrame
        Losses of <element> in stables and storage by compound [kg <element>]
    manure.<element>_to_spread : pandas.DataFrame
        <element> available to spread [kg <element>]

    <element> is 'VS' (volatile solids), 'N' (nitrogen), 'P' (phosphorous) and 'K' (potassium).
    (ONLY 'N' IMPLEMENTED AT THE MOMENT)

    '''
    # Set of ID attributes in class
    id_attr = set(['species','breed','prod_system','sub_system','animals'])
    module_name = 'AnimalHerd'

    # Store the number of (defining) animals
    x: np.ndarray

    # Defined in subclasses
    x_is: str # defining what x represents, e.g. 'cows', 'total hens', etc.
    species: str
    animals: list[str]

    def __init__(self,par,index,**kwargs):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = DataAttr(self)

        self.par = par
        self.index = index

        self.breed = kwargs['breed'] or 'none'
        self.prod_system = kwargs['prod_system'] or 'none'
        self.sub_system = kwargs['sub_system'] or 'none'

    def __repr__(self):
        return f'''
AnimalHerd
----------
species              {self.species}
breed                {self.breed}
production system    {self.prod_system}
sub-system           {self.sub_system}
animals              {self.animals}
'''


    @abstractmethod
    def calculate_herd(self):
        """Calculate the herd and store values in data_attr."""
        pass

    @abstractmethod
    def _calculate_feed_req(self, ps, ani) -> tuple[Literal['E', 'DM'], Any]:
        """
        Calculate the feed energy [MJ] or dry matter [kg DM] requirements.
        The first item in the tuple describes which type of feed requirement the
        animal herd specifies, and the second is the value thereof.
        """
        pass

    def calculate(self,verbose=False):
        '''Calculates herd structure and production based on a vector ('x') of animal numbers or production as defined
        by 'x_is'. Index of 'x' is retained in the output and can be used as filters for the ParameterRetriever.

        Parameters
        ----------
        verbose : Bool
            If True, prints messages on progress

        Returns
        -------
        Nothing. Stores output as attrubutes: 'heads', 'slaughtered_n', 'lost_n' and 'production'
        '''

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        self.par.set(
            species = self.species,
            breed = self.breed,
            prod_system = self.prod_system,
            sub_system = self.sub_system,
            **self.index.to_frame().to_dict('list')
        )

        # Set x (i.e. nr of defining animals) to ones
        self.x = np.ones(len(self.index))

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str=f'AnimalHerd ({self.species}, {self.breed}, {self.prod_system}, {self.sub_system})')

        vprint('Calculating herd structure ...')
        self.calculate_herd()

        vprint('Calculating feed energy or DM requirements ...')
        self.calculate_feed_req()

        vprint('Calculating production ...')
        self.calculate_production()

        vprint(type='end')

    def scale(self,new_x,x_is):
        '''Scales all data attributes based on new_x

        Parameters
        ----------
        new_x : numpy.array or pandas.Series

        x_is : str
            String defining what x is. Must be one of valid herd types:
            'cows','sows','sows+gilts','broilers','total hens','total horses','ewes+rams','meat','milk'

        '''

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        self.par.set(
            species = self.species,
            breed = self.breed,
            prod_system = self.prod_system,
            sub_system = self.sub_system,
            **self.index.to_frame().to_dict('list')
        )
        p = self.par.get

        valid = ['cows','sows','sows+gilts','broilers','total hens','total horses','ewes+rams','meat','milk']
        if x_is not in valid:
            raise ValueError("x_is must be one of %r." % valid)

        # Check so that new_x length and index match AnimalHerd object
        if len(new_x) != len(self.index):
            raise TypeError('Length of x does not match length of herd\'s index!')
        if hasattr(new_x,'index'):
            if (new_x.index != self.index).any():
                raise TypeError('Index of x does not match herd\'s index!')

        if x_is == 'sows+gilts':
            new_x = new_x / (1+ (p('recruitment_rate')/100 * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25))
            x_is = 'sows'

        # Get 'old_x'
        if x_is in ['milk','meat']:
            old_x = self.data_attr.get('production').loc[:,(slice(None),x_is)].sum(axis=1)
        elif x_is == 'total horses':
            old_x = self.data_attr.get('heads').sum(axis=1)
        elif x_is == 'ewes+rams':
            old_x = self.data_attr.get('heads').loc[:,(slice(None), ['ewes', 'rams'])].sum(axis=1)
        elif x_is == 'total hens':
            old_x = self.data_attr.get('heads').drop('laying chicks', level='animal', axis=1).sum(axis=1)
        else:
            old_x = self.data_attr.get('heads').loc[:,(self.prod_system,x_is)]

        # Update data attributes
        for attr in self.data_attr:
            if self.data_attr[attr]['scalable']:
                if self.data_attr.get(attr) is not None:
                    self.data_attr.add(
                        self.data_attr.get(attr).mul(new_x/old_x, axis=0),
                        name = attr,
                        **self.data_attr[attr]
                    )

    def make_static(self):
        '''Returns a StaticAnimalHerd object that retains all ID and data attributes but
        has no methods or ParameterRetriever'''

        # Create a StaticAnimalHerd obejct and populate with data
        obj = StaticAnimalHerd()

        obj.id_attr = self.id_attr.copy()
        obj.data_attr = DataAttr(obj)

        # Set ID attributes
        for attr in obj.id_attr:
            setattr(obj,attr,getattr(self,attr))

        # Set data attributes
        for attr in self.data_attr:
            if self.data_attr.get(attr) is not None:
                obj.data_attr.add(data=self.data_attr.get(attr).copy(), name=attr, **self.data_attr[attr])
            else:
                obj.data_attr.add(data=None, name=attr, **self.data_attr[attr])

        return obj

    def calculate_feed_req(self):

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        self.par.set(
            species = self.species,
            breed = self.breed,
            prod_system = self.prod_system,
            sub_system = self.sub_system,
            **self.index.to_frame().to_dict('list')
        )

        # Remove 'milk_to_calves' attribute if it exists
        if 'milk_to_calves' in self.data_attr:
            self.data_attr.remove('milk_to_calves')


        df_req = pd.DataFrame(
            index = self.index,
            columns = self.data_attr.get('heads').columns,
            dtype = float
        )
        pss = list(df_req.columns.get_level_values('prod_system'))
        anis = list(df_req.columns.get_level_values('animal'))
        if self.species == 'cattle':
            # Make sure calves are hendeled first to get milk from cows
            # to calves
            anis.insert(0, anis.pop(anis.index('calves')))

        feed_req_type = None
        for ani, ps in zip(anis, pss):
            self.par.set(
                prod_system = ps,
                animal = ani
            )

            # Calculate feed requirements (energy [MJ] or dry matter [kg DM])
            (feed_req_type, req) = self._calculate_feed_req(ps, ani)

            df_req.loc[:,(ps,ani)] = req

        # No animals found in animal-herd, we can exit.
        # We should maybe raise an Exception here.
        if feed_req_type is None:
            return

        # If herd has a method to calculate energy requirements of animals
        # energy requirements are calculated from live weights, growth rates,
        # gestation, lactation, etc.
        # Otherwise dry matter feed requirements are calculated from feed
        # conversion ratios or a fixed feed intake per animal.
        if feed_req_type == "E":
            self.data_attr.add(
                (df_req * self.data_attr.get('heads')),
                name = 'feed_E_req',
                unit = 'MJ/year',
                orig = 'AnimalHerd',
                desc = 'Total feed requirements in terms of energy. Type of energy differ by species'
            )
        else:
            self.data_attr.add(
                (df_req * self.data_attr.get('heads')),
                name = 'feed_DM_req',
                unit = 'kg DM/year',
                orig = 'AnimalHerd',
                desc = 'Total feed requirements in terms of dry matter'
            )

    def calculate_production(self):

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        self.par.set(
            species = self.species,
            breed = self.breed,
            prod_system = self.prod_system,
            sub_system = self.sub_system,
            **self.index.to_frame().to_dict('list')
        )

        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        # Define output products
        prs = ['meat']
        if self.species == 'cattle':
            prs.append('milk')
        if (self.species == 'poultry') and (self.breed == 'layer'):
            prs.append('eggs')
        if (self.species == 'sheep') and (self.sub_system == 'other sheep'):
            prs.append('heads')
        if self.species == 'horses':
            prs.append('heads')

        # Get ouput production systems
        pss = self.data_attr.get('heads').columns.get_level_values('prod_system').unique()
        # Get animals
        anis = self.animals

        # Create production DF
        production = pd.DataFrame(
            index = self.index,
            columns = pd.MultiIndex.from_tuples([(ps,ani,pr) for ps in pss for ani in anis for pr in prs], names=['prod_system','animal','animal_prod']),
            dtype = float
            )

        # Calculate meat production [kg CW]
        production.loc[:,(slice(None),slice(None),'meat')] = \
            pd.concat({'meat': self.data_attr.get('slaughtered_n')}, names=['animal_prod'], axis=1).reorder_levels(['prod_system','animal','animal_prod'], axis=1) * \
            np.array([p('slaughter_weight', animal=ani, prod_system=ps) for ps in pss for ani in anis]).T

        # Calculate raw milk production [kg ECM]
        # kg ECM = kg milk x 0.25 + kg fat x 12.2 + kg protein x 7.7
        if 'milk' in prs:
            production.loc[:,(slice(None),slice(None),'milk')] = \
                pd.concat([
                    pd.concat({'milk': self.data_attr.get('heads').loc[:,[(ps,'cows')]]}, names=['animal_prod'], axis=1).reorder_levels(['prod_system','animal','animal_prod'], axis=1).mul(
                    (p('milk_prod', prod_system=ps) - self.data_attr.get('milk_to_calves')).values.clip(0) * (1 - p('milk_loss')/100) *
                    (0.25 + p('milk_fat', prod_system=ps)/100*12.2 + p('milk_protein', prod_system=ps)/100*7.7),
                    axis = 0
                )
                    for ps in pss
                    ], axis=1)

        # Calculate egg production [kg]
        if 'eggs' in prs:
            production.loc[:,(slice(None),slice(None),'eggs')] = \
                pd.concat({'eggs': self.data_attr.get('heads')}, names=['animal_prod'], axis=1).reorder_levels(['prod_system','animal','animal_prod'], axis=1) * \
                np.array([p('egg_production', animal=ani, prod_system=ps) for ps in pss for ani in anis]).T

        # Calculate total number of heads
        if 'heads' in prs:
            production.loc[:,(slice(None),slice(None),'heads')] = (
                pd.concat({'heads': self.data_attr.get('heads')}, names=['animal_prod'], axis=1)
                .reorder_levels(['prod_system','animal','animal_prod'], axis=1)
            )

        # Fill NaNs in production DataFrame and set column index
        production = production.fillna(0)

        # Add data attribute
        self.data_attr.add(
            production,
            name = 'production',
            unit = 'kg/year',
            orig = 'AnimalHerd',
            desc = 'Total production of animal products'
        )

class StaticAnimalHerd():
    '''Class used to create static copys of animal herd objects. These stores all attributes except 'par'
    but does not inherit any methods'''

    id_attr: set[str]
    data_attr: DataAttr
    module_name = 'AnimalHerd'

    def __repr__(self):
        return AnimalHerd.__repr__(self).replace('AnimalHerd','StaticAnimalHerd')

from .animal_herd_cattle import CattleHerd
from .animal_herd_pig import PigHerd
from .animal_herd_poultry_broiler import BroilerHerd
from .animal_herd_poultry_layer import LayerHerd
from .animal_herd_horse import HorseHerd
from .animal_herd_sheep import SheepHerd

def make_herds(
        regions,
        class_map = {
            ('cattle', 'dairy') : 'CattleHerd',
            ('cattle', 'beef') : 'CattleHerd',
            ('pigs', 'none') : 'PigHerd',
            ('poultry', 'layer') : 'LayerHerd',
            ('poultry', 'broiler') : 'BroilerHerd',
            ('sheep', 'none') : 'SheepHerd',
            ('horses', 'ponies and Icelandic horses') : 'HorseHerd',
            ('horses', 'cold blooded horses') : 'HorseHerd',
            ('horses', 'riding horses') : 'HorseHerd',
            ('horses', 'trotters and racehorses') : 'HorseHerd'
        },
        sub_systems = {
            ('cattle') : ['ley based'],
            ('cattle','dairy','conventional') : ['maize based'],
            ('cattle','beef','conventional') : ['maize based'],
            ('sheep') : ['autumn lamb', 'spring lamb', 'winter lamb', 'other sheep']
        }
        ):
    '''Helper function to instantiate AnimalHerd objects and put them in a pandas.Series.

    Parameters
    ----------
    regions : Regions object
    class_map : dict
    sub_systems : dict

    Returns
    -------
    pandas.Series : A series containing AnimalHerd objects representing all animals
                    in regions 'x0_animals' data attribute
    '''

    # Create Series to store AnimalHerd objects
    herds = pd.Series(
        data=[],
        index=pd.MultiIndex(
            levels=[[]]*4,
            codes=[[]]*4,
            names=['species','breed','prod_system','sub_system']
        ),
        dtype = object
    )

    for (sp,br,ps) in regions.data_attr.get('x0_animals').groupby(['species','breed','prod_system']).sum().index:

        if (sp,br) in class_map:

            herd_class_name = class_map[(sp,br)]
            herd_class = globals()[herd_class_name]

            # Get sub-systems
            # Get sub-systems
            sss = []
            for k,v in sub_systems.items():
                if isinstance(k, str):
                    for v_ in v:
                        if sp == k:
                            sss += [v_] if v_ not in sss else []
                elif len(k)==2:
                    for v_ in v:
                        if (sp,br) == k:
                            sss += [v_] if v_ not in sss else []
                elif len(k)==3:
                    for v_ in v:
                        if (sp,br,ps) == k:
                            sss += [v_] if v_ not in sss else []
            if len(sss) == 0:
                sss = ['none']

            for ss in sss:
                herds.loc[(sp,br,ps,ss)] = \
                herd_class(
                    par = ParameterRetriever(herd_class_name),
                    index = regions.data_attr.get('x0_animals').index.get_level_values('region').unique(),
                    breed = br,
                    prod_system = ps,
                    sub_system = ss
                )
        else:
            print(f'{sp}, {br} not found in par_map. No AnimalHerd object created')

    return herds


def concat_herds(herds):
    '''Combines multiple AnimalHerd objects

    Parameters
    ----------
    herds : itterable of AnimalHerd objects

    Returns
    -------
    StaticAnimalHerd object'''
    res_herd = StaticAnimalHerd()

    res_herd.id_attr = AnimalHerd.id_attr
    for attr in AnimalHerd.id_attr:
        setattr(res_herd,attr,'aggregated')

    res_herd.data_attr = DataAttr(res_herd)

    # Check presence of data attributes in AnimalHerd objects
    # Only attributes present in all AnimalHerd objects are
    # retained in the combined StaticAnimalHerd object
    data_attr_union = set.union(*[set(h.data_attr) for h in herds])
    data_attr_in_all = set.intersection(*[set(h.data_attr) for h in herds])
    data_attr_in_some = data_attr_union - data_attr_in_all
    if len(data_attr_in_some) > 0:
        pass
        # Should a warning be printed here?
        # warnings.warn(f'Data attributes {data_attr_in_some} not pressent in all AnimalHerds and therfore not retained.')

    # Go through data attributes and concatenate
    for attr in data_attr_in_all:
        scalable = herds.iloc[0].data_attr[attr]['scalable']
        # Only include scalable data attributes for now...
        # Potentially rethink aggregation to be
        # able to include also non-scalable data
        if scalable:

            df = pd.concat(
                [
                    pd.concat({herd.species:
                        pd.concat({herd.breed:
                            pd.concat({herd.sub_system: herd.data_attr.get(attr)},
                                names=['sub_system'],axis=1)},
                            names=['breed'],axis=1)},
                        names=['species'],axis=1)
                    if herd.data_attr.get(attr) is not None else pd.DataFrame() for herd in herds
                ],
                axis=1
            )

            # Group and sum columns to avoid duplicates
            df = df.T.groupby(df.columns.names).sum().T

            # Add data attribute
            metadata = herds.iloc[0].data_attr[attr]
            if 'Herd' in metadata['orig'] and metadata['orig'] != 'AnimalHerd':
                # Replace specific herd module name
                metadata['orig'] = '<Spec.>Herd'
            res_herd.data_attr.add(df, name=attr, **metadata)

    return res_herd
