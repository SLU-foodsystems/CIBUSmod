import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class AquacultureHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','aquaculture fish')

    has_manure = False

    def __init__(self,par,index,**kwargs):

        self.species = kwargs['species']
        self.animals = ['juvenile fish', 'adult fish']
        self.products = ['fish']

        self.x_is = 'fish'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):

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
        
        # x is kg production (round weight)
        prod = self.x

        inserted_adult = prod / p('slaughter_weight', animal='adult fish') / (1 - p('mortality', animal='adult fish')/100)
        inserted_juvenile = inserted_adult / (1 - p('mortality', animal='juvenile fish')/100)

        lost_adult = inserted_adult * (p('mortality', animal='adult fish')/100)
        lost_juvenile = inserted_juvenile * (p('mortality', animal='juvenile fish')/100)

        slaughtered_adult = inserted_adult - lost_adult
        slaughtered_juvenile = np.zeros(len(self.index))

        lost_lw_adult = lost_adult * p('slaughter_weight', animal='adult fish') / 2 # Assume losses at half slaughter weight [kg round weight]
        lost_lw_juvenile = lost_juvenile * p('weight_delivery', animal='juvenile fish') / 2 # Assume losses at half delivery weight [kg round weight]

        heads_adult = (inserted_adult - lost_adult/2) * (p('months_production_cycle', animal='adult fish')/12)
        heads_juvenile = (inserted_juvenile - lost_juvenile/2) * (p('age_delivery', animal='juvenile fish')/12)

        # Create DataFrames
        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([
                (ps, ani)
                for ps in [self.prod_system]
                for ani in self.animals
            ], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
        )

        heads, inserted_n, slaughtered_n, lost_n, lost_lw = \
            [empty_df.copy() for i in range(5)]
        
        # Assign values
        heads.loc[:,:] = np.array([
            heads_juvenile,
            heads_adult
        ]).T
        inserted_n.loc[:,:] = np.array([
            inserted_juvenile,
            inserted_adult
        ]).T
        slaughtered_n.loc[:,:] = np.array([
            slaughtered_juvenile,
            slaughtered_adult
        ]).T
        lost_n.loc[:,:] = np.array([
            lost_juvenile,
            lost_adult            
        ]).T
        lost_lw.loc[:,:] = np.array([
            lost_lw_juvenile,
            lost_lw_adult            
        ]).T

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'AquacultureHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            inserted_n,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'AquacultureHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'AquacultureHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'AquacultureHerd',
            desc = 'Total number of heads lost'
        )
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'AquacultureHerd',
            desc = 'Total live weight of lost animals'
        )

        return None

    def _calculate_feed_req(self):

        for ps,ani in self.data_attr.get('heads').columns:
            self.par.set(
                prod_system = ps,
                animal = ani
            )

            # Get dry matter requirements
            DM_req = self._calculate_DM_req(ps,ani) # kg DM

            # Append requirements to appropriate 'feed_req_*' DataFrames
            self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'DM')] = DM_req

        return None

    def _calculate_DM_req(self,ps,ani):
        
        p = self.par.get

        inserted = self.data_attr.get('inserted_n').loc[:,(ps,ani)]
        lost = self.data_attr.get('lost_n').loc[:,(ps,ani)]

        if ani == 'juvenile fish':
            feed_req = p('eFCR') * (inserted - lost) * (p('weight_delivery')/1_000)
        else:
            feed_req = p('eFCR') * (inserted - lost) * p('slaughter_weight')

        return feed_req
