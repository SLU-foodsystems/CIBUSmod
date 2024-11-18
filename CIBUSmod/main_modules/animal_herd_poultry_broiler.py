import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class BroilerHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','broiler poultry')

    def __init__(self,par,index,**kwargs):

        self.species = 'poultry'
        self.breed = 'broiler'
        self.animals = ['broilers','breeding hens','breeding roosters']

        self.x_is = 'broilers'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates broiler herd structure and slaughters based on x (i.e. number of
        broilers in terms of animal places).

        Parameters
        ----------
        None

        Returns
        -------
        Nothing.
        Sets data attributes heads, slaughtered_n and lost_n'''

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

        idx_len = len(self.index)

        # Get number of broilers. This is in terms of number of animal places
        # and the average number of live animals at a given moment is lower as
        # facilities need time for cleaning between rounds
        broilers = self.x

        # Calculate number of inserted animals
        inserted_broilers = broilers * p('rounds_per_year')
        inserted_parent_hens = (
            inserted_broilers /
            p('eggs_per_breeding_hen') /
            (1 - p('mortality', animal='breeding hens')/100)
        )
        inserted_parent_roosters = inserted_parent_hens / p('hens_per_rooster')
        inserted_grandparent_hens = (
            (inserted_parent_hens * 2) / # Assumes 50/50 sex ratio
            p('eggs_per_breeding_hen') /
            (1 - p('mortality', animal='breeding hens')/100)
        )
        inserted_grandparent_roosters = inserted_grandparent_hens / p('hens_per_rooster')

        inserted_breeding_hens = (inserted_parent_hens + inserted_grandparent_hens)
        inserted_breeding_roosters = (inserted_parent_roosters + inserted_grandparent_roosters)

        # Calculate number of parent animals
        breeding_hens = inserted_breeding_hens * p('slaughter_age', animal='breeding hens')/365.25
        breeding_roosters = inserted_breeding_roosters * p('slaughter_age', animal='breeding hens')/365.25

        # Calculate lost animals
        self.par.set(animal='broilers')
        lost_broilers = (
            inserted_broilers *
            (
                (p('mortality')/100) +
                (1-p('mortality')/100)*(p('rejections_at_slaughter')/100)
            )
        )

        self.par.set(animal='breeding hens')
        lost_breeding_hens = (
            inserted_breeding_hens *
            (
                (p('mortality')/100) +
                (1-p('mortality')/100)*(p('rejections_at_slaughter')/100)
            )
        )

        self.par.set(animal='breeding roosters')
        lost_breeding_roosters = (
            inserted_breeding_roosters *
            (
                (p('mortality')/100) +
                (1-p('mortality')/100)*(p('rejections_at_slaughter')/100)
            )
        )

        # Calculate number of slaughtered animals
        slaughtered_broilers = inserted_broilers - lost_broilers
        slaughtered_breeding_hens = inserted_breeding_hens - lost_breeding_hens
        slaughtered_breeding_roosters = inserted_breeding_roosters - lost_breeding_roosters

        # Create output DataFrames
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, inserted_n, slaughtered_n, lost_n  = [empty_df.copy() for i in range(4)]

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals)
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)

            heads.loc[:,(ps,slice(None))] = \
                np.array([
                    broilers[sel],
                    breeding_hens[sel],
                    breeding_roosters[sel]
                ]).T

            inserted_n.loc[:,(ps,slice(None))] =\
                np.array([
                    inserted_broilers[sel],
                    inserted_breeding_hens[sel],
                    inserted_breeding_roosters[sel]
                ]).T

            slaughtered_n.loc[:,(ps,slice(None))] = \
                np.array([
                    slaughtered_broilers[sel],
                    slaughtered_breeding_hens[sel],
                    slaughtered_breeding_roosters[sel]
                ]).T

            lost_n.loc[:,(ps,slice(None))] = \
                np.array([
                    lost_broilers[sel],
                    lost_breeding_hens[sel],
                    lost_breeding_roosters[sel]
                ]).T

            n += 1

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'BroilerHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            inserted_n,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'BroilerHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'BroilerHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'BroilerHerd',
            desc = 'Total number of heads lost'
        )

    def _calculate_feed_req(self):

        # Get production systems and animals present
        pss = list(self.data_attr.get("heads").columns.get_level_values('prod_system'))
        anis = list(self.data_attr.get("heads").columns.get_level_values('animal'))

        for ani, ps in zip(anis, pss):
            self.par.set(
                prod_system = ps,
                animal = ani
            )

            # Calculate dry matter requirements
            DM_req = self._calculate_DM_req(ps, ani)

            # Get number of heads of animal = ani & production system = ps
            heads = self.data_attr.get('heads').loc[:,(ps,ani)]

            # Append requirements scaled to number of heads to appropriate 'feed_req_*' DataFrames
            self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'DM')] = DM_req * heads

            # NOTE: THIS METHOD ONLY CALCULATES DM REQUIREMENTS AND THEREFORE RELY ON
            # STRICTLY DEFINING FEED RATIONS WITH 'share_in_ration' PARAMETER

        print('[DM]', sep='', end=' ')

    def _calculate_DM_req(self,ps,ani):

        p = self.par.get

        if ani=='broilers':
            feed_req = (
                p('rounds_per_year') *
                p('feed_conversion_ratio') *
                p('slaughter_weight') *
                p('live_weight_per_CW')
            )
        else:
            feed_req = p('feed_per_animal') / ( p('slaughter_age') / 365.25 )


        return feed_req
