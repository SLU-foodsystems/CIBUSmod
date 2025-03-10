import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class LayerHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','layer poultry')

    def __init__(self,par,index,**kwargs):

        self.species = 'poultry'
        self.breed = 'layer'
        self.animals = ['laying chicks','laying hens (16-28 weeks)','laying hens (29-59 weeks)',
                        'laying hens (>59 weeks)','breeding hens and roosters']
        self.products = ['meat', 'eggs']

        self.x_is = 'total hens'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates layer herd structure and slaugthers/losses based on x (i.e. number of
        total laying hens on average over the year).

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
        s = self.par.set

        idx_len = len(self.index)

        # Get total number of hens (incl. parents).
        total_hens = self.x

        # Assume laying hens = total hens and adjust at the end
        laying_hens = total_hens

        # Distribute to age groups assuming distribution
        # according to time spans

        s(animal='laying hens (16-28 weeks)')
        laying_hens_16_28 = (
            laying_hens *
            (p('weeks_in_stage')  / p('time_in_hen_house'))
        )

        s(animal='laying hens (29-59 weeks)')
        laying_hens_29_59 = (
            laying_hens *
            (p('weeks_in_stage') / p('time_in_hen_house'))
        )

        laying_hens_60 = (
            laying_hens -
            (laying_hens_16_28 + laying_hens_29_59)
        )

        # Calculate number of animals entering each stage
        # assuming that mortality at each stage occurs
        # after half of the time in the stage
        s(animal='laying hens (16-28 weeks)')
        inserted_laying_hens_16_28 = (
            laying_hens_16_28 /
            (p('weeks_in_stage')/52) /
            (1+0.5*p('mortality')/100)
        )
        lost_laying_hens_16_28 = inserted_laying_hens_16_28 * p('mortality')/100
        lw_lost_laying_hens_16_28 = lost_laying_hens_16_28 * p('average_live_weight')

        s(animal='laying hens (29-59 weeks)')
        inserted_laying_hens_29_59 = (
            laying_hens_29_59 /
            (p('weeks_in_stage')/52) /
            (1+0.5*p('mortality')/100)
        )
        lost_laying_hens_29_59 = inserted_laying_hens_29_59 * p('mortality')/100
        lw_lost_laying_hens_29_59 = lost_laying_hens_29_59 * p('average_live_weight')

        s(animal='laying hens (>59 weeks)')
        inserted_laying_hens_60 = (
            laying_hens_60 /
            (p('weeks_in_stage')/52) /
            (1+0.5*p('mortality')/100)
        )
        lost_laying_hens_60 = (
            inserted_laying_hens_60 * p('mortality')/100 +
            (
                inserted_laying_hens_60 *
                (1 - p('mortality')/100) *
                p('rejections_at_slaughter')/100
            )
        )
        lw_lost_laying_hens_60 = lost_laying_hens_60 * p('average_live_weight')
        slaughtered_laying_hens_60 = inserted_laying_hens_60 - lost_laying_hens_60

        s(animal='laying chicks')
        inserted_chicks = (
            inserted_laying_hens_16_28 /
            (1 - p('mortality')/100)
        )
        chicks = (
            inserted_chicks *
            (1 + 0.5 * p('mortality')/100) *
            (p('weeks_in_stage') / 52)
        )
        lost_chicks = inserted_chicks * p('mortality')/100
        lw_lost_chicks = lost_chicks * p('average_live_weight')

        # Calculate number of parent animals needed
        self.par.remove('animal')
        parents = inserted_chicks / p('chicks_per_breeding hen') * (1 + 1/p('roosters_per_breeding_hen'))

        # Create zero arrays for animals with no assumed slaughter/losses
        zeros = np.array([0]*idx_len)
        slaughtered_chicks = zeros
        slaughtered_laying_hens_16_28 = zeros
        slaughtered_laying_hens_29_59 = zeros
        slaughtered_parents = zeros
        lost_parents = zeros
        lw_lost_parents = zeros
        inserted_parents = zeros

        # Create output DataFrames
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, inserted_n, slaughtered_n, lost_n, lost_lw  = [empty_df.copy() for i in range(5)]
        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals)
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)

            heads.loc[:,(ps,slice(None))] = \
                np.array([
                    chicks[sel],
                    laying_hens_16_28[sel],
                    laying_hens_29_59[sel],
                    laying_hens_60[sel],
                    parents[sel]
                ]).T

            inserted_n.loc[:,(ps,slice(None))] = \
                np.array([
                    inserted_chicks[sel],
                    inserted_laying_hens_16_28[sel],
                    inserted_laying_hens_29_59[sel],
                    inserted_laying_hens_60[sel],
                    inserted_parents[sel]
                ]).T

            slaughtered_n.loc[:,(ps,slice(None))] = \
                np.array([
                    slaughtered_chicks[sel],
                    slaughtered_laying_hens_16_28[sel],
                    slaughtered_laying_hens_29_59[sel],
                    slaughtered_laying_hens_60[sel],
                    slaughtered_parents[sel]
                ]).T

            lost_n.loc[:,(ps,slice(None))] = \
                np.array([
                    lost_chicks[sel],
                    lost_laying_hens_16_28[sel],
                    lost_laying_hens_29_59[sel],
                    lost_laying_hens_60[sel],
                    lost_parents[sel]
                ]).T

            lost_lw.loc[:,(ps,slice(None))] = \
                np.array([
                    lw_lost_chicks[sel],
                    lw_lost_laying_hens_16_28[sel],
                    lw_lost_laying_hens_29_59[sel],
                    lw_lost_laying_hens_60[sel],
                    lw_lost_parents[sel]
                ]).T

            n += 1

        # Adjust to include parents in total hens (i.e. x = 1 --> parents + laying hens = 1)
        adj_factor = total_hens / heads.drop('laying chicks', level='animal', axis=1).sum(axis=1)
        heads = heads.mul(adj_factor, axis=0)
        inserted_n = inserted_n.mul(adj_factor, axis=0)
        slaughtered_n = slaughtered_n.mul(adj_factor, axis=0)
        lost_n = lost_n.mul(adj_factor, axis=0)

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'LayerHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            inserted_n,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'LayerHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'LayerHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'LayerHerd',
            desc = 'Total number of heads lost'
        )
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'LayerHerd',
            desc = 'Total live weight of lost animals'
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
        feed_req = p('feed_per_head')

        return feed_req

