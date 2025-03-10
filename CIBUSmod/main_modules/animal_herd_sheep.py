import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class SheepHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','sheep')

    def __init__(self,par,index,**kwargs):

        self.species = 'sheep'
        self.animals = ['ewes','rams','lambs']
        self.products = ['meat'] + (
            ['heads'] if kwargs.get('sub_system') == 'other sheep' else []
        )

        self.x_is = 'ewes+rams'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates sheep herd structure and slaughters as a fraction of the number of ewes.
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
        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        idx_len = len(self.index)

        # Calculate number of ewes and rams
        ewes = self.x / (1 + p('rams_per_ewe'))
        rams = self.x - ewes

        lambs_born = (
            ewes *
            (p('fertility')/100) *
            p('lambs_per_ewe')
        )

        lambs_to_recruitment = ewes * (p('recruitment_rate')/100)
        lambs_to_recruitment_rams = rams * (p('recruitment_rate_rams')/100)

        ewes_lost = ewes * (p('mortality', animal='ewes')/100)
        lw_ewes_lost = ewes_lost * p('slaughter_weight') * p('live_weight_per_CW')

        rams_lost = rams * (p('mortality', animal='rams')/100)
        lw_rams_lost = rams_lost * p('slaughter_weight') * p('live_weight_per_CW')

        lambs_lost = lambs_born * (p('mortality', animal='lambs')/100)
        lw_lambs_lost = lambs_lost * ((p('birth_weight') + p('slaughter_weight'))/2) * p('live_weight_per_CW')

        ewes_to_slaughter = lambs_to_recruitment - ewes_lost
        rams_to_slaughter = lambs_to_recruitment_rams - rams_lost
        lambs_to_slaughter = lambs_born - lambs_to_recruitment - lambs_lost

        # Calculate average number of live lambs over the year
        lambs = (
            # Lost lambs are assumed to live to half their slaughter age
            (lambs_lost/2 + lambs_to_slaughter) * p('slaughter_age', animal='lambs') / 365.25 +
            lambs_to_recruitment * p('age_at_first_lambing') / 12
        )

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
                    ewes[sel],
                    rams[sel],
                    lambs[sel]
                ]).T

            inserted_n.loc[:,(ps,slice(None))] = \
                np.array([
                    lambs_to_recruitment[sel],
                    lambs_to_recruitment_rams[sel],
                    lambs_born[sel]
                ]).T

            slaughtered_n.loc[:,(ps,slice(None))] = \
                    np.array([
                        ewes_to_slaughter[sel],
                        rams_to_slaughter[sel],
                        lambs_to_slaughter[sel]
                    ]).T

            lost_n.loc[:,(ps,slice(None))] = \
                    np.array([
                        ewes_lost[sel],
                        rams_lost[sel],
                        lambs_lost[sel]
                    ]).T

            lost_lw.loc[:,(ps,slice(None))] = \
                    np.array([
                        lw_ewes_lost[sel],
                        lw_rams_lost[sel],
                        lw_lambs_lost[sel]
                    ]).T

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'SheepHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            heads,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'SheepHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'SheepHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'SheepHerd',
            desc = 'Total number of heads lost'
        )
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'SheepHerd',
            desc = 'Total live weight of lost animals'
        )

        return None

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
        '''Calculates feed DM requirements from fixed intake per head or lifetime'''

        p = self.par.get

        if ani == 'lambs':
            feed_req = (
                self.data_attr.get('inserted_n').loc[:,(ps,ani)] -
                self.data_attr.get('lost_n').loc[:,(ps,ani)] * 0.5 # 50% feed req. for lost lambs
            ) * p('feed_per_lifetime')
        else:
            feed_req = p('feed_per_head')

        return feed_req
