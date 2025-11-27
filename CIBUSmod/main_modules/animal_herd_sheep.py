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

        lambs_to_replacement = ewes * (p('replacement_rate')/100)
        lambs_to_replacement_rams = rams * (p('replacement_rate_rams')/100)

        ewes_lost = ewes * (p('mortality', animal='ewes')/100)
        lw_ewes_lost = ewes_lost * p('slaughter_weight') * p('live_weight_per_CW')

        rams_lost = rams * (p('mortality', animal='rams')/100)
        lw_rams_lost = rams_lost * p('slaughter_weight') * p('live_weight_per_CW')

        lambs_lost = lambs_born * (p('mortality', animal='lambs')/100)
        lw_lambs_lost = lambs_lost * (p('birth_weight') + p('slaughter_weight')*p('live_weight_per_CW'))/2

        ewes_to_slaughter = lambs_to_replacement - ewes_lost
        rams_to_slaughter = lambs_to_replacement_rams - rams_lost
        lambs_to_slaughter = lambs_born - lambs_to_replacement - lambs_lost

        # CALCULATE LIVE WEIGHT GAINS
        # These are in terms of total weight gain in the herd
        # per animal category and year [kg/year]
        lwg_ewes = np.zeros(len(ewes))
        lwg_rams = np.zeros(len(rams))

        lwg_lambs = (
            (
                lambs_to_replacement * p('slaughter_weight', animal='ewes') * p('live_weight_per_CW')
                + lambs_to_replacement_rams * p('slaughter_weight', animal='rams') * p('live_weight_per_CW')
                + lambs_to_slaughter * p('slaughter_weight', animal='lambs') * p('live_weight_per_CW')
                + lambs_lost * (p('slaughter_weight', animal='lambs')/2) * p('live_weight_per_CW')
            )
            - lambs_born*p('birth_weight')
        )

        # Calculate average number of live lambs over the year
        lambs = (
            # Lost lambs are assumed to live to half their slaughter age
            (lambs_lost/2 + lambs_to_slaughter) * p('slaughter_age', animal='lambs') / 365.25 +
            (lambs_to_replacement + lambs_to_replacement_rams) * p('age_at_first_lambing') / 12
        )

        # Create output DataFrames
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, inserted_n, lwg, slaughtered_n, lost_n, lost_lw  = [empty_df.copy() for i in range(6)]
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
                    lambs_to_replacement[sel],
                    lambs_to_replacement_rams[sel],
                    lambs_born[sel]
                ]).T
            
            lwg.loc[:,(ps,slice(None))] = \
                    np.array([
                        lwg_ewes[sel],
                        lwg_rams[sel],
                        lwg_lambs[sel]
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
            inserted_n,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'SheepHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            lwg,
            name = 'lwg',
            unit = 'kg LW',
            orig = 'SheepHerd',
            desc = 'Total live weight gains used in calculating nutrient retention in animals'
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

        # Get available paramters
        pars = self.par.data.index.get_level_values('parameter')

        for ani, ps in zip(anis, pss):
            self.par.set(
                prod_system = ps,
                animal = ani
            )

            # Get number of heads of animal = ani & production system = ps
            heads = self.data_attr.get('heads').loc[:,(ps,ani)]

            if (
                'maintanance_energy_factor' in pars and
                'gestation_energy_add' in pars and
                'lactation_energy_factor' in pars and
                'growth_energy_factor' in pars and
                'energy_share_before_weaning_from_milk' in pars
            ):
                # Calculate metabolizable energy (ME) requirements and append to feed_req DataFrame
                ME_req = self._calculate_ME_req(ps, ani)
                self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'ME')] = ME_req * heads
            else:
                # Calculate dry matter requirements and append to feed_req DataFrame
                DM_req = self._calculate_DM_req(ps, ani)
                self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'DM')] = DM_req * heads

    def _calculate_DM_req(self,ps,ani):
        '''Calculates feed DM requirements from fixed intake per head or lifetime'''

        p = self.par.get

        if ani == 'lambs':
            feed_req = (
                self.data_attr.get('inserted_n').loc[:,(ps,ani)] -
                self.data_attr.get('lost_n').loc[:,(ps,ani)] * 0.5 # 50% feed req. for lost lambs
            ) * p('feed_per_lifetime') / self.data_attr.get('heads').loc[:,(ps,ani)]
        else:
            feed_req = p('feed_per_head')

        return feed_req

    def _calculate_ME_req(self,ps,ani):
        '''Calculates Metabolizable Energy (ME) requrements for sheep based on
        Spörndly, R. (ed.). (2003). Fodertabeller för idisslare 2003. HUV Rapport 257. SLU'''

        p = self.par.get
        
        heads = self.data_attr.get('heads').loc[:,(ps,ani)]
        lwg = self.data_attr.get('lwg').loc[:,(ps,ani)]

        lambs_born = self.data_attr.get('inserted_n').loc[:,(ps,'lambs')]
        lambs_slaughtered = self.data_attr.get('slaughtered_n').loc[:,(ps,'lambs')]
        lambs_lost = self.data_attr.get('lost_n').loc[:,(ps,'lambs')]

        # If no animals return zero array
        if heads.sum() == 0:
            return np.zeros(len(self.index))

        # Get average live weight [kg] and growth rate [kg/day] for calculating energy requirements
        growth_rate = lwg / heads / 365.25
        if ani in ['ewes','rams']:
            live_weight = p('slaughter_weight')*p('live_weight_per_CW')
        elif ani == 'lambs':
            # Calculate average final (i.e. slaughter, lost or replacing ewe/ram)
            # weight of lambs to calculate average live weight of lambs in herd
            avg_end_weight = (
                p('slaughter_weight')*p('live_weight_per_CW') * lambs_slaughtered
                + (p('slaughter_weight')*p('live_weight_per_CW')/2) * lambs_lost
                + p('slaughter_weight', animal='ewes')*p('live_weight_per_CW') * (lambs_born-lambs_slaughtered-lambs_lost)
            ) / lambs_born
            self.par.set(animal=ani)
            live_weight = (p('birth_weight') + avg_end_weight) / 2

        # Get share of energy from milk for lambs
        if ani == 'lambs':
            # Calculate average final (i.e. slaughter, lost or replacing ewe/ram)
            # age of lambs to calculate share of time suckling
            avg_end_age = (
                p('slaughter_age') * lambs_slaughtered
                + (p('slaughter_age')/2) * lambs_lost
                + (p('age_at_first_lambing')*30.44) * (lambs_born-lambs_slaughtered-lambs_lost)
            ) / lambs_born # [days]
            # Calcualte energy share from milk as share of time suckling
            # times share of energy from milk during the suckling period
            E_share_milk = (p('weaning_age')/avg_end_age) * (p('energy_share_before_weaning_from_milk')/100)
        else:
            E_share_milk = 0

        # Calculate energy for maintenance and growth besed on supplied factors
        # and average live weigh and growth rate
        E_maintenance = p('maintanance_energy_factor') * live_weight**0.75
        E_growth = p('growth_energy_factor') * live_weight**0.75 * growth_rate

        if ani == 'ewes':
            E_lactation = p('lactation_energy_factor') * (p('fertility')/100) * p('weaning_age')
            E_gestation = p('gestation_energy_add') * (p('fertility')/100)
        else:
            E_lactation = 0
            E_gestation = 0

        # Total ME req. [MJ/year] (excl. gestation)
        E_req_final = (E_maintenance + E_growth)*(1-E_share_milk)*365.25 + E_lactation + E_gestation
        E_req_final = np.nan_to_num(E_req_final)

        return E_req_final