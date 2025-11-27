import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class GoatHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','goat')

    def __init__(self,par,index,**kwargs):

        self.species = 'sheep'
        self.animals = ['does','bucks','kids']
        self.products = ['meat','milk']

        self.x_is = 'does'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates goat herd structure and slaughters as a fraction of the number of does.
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

        # Calculate number of does and bucks
        does = self.x / (1 + p('bucks_per_doe'))
        bucks = self.x - does

        kids_born = (
            does *
            (p('fertility')/100) *
            p('kids_per_doe')
        )

        kids_to_replacement = does * (p('replacement_rate')/100)
        kids_to_replacement_bucks = bucks * (p('replacement_rate_bucks')/100)

        does_lost = does * (p('mortality', animal='does')/100)
        lw_does_lost = does_lost * p('slaughter_weight') * p('live_weight_per_CW')

        bucks_lost = bucks * (p('mortality', animal='bucks')/100)
        lw_bucks_lost = bucks_lost * p('slaughter_weight') * p('live_weight_per_CW')

        kids_lost = kids_born * (p('mortality', animal='kids')/100)
        lw_kids_lost = kids_lost * (p('birth_weight') + p('slaughter_weight')*p('live_weight_per_CW'))/2

        does_to_slaughter = kids_to_replacement - does_lost
        bucks_to_slaughter = kids_to_replacement_bucks - bucks_lost
        kids_to_slaughter = kids_born - kids_to_replacement - kids_lost

        # CALCULATE LIVE WEIGHT GAINS
        # These are in terms of total weight gain in the herd
        # per animal category and year [kg/year]
        lwg_does = np.zeros(len(does))
        lwg_bucks = np.zeros(len(bucks))

        lwg_kids = (
            (
                kids_to_replacement * p('slaughter_weight', animal='does') * p('live_weight_per_CW')
                + kids_to_replacement_bucks * p('slaughter_weight', animal='bucks') * p('live_weight_per_CW')
                + kids_to_slaughter * p('slaughter_weight', animal='kids') * p('live_weight_per_CW')
                + kids_lost * (p('slaughter_weight', animal='kids')/2) * p('live_weight_per_CW')
            )
            - kids_born*p('birth_weight')
        )

        # Calculate average number of live kids over the year
        kids = (
            # Lost kids are assumed to live to half their slaughter age
            (kids_lost/2 + kids_to_slaughter) * p('slaughter_age', animal='kids') / 365.25 +
            (kids_to_replacement + kids_to_replacement_bucks) * p('age_at_first_kidding') / 12
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
                    does[sel],
                    bucks[sel],
                    kids[sel]
                ]).T

            inserted_n.loc[:,(ps,slice(None))] = \
                np.array([
                    kids_to_replacement[sel],
                    kids_to_replacement_bucks[sel],
                    kids_born[sel]
                ]).T
            
            lwg.loc[:,(ps,slice(None))] = \
                    np.array([
                        lwg_does[sel],
                        lwg_bucks[sel],
                        lwg_kids[sel]
                    ]).T

            slaughtered_n.loc[:,(ps,slice(None))] = \
                    np.array([
                        does_to_slaughter[sel],
                        bucks_to_slaughter[sel],
                        kids_to_slaughter[sel]
                    ]).T

            lost_n.loc[:,(ps,slice(None))] = \
                    np.array([
                        does_lost[sel],
                        bucks_lost[sel],
                        kids_lost[sel]
                    ]).T

            lost_lw.loc[:,(ps,slice(None))] = \
                    np.array([
                        lw_does_lost[sel],
                        lw_bucks_lost[sel],
                        lw_kids_lost[sel]
                    ]).T

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'GoatHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            inserted_n,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'GoatHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            lwg,
            name = 'lwg',
            unit = 'kg LW',
            orig = 'GoatHerd',
            desc = 'Total live weight gains used in calculating nutrient retention in animals'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'GoatHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'GoatHerd',
            desc = 'Total number of heads lost'
        )
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'GoatHerd',
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

        if ani == 'kids':
            feed_req = (
                self.data_attr.get('inserted_n').loc[:,(ps,ani)] -
                self.data_attr.get('lost_n').loc[:,(ps,ani)] * 0.5 # 50% feed req. for lost kids
            ) * p('feed_per_lifetime') / self.data_attr.get('heads').loc[:,(ps,ani)]
        else:
            feed_req = p('feed_per_head')

        return feed_req

    def _calculate_ME_req(self,ps,ani):
        '''Calculates Metabolizable Energy (ME) requrements for goats based on
        Spörndly, R. (ed.). (2003). Fodertabeller för idisslare 2003. HUV Rapport 257. SLU'''

        p = self.par.get
        
        heads = self.data_attr.get('heads').loc[:,(ps,ani)]
        lwg = self.data_attr.get('lwg').loc[:,(ps,ani)]

        kids_born = self.data_attr.get('inserted_n').loc[:,(ps,'kids')]
        kids_slaughtered = self.data_attr.get('slaughtered_n').loc[:,(ps,'kids')]
        kids_lost = self.data_attr.get('lost_n').loc[:,(ps,'kids')]

        # If no animals return zero array
        if heads.sum() == 0:
            return np.zeros(len(self.index))

        # Get average live weight [kg] and growth rate [kg/day] for calculating energy requirements
        growth_rate = lwg / heads / 365.25
        if ani in ['does','bucks']:
            live_weight = p('slaughter_weight')*p('live_weight_per_CW')
        elif ani == 'kids':
            # Calculate average final (i.e. slaughter, lost or replacing ewe/ram)
            # weight of lambs to calculate average live weight of lambs in herd
            avg_end_weight = (
                p('slaughter_weight')*p('live_weight_per_CW') * kids_slaughtered
                + (p('slaughter_weight')*p('live_weight_per_CW')/2) * kids_lost
                + p('slaughter_weight', animal='does')*p('live_weight_per_CW') * (kids_born-kids_slaughtered-kids_lost)
            ) / kids_born
            self.par.set(animal=ani)
            live_weight = (p('birth_weight') + avg_end_weight) / 2

        # Get share of energy from milk for lambs
        if ani == 'kids':
            # Calculate average final (i.e. slaughter, lost or replacing ewe/ram)
            # age of lambs to calculate share of time suckling
            avg_end_age = (
                p('slaughter_age') * kids_slaughtered
                + (p('slaughter_age')/2) * kids_lost
                + (p('age_at_first_kidding')*30.44) * (kids_born-kids_slaughtered-kids_lost)
            ) / kids_born # [days]
            # Calcualte energy share from milk as share of time suckling
            # times share of energy from milk during the suckling period
            E_share_milk = (p('weaning_age')/avg_end_age) * (p('energy_share_before_weaning_from_milk')/100)
        else:
            E_share_milk = 0

        # Calculate energy for maintenance and growth besed on supplied factors
        # and average live weigh and growth rate
        E_maintenance = p('maintanance_energy_factor') * live_weight**0.75
        E_growth = p('growth_energy_factor') * live_weight**0.75 * growth_rate

        if ani == 'does':
            E_lactation = p('lactation_energy_factor') * p('weaning_age')
            E_gestation = p('gestation_energy_add')
        else:
            E_lactation = 0
            E_gestation = 0

        # Total ME req. [MJ/year] (excl. gestation)
        E_req_final = (E_maintenance + E_growth)*(1-E_share_milk)*365.25 + E_lactation + E_gestation
        E_req_final = np.nan_to_num(E_req_final)

        return E_req_final