import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class GoatHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','goat')

    def __init__(self,par,index,**kwargs):

        self.species = 'goats'
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
        does = self.x
        bucks = does * p('bucks_per_doe')

        kids_born = (
            does *
            (p('fertility')/100) *
            p('kids_per_doe')
        )

        kids_to_replacement = does * (p('replacement_rate')/100)
        kids_to_replacement_bucks = bucks * (p('replacement_rate_bucks')/100)

        kids_culled = (
            kids_born -
            kids_to_replacement -
            kids_to_replacement_bucks
        ) * (p('surplus_kids_culled')/100)

        does_lost = does * (p('mortality', animal='does')/100)
        lw_does_lost = does_lost * p('slaughter_weight') * p('live_weight_per_CW')

        bucks_lost = bucks * (p('mortality', animal='bucks')/100)
        lw_bucks_lost = bucks_lost * p('slaughter_weight') * p('live_weight_per_CW')

        kids_lost = (
            (kids_born - kids_culled) * (p('mortality', animal='kids')/100) +
            kids_culled
        )
        lw_kids_lost = (
            (kids_lost - kids_culled) * (p('birth_weight') + p('slaughter_weight')*p('live_weight_per_CW'))/2 +
            kids_culled * p('birth_weight')
        )

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
                + (kids_lost - kids_culled) * (p('slaughter_weight', animal='kids')/2) * p('live_weight_per_CW')
            )
            - kids_born*p('birth_weight')
        )

        # Calculate average number of live kids over the year
        kids = (
            # Lost kids are assumed to live to half their slaughter age
            ((kids_lost - kids_culled)/2 + kids_to_slaughter) * p('slaughter_age', animal='kids') / 365.25 +
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

        # Remove 'milk_to_kids' attribute if it exists
        if 'milk_to_kids' in self.data_attr:
            self.data_attr.remove('milk_to_kids')

        # Get production systems and animals present
        pss = list(self.data_attr.get("heads").columns.get_level_values('prod_system'))
        anis = list(self.data_attr.get("heads").columns.get_level_values('animal'))

        # Make sure kids are hendeled first to get milk from does
        # to kids
        anis.insert(0, anis.pop(anis.index('kids')))

        # Get available paramters
        pars = self.par.data.index.get_level_values('parameter')

        for ani, ps in zip(anis, pss):
            self.par.set(
                prod_system = ps,
                animal = ani
            )

            # Get number of heads of animal = ani & production system = ps
            heads = self.data_attr.get('heads').loc[:,(ps,ani)]

            # Calculate metabolizable energy (ME) requirements and append to feed_req DataFrame
            ME_req = self._calculate_ME_req(ps, ani)
            self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'ME')] = ME_req * heads

    def _calculate_ME_req(self,ps,ani):
        '''Calculates Metabolizable Energy (ME) requrements for goats based on
        Spörndly, R. (ed.). (2003). Fodertabeller för idisslare 2003. HUV Rapport 257. SLU'''

        p = self.par.get
        
        heads = self.data_attr.get('heads').loc[:,(ps,ani)]
        lwg = self.data_attr.get('lwg').loc[:,(ps,ani)]

        kids_born = self.data_attr.get('inserted_n').loc[:,(ps,'kids')]
        kids_slaughtered = self.data_attr.get('slaughtered_n').loc[:,(ps,'kids')]
        kids_culled = (
            kids_born -
            self.data_attr.get('inserted_n').loc[:,(ps,'does')] -
            self.data_attr.get('inserted_n').loc[:,(ps,'bucks')]
        ) * (p('surplus_kids_culled')/100)
        kids_lost = (
            self.data_attr.get('lost_n').loc[:,(ps,'kids')] -
            kids_culled
        )

        # If no animals return zero array
        if heads.sum() == 0:
            return np.zeros(len(self.index))

        # Get average live weight [kg] and growth rate [kg/day] for calculating energy requirements
        growth_rate = lwg / heads / 365.25
        if ani in ['does','bucks']:
            live_weight = p('slaughter_weight')*p('live_weight_per_CW')
        elif ani == 'kids':
            # Calculate average final (i.e. slaughter, lost or replacing doe/buck)
            # weight of kids to calculate average live weight of kids in herd
            # Kids culled at birth excluded as zero feed demand assumed for those
            avg_end_weight = (
                p('slaughter_weight')*p('live_weight_per_CW') * kids_slaughtered
                + (p('slaughter_weight')*p('live_weight_per_CW')/2) * kids_lost
                + p('slaughter_weight', animal='does')*p('live_weight_per_CW') * (kids_born-kids_slaughtered-kids_culled-kids_lost)
            ) / (kids_born - kids_culled)
            self.par.set(animal=ani)
            live_weight = (p('birth_weight') + avg_end_weight) / 2

        # Get share of energy from milk for kids
        if ani == 'kids':
            # Calculate average final (i.e. slaughter, lost or replacing doe/buck)
            # age of kids to calculate share of time suckling
            avg_end_age = (
                p('slaughter_age') * kids_slaughtered
                + (p('slaughter_age')/2) * kids_lost
                + (p('age_at_first_kidding')*30.44) * (kids_born-kids_slaughtered-kids_lost-kids_culled)
            ) / (kids_born - kids_culled) # [days]
            # Calcualte energy share from milk as share of time suckling
            # times share of energy from milk during the suckling period
            E_share_milk = (p('weaning_age')/avg_end_age) * (p('energy_share_before_weaning_from_milk')/100)
        else:
            E_share_milk = 0

        # Calculate energy for maintenance and growth besed on supplied factors
        # and average live weigh and growth rate
        E_maintenance = p('maintanance_energy_factor') * live_weight**0.75
        E_growth = p('growth_energy_factor') * growth_rate

        if ani == 'does':
            # ME req. for lactation [MJ/day]
            # Milk production is taken as the maximum of 'milk_prod' parameter and
            # calculated milk to kids
            # Milk in kg ECM: milk kg x 0,25 + fat kg x 12,2 + protein kg x 7,7 = kg ECM
            milk_prod = np.maximum(
                p('milk_prod'),
                self.data_attr.get('milk_to_kids').sum(axis=1).values
            )
            milk = milk_prod * (0.25 + p('milk_fat')/100*12.2 + p('milk_protein')/100*7.7)
            E_lactation = p('lactation_energy_factor') * milk

            # ME req. for gestation [MJ/year]
            E_gestation = p('gestation_energy_add') * (p('fertility')/100)
        else:
            E_lactation = 0
            E_gestation = 0

        # Total ME req. [MJ/year] (excl. gestation)
        E_req_final = (E_maintenance + E_growth)*(1-E_share_milk)*365.25 + E_lactation + E_gestation
        E_req_final = np.nan_to_num(E_req_final)

        
        if ani == 'kids':
            # Calculate milk to kids and store data attribute
            milk_to_kids = pd.DataFrame(
                (
                    (
                        (E_maintenance + E_growth) * E_share_milk
                        * self.data_attr.get('heads').loc[:,(ps, ani)]
                        * 365.25
                    ) / p('energy_in_milk_to_kids')
                ).values,
                index = self.index,
                columns = pd.Index([ps], name='prod_system')
            )
            self.data_attr.add(
                milk_to_kids,
                name = 'milk_to_kids',
                unit = 'kg/year',
                orig = 'GoatHerd',
                desc = 'Milk fed to kids'
            )

        return E_req_final