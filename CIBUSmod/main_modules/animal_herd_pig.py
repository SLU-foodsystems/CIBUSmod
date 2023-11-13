import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class PigHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','pig')

    def __init__(self,par,index,**kwargs):
        
        self.species = 'pigs'
        self.animals = ['sows','boars','piglets','gilts','growing pigs','finishing pigs']
        
        self.x_is = 'sows+gilts'
        
        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates pig herd structure and slaughters based on x (i.e. number of sows).
        
        Parameters
        ----------
        None

        Returns
        -------
        Nothing.
        Sets data attributes self.heads, self.slaughtered_n and self.lost_n'''

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
        sows = self.x / (1+ (p('recruitment_rate')/100 * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25))

        tmp_piglets_born = sows * p('litters_per_sow') * p('live_per_litter')
        tmp_piglets_weaned = tmp_piglets_born * (1 - p('mortality_0towean')/100)
        tmp_piglets_delivered = tmp_piglets_weaned * (1 - p('mortality_weantodelivery')/100)
        
        piglets_lost = (sows * p('litters_per_sow') * p('dead_per_litter')) + (tmp_piglets_born - tmp_piglets_delivered)

        # Calculate avg. number of live piglets assuming a 50% weight on lost animals
        piglets = (
            (tmp_piglets_weaned + (tmp_piglets_weaned - tmp_piglets_born)*0.5) * p('weaning_age') +
            (tmp_piglets_delivered + (tmp_piglets_delivered - tmp_piglets_weaned)*0.5) * p('post_weaning_nursing_period')
        ) / 365.25

        

        tmp_growers_to_recruitment = sows * p('recruitment_rate')/100
        growers_lost = tmp_piglets_delivered * p('mortality',animal='growing pigs')/100
        tmp_growers_to_finishing = tmp_piglets_delivered - growers_lost - tmp_growers_to_recruitment
        growers = (tmp_growers_to_finishing + tmp_growers_to_recruitment + growers_lost*0.5) * p('growing_period') / 365.25
        gilts = tmp_growers_to_recruitment * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25
        
        finishers_lost = tmp_growers_to_finishing * p('mortality',animal='finishing pigs')/100
        finishers_to_slaughter = tmp_growers_to_finishing - finishers_lost
        finishers = (finishers_to_slaughter + finishers_lost*0.5) * p('finishing_period') / 365.25

        sows_lost = sows * p('mortality',animal='sows')/100
        sows_to_slaughter = tmp_growers_to_recruitment - sows_lost

        boars = (sows + gilts) * p('boars_per_sows+gilts')

        boars_to_slaughter = np.zeros(len(sows))
        piglets_to_slaughter = np.zeros(len(sows))
        gilts_to_slaughter = np.zeros(len(sows))
        growers_to_slaughter = np.zeros(len(sows))

        boars_lost = np.zeros(len(sows))
        gilts_lost = np.zeros(len(sows))

        # CALCULATE LIVE WEIGHT GAINS
        # These are in terms of total weight gain in the herd
        # per animal category and year [kg/year]

        # Assumes same growth rate for growers and finishers
        growth_rate_growers_and_finishers = (
            (p('live_weight_slaughter') - p('live_weight_delivery')) / 
            (p('growing_period') + p('finishing_period'))
        ) # kg/head/day
        live_weight_after_growing_period = (
            p('live_weight_delivery') + 
            growth_rate_growers_and_finishers * p('growing_period')
        ) # kg/head
        
        lwg_piglets = (p('live_weight_delivery') - p('birth_weight')) / (p('weaning_age') + p('post_weaning_nursing_period')) * 365.25
        lwg_growers = growth_rate_growers_and_finishers * growers * 365.25
        lwg_finishers = growth_rate_growers_and_finishers * finishers * 365.25
        lwg_gilts = (
            (p('live_weight', animal='sows') - live_weight_after_growing_period) / 
            (p('age_at_first_farrowing') - p('growing_period') - 
             p('post_weaning_nursing_period') - p('weaning_age'))
        ) * gilts * 365.25

        # lwg for sows includes growth of fetus
        lwg_sows = p('litters_per_sow') * (p('live_per_litter') + p('dead_per_litter')) * p('birth_weight') * sows
        lwg_boars = np.zeros(len(sows))

        # Create output DataFrames
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, lwg, inserted_n, slaughter_n, lost_n  = [empty_df.copy() for i in range(5)]

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals) 
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)

            heads.loc[:,(ps,slice(None))] = \
                np.array([
                    sows[sel],
                    boars[sel],
                    piglets[sel],
                    gilts[sel],
                    growers[sel],
                    finishers[sel]
                ]).T
            
            lwg.loc[:,(ps,slice(None))] = \
                np.array([
                    lwg_sows[sel],
                    lwg_boars[sel],
                    lwg_piglets[sel],
                    lwg_gilts[sel],
                    lwg_growers[sel],
                    lwg_finishers[sel]
                ]).T
            
            inserted_n.loc[:,(ps,slice(None))] = \
                np.array([
                    (sows_to_slaughter+sows_lost)[sel],
                    (boars_to_slaughter+boars_lost)[sel],
                    tmp_piglets_born[sel],
                    tmp_growers_to_recruitment[sel],
                    tmp_piglets_delivered[sel],
                    tmp_growers_to_finishing[sel]
                ]).T
            
            slaughter_n.loc[:,(ps,slice(None))] = \
                np.array([
                    sows_to_slaughter[sel],
                    boars_to_slaughter[sel],
                    piglets_to_slaughter[sel],
                    gilts_to_slaughter[sel],
                    growers_to_slaughter[sel],
                    finishers_to_slaughter[sel]
                ]).T
            
            lost_n.loc[:,(ps,slice(None))] = \
                np.array([
                    sows_lost[sel],
                    boars_lost[sel],
                    piglets_lost[sel],
                    gilts_lost[sel],
                    growers_lost[sel],
                    finishers_lost[sel]
                ]).T

            n += 1

        self.heads = heads
        self.lwg = lwg
        self.inserted_n = inserted_n
        self.slaughtered_n = slaughter_n
        self.lost_n = lost_n
        self.data_attr.update(['heads','lwg','inserted_n','slaughtered_n','lost_n'])

    def calculate_feed_E_req(self,ps,ani):
        '''Calculates Net Energy (NEs [sows and boars] or NEv [other pigs]) requrements for pigs based on
        [1] Simonsson, A. (2006). Fodermedel och näringsrekommendationer för gris. HUV Rapport 266. SLU
        [2] Göransson, L., Lindberg, J.E. (2011). Näringsrekommendationer ver. 2011.1 - Energi'''

        p = self.par.get

        # Get average live weight [kg] and growth rate [kg/day]
        if ani in ['sows','boars']:
            live_weight = p('live_weight')
        elif ani == 'gilts':
            growth_rate = self.lwg.loc[:,(ps,ani)] / self.heads.loc[:,(ps,ani)] / 365.25
            live_weight = (
                2*p('live_weight', animal='sows') - 
                growth_rate * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age'))
            ) / 2
        elif ani == 'piglets':
            # After weaning
            growth_rate = (p('live_weight_delivery') - p('live_weight_weaning')) / p('post_weaning_nursing_period')
        elif ani in ['growing pigs','finishing pigs']:
            growth_rate = self.lwg.loc[:,(ps,ani)] / self.heads.loc[:,(ps,ani)] / 365.25
            
        if ani == 'sows':
            E_weaning_to_insemination = 55 * (365.25 - (p('weaning_age') + p('gestation_period')) * p('litters_per_sow')) # [2] 50-60 MJ NEs/day
            E_gestation = 23 * p('gestation_period') * p('litters_per_sow') # [2] Tabell 4
            E_lactation_first_8_days = 53 * 8 * p('litters_per_sow') # [2] Tabell 5
            E_lactation_8_days_to_weaning = (p('weaning_age') - 8) * (p('live_per_litter') * 5.46 + live_weight * 0.058) * p('litters_per_sow') # Derived from [1] Tabell 12 (assuming NE = ME * 0.75)
            
            E_req = (E_weaning_to_insemination + E_gestation + E_lactation_first_8_days + E_lactation_8_days_to_weaning)

        if ani == 'boars':
            E_req = 16.5 + live_weight * 0.033 * 365.25 # Derived from [1] Tabell 2 (assuming NE = ME * 0.75)

        if ani == 'gilts':
            E_req = 23.3 * 365.25 # [2] Tabell 6 >60 kg (gilts are treated as 'growing pigs' for the growing period)
        
        if ani == 'piglets':
            E_req = p('feed_energy_per_growth') * growth_rate * (p('post_weaning_nursing_period') / (p('weaning_age') + p('post_weaning_nursing_period'))) * 365.25
            # E_req = 2.2 + live_weight * 0.41 * (p('post_weaning_nursing_period') / (p('weaning_age') + p('post_weaning_nursing_period'))) * 365.25 # Derived from [2] Tabell 2

        if ani in ['growing pigs','finishing pigs']:
            E_req = p('feed_energy_per_growth') * growth_rate * 365.25

        E_req = np.nan_to_num(E_req)

        return E_req