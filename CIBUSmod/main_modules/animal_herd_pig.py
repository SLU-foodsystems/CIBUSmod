import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class PigHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','pig')

    def __init__(self,par,index,**kwargs):

        self.species = 'pigs'
        self.animals = ['sows','boars','piglets','gilts','growing pigs','finishing pigs']
        self.products = ['meat']

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
        sows = self.x / (1+ (p('replacement_rate')/100 * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25))

        tmp_piglets_born = sows * p('litters_per_sow') * p('live_per_litter')
        tmp_piglets_weaned = tmp_piglets_born * (1 - p('mortality_0towean')/100)
        tmp_piglets_delivered = tmp_piglets_weaned * (1 - p('mortality_weantodelivery')/100)

        tmp_piglets_lost_stillborn = sows * p('litters_per_sow') * p('dead_per_litter')
        tmp_piglets_lost_0towean = tmp_piglets_born - tmp_piglets_weaned
        tmp_piglets_lost_weantodelivery = tmp_piglets_weaned - tmp_piglets_delivered
        piglets_lost = tmp_piglets_lost_stillborn + tmp_piglets_lost_0towean + tmp_piglets_lost_weantodelivery

        # Calculate avg. number of live piglets assuming a 50% weight on lost animals
        piglets = (
            (tmp_piglets_weaned + (tmp_piglets_weaned - tmp_piglets_born)*0.5) * p('weaning_age') +
            (tmp_piglets_delivered + (tmp_piglets_delivered - tmp_piglets_weaned)*0.5) * p('post_weaning_nursing_period')
        ) / 365.25



        tmp_growers_to_replacement = sows * p('replacement_rate')/100
        growers_lost = tmp_piglets_delivered * p('mortality',animal='growing pigs')/100
        tmp_growers_to_finishing = tmp_piglets_delivered - growers_lost - tmp_growers_to_replacement
        growers = (tmp_growers_to_finishing + tmp_growers_to_replacement + growers_lost*0.5) * p('growing_period') / 365.25
        gilts = tmp_growers_to_replacement * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25

        finishers_lost = tmp_growers_to_finishing * p('mortality',animal='finishing pigs')/100
        finishers_to_slaughter = tmp_growers_to_finishing - finishers_lost
        finishers = (finishers_to_slaughter + finishers_lost*0.5) * p('finishing_period') / 365.25

        sows_lost = sows * p('mortality',animal='sows')/100
        sows_to_slaughter = tmp_growers_to_replacement - sows_lost

        boars = (sows + gilts) * p('boars_per_sows+gilts')

        boars_to_slaughter = np.zeros(len(sows))
        piglets_to_slaughter = np.zeros(len(sows))
        gilts_to_slaughter = np.zeros(len(sows))
        growers_to_slaughter = np.zeros(len(sows))

        boars_lost = np.zeros(len(sows)) # No losses
        gilts_lost = np.zeros(len(sows)) # No losses

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

        lwg_piglets = (p('live_weight_delivery') - p('birth_weight')) / (p('weaning_age') + p('post_weaning_nursing_period')) * piglets * 365.25
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

        # CALCULATE LIVE WEIGHTS FOR LOST ANIMALS
        # Animals are assumed to be lost half way through each stage

        tmp_lw_piglets_lost_stillborn = (
            p('birth_weight') # kg/head
        ) * tmp_piglets_lost_stillborn # --> kg
        tmp_lw_piglets_lost_0towean = (
            p('birth_weight') + # kg/head
            (p('weaning_age')/2) * # days
            (lwg_piglets / piglets / 365.25) # kg/head/day
        ) * tmp_piglets_lost_0towean # --> kg
        tmp_lw_piglets_lost_weantodelivery = (
            p('live_weight_weaning') + # kg/head
            (p('post_weaning_nursing_period')/2) * # days
            (lwg_piglets / piglets / 365.25) # kg/head/day
        ) * tmp_piglets_lost_weantodelivery # --> kg
        lw_piglets_lost = tmp_lw_piglets_lost_stillborn + tmp_lw_piglets_lost_0towean + tmp_lw_piglets_lost_weantodelivery

        lw_growers_lost = (
            p('live_weight_delivery') + # kg/head
            (p('growing_period')/2) * # days
            growth_rate_growers_and_finishers # kg/head/day
        ) * growers_lost

        lw_finishers_lost = (
            live_weight_after_growing_period + # kg/head
            (p('finishing_period')/2) * # days
            growth_rate_growers_and_finishers # kg/head/day
        ) * finishers_lost

        lw_gilts_lost = np.zeros(len(sows)) # No losses
        lw_sows_lost = p('live_weight', animal='sows') * sows_lost
        lw_boars_lost = np.zeros(len(sows)) # No losses

        # Create output DataFrames
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, lwg, inserted_n, slaughtered_n, lost_n, lost_lw  = [empty_df.copy() for i in range(6)]

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
                    tmp_growers_to_replacement[sel],
                    tmp_piglets_delivered[sel],
                    tmp_growers_to_finishing[sel]
                ]).T

            slaughtered_n.loc[:,(ps,slice(None))] = \
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

            lost_lw.loc[:,(ps,slice(None))] = \
                np.array([
                    lw_sows_lost[sel],
                    lw_boars_lost[sel],
                    lw_piglets_lost[sel],
                    lw_gilts_lost[sel],
                    lw_growers_lost[sel],
                    lw_finishers_lost[sel]
                ]).T

            n += 1

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'PigHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            lwg,
            name = 'lwg',
            unit = 'kg LW',
            orig = 'PigHerd',
            desc = 'Total live weight gains used in calculating nutrient retention in animals'
        )
        self.data_attr.add(
            inserted_n,
            name = 'inserted_n',
            unit = 'heads/year',
            orig = 'PigHerd',
            desc = 'Total number of heads inserted'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'PigHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'PigHerd',
            desc = 'Total number of heads lost'
        )
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'PigHerd',
            desc = 'Total live weight of lost animals'
        )

    def _calculate_feed_req(self):

        p = self.par.get

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

            # Calculate net energy requirements
            NE_req = self._calculate_NE_req(ps, ani)
            self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'NE')] = NE_req * heads

            for mm in ['max','min']:
                if 'f_feed_par' in self.par.data.index.names:
                    if f'{mm}_feed_par_per_NE' in pars:
                        fps = self.par.get_unique('feed_par', qry=f"parameter == '{mm}_feed_par_per_NE'")
                        for fp in fps:
                            self.par.set(feed_par = fp)
                            feed_par_mm = NE_req * p(f'{mm}_feed_par_per_NE')
                            self.data_attr.get(f'feed_req_{mm}').loc[:,(ps,ani,fp)] = feed_par_mm * heads

        print('[NE]', sep='', end=' ')

    def _calculate_NE_req(self,ps,ani):
        '''Calculates Net Energy (NEs [sows and boars] or NEv [other pigs]) requrements for pigs based on
        [1] Simonsson, A. (2006). Fodermedel och näringsrekommendationer för gris. HUV Rapport 266. SLU
        [2] Göransson, L., Lindberg, J.E. (2011). Näringsrekommendationer ver. 2011.1 - Energi'''
        # TODO: live_weight and growth_rate are possible unbound here.

        p = self.par.get

        E_req = None

        # Get average live weight [kg] and growth rate [kg/day]
        if ani in ['sows','boars']:
            live_weight = p('live_weight')
        elif ani == 'gilts':
            growth_rate = self.data_attr.get('lwg').loc[:,(ps,ani)] / self.data_attr.get('heads').loc[:,(ps,ani)] / 365.25
            live_weight = (
                2*p('live_weight', animal='sows') -
                growth_rate * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age'))
            ) / 2
        elif ani == 'piglets':
            # After weaning
            growth_rate = (p('live_weight_delivery') - p('live_weight_weaning')) / p('post_weaning_nursing_period')
        elif ani in ['growing pigs','finishing pigs']:
            growth_rate = self.data_attr.get('lwg').loc[:,(ps,ani)] / self.data_attr.get('heads').loc[:,(ps,ani)] / 365.25

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

        if E_req is None:
            raise AssertionError("Reached end of function without defining E_req.")

        E_req = np.nan_to_num(E_req)

        return E_req

