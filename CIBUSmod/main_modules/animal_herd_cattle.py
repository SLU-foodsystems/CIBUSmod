import warnings
import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class CattleHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','cattle')

    def __init__(self,par,index,**kwargs):
        
        self.species = 'cattle'
        self.animals = ['cows','breeding bulls','calves','heifers','steers','bulls']

        self.x_is = 'cows'
        
        super().__init__(par,index,**kwargs)
        
    def calculate_herd(self):
        '''Calculates cattle herd structure and slaughters based on the number of cows.
        
        Parameters
        ----------
        cows : pandas.Series
            A pandas series contaning the number of cows.
        **kwargs : str or list
            Keyword arguments to be passed on as filters to the ParameterRetriever.

        Returns
        -------
        tuple of pandas.DataFrames
            The order of DataFrames are (heads, slaughtered_n, lost_n)    
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
        cows = self.x

        # Check if there is any redistribution of animals across production systems.
        # If so extend rows in 'cows' to make room for animals in production systems
        # to which animals are redistributed.
        if self.prod_system in self.par.get_unique('from_ps'):
            redist = True
            to_ps = self.par.get_unique('to_ps', qry=f'f_from_ps == "{self.prod_system}"')

            # Add zeros to cows array
            cows = np.concatenate((
                cows,
                np.zeros(idx_len*len(to_ps))
            ))

            # Update ParameterRetriever prod_system filter
            self.par.remove(list(self.index.names + ['prod_system']))
            self.par.set(prod_system = [self.prod_system]*idx_len + [ps for ps in to_ps for n in range(idx_len)])

            # Update ofther filters from index (need to store and check 'filters_from_index' value!?!?!)
            for idx in self.index.names:
                self.par.set(**{idx : list(self.index.get_level_values(idx))*(len(to_ps)+1)})

        else:
            redist = False
        
        # Calves born per year per cow
        tmp_calves_per_year = \
            ( 12/p('calving_interval') * (1-p('recruitment_rate')/100) + (p('recruitment_rate')/100) ) \
            * ( 1*(1-p('twin_birth')/100) + 2*(p('twin_birth')/100) ) * (1-p('stillborn_calf')/100)

        # Total no. of calves born per year
        tmp_calves_born = cows * tmp_calves_per_year
        # ... of which male
        tmp_male_calves_born = tmp_calves_born * p('ratio_male_calf')/100
        # ... of which female
        tmp_female_calves_born = tmp_calves_born - tmp_male_calves_born

        self.par.set(animal = 'calves')
        
        # No. male calves surviving past weaning
        tmp_male2weaned = tmp_male_calves_born * (1-p('mortality_male_0towean')/100)
        # No. female calves surviving past weaning
        tmp_female2weaned = tmp_female_calves_born * (1-p('mortality_female_0towean')/100)

        # Handle redistribution of calves from one production system to another.
        tmp_male2weaned_before_redist = tmp_male2weaned
        tmp_female2weaned_before_redist = tmp_female2weaned

        if redist:
            # Redistribute calves across production systems
            fps = self.prod_system
            fsel = range(0, idx_len)
            n = 1
            
            for tps in to_ps:
                tsel = range(n*idx_len, (n+1)*idx_len)

                # Redistribute male calves
                tmp_redist = (tmp_male2weaned_before_redist * np.nan_to_num(p('redist_male_calves', from_ps=fps, to_ps=tps)/100))[fsel]
                tmp_male2weaned[fsel] = tmp_male2weaned[fsel] - tmp_redist
                tmp_male2weaned[tsel] = tmp_male2weaned[tsel] + tmp_redist

                # Redistribute female calves
                tmp_redist = (tmp_female2weaned_before_redist * np.nan_to_num(p('redist_female_calves', from_ps=fps, to_ps=tps)/100))[fsel]
                tmp_female2weaned[fsel] = tmp_female2weaned[fsel] - tmp_redist
                tmp_female2weaned[tsel] = tmp_female2weaned[tsel] + tmp_redist

                n += 1

        # No. male calves surviving to slaughter
        tmp_male2end = tmp_male2weaned * (1-p('mortality_male_weantoslaught')/100)
        # No. female calves surviving to slaughter/recruitment
        tmp_female2end = tmp_female2weaned * (1-p('mortality_female_weantoslaught')/100)

        # No. dead male calves
        tmp_male_calves2dead = tmp_male_calves_born - tmp_male2weaned_before_redist + tmp_male2weaned - tmp_male2end
        # No. dead female calves
        tmp_female_calves2dead = tmp_female_calves_born - tmp_female2weaned_before_redist + tmp_female2weaned - tmp_female2end
        
        # No. female calves --> recruitment heifers
        tmp_calves2recruitment = cows * p('recruitment_rate')/100

        # No. calves slaughtered before 1 year
        tmp_calves2slaughter = (tmp_male2end + tmp_female2end - tmp_calves2recruitment) * p('slaughter_share_as_calf')/100
        # No. male calves slaughtered before 1 year (return 0 if div. by 0)
        tmp_male_calves2slaughter = np.divide(
            tmp_calves2slaughter * (tmp_male2end),
            (tmp_male2end + tmp_female2end - tmp_calves2recruitment),
            out = np.zeros_like(tmp_calves2slaughter),
            where = (tmp_male2end + tmp_female2end - tmp_calves2recruitment) != 0
        )

        # No. female calves slaughtered before 1 year
        tmp_female_calves2slaughter = tmp_calves2slaughter - tmp_male_calves2slaughter

        # No. calves --> heifers for slaughter
        tmp_calves2heifer = tmp_female2end - tmp_calves2recruitment - tmp_female_calves2slaughter
        # No. calves --> steers for slaughter
        tmp_calves2steer = (tmp_male2end - tmp_male_calves2slaughter) * p('slaughter_share_male_as_steers')/100
        # No. calves --> bulls for slaughter 
        tmp_calves2bull = tmp_male2end - tmp_male_calves2slaughter - tmp_calves2steer

        # CALCULATE AVERAGE ANNUAL NUMBER OF ANIMALS
        breeding_bulls = np.zeros(len(cows))
        
        self.par.set(animal='calves')
        calves = (
            (tmp_male_calves_born-tmp_male2weaned) * p('mortality_male_0towean_age') + \
            (tmp_female_calves_born-tmp_female2weaned) * p('mortality_female_0towean_age') + \
            (tmp_male2weaned-tmp_male2end) * p('mortality_male_weantoslaught_age') + \
            (tmp_female2weaned-tmp_female2end) * p('mortality_female_weantoslaught_age') + \
            (tmp_calves2recruitment+tmp_calves2heifer+tmp_calves2steer+tmp_calves2bull) * 365.25 + \
            tmp_calves2slaughter * p('slaughter_age') * 30.44
        )/365.25
        
        heifers = \
            tmp_calves2recruitment * ((p('AFC')-12)/12) + \
            tmp_calves2heifer * ((p('slaughter_age',animal='heifers')-12)/12)

        steers = tmp_calves2steer * ((p('slaughter_age',animal='steers')-12)/12)

        bulls = tmp_calves2bull * ((p('slaughter_age',animal='bulls')-12)/12)

        cows2lost = cows * p('mortality',animal='cows')/100
        breeding_bulls2lost = np.zeros(len(cows))
        calves2lost = tmp_male_calves2dead + tmp_female_calves2dead
        heifers2lost = np.zeros(len(cows))
        steers2lost = np.zeros(len(cows))
        bulls2lost = np.zeros(len(cows))
        
        cows2slaughter = cows * p('recruitment_rate')/100 - cows2lost
        breeding_bulls2slaughter = np.zeros(len(cows))
        calves2slaughter = tmp_calves2slaughter
        heifers2slaughter = tmp_calves2heifer
        steers2slaughter = tmp_calves2steer
        bulls2slaughter = tmp_calves2bull

        # CALCULATE LIVE WEIGHT GAINS
        # These are in terms of total weight gain in the herd
        # per animal category and year [kg/year]
        lw_calves_start = p('birth_weight', animal='calves')
        lw_calves_weaning = p('live_weight_weaning', animal='calves') # kg/head
        lw_calves_slaughter = p('live_weight_slaughter', animal='calves')

        lwg_heifers = (
            # For recruitment
            (
                (p('live_weight', animal='cows') - lw_calves_weaning) / 
                (p('AFC', animal='heifers')*30.44 - p('weaning_age', animal='calves'))
            ) * #  -> kg/head/day
            tmp_calves2recruitment * ((p('AFC')-12)/12) + # -> kg/day
            # For slaughter
            (
                (p('live_weight_slaughter', animal='heifers') - lw_calves_weaning) / 
                (p('slaughter_age', animal='heifers')*30.44 - p('weaning_age', animal='calves'))
            ) * # -> kg/head/day
            tmp_calves2heifer * ((p('slaughter_age',animal='heifers')-12)/12) # -> kg/day
        ) * 365.25 # -> kg/year

        lwg_steers = (
            (p('live_weight_slaughter', animal='steers') - lw_calves_weaning) / 
            (p('slaughter_age', animal='steers')*30.44 - p('weaning_age', animal='calves')) # -> kg/head/day
        ) * steers * 365.2 # -> kg/year

        lwg_bulls = (
            (p('live_weight_slaughter', animal='bulls') - lw_calves_weaning) / 
            (p('slaughter_age', animal='bulls')*30.44 - p('weaning_age', animal='calves')) # -> kg/head/day
        ) * bulls * 365.25 # -> kg/year

        lw_calves_1yr = (
            ((lwg_heifers + lwg_steers + lwg_bulls) / (heifers + steers + bulls) / 365.25) # -> kg/head/day
            * (365.25 - p('weaning_age', animal='calves')) # -> kg/head
            + lw_calves_weaning # -> kg/head
        )

        lwg_calves = (
            (
                # Calves reaching 1 year
                (lw_calves_1yr - lw_calves_start) * # -> kg/head
                (tmp_calves2recruitment + tmp_calves2heifer +
                tmp_calves2heifer + tmp_calves2steer + tmp_calves2bull) # -> kg/year
                +
                # Calves to slaughter
                (lw_calves_slaughter - lw_calves_start) /  # -> kg/head
                p('slaughter_age', animal='calves') * 12 * # -> kg/head/year
                tmp_calves2slaughter # -> kg/year
            ) / (tmp_calves2recruitment + tmp_calves2heifer + tmp_calves2slaughter +
             tmp_calves2heifer + tmp_calves2steer + tmp_calves2bull) # -> kg/head/year
        ) * calves # -> kg/year

        # lwg for cows includes fetus growth
        lwg_cows = (12/p('calving_interval', animal = 'cows')) * p('birth_weight', animal='calves') * cows # -> kg/year
        lwg_breeding_bulls = np.zeros(len(cows))

        # Create output DataFrames
        pss = [self.prod_system]+list(to_ps) if redist else [self.prod_system] # Output production systems (==[self.prod_system] if no redistribution of animals)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, lwg, slaughtered_n, lost_n  = [empty_df.copy() for i in range(4)]

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals) 
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)

            heads.loc[:,(ps,slice(None))] = \
                np.array([
                    cows[sel], 
                    breeding_bulls[sel], 
                    calves[sel], 
                    heifers[sel], 
                    steers[sel], 
                    bulls[sel]
                ]).T
            
            lwg.loc[:,(ps,slice(None))] = \
                np.array([
                    lwg_cows[sel], 
                    lwg_breeding_bulls[sel], 
                    lwg_calves[sel], 
                    lwg_heifers[sel], 
                    lwg_steers[sel], 
                    lwg_bulls[sel]
                ]).T
            
            slaughtered_n.loc[:,(ps,slice(None))] = \
                np.array([
                    cows2slaughter[sel], 
                    breeding_bulls2slaughter[sel], 
                    calves2slaughter[sel], 
                    heifers2slaughter[sel], 
                    steers2slaughter[sel], 
                    bulls2slaughter[sel]
                ]).T
            
            lost_n.loc[:,(ps,slice(None))] = \
                np.array([
                    cows2lost[sel],
                    breeding_bulls2lost[sel],
                    calves2lost[sel],
                    heifers2lost[sel],
                    steers2lost[sel],
                    bulls2lost[sel]
                ]).T

            n += 1
        
        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'CattleHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            lwg,
            name = 'lwg',
            unit = 'kg LW',
            orig = 'CattleHerd',
            desc = 'Total live weight gains used in calculating nutrient retention in animals'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'CattleHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'CattleHerd',
            desc = 'Total number of heads lost'
        )

    def calculate_feed_E_req(self,ps,ani):
        '''Calculates Metabolizable Energy (ME) and water requrements for cattle based on
        Spörndly, R. (ed.). (2003). Fodertabeller för idisslare 2003. HUV Rapport 257. SLU'''

        p = self.par.get

        # Get average live weight [kg] and growth rate [kg/day] for calculating energy requirements
        if ani in ['cows','breeding bulls']:
            live_weight = p('live_weight')
            growth_rate = self.lwg.loc[:,(ps,ani)] / self.heads.loc[:,(ps,ani)] / 365.25
            if ani == 'cows':
                 # Subtract fetus growth
                 growth_rate -= (12/p('calving_interval', animal = 'cows')) * p('birth_weight', animal='calves') / 365.25
                 self.par.set(animal = ani)
        elif ani == 'calves':
            live_weight_1yr = p('birth_weight') + self.lwg.loc[:,(ps,ani)] / self.heads.loc[:,(ps,ani)]
            live_weight_pre_weaning = (p('live_weight_weaning') + p('birth_weight')) / 2
            growth_rate_pre_weaning = (p('live_weight_weaning') - p('birth_weight')) / p('weaning_age')
            live_weight = (live_weight_1yr + p('live_weight_weaning')) / 2
            growth_rate = (live_weight_1yr - p('live_weight_weaning')) / (365.25 - p('weaning_age'))
        else:
            growth_rate = self.lwg.loc[:,(ps,ani)] / self.heads.loc[:,(ps,ani)] / 365.25
            live_weight = (2*p('live_weight_slaughter') - growth_rate * (p('slaughter_age')*30.44 - 365.25)) / 2
              
        # Daily ME req. for maintenance [MJ/day]
        E_maintenance = p('maintanance_energy_factor') * live_weight**0.75

        # Daily ME req. for changes in body weight [MJ/day]
        if ani=='cows':
            E_growth = 35 * growth_rate # (Tabell 1)
        else:
            E_growth = (growth_rate * (6.28 + 0.0188 * live_weight)) / ((1 - 0.3 * growth_rate) * 0.435) # (Tabell 4a)
            if np.array(live_weight > 825).any() or np.array(growth_rate > 2).any():
                warnings.warn(f'Growth energy equation defined up to 825 kg LW and 2.0 kg LWG/day.')


        if ani == 'cows':
            # ME req. for lactation [MJ/day]
            # Milk in kg ECM: milk kg x 0,25 + fat kg x 12,2 + protein kg x 7,7 = kg ECM
            milk = p('milk_prod') * (0.25 + p('milk_fat')/100*12.2 + p('milk_protein')/100*7.7) / 365.25
            E_lactation = p('lactation_energy_factor') * milk

            # ME req. for gestation [MJ/year]
            E_gestation = (12/p('calving_interval')) * live_weight * p('gestation_energy_factor')
        else:
            E_lactation = 0
            E_gestation = 0
        
        # Total ME req. [MJ/day] (excl. gestation)
        if ani=='calves':
            E_from_milk = 0 # !!!! NEED TO INCLUDE THIS AND TIE TO MILK PRODUCTION !!!!
            E_pre_weaning = 0.16 * live_weight_pre_weaning + 12.5 * growth_rate_pre_weaning - E_from_milk # Equation deduced from (Tabell 3)
            E_post_weaning = (E_maintenance + E_growth)

            # Total requirements
            E_req = (E_pre_weaning * p('weaning_age') + E_post_weaning * (365.25 - p('weaning_age'))) / 365.25
        else:
            E_req = (E_maintenance + E_growth + E_lactation)

        # Adjust energy requirements based on factors for different animals and breeds and 
        # convert to MJ/year and add energy requirements for gestation
        E_req_final = (E_req * p('energy_adjustment_factor') + p('energy_adjustment_addend')) * 365.25 + E_gestation
        
        E_req_final = np.nan_to_num(E_req_final)

        return E_req_final