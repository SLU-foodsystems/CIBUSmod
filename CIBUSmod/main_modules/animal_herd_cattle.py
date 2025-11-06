import warnings
import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class CattleHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','cattle')

    def __init__(self,par,index,**kwargs):

        self.species = 'cattle'
        self.animals = ['cows','breeding bulls',
                        'calves, suckling', # 0 -> weaning
                        'calves, for slaughter','calves, heifer','calves, steer','calves, bull', # waening -> 1 year
                        'heifers','steers','bulls'] # 1 year ->
        self.products = ['meat', 'milk']

        self.x_is = 'cows'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates cattle herd structure, slaughtered and lost animals and live weight gains as a fraction of the number of cows.
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
        if 'f_from_ps' in self.par.data.index.names and self.prod_system in self.par.get_unique('from_ps'):
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

            # Update ofther filters from index
            for idx in self.index.names:
                self.par.set(**{idx : list(self.index.get_level_values(idx))*(len(to_ps)+1)})

        else:
            redist = False

        breeding_bulls = cows * p('breeding_bulls_per_cow')

        self.par.set(animal='cows')

        # Calculate replacement rate
        replacement_rate = p('replacement_rate')/100

        # Calves born per year per cow
        tmp_calvings_per_year = 12/p('calving_interval') * (1-replacement_rate) + replacement_rate
        tmp_calves_per_year = (
            tmp_calvings_per_year *
            ( 1*(1-p('twin_birth')/100) + 2*(p('twin_birth')/100) ) * (1-p('stillborn_calf')/100)
        )

        # Total no. of calves born and stillborn per year
        tmp_calves_born = cows * tmp_calves_per_year
        tmp_calves_stillborn = cows * tmp_calvings_per_year * (p('stillborn_calf')/100)
        # ... of which male
        tmp_male_calves_born = tmp_calves_born * p('ratio_male_calf')/100
        # ... of which female
        tmp_female_calves_born = tmp_calves_born - tmp_male_calves_born

        self.par.set(animal = 'calves, suckling')

        # No. male calves surviving past weaning
        tmp_male2weaned = tmp_male_calves_born * (1-p('mortality_male_0towean')/100)
        # No. female calves surviving past weaning
        tmp_female2weaned = tmp_female_calves_born * (1-p('mortality_female_0towean')/100)

        # No. male/female calves lost before weaning
        tmp_male2lost = tmp_male_calves_born - tmp_male2weaned
        tmp_female2lost = tmp_female_calves_born - tmp_female2weaned

        # Handle redistribution of calves from one production system to another.
        tmp_male2weaned_before_redist = tmp_male2weaned.copy()
        tmp_female2weaned_before_redist = tmp_female2weaned.copy()

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
        # No. female calves surviving to slaughter/replacement
        tmp_female2end = tmp_female2weaned * (1-p('mortality_female_weantoslaught')/100)

        # No. calves lost after weaning
        tmp_male2lost_after_weaning = tmp_male2weaned - tmp_male2end
        tmp_female2lost_after_weaning = tmp_female2weaned - tmp_female2end

        # No. female calves --> replacement heifers
        tmp_calves2replacement = cows * p('replacement_rate')/100

        # No. calves slaughtered before 1 year
        tmp_calves2slaughter = (tmp_male2end + tmp_female2end - tmp_calves2replacement) * p('slaughter_share_as_calf')/100
        # No. male calves slaughtered before 1 year (return 0 if div. by 0)
        tmp_male_calves2slaughter = np.divide(
            tmp_calves2slaughter * (tmp_male2end),
            (tmp_male2end + tmp_female2end - tmp_calves2replacement),
            out = np.zeros_like(tmp_calves2slaughter),
            where = (tmp_male2end + tmp_female2end - tmp_calves2replacement) != 0
        )

        # No. female calves slaughtered before 1 year
        tmp_female_calves2slaughter = tmp_calves2slaughter - tmp_male_calves2slaughter

        # No. calves --> heifers for slaughter
        tmp_calves2heifer = tmp_female2end - tmp_calves2replacement - tmp_female_calves2slaughter
        # No. calves --> steers for slaughter
        tmp_calves2steer = (tmp_male2end - tmp_male_calves2slaughter) * p('slaughter_share_male_as_steers')/100
        # No. calves --> breeding bulls
        tmp_calves2breeding_bulls = breeding_bulls * p('replacement_rate_bulls')/100
        # Adjust number of steers for cases where male calves aren't enough
        np.where(
            tmp_calves2steer + tmp_male_calves2slaughter + tmp_calves2breeding_bulls > tmp_male2end,
            tmp_male2end - tmp_male_calves2slaughter - tmp_calves2breeding_bulls,
            tmp_calves2steer
        )
        # No. calves --> bulls for slaughter
        tmp_calves2bull = tmp_male2end - tmp_male_calves2slaughter - tmp_calves2steer - tmp_calves2breeding_bulls

        # CALCULATE LOST ANIMALS PER CATEGORY
        cows2lost = cows * p('mortality',animal='cows')/100
        breeding_bulls2lost = np.zeros(len(cows)) # No losses assumed for breeding bulls for now...
        calves_suckling2lost = tmp_male2lost + tmp_female2lost + tmp_calves_stillborn
        tmp_calves_slaughter2lost_male = np.nan_to_num(
            tmp_male2lost_after_weaning * (
                tmp_male_calves2slaughter /
                _np_zero_to_nan(tmp_male_calves2slaughter + tmp_calves2steer + tmp_calves2bull)  # To avoid zero div errors
            )
        )
        tmp_calves_slaughter2lost_female = np.nan_to_num(
            tmp_female2lost_after_weaning * (
                tmp_female_calves2slaughter /
                _np_zero_to_nan(tmp_female_calves2slaughter + tmp_calves2replacement + tmp_calves2heifer) #  # To avoid zero div errors
            )
        )
        calves_slaughter2lost = tmp_calves_slaughter2lost_male + tmp_calves_slaughter2lost_female
        calves_heifer2lost = np.nan_to_num(
            tmp_female2lost_after_weaning * (
                (tmp_calves2replacement + tmp_calves2heifer) /
                _np_zero_to_nan(tmp_female_calves2slaughter + tmp_calves2replacement + tmp_calves2heifer)  # To avoid zero div errors
            )
        )
        calves_steer2lost = np.nan_to_num(
            tmp_male2lost_after_weaning * (
                tmp_calves2steer /
                _np_zero_to_nan(tmp_male_calves2slaughter + tmp_calves2steer + tmp_calves2bull)  # To avoid zero div errors
            )
        )
        calves_bull2lost = np.nan_to_num(
            tmp_male2lost_after_weaning * (
                tmp_calves2bull /
                _np_zero_to_nan(tmp_male_calves2slaughter + tmp_calves2steer + tmp_calves2bull) # To avoid zero div errors
            )
        )
        heifers2lost = np.zeros(len(cows)) # No losses
        steers2lost = np.zeros(len(cows)) # No losses
        bulls2lost = np.zeros(len(cows)) # No losses

        assert np.isclose(
            tmp_male_calves_born + tmp_female_calves_born + tmp_calves_stillborn - calves_suckling2lost,
            tmp_male2weaned_before_redist + tmp_female2weaned_before_redist
        ).all(), "Born calves - Lost calves != Weaned calves"

        assert np.isclose(
            tmp_male2weaned + tmp_female2weaned - calves_slaughter2lost - calves_heifer2lost - calves_steer2lost - calves_bull2lost,
            tmp_calves2slaughter + tmp_calves2replacement + tmp_calves2heifer + tmp_calves2steer + tmp_calves2bull + tmp_calves2breeding_bulls
        ).all(), "Weaned calves - Lost calves != Calves to slaughter + heifers + steers + bulls + breeding bulls"

        # CALCULATE AVERAGE ANNUAL NUMBER OF ANIMALS

        tmp_calves_suckling_male = (
            tmp_male2lost * p('mortality_male_0towean_age') +
            tmp_male2weaned_before_redist * p('weaning_age')
        ) / 365.25
        tmp_calves_suckling_female = (
            tmp_female2lost * p('mortality_female_0towean_age') +
            tmp_female2weaned_before_redist * p('weaning_age')
        ) / 365.25
        calves_suckling = tmp_calves_suckling_male + tmp_calves_suckling_female

        tmp_calves_slaughter_male = (
            tmp_calves_slaughter2lost_male * (p('mortality_male_weantoslaught_age') - p('weaning_age')) +
            tmp_male_calves2slaughter * (p('slaughter_age', animal='calves, for slaughter')*30.4 - p('weaning_age'))
        ) / 365.25
        tmp_calves_slaughter_female = (
            tmp_calves_slaughter2lost_female * (p('mortality_female_weantoslaught_age') - p('weaning_age')) +
            tmp_female_calves2slaughter * (p('slaughter_age', animal='calves, for slaughter')*30.4 - p('weaning_age'))
        ) / 365.25
        calves_slaughter = tmp_calves_slaughter_male + tmp_calves_slaughter_female

        calves_heifer = (
            calves_heifer2lost * (p('mortality_female_weantoslaught_age') - p('weaning_age')) +
            (tmp_calves2replacement + tmp_calves2heifer) * (365.25 - p('weaning_age'))
        ) / 365.25

        calves_steer = (
            calves_steer2lost * (p('mortality_male_weantoslaught_age') - p('weaning_age')) +
            tmp_calves2steer * (365.25 - p('weaning_age'))
        ) / 365.25

        calves_bull = (
            calves_bull2lost * (p('mortality_male_weantoslaught_age') - p('weaning_age')) +
            tmp_calves2bull * (365.25 - p('weaning_age'))
        ) / 365.25

        tmp_heifers_replacement = tmp_calves2replacement * ((p('AFC')-12)/12)
        tmp_heifers_slaughter = tmp_calves2heifer * ((p('slaughter_age',animal='heifers')-12)/12)
        heifers = tmp_heifers_replacement + tmp_heifers_slaughter

        steers = tmp_calves2steer * ((p('slaughter_age',animal='steers')-12)/12)

        bulls = tmp_calves2bull * ((p('slaughter_age',animal='bulls')-12)/12)

        # GET FINAL NUMBER OF ANIMALS TO SLAUGHTER
        cows2slaughter = cows * p('replacement_rate')/100 - cows2lost
        breeding_bulls2slaughter = tmp_calves2breeding_bulls
        calves_suckling2slaughter = np.zeros(len(cows)) # No slaughter
        calves_slaughter2slaughter = tmp_calves2slaughter
        calves_heifer2slaughter = np.zeros(len(cows)) # No slaughter
        calves_steer2slaughter = np.zeros(len(cows)) # No slaughter
        calves_bull2slaughter = np.zeros(len(cows)) # No slaughter
        heifers2slaughter = tmp_calves2heifer
        steers2slaughter = tmp_calves2steer
        bulls2slaughter = tmp_calves2bull

        # CALCULATE LIVE WEIGHT GAINS
        # These are in terms of total weight gain in the herd
        # per animal category and year [kg/year]
        self.par.remove('animal')
        lw_calves_start = p('birth_weight')
        weaning_age = p('weaning_age') # days

        tmp_lwg_heifers = np.nan_to_num( # <----
            (
                # For replacement
                (
                    (p('live_weight_first_calving')) /
                    (p('AFC')*30.44)
                ) * #  -> kg/head/day
                tmp_heifers_replacement + # -> kg/day
                # For slaughter
                (
                    (p('live_weight_slaughter', animal='heifers')) /
                    (p('slaughter_age', animal='heifers')*30.44)
                ) * # -> kg/head/day
                tmp_heifers_slaughter # -> kg/day
            ) / _np_zero_to_nan(heifers) # To avoid zero div errors
        ) # kg/head/day
        lwg_calves_heifer = tmp_lwg_heifers * calves_heifer * 365.25 # -> kg/year
        lwg_heifers = tmp_lwg_heifers * heifers * 365.25 # -> kg/year

        tmp_lwg_steers = (
            (p('live_weight_slaughter', animal='steers')) /
            (p('slaughter_age', animal='steers')*30.44) # -> kg/head/day
        )
        lwg_calves_steer = tmp_lwg_steers * calves_steer * 365.2 # -> kg/year
        lwg_steers = tmp_lwg_steers * steers * 365.2 # -> kg/year

        tmp_lwg_bulls = (
            (p('live_weight_slaughter', animal='bulls')) /
            (p('slaughter_age', animal='bulls')*30.44) # -> kg/head/day
        )
        lwg_calves_bull = tmp_lwg_bulls * calves_bull * 365.25 # -> kg/year
        lwg_bulls = tmp_lwg_bulls * bulls * 365.25 # -> kg/year

        tmp_lwg_calves_slaughter_male = (
            # Male calves
            (p('live_weight_slaughter', animal='calves, for slaughter')) /
            (p('slaughter_age', animal='calves, for slaughter')*30.44) # -> kg/head/day
        )
        tmp_lwg_calves_slaughter_female = (
            # Female calves
            (p('live_weight_slaughter', animal='calves, for slaughter')) /
            (p('slaughter_age', animal='calves, for slaughter')*30.44) # -> kg/head/day
        )
        lwg_calves_slaughter = (
            tmp_lwg_calves_slaughter_male * tmp_calves_slaughter_male * 365.25 + # -> kg/year
            tmp_lwg_calves_slaughter_female * tmp_calves_slaughter_female * 365.25 # -> kg/year
        )

        lw_calves_weaning_male = lw_calves_start + (
            tmp_lwg_bulls * (tmp_calves2bull+tmp_calves2breeding_bulls) +
            tmp_lwg_steers * tmp_calves2steer +
            tmp_lwg_calves_slaughter_male * tmp_male_calves2slaughter
        ) / (tmp_calves2bull+tmp_calves2breeding_bulls+tmp_calves2steer+tmp_male_calves2slaughter) * p('weaning_age') # -> kg
        
        lw_calves_weaning_female = lw_calves_start + (
            tmp_lwg_heifers * (tmp_calves2heifer+tmp_calves2replacement) + 
            tmp_lwg_calves_slaughter_female * tmp_female_calves2slaughter
        ) / (tmp_calves2heifer+tmp_calves2replacement) * p('weaning_age') # -> kg

        tmp_lwg_calves_suckling_male = (
            # Male calves
            (lw_calves_weaning_male - lw_calves_start) / p('weaning_age') * # -> kg/head/day
            tmp_calves_suckling_male # -> kg/day
            * 365.25 # -> kg/year
        )
        tmp_lwg_calves_suckling_female = (
            # Female calves
            (lw_calves_weaning_female - lw_calves_start) / p('weaning_age') * # -> kg/head/day
            tmp_calves_suckling_female # -> kg/day
            * 365.25 # -> kg/year
        )
        lwg_calves_suckling = tmp_lwg_calves_suckling_male + tmp_lwg_calves_suckling_female

        # lwg for cows includes fetus growth
        lwg_cows_fetus = tmp_calves_per_year * p('birth_weight') * cows # -> kg/year
        lwg_cows_growth = (
            (p('live_weight', animal='cows') - p('live_weight_first_calving')) / # -> kg
            (1/replacement_rate) # -> kg/year
        )
        lwg_cows = lwg_cows_fetus + lwg_cows_growth

        lwg_breeding_bulls = (
            (
                p('live_weight_slaughter', animal='breeding bulls') -
                (lw_calves_weaning_male + tmp_lwg_bulls * (365.25 - weaning_age)) # -> kg/Head
            ) / ((p('slaughter_age') - 12) * 30.4) # -> kg/head/day
            * 365.25 # -> kg/head/year
            * breeding_bulls # -> kg/year
        )

        # CALCULATE LIVE WEIGHTS FOR LOST ANIMALS       
        lw_calves_suckling2lost = (
            # Stillborn calves
            p('birth_weight') * tmp_calves_stillborn # --> kg
            +
            # Male calves
            np.nan_to_num(
                lw_calves_start + # kg
                p('mortality_male_0towean_age') * # days
                (tmp_lwg_calves_suckling_male / _np_zero_to_nan(tmp_calves_suckling_male) / 365.25) # kg/head/day
            ) * tmp_male2lost # --> kg
            +
            # Female calves
            np.nan_to_num(
                lw_calves_start + # kg
                p('mortality_female_0towean_age') * # days
                (tmp_lwg_calves_suckling_female / _np_zero_to_nan(tmp_calves_suckling_female) / 365.25) # kg/head/day
            ) * tmp_female2lost # --> kg
        )
        tmp_calves_slaughter2lost_male
        tmp_calves_slaughter2lost_female
        
        lw_calves_slaughter2lost = (
            # Male
            np.nan_to_num(
                lw_calves_weaning_male + # kg
                (p('mortality_male_weantoslaught_age') - p('weaning_age')) * # days
                (tmp_lwg_calves_slaughter_male / _np_zero_to_nan(tmp_calves_slaughter_male) / 365.25) # kg/head/day
            ) * tmp_calves_slaughter2lost_male # --> kg
            +
            # Female
            np.nan_to_num(
                lw_calves_weaning_female + # kg
                (p('mortality_female_weantoslaught_age') - p('weaning_age')) * # days
                (tmp_lwg_calves_slaughter_female / _np_zero_to_nan(tmp_calves_slaughter_female) / 365.25) # kg/head/day
            ) * tmp_calves_slaughter2lost_female # --> kg
        )

        lw_calves_heifer2lost = np.nan_to_num(
            lw_calves_weaning_female + # kg
            (p('mortality_female_weantoslaught_age') - p('weaning_age')) * # days
            (lwg_calves_heifer / _np_zero_to_nan(calves_heifer) / 365.25) # kg/head/day
        ) * calves_heifer2lost # --> kg

        lw_calves_steer2lost = np.nan_to_num(
            lw_calves_weaning_male + # kg
            (p('mortality_male_weantoslaught_age') - p('weaning_age')) * # days
            (lwg_calves_steer / _np_zero_to_nan(calves_steer) / 365.25) # kg/head/day
        ) * calves_steer2lost # --> kg

        lw_calves_bull2lost = np.nan_to_num(
            lw_calves_weaning_male + # kg
            (p('mortality_male_weantoslaught_age') - p('weaning_age')) * # days
            (lwg_calves_bull / _np_zero_to_nan(calves_bull) / 365.25) # kg/head/day
        ) * calves_bull2lost # --> kg
        
        lw_heifers2lost = np.zeros(len(cows)) # No losses
        lw_steers2lost = np.zeros(len(cows)) # No losses
        lw_bulls2lost = np.zeros(len(cows)) # No losses

        lw_cows2lost = cows2lost * p('live_weight', animal='cows')
        lw_breeding_bulls2lost = np.zeros(len(cows)) # No losses

        # Create output DataFrames
        pss = [self.prod_system]+list(to_ps) if redist else [self.prod_system] # Output production systems (==[self.prod_system] if no redistribution of animals)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, lwg, slaughtered_n, lost_n, lost_lw  = [empty_df.copy() for i in range(5)]

        # Create dict of DataFrames for storing LW @ waening to access in _calculate_ME_req()
        self.lw_weaning = {
            k : pd.DataFrame(
                columns = pd.Index([ps for ps in pss], name='prod_system'),
                index = self.index,
                dtype = 'float64'
            ) for k in ['male','female']
        }

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals)
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)

            heads.loc[:,(ps,slice(None))] = \
                np.array([
                    cows[sel],
                    breeding_bulls[sel],
                    calves_suckling[sel],
                    calves_slaughter[sel],
                    calves_heifer[sel],
                    calves_steer[sel],
                    calves_bull[sel],
                    heifers[sel],
                    steers[sel],
                    bulls[sel]
                ]).T

            lwg.loc[:,(ps,slice(None))] = \
                np.array([
                    lwg_cows[sel],
                    lwg_breeding_bulls[sel],
                    lwg_calves_suckling[sel],
                    lwg_calves_slaughter[sel],
                    lwg_calves_heifer[sel],
                    lwg_calves_steer[sel],
                    lwg_calves_bull[sel],
                    lwg_heifers[sel],
                    lwg_steers[sel],
                    lwg_bulls[sel]
                ]).T

            slaughtered_n.loc[:,(ps,slice(None))] = \
                np.array([
                    cows2slaughter[sel],
                    breeding_bulls2slaughter[sel],
                    calves_suckling2slaughter[sel],
                    calves_slaughter2slaughter[sel],
                    calves_heifer2slaughter[sel],
                    calves_steer2slaughter[sel],
                    calves_bull2slaughter[sel],
                    heifers2slaughter[sel],
                    steers2slaughter[sel],
                    bulls2slaughter[sel]
                ]).T

            lost_n.loc[:,(ps,slice(None))] = \
                np.array([
                    cows2lost[sel],
                    breeding_bulls2lost[sel],
                    calves_suckling2lost[sel],
                    calves_slaughter2lost[sel],
                    calves_heifer2lost[sel],
                    calves_steer2lost[sel],
                    calves_bull2lost[sel],
                    heifers2lost[sel],
                    steers2lost[sel],
                    bulls2lost[sel]
                ]).T
            
            lost_lw.loc[:,(ps,slice(None))] = \
                np.array([
                    lw_cows2lost[sel],
                    lw_breeding_bulls2lost[sel],
                    lw_calves_suckling2lost[sel],
                    lw_calves_slaughter2lost[sel],
                    lw_calves_heifer2lost[sel],
                    lw_calves_steer2lost[sel],
                    lw_calves_bull2lost[sel],
                    lw_heifers2lost[sel],
                    lw_steers2lost[sel],
                    lw_bulls2lost[sel]
                ]).T
            
            # Store LW @ weaning to access in _calculate_ME_req()
            self.lw_weaning['male'].loc[:,ps] = lw_calves_weaning_male[sel]
            self.lw_weaning['female'].loc[:,ps] = lw_calves_weaning_female[sel]

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
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'CattleHerd',
            desc = 'Total live weight of lost animals'
        )

        return None

    def _calculate_feed_req(self):

        p = self.par.get

        # Remove 'milk_to_calves' attribute if it exists
        if 'milk_to_calves' in self.data_attr:
            self.data_attr.remove('milk_to_calves')

        # Get production systems and animals present
        pss = list(self.data_attr.get("heads").columns.get_level_values('prod_system'))
        anis = list(self.data_attr.get("heads").columns.get_level_values('animal'))

        # Make sure calves are hendeled first to get milk from cows
        # to calves
        anis.insert(0, anis.pop(anis.index('calves, suckling')))

        # Get available paramters
        pars = self.par.data.index.get_level_values('parameter')

        for ani, ps in zip(anis, pss):
            self.par.set(
                prod_system = ps,
                animal = ani
            )

            # Get number of heads of animal = ani & production system = ps
            heads = self.data_attr.get('heads').loc[:,(ps,ani)]

            # Calculate metabolizable energy requirements
            ME_req = self._calculate_ME_req(ps, ani)
            self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'ME')] = ME_req * heads

            # Calculate protein requirements in terms of AAT
            if 'AAT_factor' in pars:
                AAT_min = ME_req * p('AAT_factor')
                self.data_attr.get('feed_req_min').loc[:,(ps,ani,'AAT')] = AAT_min * heads

            # Calculate min and max PBV
            if (
                'min_PBV' in pars and
                'max_PBV' in pars and
                'min_PBV_per_ME' in pars and
                'max_PBV_per_ME' in pars
            ):
                PBV_min = p('min_PBV') * 365.25 + p('min_PBV_per_ME') * ME_req
                PBV_max = p('max_PBV') * 365.25 + p('max_PBV_per_ME') * ME_req
                self.data_attr.get('feed_req_min').loc[:,(ps,ani,'PBV')] = PBV_min * heads
                self.data_attr.get('feed_req_max').loc[:,(ps,ani,'PBV')] = PBV_max * heads

            # Get maximum dry matter intake
            if 'max_DMI' in pars:
                DM_max = p('max_DMI') * 365.25
                self.data_attr.get('feed_req_max').loc[:,(ps,ani,'DM')] = DM_max * heads

    def _calculate_ME_req(self,ps,ani):
        '''Calculates Metabolizable Energy (ME) requrements for cattle based on
        Spörndly, R. (ed.). (2003). Fodertabeller för idisslare 2003. HUV Rapport 257. SLU'''

        p = self.par.get

        # If no animals return zero array
        if self.data_attr.get('heads').loc[:,(ps,ani)].sum() == 0:
            return np.zeros(len(self.index))

        # Get average live weight [kg] and growth rate [kg/day] for calculating energy requirements
        growth_rate = self.data_attr.get('lwg').loc[:,(ps,ani)] / self.data_attr.get('heads').loc[:,(ps,ani)] / 365.25
        if ani in ['cows']:
            live_weight = p('live_weight')
            if ani == 'cows':
                # Subtract fetus growth
                replacement_rate = p('replacement_rate')/100
                calves_per_year = \
                    ( 12/p('calving_interval') * (1-replacement_rate) + replacement_rate ) \
                    * ( 1*(1-p('twin_birth')/100) + 2*(p('twin_birth')/100) ) * (1-p('stillborn_calf')/100)
                growth_rate -= calves_per_year * p('birth_weight') / 365.25
        else:
            if ani == 'calves, suckling':
                live_weight = (2*p('birth_weight') + growth_rate * p('weaning_age')) / 2
            elif ani in ['calves, heifer', 'calves, steer', 'calves, bull']:
                sex = 'female' if ani == 'calves, heifer' else 'male'
                live_weight = (2*self.lw_weaning[sex].loc[:,ps].values + growth_rate * (365.25 - p('weaning_age'))) / 2
            else:
                live_weight = (2*p('live_weight_slaughter') - growth_rate * (p('slaughter_age')*30.44 - 365.25)) / 2

        if ani == 'calves, suckling':
            E_req_tot = (0.16 * live_weight + 12.5 * growth_rate) # Equation deduced from (Tabell 3)

            # Share of energy from milk
            E_from_milk = E_req_tot * (p('energy_share_before_weaning_from_milk')/100)

            # Calculate milk to calves and store data attribute
            milk_to_calves = pd.DataFrame(
                ((E_from_milk * self.data_attr.get('heads').loc[:,(ps, ani)] * 365.25) / p('energy_in_milk_to_calves')).values,
                index = self.index,
                columns = pd.Index([ps], name='prod_system')
            )
            self.data_attr.add(
                milk_to_calves,
                name = 'milk_to_calves',
                unit = 'kg/year',
                orig = 'CattleHerd',
                desc = 'Milk fed to calves'
            )

            # Subtract energy from milk to get energy from feeds and convert to MJ/year
            E_req = (E_req_tot - E_from_milk) * 365.25
            return E_req

        # Daily ME req. for maintenance [MJ/day]
        E_maintenance = p('maintanance_energy_factor') * live_weight**0.75

        # Daily ME req. for changes in body weight [MJ/day]
        if ani in ['cows', 'breeding bulls']:
            E_growth = 35 * growth_rate # (Tabell 1) - This factor is for older dairy cows. Here the same is assumed for beef cows and breeding bulls
        else:
            E_growth = (growth_rate * (6.28 + 0.0188 * live_weight)) / ((1 - 0.3 * growth_rate) * 0.435) # (Tabell 4a)
            if np.array(live_weight > 825).any() or np.array(growth_rate > 2).any():
                warnings.warn('Growth energy equation defined up to 825 kg LW and 2.0 kg LWG/day.')

        if ani == 'cows':
            # ME req. for lactation [MJ/day]
            # Milk production is taken as the maximum of 'milk_prod' parameter and
            # calculated milk to calves
            # Milk in kg ECM: milk kg x 0,25 + fat kg x 12,2 + protein kg x 7,7 = kg ECM
            milk_prod = np.maximum(
                p('milk_prod'),
                self.data_attr.get('milk_to_calves').sum(axis=1).values
            )
            milk = milk_prod * (0.25 + p('milk_fat')/100*12.2 + p('milk_protein')/100*7.7) / 365.25
            E_lactation = p('lactation_energy_factor') * milk

            # ME req. for gestation [MJ/year]
            E_gestation = (12/p('calving_interval')) * live_weight * p('gestation_energy_factor')

            # Subtract maintanance energy requirements for the time between
            # slaughter of cow to first calving of replacement heifer
            # Mainly applicable in suckler cow systems where the cow is
            # slaughtered after weaning (autumn) but first calving of
            # replacing heifer is in the spring.
            time_lag = p('time_lag_replacement')
            time_lag_share = time_lag / (time_lag + ((1/replacement_rate) * 12))
            E_maintenance *= (1-time_lag_share)
        else:
            E_lactation = 0
            E_gestation = 0

        # Total ME req. [MJ/day] (excl. gestation)
        E_req = (E_maintenance + E_growth + E_lactation)

        # Adjust energy requirements based on factors for different animals and breeds and
        # convert to MJ/year and add energy requirements for gestation
        E_req_final = (E_req * p('energy_adjustment_factor') + p('energy_adjustment_addend')) * 365.25 + E_gestation

        E_req_final = np.nan_to_num(E_req_final)

        return E_req_final

# Function to convert all zeros in np.array to np.nan
# in order to avoid div by zero problems
def _np_zero_to_nan(x):
    return np.where(x == 0, np.nan, x)
