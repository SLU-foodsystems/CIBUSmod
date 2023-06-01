import warnings
import pandas as pd
import numpy as np

from ..utils.verbose_print import verbose_init
from ..utils.misc import rgetattr, rsetattr
from ..utils.misc import Container

from .feed_mgmt import Feed
from .manure_mgmt import Manure

class AnimalHerd(object):
    '''Class that handels animal herd structure, feed requirements, production etc.
    
    Parameters
    ----------
    par : ParameterRetriever object
    index : pandas.Index or pandas.MultiIndex
        Index for the rows. This is also passed on to the ParameterRetriever
    **kwargs : str or list
        Keyword arguments to be passed on as filters to the ParameterRetriever, along with
        the index, self.species. Special cases are 'breed', 'prod_system' and 'sub_system'
        which if supplied are also stored as attributes in the AnimalHerd object.
        
    Attributes set on init
    ----------------------
    index : pandas.Index or padnas.MultiIndex
        Index for rows

    species : str
        Animal species (depends on which AnimalHerd sub-class is used)
    breed : str
        Specifies the breed (e.g. dairy or beef breed)
    prod_system : str
        Specifies the production system which links feed
    sub_system : str

    animals : list
        List of str specifying animal categories in the herd (depends on which AnimalHerd
        sub-class is used)

    Attributes set by AnimalHerd.calculate()
    ----------------------------------------
    heads : pandas.DataFrame
        Yearly average number of heads per animal [heads]
    slaughtered_n : pandas.DataFrame
        Number of slaughtered animals per year [heads]
    lost_n : pandas.DataFrame
        Number of lost animals per year [heads]
    production : pandas.DataFrame
        Production per year for each animal and product  [kg]
    
    Attributes set by FeedMgmt.calculate()
    --------------------------------------
    feed.energy_req : pandas.DataFrame
        Total energy requirements per year for each animal type [MJ]
        Energy is expressed differently across animal species. E.g. for cattle energy is in
        terms of Metabolizable Energy (ME) and for pigs energy is in terms of Net Energy (NE)
    feed.consumption : pandas.DataFrame
        Feed demand per year for each animal type and feed (in terms of feed consumed) [kg DM]

    Attributes set by ManureMgmt.calculate()
    ----------------------------------------
    manure.<element>_excr : pandas.DataFrame
        Manure excretion by animals [kg <element>]
    manure.<element>_loss : pandas.DataFrame
        Losses of <element> in stables and storage by compound [kg <element>]
    manure.<element>_to_spread : pandas.DataFrame
        <element> available to spread [kg <element>]

    <element> is 'VS' (volatile solids), 'N' (nitrogen), 'P' (phosphorous) and 'K' (potassium).
    (ONLY 'N' IMPLEMENTED AT THE MOMENT)
    
    '''
    # Set of ID attributes in class
    id_attr = set(['species','breed','prod_system','sub_system','animals'])

    def __init__(self,par,index,**kwargs):

        # Set to keep track of data attributes that have been assigned
        self.data_attr = set()
        
        self.par = par
        self.index = index

        for att in ['breed','prod_system','sub_system']:
            if not hasattr(self,att):
                if att in kwargs:
                    setattr(self,att,kwargs[att])
                else:
                    setattr(self,att,'none')
            
    def __repr__(self):
        return f'''
AnimalHerd
----------
species              {self.species}
breed                {self.breed}
production system    {self.prod_system}
sub-system           {self.sub_system}
animals              {self.animals}
'''
    
    def calculate(self,verbose=False):
        '''Calculates herd structure and production based on a vector ('x') of animal numbers or production as defined
        by 'x_is'. Index of 'x' is retained in the output and can be used as filters for the ParameterRetriever.

        Parameters
        ----------
        verbose : Bool
            If True, prints messages on progress

        Returns
        -------
        Nothing. Stores output as attrubutes: 'heads', 'slaughtered_n', 'lost_n' and 'production'
        '''

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        self.par.set(
            species = self.species,
            breed = self.breed,
            prod_system = self.prod_system,
            sub_system = self.sub_system
        )
        for i in self.index.names:
            self.par.set(
                **{i : self.index.get_level_values(i).values}
            )

        # Set x (i.e. nr of defining animals) to ones
        self.x = np.ones(len(self.index))

        # Define functions to print progress messages if verbose==True
        vprint = verbose_init(verbose, id_str=f'AnimalHerd ({self.species}, {self.breed}, {self.prod_system}, {self.sub_system})')
        
        vprint('Calculating herd structure ...')
        self.calculate_herd()

        vprint('Calculating production ...')
        self.calculate_production()

        vprint(type='end')

    def scale(self,new_x,x_is):
        '''Scales all data attributes based on new_x
        
        Parameters
        ----------
        new_x : numpy.array or pandas.Series

        x_is : str
            Strimng defining what x is. Must be in 
        
        '''

        # Clear and set filters for ParameterRetriever
        self.par.clear()
        self.par.set(
            species = self.species,
            breed = self.breed,
            prod_system = self.prod_system,
            sub_system = self.sub_system
        )
        for i in self.index.names:
            self.par.set(
                **{i : self.index.get_level_values(i).values}
            )
        p = self.par.get

        valid = ['cows','sows','sows+gilts','broilers','meat','milk']
        if x_is not in valid:
            raise ValueError("x_is must be one of %r." % valid)

        # Check so that new_x length and index match AnimalHerd object
        if len(new_x) != len(self.index):
            raise TypeError('Length of x does not match length of herd\'s index!')
        if hasattr(new_x,'index'):
            if (new_x.index != self.index).any():
                raise TypeError('Index of x does not match herd\'s index!')

        if x_is == 'sows+gilts':
            new_x = new_x / (1+ (p('recruitment_rate')/100 * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25))
            x_is = 'sows'

        # Get 'old_x'
        if x_is in ['milk','meat']:
            if x_is == 'milk':
                old_x = self.production.loc[:,(slice(None),'milk')].sum(axis=1)
            elif x_is == 'meat':
                old_x = self.production.loc[:,(slice(None),'meat')].sum(axis=1)
        elif x_is == 'sows+gilts':
            old_x = self.heads.loc[:,(self.prod_system,['sows','gilts'])].sum(axis=1)
        else:
            old_x = self.heads.loc[:,(self.prod_system,x_is)]
        
        # Update data attributes
        for attr in self.data_attr:
            if rgetattr(self, attr) is not None:
                rsetattr(self, attr, rgetattr(self, attr).mul(new_x/old_x, axis=0))
    
    def make_static(self):
        '''Returns a StaticAnimalHerd object that retains all ID and data attributes but
        has no methods or ParameterRetriever'''

        # Create a StaticAnimalHerd obejct and populate with data
        obj = StaticAnimalHerd()

        obj.index = self.index.copy()
        obj.feed = Feed()
        obj.manure = Manure()
        obj.id_attr = self.id_attr.copy()
        obj.data_attr = self.data_attr.copy()

        # Set ID attributes
        for attr in obj.id_attr:
            setattr(obj,attr,getattr(self,attr))
        
        # Set data attributes
        for attr in obj.data_attr:
            if rgetattr(self, attr) is not None:
                rsetattr(obj, attr, rgetattr(self, attr).copy())
            else:
                rsetattr(obj, attr, None)

        return obj

    def calculate_production(self):
        
        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        # Define output products
        prs = ['meat']
        if 'cows' in self.animals:
            prs.append('milk')
        if 'laying hen' in self.animals:
            prs.append('eggs')
        # Get ouput production systems
        pss = self.heads.columns.get_level_values('prod_system').unique()
        # Get animals
        anis = self.animals

        # Create production DF
        production = pd.DataFrame(
            index = self.index,
            columns = pd.MultiIndex.from_tuples([(ps,ani,pr) for ps in pss for ani in anis for pr in prs], names=['prod_system','animal','animal_prod'])
            )
        
        # Calculate meat production [kg CW]
        production.loc[:,(slice(None),slice(None),'meat')] = \
            pd.concat({'meat': self.slaughtered_n}, names=['animal_prod'], axis=1).reorder_levels(['prod_system','animal','animal_prod'], axis=1) * \
            np.array([p('slaughter_weight', animal=ani, prod_system=ps) for ps in pss for ani in anis]).T

        # Calculate raw milk production [kg ECM]
        # kg ECM = kg milk x 0.25 + kg fat x 12.2 + kg protein x 7.7
        if 'milk' in prs:
            production.loc[:,(slice(None),slice(None),'milk')] = \
                pd.concat([
                    pd.concat({'milk': self.heads.loc[:,[(ps,'cows')]]}, names=['animal_prod'], axis=1).reorder_levels(['prod_system','animal','animal_prod'], axis=1) * \
                    (p('milk_prod', prod_system=ps) * p('milk_to_dairy', prod_system=ps)/100) * \
                    (0.25 + p('milk_fat', prod_system=ps)/100*12.2 + p('milk_protein', prod_system=ps)/100*7.7) \
                    for ps in pss
                    ], axis=1)

        # Calculate egg production
        if 'eggs' in prs:
            pass
        
        # Fill NaNs in production DataFrame and set column index
        production = production.fillna(0)

        self.production = production

    def check_ration(self):
        '''Dummy method to pass feed ration feasibility check if a method is not provided in the species-specific sub-class'''
        pass

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
                self.par.set(**{idx : list(self.index.get_level_values(idx))*len(to_ps)})

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
        
        # No. female calves --> recruitments heifers
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

        # Create output DataFrames
        pss = [self.prod_system]+list(to_ps) if redist else [self.prod_system] # Output production systems (==[self.prod_system] if no redistribution of animals)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, slaughter_n, lost_n  = (empty_df.copy(),empty_df.copy(),empty_df.copy())

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals) 
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)
            heads.loc[:,(ps,slice(None))] = \
                np.array([cows[sel],breeding_bulls[sel],calves[sel],heifers[sel],steers[sel],bulls[sel]]).T
            slaughter_n.loc[:,(ps,slice(None))] = \
                np.array([cows2slaughter[sel],breeding_bulls2slaughter[sel],calves2slaughter[sel],heifers2slaughter[sel],steers2slaughter[sel],bulls2slaughter[sel]]).T
            lost_n.loc[:,(ps,slice(None))] = \
                np.array([cows2lost[sel],breeding_bulls2lost[sel],calves2lost[sel],heifers2lost[sel],steers2lost[sel],bulls2lost[sel]]).T

            n += 1
        
        self.heads = heads
        self.slaughtered_n = slaughter_n
        self.lost_n = lost_n
        self.data_attr.update(['heads','slaughtered_n','lost_n'])

    def calculate_feed_energy_req(self,ani):
        '''Calculates Metabolizable Energy (ME) and water requrements for cattle based on
        Spörndly, R. (ed.). (2003). Fodertabeller för idisslare 2003. HUV Rapport 257. SLU'''

        p = self.par.get

        # Get average live weight [kg] and growth rate [kg/day] of animal
        if ani in ['cows','breeding bulls']:
            live_weight = p('live_weight')
            growth_rate = 0
        elif ani == 'calves':
            live_weight_pre_weaning = (p('live_weight_weaning') + p('birth_weight')) / 2
            growth_rate_pre_weaning = (p('live_weight_weaning') - p('birth_weight')) / p('weaning_age')
            live_weight = (p('live_weight_1yr') + p('live_weight_weaning')) / 2
            growth_rate = (p('live_weight_1yr') - p('live_weight_weaning')) / (365.25 - p('weaning_age'))
        else:
            live_weight = (p('live_weight_slaughter', animal=ani) + p('live_weight_1yr', animal='calves')) / 2
            growth_rate = (p('live_weight_slaughter', animal=ani) - p('live_weight_1yr', animal='calves')) / (p('slaughter_age', animal=ani) * 30.4 - 365.25)
            self.par.set(animal=ani)
              
        # Daily ME req. for maintenance [MJ/day]
        E_maintenance = p('maintanance_energy_factor') * live_weight**0.75

        # Daily ME req. for changes in body weight [MJ/day]
        if ani=='cows':
            E_growth = 35 * growth_rate # (Tabell 1)
        else:
            E_growth = (growth_rate * (6.28 + 0.0188 * live_weight)) / ((1 - 0.3 * growth_rate) * 0.435) # (Tabell 4a)

        if ani == 'cows':
            # ME req. for lactation [MJ/day]
            # Milk in kg ECM: milk kg x 0,25 + fa tkg x 12,2 + protein kg x 7,7 = kg ECM
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

        return np.nan_to_num(E_req_final)

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

        # Provide shorthand 'p()' to get parameters
        p = self.par.get

        idx_len = len(self.index)
        sows = self.x / (1+ (p('recruitment_rate')/100 * (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age')) / 365.25))

        tmp_piglets_born = sows * p('litters_per_sow') * p('live_per_litter')
        tmp_piglets_weaned = tmp_piglets_born * (1 - p('mortality_0towean')/100)
        tmp_piglets_delivered = tmp_piglets_weaned * (1 - p('mortality_weantodelivery')/100)
        
        piglets_lost = (sows * p('litters_per_sow') * p('dead_per_litter')) + (tmp_piglets_delivered - tmp_piglets_born)

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

        # Create output DataFrames
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
            )
        heads, slaughter_n, lost_n  = (empty_df.copy(),empty_df.copy(),empty_df.copy())

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals) 
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)
            heads.loc[:,(ps,slice(None))] = \
                np.array([sows[sel],boars[sel],piglets[sel],gilts[sel],growers[sel],finishers[sel]]).T
            slaughter_n.loc[:,(ps,slice(None))] = \
                np.array([sows_to_slaughter[sel],boars_to_slaughter[sel],piglets_to_slaughter[sel],gilts_to_slaughter[sel],growers_to_slaughter[sel],finishers_to_slaughter[sel]]).T
            lost_n.loc[:,(ps,slice(None))] = \
                np.array([sows_lost[sel],boars_lost[sel],piglets_lost[sel],gilts_lost[sel],growers_lost[sel],finishers_lost[sel]]).T

            n += 1

        self.heads = heads
        self.slaughtered_n = slaughter_n
        self.lost_n = lost_n
        self.data_attr.update(['heads','slaughtered_n','lost_n'])

    def calculate_feed_energy_req(self,ani):
        '''Calculates Net Energy (NEs [sows and boars] or NEv [other pigs]) requrements for pigs based on
        [1] Simonsson, A. (2006). Fodermedel och näringsrekommendationer för gris. HUV Rapport 266. SLU
        [2] Göransson, L., Lindberg, J.E. (2011). Näringsrekommendationer ver. 2011.1 - Energi'''

        p = self.par.get

        # Get average live weight [kg] and growth rate [kg/day]
        if ani in ['sows','boars']:
            live_weight = p('live_weight')
        # elif ani == 'gilts':
        #     growth_rate_growing_period = (p('live_weight_slaughter') - p('live_weight_delivery')) / (p('growing_period') + p('finishing_period'))
        #     live_weight_after_growing_period = p('live_weight_delivery') + growth_rate_growing_period * p('growing_period')
        #     live_weight = (p('live_weight', ani='sows') + live_weight_after_growing_period) / 2
        #     growth_rate = (p('live_weight', ani='sows') - live_weight_after_growing_period) / (p('age_at_first_farrowing') - p('growing_period') - p('post_weaning_nursing_period') - p('weaning_age'))
        #     self.par.set(ani=ani)
        elif ani == 'piglets':
            # live_weight_pre_weaning = (p('live_weight_weaning') + p('birth_weight')) / 2
            # growth_rate_pre_weaning = (p('live_weight_weaning') - p('birth_weight')) / p('weaning_age')
            live_weight = (p('live_weight_delivery') + p('live_weight_weaning')) / 2
            growth_rate = (p('live_weight_delivery') - p('live_weight_weaning')) / p('post_weaning_nursing_period')
        elif ani in ['growing pigs','finishing pigs']:
            growth_rate = (p('live_weight_slaughter') - p('live_weight_delivery')) / (p('growing_period') + p('finishing_period'))
            # if ani == 'growing pigs':
            #     live_weight = (p('live_weight_delivery') * 2 + growth_rate * p('growing_period')) / 2
            # else:
            #     live_weight = (p('live_weight_delivery') * 2 + growth_rate * p('growing_period') * 2 + growth_rate * p('finishing_period')) / 2
            
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

        return np.nan_to_num(E_req)

class SheepHerd(AnimalHerd):
    pass

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
        Sets data attributes self.heads, self.slaughtered_n and self.lost_n'''

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
            (inserted_parent_hens + inserted_parent_roosters) /
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
        heads, slaughter_n, lost_n  = (empty_df.copy(),empty_df.copy(),empty_df.copy())

        # Populate dataframes by distributing rows according to output production systems (i.e. after redistribution of animals) 
        n = 0
        for ps in pss:
            sel = range(n*idx_len, (n+1)*idx_len)
            heads.loc[:,(ps,slice(None))] = \
                np.array([broilers[sel],breeding_hens[sel],breeding_roosters[sel]]).T
            slaughter_n.loc[:,(ps,slice(None))] = \
                np.array([slaughtered_broilers[sel],slaughtered_breeding_hens[sel],slaughtered_breeding_roosters[sel]]).T
            lost_n.loc[:,(ps,slice(None))] = \
                np.array([lost_broilers[sel],lost_breeding_hens[sel],lost_breeding_roosters[sel]]).T

            n += 1

        self.heads = heads
        self.slaughtered_n = slaughter_n
        self.lost_n = lost_n
        self.data_attr.update(['heads','slaughtered_n','lost_n'])

    def calculate_feed_req(self,ani):

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

        return(feed_req)


class LayerHerd(AnimalHerd):
    pass

class ReindeerHerd(AnimalHerd):
    pass

class HorsesHerd(AnimalHerd):
    pass

class StaticAnimalHerd(Container):
    '''Class used to create static copys of animal her objects. These stores all attributes except 'par'
    but does not inherit any methods'''

    def __repr__(self):
        return AnimalHerd.__repr__(self).replace('AnimalHerd','StaticAnimalHerd')