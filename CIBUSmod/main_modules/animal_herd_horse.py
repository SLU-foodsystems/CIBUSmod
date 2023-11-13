import pandas as pd

from .animal_herd import AnimalHerd

class HorseHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','horse')

    def __init__(self,par,index,**kwargs):
            
        self.species = 'horses'
        self.animals = ['low-performing horses','medium-performing horses','broodmares','young horses']

        self.x_is = 'total horses'
        
        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        '''Calculates horse herd structure x (i.e. total number of horses).
        
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

        # Get total number of horses
        total_horses = self.x

        # Calculate number of different animals
        pss = [self.prod_system] # Output production systems (==[self.prod_system] as no redistribution of animals in this class)

        heads = pd.DataFrame(
            index = self.index,
            columns = pd.MultiIndex.from_tuples([(ps, ani) for ps in pss for ani in self.animals], names=['prod_system','animal'])
        )

        heads = (
            (self.par.get_from_frame('share_of_horses',heads)/100)
            .mul(total_horses, axis=0)
        )

        # Assume no slaughter and losses for now...
        slaughtered_n = heads * 0
        lost_n = heads * 0

        self.heads = heads
        self.slaughtered_n = slaughtered_n
        self.lost_n = lost_n
        self.data_attr.update(['heads','slaughtered_n','lost_n'])

    def calculate_feed_E_req(self,ps,ani):

        p = self.par.get

        # Calculate maintenance energy req.
        E_maint = p('live_weight')**0.75 * p('maintenance_energy_factor') * 365.25

        # Get activity adjustment factor
        f_acti = p('energy_adjustment_factor')

        if ani=='broodmares':
            # Get gestation adjustment factor
            f_gest = p('foals_per_year') * p('gestation_energy_factor')
            # Get lactation adjustment factor
            f_lact = p('foals_per_year') * p('lactation_energy_factor')
        else:
            f_gest = 0
            f_lact = 0

        E_req = E_maint * (1 + (f_acti + f_gest + f_lact))

        return E_req