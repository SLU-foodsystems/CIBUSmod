import pandas as pd

from .animal_herd import AnimalHerd

class HorseHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','horse')

    def __init__(self,par,index,**kwargs):

        self.species = 'horses'
        self.animals = ['low-performing horses','medium-performing horses','broodmares','young horses']
        self.products = ['meat', 'heads']

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
        lost_lw = heads * 0

        # Add data attributes
        self.data_attr.add(
            heads,
            name = 'heads',
            unit = 'heads',
            orig = 'HorseHerd',
            desc = 'Total average number of heads over a year'
        )
        self.data_attr.add(
            slaughtered_n,
            name = 'slaughtered_n',
            unit = 'heads/year',
            orig = 'HorseHerd',
            desc = 'Total number of heads slaughtered'
        )
        self.data_attr.add(
            lost_n,
            name = 'lost_n',
            unit = 'heads/year',
            orig = 'HorseHerd',
            desc = 'Total number of heads lost'
        )
        self.data_attr.add(
            lost_lw,
            name = 'lost_lw',
            unit = 'kg/year',
            orig = 'HorseHerd',
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

            # Calculate metabolizable energy requirements
            ME_req = self._calculate_ME_req(ps, ani)
            self.data_attr.get('feed_req_eq').loc[:,(ps,ani,'ME')] = ME_req * heads

            # Get maximum dry matter intake
            if 'max_DMI' in pars:
                DM_max = p('max_DMI') * 365.25
                self.data_attr.get('feed_req_max').loc[:,(ps,ani,'DM')] = DM_max * heads

    def _calculate_ME_req(self,ps,ani):

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

