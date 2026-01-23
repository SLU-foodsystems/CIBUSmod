import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class FisheriesHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','wild caught fish')

    has_manure = False

    def __init__(self,par,index,**kwargs):

        self.species = kwargs['species']
        self.animals = ['wild fish']
        self.products = ['fish']

        self.x_is = 'fish'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        # No herd structure

        # Create DataFrames
        empty_df = pd.DataFrame(
            columns = pd.MultiIndex.from_tuples([
                (ps, ani)
                for ps in [self.prod_system]
                for ani in self.animals
            ], names=['prod_system','animal']),
            index = self.index,
            dtype = 'float64'
        )

        # Assign values
        heads = empty_df.copy()

        # Add data attributes
        self.data_attr.add(
            heads, # Empty DataFrame, needed in other modules
            name = 'heads',
            unit = 'heads',
            orig = 'AquacultureHerd',
            desc = 'Total average number of heads over a year'
        )

        return None

    def _calculate_feed_req(self):
        # No feed requirements
        return None

