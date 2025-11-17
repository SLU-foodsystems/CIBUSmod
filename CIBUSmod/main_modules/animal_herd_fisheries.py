import pandas as pd
import numpy as np

from .animal_herd import AnimalHerd

class FisheriesHerd(AnimalHerd):
    AnimalHerd.__doc__.replace('animal','wild caught fish')

    def __init__(self,par,index,**kwargs):

        self.species = kwargs['species']
        self.animals = ['fish']
        self.products = ['fish']

        self.x_is = 'fish'

        super().__init__(par,index,**kwargs)

    def calculate_herd(self):
        # No herd structure
        return None

    def _calculate_feed_req(self):
        # No feed requirements
        return None

