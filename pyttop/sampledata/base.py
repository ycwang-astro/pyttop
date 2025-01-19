# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:08:23 2024

@author: Yu-Chen Wang
"""

from ..table import Data
from collections.abc import Iterable
import numpy as np

SEED = 42

class DataGenerator():
    def __init__(self):
        self.seed = SEED
    
    def gen_data_from_attr(self, keys, frac=1, seed=None, name=None):
        '''
        Generates a Data from attributes of self.

        Parameters
        ----------
        keys : Iterable
            Names of the attributes.
        frac : optional
            Only return a fraction of the rows.
        seed : int, optional
            Random seed.
        name : str, optional
            Name of the output data. The default is None.

        Returns
        -------
        Data

        '''
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed=seed)
        sel = rng.choice(self.N, int(self.N * frac), replace=False)
        data = {k: getattr(self, k)[sel] for k in keys}
        return Data(data,
                    name=name)
    
    def get_data(self, name):
        '''
        An attribute of self should be defined as below:
        attribute name: dataset name
        attribute value: (<a list of attribute names>, <fraction>)

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        Data

        '''
        try:
            kwargs = getattr(self, name)
            assert (isinstance(kwargs, dict)
                    and isinstance(kwargs['keys'], Iterable) 
                    # and isinstance(kwargs['frac'], (int, float)),
                    )
        except (AttributeError, AssertionError, TypeError, KeyError) as e:
            raise ValueError(f"dataset '{name}' not found.") from e
        return self.gen_data_from_attr(**kwargs, name=f'{self.examplename}.{name}')

