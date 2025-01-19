# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:27:51 2024

@author: Yu-Chen Wang
"""

from .lgm import LittleGreenMen


examples = {
    'LGM': LittleGreenMen,
    }

generators = {}

def get_example(identifier, **kwargs):
    names = identifier.split('.')
    if len(names) == 2:
        example_name, data_name = names
        if example_name in examples:
            generator = examples[example_name]
            if generator not in generators:
                generators[generator] = generator(**kwargs)
            datasets = generators[generator]
            return datasets.get_data(data_name)
        else:
            raise ValueError(f"name not found: {example_name}")
    else:
        raise ValueError(f"example not found: '{identifier}'")

def reset():
    generators.clear()
