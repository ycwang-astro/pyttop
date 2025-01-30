# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:59:35 2025

@author: Yu-Chen Wang

general example datasets
"""

from .base import DataGenerator
import numpy as np

class PlotExample1(DataGenerator):
    examplename = 'P1'
    def __init__(self):
        super().__init__()
        
        n1 = 2000
        x1 = self.rng.normal(size=n1) * 10
        y1 = .5 * x1 + self.rng.normal(size=n1) * 2
        z1 = self.rng.normal(size=n1) * 20
        m1 = x1 + self.rng.normal(size=n1) * 1
        c1 = np.full(n1, 'A')

        n2 = 1500
        x2 = self.rng.normal(size=n2) * 10
        y2 = .5 * x2 + 3 + self.rng.normal(size=n2) * 2
        z2 = self.rng.normal(size=n2) * 20
        m2 = x2 + 30 + self.rng.normal(size=n2) * 1
        c2 = np.full(n2, 'B')
        
        self.N = n1 + n2
        
        self.x = np.concatenate((x1, x2))
        self.y = np.concatenate((y1, y2))
        self.z = np.concatenate((z1, z2))
        self.m = np.concatenate((m1, m2))
        self.obj_class = np.concatenate((c1, c2))

    @property
    def _main_(self):
        return dict(
            keys = ['x', 'y', 'z', 'm', 'obj_class'],
            )
