# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:54:16 2024

@author: Yu-Chen Wang
"""

from .base import DataGenerator
import numpy as np

surnames = [
    # generated from language models
    'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
    'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Roberts',
    'Lee', 'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Adams',
    'Hill', 'Green', 'Baker', 'Gonzalez', 'Nelson', 'Carter', 'Mitchell', 'Perez', 'Evans', 'Edwards',
    'Collins', 'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy',
    'Bailey', 'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Flores', 'Kelly', 'Barnes',
    'Chavez', 'Graham', 'Hall', 'Alvarez', 'Garner', 'Knight', 'Stone', 'Hudson', 'Douglas', 'Cameron',
    'Duncan', 'Hughes', 'Washington', 'Simmons', 'Foster', 'Bryant', 'Russell', 'Chang', 'Sanders', 'Parker',
    'Graham', 'Patterson', 'Perkins', 'Hughes', 'Phillips', 'Gibson', 'Barnett', 'Reyes', 'Johnston',
    'Matthews', 'West', 'Payne', 'Bryant', 'Guzman', 'Hawkins', 'Lynch', 'Bryan', 'Murray'
]

class LittleGreenMen(DataGenerator):
    def __init__(self, N=5000):
        super().__init__()
        self.examplename = 'LGM'
        self.N = N
        self.rng = np.random.default_rng(seed=self.seed)
        self.id = self.rng.permutation(self.N)
        # self.ra = self.rng.uniform(0, 24, self.N)
        self.ra = self.rng.uniform(0, 360, self.N)
        self.dec = np.rad2deg(np.arcsin(2 * self.rng.uniform(size=self.N) - 1))
        sexes = ['Male', 'Female', 'Both', 'Neither']
        self.sex = self.rng.choice(sexes, size=self.N, replace=True)
        self.age = np.ceil(-np.log(self.rng.uniform(size=self.N)) * 100).astype(int)
        self.height = 5 + self.age * np.exp(-self.age) * np.e * 10 + self.rng.normal(size=self.N)
        self.weight = 10 * self.height**1.1 - .1 * self.age + .1 * self.rng.normal(size=self.N)
        self.area = np.abs(self.dec) * .1 + self.rng.normal(size=self.N) + 20
        self.surname = self.rng.choice(surnames, size=self.N, replace=True)
        
        self.age[self.rng.choice(self.N, self.N // 8, replace=False)] = -99
        self.sex[self.rng.choice(self.N, self.N // 7, replace=False)] = 'N/A'
        
        # in_an_area = np.where((self.ra < 5) & (self.ra > 4) & (self.dec < 70) & (self.dec > 50))[0]
        in_an_area = np.where((self.ra < 90) & (self.ra > 75) & (self.dec < 70) & (self.dec > 50))[0]
        self.surname[self.rng.choice(in_an_area, size=np.min((10, len(in_an_area))), replace=False)] = 'Smith'
        
    @property
    def addr(self):
        return dict(
            keys = ['id', 'ra', 'dec'],
            )
    
    @property
    def name(self):
        return dict(
            keys = ['id', 'surname'],
            frac = .7,
            )
    
    @property
    def bio(self):
        return dict(
            keys = ['id', 'sex', 'age', 'height', 'weight'],
            )
    
    @property
    def house(self):
        return dict(
            keys = ['ra', 'dec', 'area'],
            )
        
