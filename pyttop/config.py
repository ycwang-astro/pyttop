# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:25:17 2025

@author: Yu-Chen Wang
"""

from dataclasses import dataclass, fields

@dataclass
class Config:
    data_name_repr_maxlen: int = 100
    
    def reset(self):
        for f in fields(self):
            setattr(self, f.name, f.default)
    
    # def __repr__(self):
    #     return '<pyttop config>'
    
config = Config()
