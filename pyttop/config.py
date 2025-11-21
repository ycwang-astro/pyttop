# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:25:17 2025

@author: Yu-Chen Wang

Configuration system for PyTTOP. 
This controls the behavior of PyTTOP.
"""

from dataclasses import dataclass, field, fields, asdict, is_dataclass, MISSING
from typing import Optional, Dict, Any

def _reset_dataclass(obj):
    for f in fields(obj):
        val = getattr(obj, f.name)
        if is_dataclass(val):
            _reset_dataclass(val)
        else:
            if f.default is not MISSING:
                setattr(obj, f.name, f.default)
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                setattr(obj, f.name, f.default_factory())  # type: ignore[misc]

# Group configs

@dataclass
class BaseConfig:
    def reset(self):
        """
        Reset all config groups to their default values.
        """
        _reset_dataclass(self)


@dataclass
class PlotConfig(BaseConfig):
    """
    Plot-related defaults and behavior toggles.
    """
    defaults_hist: dict = field(default_factory=lambda: {
        'histtype': 'step',
        'linewidth': 1.3,
        })


@dataclass
class DisplayConfig(BaseConfig):
    """
    Human-facing display options (repr, summaries).
    """
    data_name_maxlen: int = 100          # Truncation length for Data name in repr

# Top-level config

@dataclass
class Config(BaseConfig):
    """
    Global configuration accessor collected into logical groups.
    """
    plot: PlotConfig = field(default_factory=PlotConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def to_dict(self) -> dict:
        """
        Convert the entire configuration tree to a plain dict.
        """
        return asdict(self)
    
# @dataclass
# class Config:
#     data_name_repr_maxlen: int = 100
    
#     def reset(self):
#         for f in fields(self):
#             setattr(self, f.name, f.default)
    
    def __repr__(self):
        return '<pyttop config>'
    
# Singleton instance
config = Config()
