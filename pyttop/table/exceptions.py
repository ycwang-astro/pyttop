# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 17:47:03 2025

@author: Yu-Chen Wang

exceptions
"""


class FailedToLoadError(Exception):
    pass

class FailedToEvaluateError(Exception):
    pass

class SubsetError(Exception):
    pass

class SubsetInconsistentError(SubsetError):
    pass

class MergeError(Exception):
    pass

class SubsetMergeError(MergeError):
    pass

class SubsetNotFoundError(LookupError):
    def __init__(self, name, kind='path', suggest_names=None):
        if kind == 'path':
            suggest_str = " (did you mean: '{}')".format("', '".join(suggest_names)) if suggest_names else ''
            info = f"'{name}'. Maybe missing/incorrect group name or incorrect subset name{suggest_str}?"
        elif kind in ['subset', 'name']:
            info = f"'{name}'"
            if suggest_names:
                info += ". Did you mean: '{}'".format("', '".join(suggest_names))
        else:
            raise ValueError(f"unknown kind '{kind}'")
        super().__init__(info)

class GroupNotFoundError(LookupError):
    def __init__(self, name, suggest_names=None):
        info = f"'{name}'"
        if suggest_names:
            info += ". Did you mean: '{}'".format("', '".join(suggest_names))
        super().__init__(info)

class ColumnNotFoundError(LookupError):
    pass


