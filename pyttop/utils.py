# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang
"""

import numpy as np
import warnings
import pickle
import os
from functools import wraps, reduce
from operator import iand, ior

from typing import Union, Sequence

#%% array/Iterable operation

def find_idx(array, values):
    '''
    Find the indexes of values in array.
    
    If not found, will return -l-1, which is out of
    the range of array.

    Parameters
    ----------
    array : Iterable
        .
    values : Iterable
        .

    Returns
    -------
    idx : np.ndarray (int)
    
    found : np.ndarray (bool)

    '''
    l = len(array)
    sorter = np.argsort(array)
    ss = np.searchsorted(array, values, sorter=sorter)
    isin = np.isin(values, array)
    not_found = (ss==l) | (~isin)
    found = ~not_found
    ss[not_found] = -1
    idx = sorter[ss]
    idx[not_found] = -l-1
    return idx, found

def find_eq(array, values):
    '''
    Return an boolean array indicating whether each row in ``values`` is 
    equal to each row in ``array``.

    Parameters
    ----------
    array : array of shape (N1, M)
        .
    values : array of shape (N2, M)
        .

    Returns
    -------
    eq : boolean array of shape (N2, N1)
        An array where the ``eq[i, j]`` element is ``np.all(values[i] == array[j])``.

    '''
    eq = np.all(values[:, np.newaxis, :] == array, axis=2)
    # same as [[np.all(a[i] == b[j]) for j in range(array.shape[0])] for i in range(values.shape[0])]
    return eq

def find_dup(arr):
    if np.ma.is_masked(arr):
        arr = arr[~arr.mask]
    arr, counts = np.unique(arr, return_counts=True)
    dup_vals = arr[counts != 1]
    return dup_vals

# # testing find_eq
# for i in range(values.shape[0]): 
#     for j in range(array.shape[0]): 
#         assert find_eq(array, values)[i, j] == np.all(values[i] == array[j])

def grid(x, y, flat=False):
    if flat:
        xx = [xi for yi in y for xi in x]
        yy = [yi for yi in y for xi in x]
    else:
        xx = [[xi for xi in x] for yi in y]
        yy = [[yi for xi in x] for yi in y]
    return xx, yy

def bitwise_all(iterable):
    '''
    Return the bitwise all of an iterable. 
    For example, ``bitwize_all([a, b, c])`` is equivalent to ``a & b & c``.
    '''
    return reduce(iand, iterable)

def bitwise_or(iterable):
    return reduce(ior, iterable)

#%% basic types

# Modified dictionary. If each value has the same length, it is similar to pandas.DataFrame, but simpler.
class objdict(dict):
    # author: Senko Rašić
    # from https://goodcode.io/articles/python-dict-object/?msclkid=daff3822c47111eca4f572e5716ccae3
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

class SummaryDict(dict):
    def __init__(self, *args, dict_name='dict', 
                 element_names: Union[Sequence[str], str] = None, 
                 depth: int = None, 
                 join_str=' with ',
                 **kwargs):
        if isinstance(element_names, str):
            element_names = [element_names]
        if depth is None:
            if element_names is not None:
                depth = len(element_names)
            else: # depth is None and element_names is None
                depth = 1
        if element_names is None:
            element_names = ['elements']
        if depth < 0:
            raise ValueError
        self.depth = depth
        self.dict_name = dict_name
        self.element_names = element_names
        self.join_str = join_str
        
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def _count_elements(d, depth=1, this_depth=1, counts=None, max_depth=10,
                        # known_ds: list = None
                        ):
        if this_depth > max_depth:
            raise RecursionError(f'max recursion depth exceeded: {max_depth}')
            
        if counts is None:
            counts = [0] * depth # initialize a list of element counts
            
        # if known_ds is None:
        #     known_ds = [d]
        
        if not isinstance(d, dict) or this_depth > depth:
            return counts
    
        counts[this_depth - 1] += len(d)
        
        for value in d.values():
            # if value not in known_ds and isinstance(value, dict):
            #     known_ds.append(value)
            #     SummaryDict._count_elements(value, depth, this_depth + 1, counts, known_ds)
            if isinstance(value, dict):
                SummaryDict._count_elements(value, depth, this_depth + 1, counts, max_depth=max_depth)
    
        return counts        
    
    def __repr__(self):
        element_counts = self.__class__._count_elements(
            self, depth=self.depth)
        summary_strs = []
        for count, name in zip(element_counts, self.element_names):
            if count:
                summary_strs.append(f'{count} {name}')
        if summary_strs:
            summary_str = self.join_str + ', '.join(summary_strs)
        else:
            summary_str = ''
        return f"<{self.dict_name}{summary_str}>"

class DeprecationError(DeprecationWarning):
    pass

#%% interactive functions

def pause_and_warn(message=' ', choose='Proceed?', default = 'n', yes_message='', no_message='raise', warn=True, timeout=None):
    '''
    calling this function will do something like this:
            [print]  <message>
            [print]  <choose> y/n >>> 
    default choice is <default>
    if yes:
            [print] <yes_message>
    if no:
            [print] <no_message>
        if no_message is 'raise':
            [raise] Error: <message>
    [return] the choise, True for yes, False for no.
    '''
    print('{:-^40}'.format('[WARNING]'))
    
    if isinstance(message, Exception):
        message = str(type(message)).replace('<class \'','').replace('\'>', '')+': '+'. '.join(message.args)
    if warn:
        warnings.warn(message, stacklevel=3)
    print(message)
    
    question = '{} {} >>> '.format(choose, '[y]/n' if default == 'y' else 'y/[n]')
    if timeout is None:
        cont = input(question)
    else:
        raise NotImplementedError
    if not cont in ['y', 'n']:
        cont = default
    if cont == 'y':
        print(yes_message)
        return True
    elif cont == 'n':
        if no_message == 'raise':
            raise RuntimeError(message)
        else:
            print(no_message)
            return False

#%% file IO

def save_pickle(fname, *data, yes=False, ext=False):
    '''
    save data to fname

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    *data : TYPE
        DESCRIPTION.
    yes : bool
        if ``True``, file will be overwritten without asking.
    ext : bool
        if ``True``, file name will always end with ".pkl"; otherwise use original fname given
    '''
    if ext and not '.pkl' in fname:
        fname+='.pkl'
    if os.path.exists(fname):
        if os.path.isdir(fname):
            raise ValueError('fname should be the file name, not the directory!')
        if yes:
            print(f'OVERWRITTEN: {fname}')
        else:
            pause_and_warn('File "{}" already exists!'.format(fname), choose='overwrite existing files?',
                           default='n', yes_message='overwritten', no_message='raise')
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(fname):
    '''
    load pkl and return. 
    If there is only one object in the pkl, will return it.
    Otherwise, return a tuple of the objects.

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # if fname[-4:] != '.pkl':
    #     fname+='.pkl'
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        if len(data) == 1:
            return data[0]
        else:
            return data

# def save_zip(fname, ext='zip', overwrite=False)

#%% wrappers
def keyword_alias(state='deprecated', /, **aliases):
    '''
    Returns wrapper for alias of keyword argument.

    Parameters
    ----------
    state : str
        POSITIONAL-ONLY argument.
        Three states: 
            'accepted' (no warnings)
            'deprecated' (warnings)
            'removed' (error)
    **aliases : old = new
        old (deprecated) name and new name
    '''
    def wrapper(f):
        @wraps(f)
        def fnew(*args, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"Both {old} and {new} found in arguments; use {new} only.")
                    kwargs[new] = kwargs.pop(old)
                    if state == 'accepted':
                        pass # accepted alias
                    elif state == 'deprecated':
                        warnings.warn(f"argument '{old}' is deprecated; use '{new}' instead",
                                      category=FutureWarning, # this is a warning for end-users rather than programmers
                                      stacklevel=2)
                    elif state == 'removed':
                        raise DeprecationError(f"argument '{old}' is deprecated; use '{new}' instead")
            return f(*args, **kwargs)
        return fnew
    return wrapper

