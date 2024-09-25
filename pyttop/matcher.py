# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang

Built-in matchers.
"""

import numpy as np
from .utils import find_idx, find_eq, find_dup
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import UnitTypeError
import warnings
from collections.abc import Iterable

class UnsafeMatchingWarning(Warning):
    pass
    # def __init__(self, data, **kwargs)

class DuplicationWarning(UnsafeMatchingWarning):
    pass

class ExactMatcher():
    def __init__(self, value, value1=None):
        '''
        Used to match `pyttop.table.Data` objects `data1` to `data`.
        Match records with exact values.
        This should be passed to method `data.match()`.
        See `help(data.match)`.

        Parameters
        ----------
        value : str or Iterable
            Specify values for `data` used to match catalogs. Possible inputs are:
            - str, name of the field used for matching.
            - Iterable, values for `data`. `len(value)` should be equal to `len(data)`.
        value1 : str or Iterable, optional
            Specify values for `data1` used to match catalogs. Possible inputs are:
            - str, name of the field used for matching.
            - Iterable, values for `data1`. `len(value1)` should be equal to `len(data1)`.
            
            If not given and ``value`` is a string, ``value1`` set to the same as ``value``.
        '''
        self.value = value
        self.value1 = value1
        
        self.value_name, self.value1_name = f'"{value}"' if isinstance(value, str) else value, f'"{value1}"' if isinstance(value1, str) else value1 # before evaluation, let the input be the names
        
        if self.value1 is None:
            if isinstance(self.value, str):
                self.value1 = self.value
            else:
                raise TypeError("argument missing: 'value1'")
    
    def get_values(self, data, data1, verbose=True):
        valuetype, value1type = type(self.value), type(self.value1)
        if isinstance(self.value, str):
            self.value = data[self.value]
        elif isinstance(self.value, Iterable):
            if not isinstance(self.value, np.ndarray): # Column, MaskedArray, etc. are instances of np.ndarray but will be converted by np.array(), so we need this condition
                self.value = np.array(self.value)
        else:
            raise TypeError(f"expected str or Iterable for 'value', got '{type(self.value)}'")
        if isinstance(self.value1, str):
            self.value1 = data1[self.value1]
        elif isinstance(self.value1, Iterable):
            if not isinstance(self.value1, np.ndarray): # Column, MaskedArray, etc. are instances of np.ndarray but will be converted by np.array(), so we need this condition
                self.value1 = np.array(self.value1)
        else:
            raise TypeError(f"expected str or Iterable for 'value1', got '{type(self.value1)}'")
        
        if hasattr(self.value, 'name'):
            self.value_name = f'"{self.value.name}"'
        else:
            self.value_name = valuetype
        if hasattr(self.value1, 'name'):
            self.value1_name = f'"{self.value1.name}"'
        else:
            self.value1_name = value1type
            
        dup_vals = find_dup(self.value)
        if dup_vals.size > 0:
            warnings.warn(f"Duplications found for data '{data.name}' while matching '{data1.name}' to it: the same row of '{data1.name}' may be matched to multiple rows in '{data.name}'.",
                          stacklevel=3, category=DuplicationWarning)
        dup_vals = find_dup(self.value1)
        if dup_vals.size > 0:
            warnings.warn(f"Duplications found for data '{data1.name}' while matching to '{data.name}': there may be multiple rows in '{data1.name}' that can be matched to a row in '{data.name}', and only one will be returned by the matcher.",
                          stacklevel=3, category=DuplicationWarning)
        missings = [] # whether the coord is missing
        not_missing_ids = [] # the indices of those that are not missing
        for valuei, datai in [[self.value, data], [self.value1, data1]]:
            if np.ma.is_masked(valuei): #datai.t.masked:
                # NOTE: it should not matter whether datai.t is masked; it is valuei that matters. A table that is not "masked" can have masked colums; valuei can also be user-specified rather than from datai.t
                missingi = valuei.mask
            else:
                missingi = np.full(len(datai), False)
            not_missing_idi = np.arange(len(datai), dtype=int)[~missingi]
            missings.append(missingi)
            not_missing_ids.append(not_missing_idi)
        
        self.missing, self.missing1 = missings
        self.not_missing_id, self.not_missing_id1 = not_missing_ids
    
    def match(self):
        l = len(self.missing)
        idx = np.full(self.missing.shape, -l-1)
        matched = np.full(self.missing.shape, False)
        idx_nm, matched_nm = find_idx(self.value1[~self.missing1], self.value[~self.missing])
        matched[~self.missing] = matched_nm
        idx[matched] = self.not_missing_id1[idx_nm[matched_nm]]
        return idx, matched
    
    def __repr__(self):
        return f'ExactMatcher({self.value_name}, {self.value1_name})'


class SkyMatcher():
    def __init__(self, thres=1, coord=None, coord1=None, unit=u.deg, unit1=u.deg):
        '''
        Used to match `pyttop.table.Data` objects `data1` to `data`.
        Match records with nearest coordinates.
        This should be passed to method `data.match()`.
        See `help(data.match)`.

        Parameters
        ----------
        thres : number, optional
            Threshold in arcsec. The default is 1.
        coord : str or astropy.coordinates.SkyCoord, optional
            Specify coordinate for the base data. Possible inputs are:
            - astropy.coordinates.SkyCoord (recommended), the coordinate object.
            - str, should be like 'RA-DEC', which specifies the column name for RA and Dec.
            - None (default), will try ['ra', 'RA'] and ['DEC', 'Dec', 'dec'].
            The default is None.
        coord1 : str or astropy.coordinates.SkyCoord, optional
            Specify coordinate for the matched data. Possible inputs are:
            - astropy.coordinates.SkyCoord (recommended), the coordinate object.
            - str, should be like 'RA-DEC', which specifies the column name for RA and Dec.
            - None (default), will try ['ra', 'RA'] and ['DEC', 'Dec', 'dec'].
            The default is None.
        unit : astropy.units.core.Unit or list/tuple/array of it
            If astropy.coordinates.SkyCoord object is not given for coord, 
            this is used to specify the unit of coord.
            The default is astropy.units.deg.
        unit1 : astropy.units.core.Unit or list/tuple/array of it
            If astropy.coordinates.SkyCoord object is not given for coord1, 
            this is used to specify the unit of coord1.
            The default is astropy.units.deg.
           
        Notes
        -----
        The data columns for RA, Dec may already have units (e.g. ``data.t['RA'].unit``).
        In this case, any input for ``unit`` or ``unit1`` is ignored, and the units recorded
        in the columns are used.
        '''
        self.thres = thres
        self.coord = coord
        self.coord1 = coord1
        self.unit = unit
        self.unit1 = unit1
    
    def get_values(self, data, data1, verbose=True):
        # TODO: this method has not been debugged!
        # USE WITH CAUTION!
        ra_names = np.array(['ra', 'RA'])
        dec_names = np.array(['DEC', 'Dec', 'dec'])
        coords = []
        missings = [] # whether the coord is missing
        not_missing_ids = [] # the indices of those that are not missing
        for coordi, datai, uniti in [[self.coord, data, self.unit], [self.coord1, data1, self.unit1]]:
            if coordi is None or isinstance(coordi, str):
                if coordi is None: # auto decide ra, dec
                    found_ra = np.isin(ra_names, datai.colnames)
                    if not np.any(found_ra):
                        raise KeyError(f'RA for {datai.name} not found.')
                    self.ra_name = ra_names[np.where(found_ra)][0]
                    ra = datai.t[self.ra_name]
    
                    found_dec = np.isin(dec_names, datai.colnames)
                    if not np.any(found_dec):
                        raise KeyError(f'Dec for {datai.name} not found.')
                    self.dec_name = dec_names[np.where(found_dec)][0]
                    dec = datai.t[self.dec_name]
                    
                    if verbose: print(f"[SkyMatcher] Data {datai.name}: found RA name '{self.ra_name}' and Dec name '{self.dec_name}'.")
            
                else: # type(coordi) is str:
                    self.ra_name, self.dec_name = coordi.split('-')
                    ra = datai.t[self.ra_name]
                    dec = datai.t[self.dec_name]
                
                # check missing values for ra and dec
                # TODO: below NOT TESTED
                missingi = np.full(len(datai), False)
                if np.ma.is_masked(ra):
                    missingi |= ra.mask
                if np.ma.is_masked(dec): # datai.t.masked or 
                    missingi |= dec.mask
                    
                not_missing_idi = np.arange(len(datai), dtype=int)[~missingi]
                
                try:
                    coordi = SkyCoord(ra=ra[~missingi], dec=dec[~missingi], unit=uniti)
                except UnitTypeError as e:
                    info = e.args[0]
                    which_coor = self.ra_name if 'Longitude' in info else self.dec_name
                    got_unit =  info.split('set it to ')[-1]
                    raise UnitTypeError(f"Unrecognized unit for column '{which_coor}': expected units equivalent to 'rad', got {got_unit}"\
                                        f" Try manually setting {datai.__repr__()}.t['{which_coor}'].unit") from e
            
            elif type(coordi) is SkyCoord:
                self.ra_name, self.dec_name = None, None
                coordi = coordi
                
                missingi = np.full(len(datai), False)
                not_missing_idi = np.arange(len(datai), dtype=int)[~missingi]
            
            else:
                raise TypeError(f"Unsupported type for coord/coord1: expected str or astropy.coordinates.SkyCoord, got {type(coordi)}")
                
            coords.append(coordi)
            missings.append(missingi)
            not_missing_ids.append(not_missing_idi)
        
        self.coord, self.coord1 = coords
        self.missing, self.missing1 = missings
        self.not_missing_id, self.not_missing_id1 = not_missing_ids
        
        # check duplicates for coordinates
        _, counts = np.unique(np.stack([self.coord.ra, self.coord.dec]), axis=1, return_counts=True)
        if np.any(counts != 1):
            warnings.warn(f"Duplications found for data '{data.name}' while matching '{data1.name}' to it: the same row of '{data1.name}' may be matched to multiple rows in '{data.name}'.",
                          stacklevel=3, category=DuplicationWarning)
        _, counts = np.unique(np.stack([self.coord1.ra, self.coord1.dec]), axis=1, return_counts=True)
        if np.any(counts != 1):
            warnings.warn(f"Duplications found for data '{data1.name}' while matching to '{data.name}': there may be multiple rows in '{data1.name}' that can be matched to a row in '{data.name}', and only one will be returned by the matcher.",
                          stacklevel=3, category=DuplicationWarning)
        
    def match(self):
        l = len(self.missing)
        idx = np.full(self.missing.shape, -l-1)
        matched = np.full(self.missing.shape, False)
        idx_nm, d2d, d3d = self.coord.match_to_catalog_sky(self.coord1)
        idx[~self.missing] = self.not_missing_id1[idx_nm]
        matched[~self.missing] = d2d.arcsec < self.thres
        return idx, matched
    
    def explore(self, data, data1):
        '''
        Plot as simple histogram to 
        check the distribution of the minimum (2-d) sky separation.

        Parameters
        ----------
        data : ``pyttop.table.Data``
            The base data of the match.
        data1 : ``pyttop.table.Data``
            The data to be matched to ``data1``.

        Returns
        -------
        None.

        '''
        self.get_values(data, data1)
        idx, d2d, d3d = self.coord.match_to_catalog_sky(self.coord1)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(np.log10(d2d.arcsec), bins=min((200, len(data)//20)), histtype='step', linewidth=1.5, log=True)
        plt.axvline(np.log10(self.thres), color='r', linestyle='--')
        plt.xlabel('lg (d / arcsec)')
        plt.title(f"Min. distance to '{data1.name}' objects for each '{data.name}' object\nthreshold={self.thres}\"")
        return d2d.arcsec
        
    def __repr__(self):
        # TODO: show more information here 
        return f'<SkyMatcher with thres={self.thres}>'

class IdentityMatcher():
    def __init__(self):
        '''
        Used to match ``pyttop.table.Data`` objects ``data1`` to ``data``.
        Directly match records row by row, i.e. row #1 matched to row #1, row #2 matched to row #2, etc.
        Only possible if ``len(data1) == len(data)``.
        This should be passed to method `data.match()`.
        See ``help(data.match)``.
        '''
    
    def get_values(self, data, data1, verbose=True):
        if len(data) != len(data1):
            raise ValueError(f'IdentityMatcher can only be used to match data with the same number of rows ({len(data)} != {len(data1)})')
        self.len = len(data)
    
    def match(self):
        idx = np.arange(self.len)
        matched = np.full((self.len,), True)
        return idx, matched    
    
    def __repr__(self):
        return '<IdentityMatcher>'
    
