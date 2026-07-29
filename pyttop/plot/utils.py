# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:09:35 2026

@author: Yu-Chen Wang
"""

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

__all__ = [
    'merged_legend',
    ]

def merged_legend(ax=None, pad=0.3, **kwargs):
    '''
    Create a legend with identically labelled artists merged into one entry.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes used to create the legend. 
        The default is None (current Axes).
    pad : float, optional
        Padding between artists within a merged legend entry, in fractions
        of the legend font size. The default is 0.3.
    **kwargs : 
        Additional keyword arguments passed to ``ax.legend``.

    Returns
    -------
    matplotlib.legend.Legend
        The created legend.
    '''
    if ax is None:
        ax = plt.gca()
        
    handles, labels = ax.get_legend_handles_labels()
    
    label_handles = defaultdict(list) # {label: handles}
    for h, l in zip(handles, labels):
        label_handles[l].append(h)
    
    merged_labels = list(label_handles.keys())
    merged_handles = [hs[0] if len(hs) == 1 else tuple(hs) for hs in label_handles.values()]
    
    handler_map = kwargs.pop('handler_map', {})
    if handler_map is None:
        handler_map = {}
    handler_map = dict(handler_map)
    
    if tuple not in handler_map:
        handler_map[tuple] = HandlerTuple(ndivide=None, pad=pad)
        
    return ax.legend(merged_handles, merged_labels, handler_map=handler_map, **kwargs)
    
