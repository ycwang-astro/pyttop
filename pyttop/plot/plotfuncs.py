# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:04:11 2024

@author: Yu-Chen Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from .base import plotFunc, plotFuncAx
from collections.abc import Iterable

__all__ = [
    'refline', 'annotate',
    ]

@plotFunc
def refline(x=None, y=None, xpos=.1, ypos=.1, xtxt=None, ytxt=None, xfmt='.2f', yfmt='.2f', marker='', style='through', label=None, ax=None, **lineargs):
    '''
    Plot reference line(s) and optionally marker(s) at given position(s).

    This function adds vertical and/or horizontal lines to the plot, 
    anchored at the specified `x` and/or `y` values. Optionally, 
    marker(s) can be drawn at the intersection(s), and text annotations can 
    be shown on the reference line(s) to indicate the values.

    Parameters
    ----------
    x : float or Iterable, optional
        The x-coordinate(s) at which to draw vertical reference line(s). The default is None.
    y : float or Iterable, optional
        The y-coordinate(s) at which to draw horizontal reference line(s). The default is None.
    xpos : float or None, optional
        Relative x (horizontal) position (in axes fraction) for y-value annotation text.
        If None, no text is shown.
        The default is 0.1.
    ypos : float or None, optional
        Relative y (vertical) position (in axes fraction) for x-value annotation text.
        If None, no text is shown.
        The default is 0.1.
    xtxt : str, optional
        If not None, the x label text will be overwritten by this.
    ytxt : str, optional
        If not None, the y label text will be overwritten by this.
    xfmt : str, optional
        Format string for x label (if ``xtxt`` not specified).
        The default is ``'.2f'``.
    yfmt : str, optional
        Format string for y label (if ``ytxt`` not specified).
        The default is ``'.2f'``.
    marker : optional
        Marker style for the intersection point, if both x and y are provided.
        The default is '' (no marker).
    style : {'through', 'axis'}, optional
        Line style:
        
        - ``'through'``: line(s) extend across the full axis.
        - ``'axis'``: only plot line(s) on the left and/or beneath the point.
        
        The default is ``'through'``.
    label : str, optional
        Label assigned to the line(s), useful for legends.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot. If None, uses the current axis.
    **lineargs :
        Additional keyword arguments passed to ``ax.axhline`` and ``ax.axvline``.
    '''
    
    def _format_val(v, formatter):
        return (v - formatter.offset) / 10.**formatter.orderOfMagnitude
    
    # check input
    if style not in ['through', 'axis']:
        raise ValueError(f"'style' should be 'through' or 'axis', got '{style}'")

    artists = {}

    if ax is None:
        ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    xscale = ax.get_xscale()
    if xscale == 'log':
        dx = np.log10(xmax) - np.log10(xmin)
    else:
        dx = xmax - xmin
    ymin, ymax = ax.get_ylim()
    yscale = ax.get_yscale()
    if yscale == 'log':
        dy = np.log10(ymax) - np.log10(ymin)
    else:
        dy = ymax - ymin
    
    fig = ax.figure

    if x is None and y is None:
        raise ValueError('You should at least specify one of the parameters: "x" and "y".')

    if x is not None:
        if isinstance(x, Iterable):
            xs = x
        else:
            xs = [x]
        if not isinstance(xpos, Iterable):
            xposs = [xpos]*len(xs)
        else:
            xposs = xpos

    if y is not None:
        if isinstance(y, Iterable):
            ys = y
        else:
            ys = [y]
        if not isinstance(ypos, Iterable):
            yposs = [ypos]*len(ys)
        else:
            yposs = ypos

    if x is None:
        xs = [xmax]*len(ys)
    if y is None:
        ys = [ymax]*len(xs)

    plotx, ploty = False, False
    if x is not None:
        plotx = True
    if y is not None:
        ploty = True

    if plotx:
        for i, info in enumerate(zip(xs, xposs, ys)):
            x, xpos, y = info
            if style == 'through':
                lineymax = 1
            elif style == 'axis':
                lineymax = (np.log10(y)-np.log10(ymin))/dy if yscale == 'log' else (y-ymin)/dy
            if i != 0:
                label = None
            artists['vline'] = ax.axvline(x, ymax=lineymax, label=label, **lineargs)
            if xpos is not None:
                fig.canvas.draw() # makes sure the ScalarFormatter has been set
                x_fmter = ax.xaxis.get_major_formatter()
                if xtxt is None:
                    xtxt1 = f'{_format_val(x, x_fmter):{xfmt}}'
                else:
                    xtxt1 = xtxt
                if yscale == 'log':
                    yt = ymin * (ymax/ymin)**ypos
                else:
                    yt = ymin + ypos * dy
                artists['vtext'] = ax.text(x, yt, xtxt1, horizontalalignment='center', backgroundcolor='white')

    if ploty:
        for i, info in enumerate(zip(ys, yposs, xs)):
            y, ypos, x = info
            if style == 'through':
                linexmax = 1
            elif style == 'axis':
                linexmax = (np.log10(x)-np.log10(xmin))/dx if xscale == 'log' else (x-xmin)/dx
            if i != 0 or plotx:
                label = None
            artists['hline'] = ax.axhline(y, xmax=linexmax, label=label, **lineargs)
            if ypos is not None:
                fig.canvas.draw() # makes sure the ScalarFormatter has been set
                y_fmter = ax.yaxis.get_major_formatter()
                if ytxt is None:
                    ytxt1 = f'{_format_val(y, y_fmter):{yfmt}}'
                else:
                    ytxt1 = ytxt
                if xscale == 'log':
                    xt = xmin * (xmax/xmin)**xpos
                else:
                    xt = xmin + xpos * dx
                artists['htext'] = ax.text(xt, y, ytxt1, verticalalignment='center', backgroundcolor='white')

    if plotx and ploty:
        artists['scat'] = ax.scatter(x, y, marker=marker, c='k')

    return artists

# annotate = plotFunc(_annotate)
annotate = refline

