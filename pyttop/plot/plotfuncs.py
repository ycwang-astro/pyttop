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
    'annotate'
    ]

def _annotate(x=None, y=None, xpos=.1, ypos=.1, xtxt=None, ytxt=None, xfmt='.2f', yfmt='.2f', marker='', style='through', label=None, ax=None, **lineargs):
    '''
    Plot a point with a marker,
    as well as a horizontal line and a vertical line,
    both going through the point.

    Parameters
    ----------
    x : number or Iterable object, optional
        The x position(s). The default is None.
    y : number or Iterable object, optional
        The y position(s). The default is None.
    xpos : float or None, optional
        The horizontal position of the text relative to the width of the plot.
        If it is None, no text added.
        The default is .1
    ypos : float or None, optional
        The vertical position of the text relative to the height of the plot.
        If it is None, no text added.
        The default is .1
    xtxt : str, optional
        If not None, the x label text will be overwritten by this.
    ytxt : str, optional
        If not None, the y label text will be overwritten by this.
    xfmt : str, optional
        The format string for x label (if xtxt not specified).
        The default is '.2f'.
    yfmt : str, optional
        The format string for y label (if ytxt not specified).
        The default is '.2f'.
    marker : optional
        The marker of the point.
        The default is ''.
    style : str, optional
        'through' or 'axis'.
        'through': plot line(s) across the whole axis.
        'axis': only plot line(s) on the left and/or beneath the point.
    label : str, optional
        The label for the lines.
    ax : optional
        The axis where you want to plot the lines.
    **lineargs :
        Keyword arguments for lines.
    '''

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
                plt.pause(.01) # this pause is essential. without this pause, the ScalarFormatter (got by ax.xaxis.get_major_formatter()) has not yet been set, so one always get offset == 1.
                offset = ax.xaxis.get_major_formatter().get_offset()
                if offset == '':
                    offset = 1
                else:
                    offset = float(offset)
                # offset = 1
                if xtxt is None:
                    xtxt1 = ('${:'+xfmt+'}$').format(x/offset)
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
                plt.pause(.01) # this pause is essential. without this pause, the ScalarFormatter (got by ax.xaxis.get_major_formatter()) has not yet been set, so one always get offset == 1.
                offset = ax.yaxis.get_major_formatter().get_offset()
                if offset == '':
                    offset = 1
                else:
                    offset = float(offset)
                # offset = 1
                if ytxt is None:
                    ytxt1 = ('${:'+yfmt+'}$').format(y/offset)
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

annotate = plotFunc(_annotate)


