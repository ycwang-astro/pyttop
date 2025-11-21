# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:04:11 2024

@author: Yu-Chen Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from .base import plotFunc, plotFuncAx
from .base import scatter as ptscatter
from collections.abc import Iterable

__all__ = [
    'refline', 'annotate',
    'binned_quantiles'
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

@plotFunc
def binned_quantiles(x, y, 
                     bin_size=.1, bin_dist=.1, quantiles=[.16, .50, .84], min_n=10, 
                     xmin=None, xmax=None,
                     show_scatter=True, s=None, c=None, label=None, 
                     show_bins=True, show_errorbars=True, emarker='o', es=5, ec=None, elabel=None,
                     show_fill=False, fc=None, flabel=None,
                     errkwargs={}, fillkwargs={}, **kwargs
                     ):
    """
    Plot sliding-window quantile errorbars/fill.

    This function visualizes a 2D distribution by plotting raw (x, y) points and computing
    sliding-window quantiles in x-bins. It then overlays error bars and/or filled regions 
    to represent variability (e.g. 16th–84th percentile) in y-values within each x-bin.
    
    This is useful when visualizing scatter data along with robust estimates of central tendency 
    and spread.

    Parameters
    ----------
    x, y : array-like
        Data coordinates.
    bin_size : float, optional
        Width of each sliding bin in x-units. Default is 0.1.
    bin_dist : float, optional
        Step size between consecutive bin positions (i.e., sliding window stride). Default is 0.1.
    quantiles : list of 3 floats, optional
        List of quantiles to compute within each bin. Must be in increasing order.
        Default is [0.16, 0.50, 0.84].
    min_n : int, optional
        Minimum number of data points required in a bin to compute quantiles. Default is 10.
    xmin, xmax : float, optional
        Range of x-values to include in binning. If None, inferred from data.
    show_scatter : bool, optional
        If True, plot the raw scatter points. Default is True.
    s, c : optional
        Marker size and color for scatter points.
    label : str, optional
        Label for the scatter plot.
    show_bins : bool, optional
        If True, include horizontal error bars showing bin width. Default is True.
    show_errorbars : bool, optional
        If True, show vertical error bars (quantile-based). Default is True.
    emarker : str, optional
        Marker style for error bar midpoints. Default is 'o'.
    es : float, optional
        Marker size for error bars. Default is 5.
    ec : color, optional
        Color for error bars.
    elabel : str, optional
        Label for the error bars. If None and ``show_scatter`` is False, inherits ``label``.
    show_fill : bool, optional
        If True, fills the area between lower and upper quantiles. Default is False.
    fc : color, optional
        Fill color. If None, inherits from ``ec``.
    flabel : str, optional
        Label for the fill.
    errkwargs : dict, optional
        Additional keyword arguments passed to ``plt.errorbar``.
    fillkwargs : dict, optional
        Additional keyword arguments passed to ``plt.fill_between``.
    **kwargs : dict
        Additional keyword arguments passed to the scatter plot.
    """
    # np.asarray() or np.array() will return a base np.ndarray (masks will be lost)
    x, y = np.asanyarray(x), np.asanyarray(y)
    artists = {}
    
    if show_scatter:
        ptscatter(x, y, s=s, c=c, label=label, **kwargs)
        artists['scatter'] = ptscatter.s
        # scatter = plt.scatter(x, y, s=s, c=c, label=label, **kwargs)
    
    if not show_scatter and elabel is None:
        elabel = label
    if fc is None:
        fc = ec
    
    mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
    x = x[~mask]
    y = y[~mask]
    
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    x_lefts = np.arange(xmin, xmax - bin_size + bin_dist, bin_dist)
    # print(np.min(x), np.max(x), bin_size, x_lefts)
    
    x_centers = x_lefts + bin_size / 2
    x_rights = x_lefts + bin_size
    
    ymids, yq0s, yq1s = [], [], []
    
    for left, right in zip(x_lefts, x_rights):
        assert right == left + bin_size
        in_bin = (x >= left) & (x < right)
        if np.sum(in_bin) >= min_n:
            yq0, ymid, yq1 = np.quantile(y[in_bin], quantiles)
        else:
            yq0, ymid, yq1 = np.nan, np.nan, np.nan
        ymids.append(ymid)
        yq0s.append(yq0)
        yq1s.append(yq1)
    ymids = np.array(ymids)
    yq0s = np.array(yq0s)
    yq1s = np.array(yq1s)
    
    ekwargs = dict(
        linestyle='',
        )
    ekwargs.update(errkwargs)
    
    if show_bins:
        xerr = [x_centers-x_lefts, x_rights-x_centers]
    else:
        xerr = None
    
    if show_errorbars:
        yerr = [ymids-yq0s, yq1s-ymids]
    else:
        yerr = None
    
    artists['errorbar'] = plt.errorbar(
        x_centers, ymids,
        xerr=xerr,
        yerr=yerr,
        marker=emarker, markersize=es, color=ec,
        label=elabel,
        **ekwargs,
        )
    
    if show_fill:
        fkwargs = {'alpha': .2} | fillkwargs
        artists['fill'] = plt.fill_between(
            x_centers, yq1s, yq0s, 
            color=fc, label=flabel,
            **fkwargs,
            )
    
    return artists
