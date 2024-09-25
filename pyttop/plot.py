# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:19:10 2022

@author: Yuchen Wang
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from inspect import signature, isfunction
from astrotable.utils import objdict
from collections.abc import Iterable
from copy import deepcopy

Axes = matplotlib.axes.Axes

#%% config
# Configuration that controls the behavior of Data.plot() (or Data.plots()) when a PlotFunction object is passed to it.
# This is the global and default config; customize it for each individual plot function, say `plot`, 
# by directly modifying `plot.config`.
DEFAULT_CONFIG = {
    'ax_label_kwargs_generator': # function to generate the kwargs, to be passed to axis, that sets the axis labels
        lambda labels: # input labels
            dict(zip(['xlabel', 'ylabel', 'zlabel'], labels),), 
            # returns dict like {'xlabel': xlabel, ...}
    }

#%% Class
class PlotFunction():
    def __init__(self, func, input_ax=True):
        self.func = func
        self.input_ax = input_ax
        if hasattr(func, 'ax_callback'):
            self.ax_callback = func.ax_callback
        else:
            self.ax_callback = lambda ax: None
        
        if input_ax:
            plot_func = func(Axes)
            self.func_doc = plot_func.__doc__
            self.func_name = func.__name__ # the appearent name when using this function
            self.func_defname = plot_func.__name__ # the real name in the definition of plot function
            self.func_sig = (signature(plot_func))
        else:
            self.func_doc = func.__doc__
            self.func_name = func.__name__ # the appearent name when using this function
            self.func_defname = func.__name__ # the real name in the definition of plot function
            self.func_sig = (signature(self.func))
        self.func_def = self.func_name + str(self.func_sig) + '\n\n' + self.func_name + '(axis)' + str(self.func_sig)
        
        # TODO: below may cause bugs
        self.func_def = self.func_def.replace('(self, ', '(')
        if self.func_doc is None: self.func_doc = ''
        if self.func_doc and self.func_doc[0] == '\n': 
            self.func_doc = self.func_doc[1:]
        
        # self.__call__.__func__.__doc__ = self.func_doc
        
        # config for Data.plot() or Data.plots()
        self.config = deepcopy(DEFAULT_CONFIG)
        if hasattr(self.func, 'config'):
            self.config.update(self.func.config)
    
    def _call_with_ax(self, ax):
        if self.input_ax:
            @wraps(self.func(ax))
            def plot(*args, **kwargs):
                return self.func(ax)(*args, **kwargs)
        else:
            @wraps(self.func)
            def plot(*args, **kwargs):
                ca = plt.gca()
                plt.sca(ax)
                out = self.func(*args, **kwargs)
                plt.sca(ca)
                return out
        plot.ax_callback = self.ax_callback
        return plot
    
    def __call__(self, *args, **kwargs): # direct call
        '''
        Calling the plot function modified by plotFuncAx or plotFunc. 
        ''' # For documentation, execute ``<function name>.help()``.
        
        # if f is called as f(ax), f(ax=ax):
        if len(args) == 0 and list(kwargs.keys()) == ['ax']: # f called as f(ax=ax)
            ax = kwargs['ax']
            if isinstance(ax, Axes):
                return self._call_with_ax(ax)
        elif len(kwargs) == 0 and len(args) == 1: # f called as f(ax) or f(x)
            ax = args[0]
            if isinstance(ax, Axes):
                return self._call_with_ax(ax)
        
        # seem that f not called with only one axis as input:
        ax = plt.gca()
        out = self._call_with_ax(ax)(*args, **kwargs)
        self.ax_callback(ax)
        return out
    
    def in_plot(self, *args, **kwargs): # use in Data.plot
        # do not call ax_callback.
        # plot function may be called several times in one subplot,
        # but ax_callback should be called ONLY ONCE.
        ax = plt.gca()
        return self._call_with_ax(ax)(*args, **kwargs)
        # return self.ax_callback
    
    def in_subplot_array(self, ax): # use in Data.subplot_array
        return self._call_with_ax(ax)
    
    # def help(self):
    #     print(self.func_doc)
    
    def __getattr__(self, attr):
        return getattr(self.func, attr)
        
    @property
    def __doc__(self): # manually generate doc
        return self.func_def + '\n\nFunction modified to accomodate astrotable.table.Data. Original documentaion shown below:\n\n' + self.func_doc + '\n\n'
    
    @property
    def __name__(self):
        return self.func_name

class DelayedPlot():
    def __init__(self):
        raise NotImplementedError()
        pass
    
    def __call__(self):
        pass
    
#%% stand-alone functions
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
            
#%% wrapper for plot functions
def plotFuncAx(f):
    '''
    Makes a function compatible to astrotable.table.Data. 

    Usage::
        
        @plotFuncAx
        def f(ax): # inputs axis object `ax`
            def plot_func(<your inputs ...>):
                <make the plot>
            return plot_func
    '''
    return PlotFunction(f, input_ax=True)

def plotFunc(f):
    '''
    Makes a function compatible to astrotable.table.Data. 

    Usage::
        
        @plotFunc
        def plot_func(<your inputs ...>):
            <make the plot>
    '''
    return PlotFunction(f, input_ax=False)

def plotFuncAuto(f):
    # automatically select plotFunc or plotFuncAx (or nothing to be done)
    if isinstance(f, PlotFunction):
        return f
    try: # what f(ax)(...) should be like
        # _, _temp_ax = plt.subplots()
        _f = f(Axes) # TODO (not solved): f may do something when calling this
        assert callable(_f)
    except:
        return plotFunc(f)
    else:
        return plotFuncAx(f)
        

#%% axis callbacks
def colorbar(ax):
    # TODO: automatically detect and add a colorbar
    raise NotImplementedError()
    pass

#%% plot functions

# to generate a universal colorbar for several scatter plots in the same panel,
# we need to play a trick: do not actually plot scatter in the main part;
# save it to ax_callback.
# TODO: Scatter is not elegant. Improve it.
class Scatter():
    def __init__(self):
        self.__name__ = 'scatter'
        self.params = []
        self.autobar = None
        # self.ax = None
        # self.s = None
    
    @staticmethod
    def _decide_autobar(c, x, autobar):
        # parse c input and decide autobar or not
        if not autobar or c is None:
            return False
        else:
            try:
                carr = np.asanyarray(c, dtype=float)
            except ValueError:
                return False
            else:
                if not (carr.shape == (1, 4) or carr.shape == (1, 3)) and carr.size == x.size:
                    return True
                else:
                    return False
    
    def __call__(self, ax):
        # if self.ax is not None and self.ax != ax:
        #     self.params = []
        # self.ax = ax
        def scatter(x, y, s=None, c=None, *, cmap=None, vmin=None, vmax=None, autobar=True, barlabel=None, **kwargs):
            self.autobar = self._decide_autobar(c, x, autobar)
            # self.autobar = autobar and (c is not None and len(c)==len(x))
            param = {key: value for key, value in locals().items() if key not in ('self', 'kwargs')}
            param.update(kwargs)
            self.params.append(param)
            # if self.s:
            #     return self.s
        return scatter
    
    def ax_callback(self, ax):
        try:
            if self.autobar: # decide colorbar information
                # the general parameters for the whole plot
                cs = []
                barinfo = objdict(
                    vmin = None,
                    vmax = None,
                    barlabel = None,
                    cmap = None)
                
                for param in self.params:
                    for name in ['vmin', 'vmax', 'barlabel', 'cmap']: # check consistency for different calls
                        if barinfo[name] is None:
                            barinfo[name] = param[name]
                        elif barinfo[name] != param[name]:
                            raise ValueError(f'colorbar cannot be generated due to inconsistency of "{name}": {barinfo[name]} != {param[name]}')
                        
                    cs.append(param['c'])
                
                # decide vmin, vmax
                if barinfo.vmin is None:
                    barinfo.vmin = min([np.min(c) for c in cs])
                if barinfo.vmax is None:
                    barinfo.vmax = max([np.max(c) for c in cs])
                
                param_exclude = ['cmap', 'vmin', 'vmax', 'autobar', 'barlabel']
                color_param_keys = ['vmin', 'vmax', 'cmap']
                for param in self.params:
                    param = {key: value for key, value in param.items() if key not in param_exclude}
                    colorparams = {key: value for key, value in barinfo.items() if key in color_param_keys}
                    self.s = ax.scatter(**param, **colorparams)
                
                # make colorbar
                cax = plt.colorbar(self.s, ax=ax)
                cax.set_label(barinfo.barlabel)
                
            else:
                param_exclude = ['autobar', 'barlabel']
                for param in self.params:
                    param = {key: value for key, value in param.items() if key not in param_exclude}
                    self.s = ax.scatter(**param)
        
        finally:
            self.params = []

scatter = plotFuncAx(Scatter())    

@plotFuncAx
def plot(ax):
    return ax.plot
def _plot_label(labels):
    if len(labels) == 1:
        return {'ylabel': labels[0]} # if only one arg is given, this is y axis rather than x axis
    else:
        return dict(zip(['xlabel', 'ylabel', 'zlabel'], labels),), 
plot.config['ax_label_kwargs_generator'] = _plot_label

@plotFuncAx
def hist(ax):
    @wraps(ax.hist)
    def _hist(x, *args, **kwargs):
        # Masked arrays are not supported by plt.hist.
        # let us consider this here.
        if np.ma.is_masked(x):
            x = x[~x.mask]
        return ax.hist(x, *args, **kwargs)
    return _hist

@plotFuncAx
def hist2d(ax):
    @wraps(ax.hist2d)
    def _hist2d(x, y, *args, **kwargs):
        # since plt.hist2d does not handle masked values, let us consider this here
        # (mask lost in: plt.hist2d -> np.histogram2d -> np.histogramdd -> np.atleast_2d -> call of asanyarray() in np.core.shape_base)
        mask = np.full(x.shape, False)
        if np.ma.is_masked(x):
            mask |= x.mask
        if np.ma.is_masked(y):
            mask |= y.mask
        x = x[~mask]
        y = y[~mask]
        return ax.hist2d(x, y, *args, **kwargs)
    return _hist2d

@plotFuncAx
def errorbar(ax):
    return ax.errorbar

annotate = plotFunc(_annotate)
