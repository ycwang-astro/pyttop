# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:19:10 2022

@author: Yuchen Wang
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps, update_wrapper
from inspect import signature
import textwrap
from ..utils import objdict
from ..config import config
from copy import deepcopy
import matplotlib.colors as mcolors

Axes = matplotlib.axes.Axes

__all__ = [
    'PlotFunction', 'plotFuncAx', 'plotFunc', 'plotFuncAuto',
    'scatter', 'plot', 'hist', 'hist2d', 'errorbar'
    ]

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

#%% fundamental classes
# TODO: the implementation of PlotFunction,
#       the two supported signatures [func(...) and func(ax)(...)]
#       and its support in Data.plot, Data.plots
#       might be improved to make them more elegant.
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
        self.func_defs = [
            self.func_name + str(self.func_sig),
            self.func_name + '(axis)' + str(self.func_sig),
            ]

        # TODO: below may cause bugs
        self.func_defs = [func_def.replace('(self, ', '(') for func_def in self.func_defs]
        if self.func_doc is None: self.func_doc = ''
        if self.func_doc and self.func_doc[0] == '\n':
            self.func_doc = self.func_doc[1:]
        self.func_doc = textwrap.dedent(self.func_doc)

        # self.__call__.__func__.__doc__ = self.func_doc

        # config for Data.plot() or Data.plots()
        self.config = deepcopy(DEFAULT_CONFIG)
        if hasattr(self.func, 'config'):
            self.config.update(self.func.config)
        
        update_wrapper(self, func)
        self.__doc__ = self._generate_doc()

    def _call_with_ax(self, ax, execute_callback=False):
        if self.input_ax:
            @wraps(self.func(ax))
            def plot(*args, **kwargs):
                out = self.func(ax)(*args, **kwargs)
                if execute_callback:
                    self.ax_callback(ax)
                return out
        else:
            @wraps(self.func)
            def plot(*args, **kwargs):
                ca = plt.gca()
                plt.sca(ax)
                out = self.func(*args, **kwargs)
                if execute_callback:
                    self.ax_callback(ax)
                plt.sca(ca)
                return out
        plot.ax_callback = self.ax_callback
        return plot

    def __call__(self, *args, **kwargs):
        # calling it as a standalone function
        # decide how it is called
        call_with_ax = False
        if len(args) == 0 and list(kwargs.keys()) == ['ax']: # f called as f(ax=ax)
            ax = kwargs['ax']
            if isinstance(ax, Axes):
                call_with_ax = True
        elif len(kwargs) == 0 and len(args) == 1: # f called as f(ax) or f(x)
            ax = args[0]
            if isinstance(ax, Axes):
                call_with_ax = True

        # call the plot function, and execute ax_callback
        if call_with_ax: # f is called as f(ax), f(ax=ax):
            return self._call_with_ax(ax, execute_callback=True)
        else:  # f not called with only one axis as input
            ax = plt.gca()
            out = self._call_with_ax(ax)(*args, **kwargs)
            self.ax_callback(ax)
            return out

    def call_with_ax(self, ax, execute_callback=False):
        # calling it with f(ax)(...)
        # ax_callback not executed by default
        # used in Data.plots
        # plot function may be called several times in one subplot,
        # but ax_callback should be called ONLY ONCE.
        return self._call_with_ax(ax, execute_callback=execute_callback)

    def call_without_ax(self, *args, **kwargs):
        ax = plt.gca()
        return self._call_with_ax(ax)(*args, **kwargs)
        # return self.ax_callback

    # def help(self):
    #     print(self.func_doc)

    def __getattr__(self, attr):
        return getattr(self.func, attr)

    # @property
    # def __doc__(self): # manually generate doc
    def _generate_doc(self):
        return (self._generate_notice()
                + self.func_doc + '\n\n')
    
    def _generate_notice(self):
        notice_text = (
            'This function is made compatible with ``pyttop.table.Data.plots()`` and can be called in either of the following ways:\n\n'
            + '\n\n'.join([f'- ``{func_def}``' for func_def in self.func_defs])
            + '\n\n'
            )
        return ('.. tip::\n\n'
                + textwrap.indent(notice_text, '    '))
    
    def __repr__(self):
        return f"<pyttop PlotFunction {self.func_defs[0]}>"
    
    # @property
    # def __name__(self):
    #     return self.func_name

class DelayedPlot():
    def __init__(self):
        raise NotImplementedError()
        pass

    def __call__(self):
        pass

#%% stand-alone functions

#%% wrapper for plot functions
def plotFuncAx(f):
    '''
    Makes a function compatible to pyttop.table.Data.

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
    Makes a function compatible to pyttop.table.Data.

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
        # _f = f(Axes) # TODO (not solved): f may do something when calling this
        _f = f(None)
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
        return dict(zip(['xlabel', 'ylabel', 'zlabel'], labels),)
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

        # mask = np.full(x.shape, False)
        # if np.ma.is_masked(x):
        #     mask |= x.mask
        # if np.ma.is_masked(y):
        #     mask |= y.mask
        mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
        x = x[~mask]
        y = y[~mask]
        return ax.hist2d(x, y, *args, **kwargs)
    return _hist2d

@plotFuncAx
def errorbar(ax):
    return ax.errorbar

#%% table.Data mixins

# def colname_kwargs(*argnames):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):

#             pass
#         return wrapper
#     return decorator

class PlotMethodsMixin():
    @staticmethod
    def _process_colname_kwargs(keys, locals, argkeys=None):
        # argkeys: these will be passed as positional arguments
        if argkeys is None:
            argkeys = []
        kwargs = locals['kwargs']
        if 'kwcols' not in kwargs:
            kwargs['kwcols'] = {}
        if 'cols' not in kwargs:
            kwargs['cols'] = []

        for key in keys:
            value = locals[key]
            if isinstance(value, str): # regarded as a column name
                if key in argkeys:
                    kwargs['cols'].append(value)
                else:
                    kwargs['kwcols'][key] = value
            else:
                kwargs[key] = value
        return kwargs

    # @wraps(plt.plot)
    def lplot(self, *args, **kwargs):
        # an real counterpart of plt.plot may be difficult to implement
        raise NotImplementedError()

    # @wraps(plt.scatter)
    def scatter(self, x, y, s=None, c=None, **kwargs):
        # TODO: docstring
        if mcolors.is_color_like(c): # this seems to be a color string
            c = (mcolors.to_rgba(c),)
        self.__class__._process_colname_kwargs(
            ['x', 'y', 's', 'c'], locals(), argkeys=['x', 'y'])
        return self.plots('scatter', **kwargs)

    # @wraps(plt.hist)
    def hist(self, x, weights=None, **kwargs):
        # TODO: docstring
        kwargs = config.plot.defaults_hist | kwargs
        self.__class__._process_colname_kwargs(
            ['x', 'weights'], locals(), argkeys=['x'])
        return self.plots('hist', **kwargs)

