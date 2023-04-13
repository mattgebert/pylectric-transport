# Programming Syntax & annotations #1
from __future__ import annotations
# Function Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
# Custom Libraries
import pylectric
from pylectric.graphing import geo_base, graphwrappers
from pylectric.analysis import mobility
# Programming Syntax & annotations #2
import warnings
from overrides import override
from abc import abstractmethod
import inspect

class hallbar_measurement(geo_base.graphable_base_dataseries):
    """Class for a hall measurement, that is magnetic field dependence of Rxx and Rxy of a device.
        Does not assume gated geometry, however does assume magnetic measurement for the purpose of Rxy data.
        
        Can provide additional data to add to object, in the form of a dictionary of data {label:numpy_array}.
    """
    
    def __init__(self, field, rxx, rxy, dataseries = {}, geom = 1, params = {}):
        # initialise super object
        super().__init__(dataseries=dataseries)
        
        ### Valid Datachecking:
        datalen = None #ensure datalength is constant.
        if isinstance(rxx, np.ndarray):
            datalen = len(rxx)
        
        #Check all arrays are numpy and correct length.
        for i in [field, rxx, rxy] + [dataseries[key] for key in dataseries]:
            if not isinstance(i,np.ndarray):
                raise TypeError("Passed are not numpy arrays.")
            elif len(i) != datalen:
                raise IndexError("Length of data arrays do not match.")
            
        self.field = field.copy()
        self.rxx = rxx.copy()
        self.rxy = rxy.copy()
        self.geom = geom
        
        # Convert rxx, rxy to rhoxx, rhoxy, sigmaxx, sigmaxy
        self._calculateTransport()
        # Copy items in params to self.
        self.params = {}
        self.params.update(params)
        return
    
    def _calculateTransport(self):
        # Convert rxx, rxy to rhoxx, rhoxy, sigmaxx, sigmaxy
        self.rhoxx = self.rxx.copy() * self.geom
        self.rhoxy = self.rxy.copy() * self.geom
        self.sigmaxx = self.rhoxx / (self.rhoxx**2 + self.rhoxy**2)
        self.sigmaxy = -self.rhoxy / (self.rhoxx**2 + self.rhoxy**2)
        
    
    def symmterise(self, full_domain=False) -> tuple[hallbar_measurement, hallbar_measurement]:
        """_summary_

        Args:
            full_domain (bool, optional): . Defaults to False.

        Raises:
            AttributeError: Field values not symmetric in data.

        Returns:
            _type_: _description_
        """
        #0. Check requirements [datapoints are evenly binned]:
        if not np.alltrue(self.field[:] == -self.field[::-1]):
            raise AttributeError("Field B is not symmetric in field about 0." +
                " Please use pylectric.signals.processing.reduction to sample data into symmetric bins.")
        
        #1. Create two clones:
        sym_clone = self.copy() #symmetric
        asym_clone = self.copy() #assymmetric
        
        #Put data together into one array.
        all_data = self.all_vars()
        
        #Symmmetrise all data:
        sym, asym = pylectric.signals.processing.symmetrise(all_data, colN=0, full_domain=full_domain) #field = 0
        
        #Reassign datasets:
        sym_clone.field = sym[:,0]
        sym_clone.rxx = sym[:,1]
        sym_clone.rxy = sym[:,2]
        
        asym_clone.field = asym[:, 0]
        asym_clone.rxx = asym[:, 1]
        asym_clone.rxy = asym[:, 2]
        
        i = 0
        for key, i in zip(self.dataseries, range(len(self.dataseries))):
            sym_clone.dataseries[key] = sym[:,3+i]
            asym_clone.dataseries[key] = asym[:,3+i]
        
        # Recalulate other transport parameters
        sym_clone._calculateTransport()
        asym_clone._calculateTransport()
                
        return sym_clone, asym_clone
    
    def copy(self):
        newobj = hallbar_measurement(field=self.field, rxx=self.rxx, rxy=self.rxy)
        newobj.dataseries = self.dataseries.copy()
        for key in newobj.dataseries:
            newobj.dataseries[key] = newobj.dataseries[key].copy()
        newobj.rhoxx = self.rhoxx.copy()
        newobj.rhoxy = self.rhoxy.copy()
        newobj.sigmaxx = self.sigmaxx.copy()
        newobj.sigmaxy = self.sigmaxy.copy()
        newobj.geom = self.geom
        newobj.params = self.params.copy()
        return newobj
    
    def __combine__(self,x):
        """Appends data together for all fields, including dataseries items.

        Args:
            x (hallbar_measurement): A secondary measurement object.

        Raises:
            TypeError: If x is not a hallbar_measurement.
            AttributeError: Geometry doesn't match
            AttributeError: Dataseries items not found in x.
            AttributeError: Extra dataseries items in x.

        Returns:
            hallbar_measurement: New object with datablocks combined.
        """
        if not isinstance(x,hallbar_measurement):
            raise TypeError("'" + str(x) + "' is not a hallbar_measurement object.")
        else:
            # check identical parameter lists:
            if self.geom != x.geom:
                raise AttributeError("Geometry of " + str(x) + " doesn't match " + str(self))
            xkeys = list(x.dataseries.keys())
            for key1 in self.dataseries.keys():
                if key1 not in xkeys:
                    raise AttributeError(key1 + " not found in " + str(x))
                else:
                    xkeys.remove(key1)
            if len(xkeys) > 0:
                raise AttributeError(xkeys + " are not found in " + str(self))
            
            # create new object
            newobj = self.copy()
            newobj.rxx = np.r_[newobj.rxx, x.rxx]
            newobj.rxy = np.r_[newobj.rxy, x.rxy]
            newobj.rhoxx = np.r_[newobj.rhoxx, x.rhoxx]
            newobj.rhoxy = np.r_[newobj.rhoxy, x.rhoxy]
            newobj.field = np.r_[newobj.field, x.field]
            newobj.sigmaxx = np.r_[newobj.sigmaxx, x.sigmaxx]
            newobj.sigmaxy = np.r_[newobj.sigmaxy, x.sigmaxy]
            for key in self.dataseries.keys():
                self.dataseries[key] = np.r_[self.dataseries[key], x.dataseries[key]]
            return newobj
    
    @override
    def __sub__(self, x, reverse=False):
        """Performs a diference operation on all rxx,rxy and dataseries data between the object and x.

        Args:
            x (hallbar_measurement): Secondary object for subtraction.
        """
        #TODO: Expand to 3D case.
        #TODO: Fix method, use pylectric.signals.processing.trim_matching
        
        # Conditions to allow subtraction:
        assert isinstance(x, hallbar_measurement)
        subdata = super().__sub__(x)
        newobj = self.copy()
        
        #Assign data to existing variables
        newobj.field = subdata[:, 0]
        newobj.rxx = subdata[:,1]
        newobj.rxy = subdata[:,2]
        for key, i in zip(self.dataseries.keys(), range(subdata.shape[1] - 3)):
            newobj.dataseries[key] = subdata[:,3+i]
        newobj._calculateTransport()
        return newobj
    
    @override
    def plot_all_data(self, axes=None, scatter=False, ** mpl_kwargs) -> graphwrappers.transport_graph:
        tg = super().plot_all_data(axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        # for i, key in zip(range(len(self.dataseries)), self.dataseries):
        #     tg.ax[2+i].set_ylabel(key)
        return tg

    @override
    def plot_dep_vars(self, axes=None, scatter=False, **mpl_kwargs) -> graphwrappers.transport_graph:
        tg = super().plot_dep_vars(axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        return tg

    @override
    def ind_vars(self):
        return self.field

    @override
    def dep_vars(self):
        return np.c_[self.rhoxx, self.rhoxy]
    
    @override
    def plot_all_dataseries(self, ax=None, scatter=False, **mpl_kwargs):
        tg = super().plot_all_dataseries(ax, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        return tg
    
    @override
    def plot_dataseries(self, key, ax=None, scatter=False, **mpl_kwargs):
        tg = super().plot_dataseries(key, ax, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        return tg
    
    @override
    def plot_dataseries_with_dep_vars(self, key, ax=None, scatter=False, **mpl_kwargs):
        tg = super().plot_dataseries_with_dep_vars(key, ax, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        return tg

    def MR_percentages(self):
        # Get zero field location
        minfield = np.min(np.abs(self.field))
        i = np.where(np.abs(self.field) == minfield)[
            0]  # get min magfield positions
        # average min field value positions if multiple, round to nearest.
        i = int(np.round(np.average(i)))
        # Prepare zero field substracted data.
        MR_rhoxx = self.rhoxx.copy()
        if self.rhoxx[i] == 0:
            raise Warning(
                "At the minimum field, Rho_xx is zero valued, giving infinite MR %. Use hallbar.plot_MR_absolute instead.")
        MR_rhoxx /= self.rhoxx[i]
        MR_rhoxy = self.rhoxy.copy()
        if self.rhoxy[i] == 0:
            raise Warning(
                "At the minimum field, Rho_xy is zero valued, giving infinite MR %. Use hallbar.plot_MR_absolute instead.")
        MR_rhoxy /= self.rhoxy[i]
        return MR_rhoxx, MR_rhoxy

    def plot_MR_percentages(self, axes=None, scatter=False, **mpl_kwargs):
        MR_rhoxx, MR_rhoxy = self.MR_percentages()
        data = np.c_[self.field, MR_rhoxx, MR_rhoxy]
        # Plots!
        tg = hallbar_measurement._graph_2Ddata(data, axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        tg.yMR_percentage(i=0, subscript="xx")
        tg.yMR_percentage(i=1, subscript="xy")
        return tg
    
    def MR_absolute(self):
        # Get zero field location
        minfield = np.min(np.abs(self.field))
        i = np.where(np.abs(self.field) == minfield)[
            0]  # get min magfield positions
        # average min field value positions if multiple, round to nearest.
        i = int(np.round(np.average(i)))
        # Prepare zero field substracted data.
        MR_rhoxx = self.rhoxx - self.rhoxx[i]
        MR_rhoxy = self.rhoxy - self.rhoxy[i]
        return MR_rhoxx, MR_rhoxy
    
    def plot_MR_absolute(self, axes=None, scatter=False, **mpl_kwargs):
        MR_rhoxx, MR_rhoxy = self.MR_absolute()
        data = np.c_[self.field, MR_rhoxx, MR_rhoxy]
        # Plots!
        tg = hallbar_measurement._graph_2Ddata(data, axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        tg.yMR_absolute(i=0, subscript="xx")
        tg.yMR_absolute(i=1, subscript="xy")
        return tg
    
    def plot_magnetoresistance_p(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_MR_percentages"""
        return self.plot_MR_percentages(axes, scatter, **mpl_kwargs)

    def plot_magnetoresistance_a(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_MR_absolute"""
        return self.plot_MR_absolute(axes, scatter, **mpl_kwargs)
    
    def plot_Shubnikov_deHass(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[1/self.field[::-81], self.rxy[::-1]]
        tg = self._graph_2Ddata(data, axes, scatter, **mpl_kwargs)
        tg.xFieldInverseT(i=-1)
        tg.yMR_absolute(i=-1, subscript="xy")
        return
     
# class hallbar_measurement_set():
#     """Class to categorise, organise and graph multiple hallbar measurements"""
#     # TODO impliment
    
#     def __init__(self, hb) -> None:
#         if isinstance(hb, hallbar_measurement):
#             self.hbs = [hb]
#         elif isinstance(hb, list) and np.all([isinstance(a, hallbar_measurement) for a in hb]):
#             self.hbs = hb
#         else:
#             raise AttributeError("hb is not an individual or list of hallbar_measurement objects.")
        
#         self.sort_method, self.sort_params = self._defaultSortMethod()
#         self.sort_accending()
        
#         return
    
#     def _defaultSortMethod():
#         default_sort_index = 0
#         method = list.sort
#         params = {'key':lambda hb: np.average(hb.dataseries[hb.dataseries.keys()[default_sort_index]]), 'reverse':False}
#         return method, params
    
#     def set_sort_handle(self, method, params):
#         """Sets the method to apply to a list, to return a sorted list.

#         Args:
#             method (function): Sorting function to apply to list. Requires 'reverse' keyword option.
#             params (kwargs): Dictionary of parameters neccessary for the function call.
#         """
#         args, varargs, kwargs = inspect.getargs(method)
#         if 'reverse' not in args:
#             raise AttributeError("Method does not contain 'reverse' argument.")
#         self.sort_method = method
#         self.sort_params = params
#         return 
    
#     def sort_accending():
        
#         return
    
#     def sort_decending():
        
#         return
    