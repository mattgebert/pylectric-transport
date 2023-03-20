# Function Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
# Custom Libraries
import pylectric
from pylectric.graphing import geo_base, graphwrappers
from pylectric.analysis import mobility
# Programming Syntax
import warnings
from overrides import override
from abc import abstractmethod

class hallbar_measurement(geo_base.graphable_base):
    """Class for a hall measurement, that is magnetic field dependence of Rxx and Rxy of a device.
        Does not assume gated geometry, however does assume magnetic measurement for the purpose of Rxy data.
        
        Can provide additional data to add to object, in the form of a dictionary of data {label:numpy_array}.
    """
    
    def __init__(self, field, rxx, rxy, dataseries = {}, geom = 1, **params):
        
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
        
        #Dataseries are other columns of data.
        self.dataseries = {}
        for key in dataseries:
            self.dataseries[key] = dataseries[key].copy()
        
        self.params = params
        
        #initialise super object
        super().__init__()
        
        return
    
    def _calculateTransport(self):
        # Convert rxx, rxy to rhoxx, rhoxy, sigmaxx, sigmaxy
        self.rhoxx = self.rxx.copy() * self.geom
        self.rhoxy = self.rxy.copy() * self.geom
        self.sigmaxx = self.rhoxx / (self.rhoxx**2 + self.rhoxy**2)
        self.sigmaxy = -self.rhoxy / (self.rhoxx**2 + self.rhoxy**2)
        
    
    def symmterise(self): # -> tuple[hallbar.hallbar_measurement, hallbar.hallbar_measurement]:
        #0. Check requirements [datapoints are evenly binned]:
        if not np.alltrue(self.field[:] == -self.field[::-1]):
            raise AttributeError("Field B is not symmetric in field about 0." +
                " Please use pylectric.signals.processing.reduction to sample data.")
        
        #1. Create two clones:
        sym_clone = self.copy() #symmetric
        asym_clone = self.copy() #assymmetric
        
        #Put data together into one array.
        all_data = np.c_[self.field, self.rxx, self.rxy]
        for key,value in self.dataseries.items():
            all_data = np.c_[all_data, value]
        
        #Symmmetrise all data:
        sym, asym = pylectric.signals.processing.symmetrise(all_data)
        
        #Reassign datasets:
        sym_clone.field = sym[:,0]
        sym_clone.rxx = sym[:,1]
        sym_clone.rxy = sym[:,2]
        
        asym_clone.field = asym[:, 0]
        asym_clone.rxx = asym[:, 1]
        asym_clone.rxy = asym[:, 2]
        
        i = 0
        for key in self.dataseries:
            sym_clone.dataseries[key] = sym[:,3+i]
            asym_clone.dataseries[key] = asym[:,3+i]
            i += 1
        
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
    
    def __add__(self,x):
        if not isinstance(x,hallbar_measurement):
            raise TypeError("'" + str(x) + "' is not a hallbar_measurement object.")
        else:
            # check identical parameter lists:
            if self.geom != x.geom:
                raise AttributeError("Geometry of " + str(x) + " doesn't match " + str(self))
            xkeys = x.dataseries.keys()
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
    def plot_all_data(self, axes = None,label=None) -> graphwrappers.transport_graph:
        tg = super().plot_all_data(axes, label)
        tg.xFieldT(i=-1)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        keys = list(self.dataseries) #indexing - not efficient for many data series, but okay for a small ammount (<10,000)
        for i in range(2, 2 + len(self.dataseries)):
            tg.ax[i].set_ylabel(keys[i-2])
        return tg

    @override
    def plot_dep_vars(self, axes=None, label=None) -> graphwrappers.transport_graph:
        tg = super().plot_dep_vars(axes, label)
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
    def extra_vars(self):
        return np.c_[*[self.dataseries[key] for key in self.dataseries]]

    def plot_MR_percentages(self, axes = None, label=None):
        #Get zero field location
        minfield = np.min(np.abs(self.field))
        i = np.where(np.abs(self.field) == minfield)[0]  # get min magfield positions
        i = int(np.round(np.average(i))) #average min field value positions if multiple, round to nearest.
        #Prepare zero field substracted data.
        MR_rxx = self.rhoxx - self.rhoxx[i]
        MR_rxx /= self.rhoxx[i]
        MR_rxy = self.rhoxy - self.rhoxy[i]
        MR_rxy /= self.rhoxy[i]
        data = np.c_[self.field, MR_rxx, MR_rxy]
        # Plots!
        tg = hallbar_measurement._plot_2Ddata(data=data, axes=axes, label=label)
        tg.xFieldT(i=-1)
        tg.yMR_percentage(i=0, subscript="xx")
        tg.yMR_percentage(i=1, subscript="xy")
        return tg
    
    def plot_MR_absolute(self, axes = None, label=None):
        # Get zero field location
        minfield = np.min(np.abs(self.field))
        i = np.where(np.abs(self.field) == minfield)[0]  # get min magfield positions
        # average min field value positions if multiple, round to nearest.
        i = int(np.round(np.average(i)))
        # Prepare zero field substracted data.
        MR_rxx = self.rhoxx - self.rhoxx[i]
        MR_rxy = self.rhoxy - self.rhoxy[i]
        data = np.c_[self.field, MR_rxx, MR_rxy]
        # Plots!
        tg = hallbar_measurement._plot_2Ddata(data, axes=axes, label=label)
        tg.xFieldT(i=-1)
        tg.yMR_absolute(i=0, subscript="xx")
        tg.yMR_absolute(i=1, subscript="xy")
        return tg
    

    def plot_magnetoresistance_p(self, axes=None, label=None):
        """Alias for plot_MR_percentages"""
        return self.plot_MR_percentages(axes, label)
    def plot_magnetoresistance_a(self, axes=None, label=None):
        """Alias for plot_MR_absolute"""
        return self.plot_MR_absolute(axes, label)
    
    def plot_Shubnikov_deHass(self,axes = None, label=None):
        data = np.c_[1/self.field[::-1], self.rxy[::-1]]
        tg = self._plot_2Ddata(data, axes=axes, label=label)
        tg.xFieldInverseT(i=-1)
        tg.yMR_absolute(i=-1, subscript="xy")
        return