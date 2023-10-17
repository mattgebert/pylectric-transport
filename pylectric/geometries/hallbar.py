# Programming Syntax & annotations #1
from __future__ import annotations
# Function Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import optimize as opt
# Custom Libraries
import pylectric
from pylectric.graphing import graphwrappers
from pylectric.analysis import mobility, hall, localisation
from pylectric.signals import processing
from pylectric.geometries import geo_base
# Programming Syntax & annotations #2
import warnings
from overrides import override
from abc import abstractmethod
# import inspect
from decimal import Decimal
from matplotlib.offsetbox import AnchoredText
import pandas as pd

class hallbar_measurement(geo_base.graphable_base_dataseries):
    """Class for a hall measurement, that is magnetic field dependence of Rxx and Rxy of a device.
        Does not assume gated geometry, however does assume magnetic measurement for the purpose of Rxy data.
        Geom parameter converts resistance to resistivity. Ie. Rho = R * Geom
        
        Can provide additional data to add to object via 'params', in the form of a dictionary of data {label:numpy_array}.
    """
    # attribute for keeping track of colours in plotting over 
    # multiple hallbar_measurement objects.
    clr_i = 0 
    COLS = ['#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf']
    
    
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
        self._geom = geom
        
        # Convert rxx, rxy to rhoxx, rhoxy, sigmaxx, sigmaxy
        self._calculateTransport()
        # Copy items in params to self.
        self.params = {}
        self.params.update(params)
        return
    
    @classmethod
    def reset_clr_counter(cls):
        hallbar_measurement.clr_i = 0
        cls.clr_i = 0
    
    def _calculateTransport(self):
        # Convert rxx, rxy to rhoxx, rhoxy, sigmaxx, sigmaxy
        self.rhoxx = self.rxx.copy() * self._geom
        self.rhoxy = self.rxy.copy() * self._geom
        self.sigmaxx = self.rhoxx / (self.rhoxx**2 + self.rhoxy**2)
        self.sigmaxy = -self.rhoxy / (self.rhoxx**2 + self.rhoxy**2)
        
    
    @property
    def geom(self):
        return self._geom
    
    @geom.setter
    def geom(self, geom):
        self._geom = geom
        self._calculateTransport() #update transport values.
    
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
        sym, asym = processing.symmetrise(all_data, colN=0, full_domain=full_domain) #field = 0
        
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
        newobj = hallbar_measurement(field=self.field, rxx=self.rxx, rxy=self.rxy, 
                    geom=self._geom)
        newobj.dataseries = self.dataseries.copy()
        for key in newobj.dataseries:
            newobj.dataseries[key] = newobj.dataseries[key].copy()
        newobj.rhoxx = self.rhoxx.copy()
        newobj.rhoxy = self.rhoxy.copy()
        newobj.sigmaxx = self.sigmaxx.copy()
        newobj.sigmaxy = self.sigmaxy.copy()
        # newobj.geom = self._geom --> Could do this,
        # then would need to calculate transport again. Instead add to constructor...
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
            if self._geom != x.geom:
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
    def plot_all_data(self, axes=None, scatter=False, **mpl_kwargs) -> graphwrappers.transport_graph:
        tg = super().plot_all_data(axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        # TODO: Comment this loop, as it should occur in subclass, but isn't working....
        for i, key in zip(range(len(self.dataseries)), self.dataseries): 
            tg.ax[2+i].set_ylabel(key)
        return tg

    @override
    def plot_dep_vars(self, axes=None, scatter=False, **mpl_kwargs) -> graphwrappers.transport_graph:
        tg = super().plot_dep_vars(axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        # tg.yResistivity(i=0, subscript="xx")
        # tg.yResistivity(i=1, subscript="xy")
        tg.yResistance(i=0, subscript="xx")
        tg.yResistance(i=1, subscript="xy")
        return tg

    @override
    def ind_vars(self):
        return self.field

    @override
    def dep_vars(self):
        return np.c_[self.rxx, self.rxy] #should match constructor, 
    
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
        # tg.yResistivity(i=0, subscript="xx")
        # tg.yResistivity(i=1, subscript="xy")
        tg.yResistance(i=0, subscript="xx")
        tg.yResistance(i=1, subscript="xy")
        return tg
    
    @override
    def to_DataFrame(self):
        return pd.DataFrame(self.all_vars(), columns=["Field","$\rho_{xx}$","$\rho_{xy}$"] + list(self.dataseries.keys()))

    @staticmethod
    def _1D_plots(data, axes=None, scatter=False, **mpl_kwargs):
        tg = hallbar_measurement._graph_2Ddata(
            data, axes, scatter, **mpl_kwargs)
        tg.xFieldT(i=-1)
        return tg

    def plot_resistance(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.rhoxx, self.rhoxy]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        return tg

    def plot_resistance_xx(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.rxx]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yResistance(i=0, subscript="xx")
        return tg
    
    def plot_resistance_xy(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.rxy]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yResistance(i=0, subscript="xy")
        return tg
    
    def plot_resistivity(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.rhoxx, self.rhoxy]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yResistivity(i=0, subscript="xx")
        tg.yResistivity(i=1, subscript="xy")
        return tg
    
    def plot_resistivity_xx(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.rhoxx]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yResistivity(i=0, subscript="xx")
        return tg

    def plot_resistivity_xy(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.rhoxy]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yResistivity(i=0, subscript="xy")
        return tg
    
    def plot_conductivity(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.sigmaxx, self.sigmaxy]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yConductivity(i=0, subscript="xx")
        tg.yConductivity(i=1, subscript="xy")
        return tg
    
    def plot_conductivity_xx(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.sigmaxx]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yConductivity(i=0, subscript="xx")
        return tg

    def plot_conductivity_xy(self, axes=None, scatter=False, **mpl_kwargs):
        data = np.c_[self.field, self.sigmaxy]
        tg = hallbar_measurement._1D_plots(data, axes, scatter, **mpl_kwargs)
        tg.yConductivity(i=0, subscript="xy")
        return tg

    def plot_R(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_resistance"""
        return self.plot_resistance(axes=axes, scatter=scatter, **mpl_kwargs)
    
    def plot_Rxx(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_resistance_xx"""
        return self.plot_resistance_xx(axes=axes, scatter=scatter, **mpl_kwargs)
    
    def plot_Rxy(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_resistance_xy"""
        return self.plot_resistance_xy(axes=axes, scatter=scatter, **mpl_kwargs)
    
    def plot_rho(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_resistivity"""
        return self.plot_resistivity(axes=axes, scatter=scatter, **mpl_kwargs)
    
    def plot_rhoxx(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_resistivity_xx"""
        return self.plot_resistivity_xx(axes=axes, scatter=scatter, **mpl_kwargs)

    def plot_rhoxy(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_resistivity_xy"""
        return self.plot_resistivity_xy(axes=axes, scatter=scatter, **mpl_kwargs)

    def plot_sigma(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_conductivity"""
        return self.plot_conductivity(axes=axes, scatter=scatter, **mpl_kwargs)

    def plot_sigmaxx(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_conductivity_xx"""
        return self.plot_conductivity_xx(axes=axes, scatter=scatter, **mpl_kwargs)

    def plot_sigmaxy(self, axes=None, scatter=False, **mpl_kwargs):
        """Alias for plot_conductivity_xy"""
        return self.plot_conductivity_xy(axes=axes, scatter=scatter, **mpl_kwargs)

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
     
    @staticmethod
    def _hall_measurement_fit(field, rxy, rxx, thickness = None, **kwargs):
        """Performs the functional fit between Field, Rxy and Rxx to determine the carrier density, 

        Parameters
        ----------
        field : numpy.ndarray
            Dataset field values
        rxy : numpy.ndarray
            Dataset Rxy values (not resistivity)
        rxx : numpy.ndarray or float
            Dataset Rxx values, or singular Rxx at B=0.
        thickness : float, optional
            Performs a 2D Hall measurement if None, otherwise
            calculates a 3D Hall measurement. None by default.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Returns a tuple of the parameters and covariance matrix.
            Parameters are:
            1 - Hall density
            2 - Hall mobility
            3 - Rxy @ B = 0
        """
        # Check if rxx has zero field value.
        if not (np.any(field > 0) and np.any(field<0)):
            warnings.warn("Field domain doesn't cover B=0. Hall mobility may be inaccurate.")
        
        # Fit
        if thickness is None:
            params, unc = hall.hall2D.hall_measurement_fit(
                field=field, rxy=rxy, rxx=rxx, **kwargs)
        else:
            params, unc = hall.hall.hall_measurement_fit(
                field=field, rxy=rxy, rxx=rxx, thickness=thickness, **kwargs)
        
        return params, unc
        
    def _hall_measurement_plot(self, params, unc, thickness=None, fitdomain=None,  
                               axes=None):
        
        #Setup kwargs for Hall function
        fit_kwargs = {
            "hall_density": params[0],
            "rxy0": params[2]
        }
        
        #If thickness provided, adjust kwarg and set function.
        if thickness is not None: # Add thickness if performing a 3D measurement.
            fit_kwargs["thickness"] = thickness
            hallfn = hall.hall.hall_resistance
        else:
            hallfn = hall.hall2D.hall_resistance
        
        # If fitdomain not passed, default to regular field.
        if fitdomain is None:
            fitdomain = [np.min(self.field), np.max(self.field)]
        
        #Setup new figure if not passed.
        ax = axes if axes is not None else plt.subplots(1,1)[1]
            
        # Plot Fit Line over display domain.
        endpoints_rxy = hallfn(fitdomain, **fit_kwargs)
        ax.plot(fitdomain, endpoints_rxy, "--", label="Fit")

        # Plot Fit Domain Endpoints
        fit_rxy = hallfn(fitdomain, **fit_kwargs)
        ax.scatter(fitdomain, fit_rxy, marker='|', s=1.5, label="Fit Domain")

        # Plot Raw Data
        ax.scatter(self.field, self.rxy, label="Data", s=0.1)

        # Labels
        tg = graphwrappers.transport_graph(ax)
        tg.xFieldT(0)
        tg.yResistance(0, subscript="xy")
        ax.set_title("Hall Density Fit")

        # Display Fit Values on Graph:
        strvals = ["{:1.2E} $\pm$ {:1.2E}".format(
            Decimal(params[i]), Decimal(unc[i])) for i in range(3)]

        dim = 2 if thickness is None else 3
        strdim = str(dim)
        ann_str = r"$n_{\mathrm{Hall}}$: "+ strvals[0]+ r" m$^{-" + strdim + r"}$" +\
            "\n" r"$\mu_{\mathrm{Hall}}$: "+ strvals[1]+ r" m$^{" + strdim + r"}$/Vs"+\
            "\n" r"$R_{xy,B=0}$: " + strvals[2] + r" $\Omega$"

        anchor = AnchoredText(ann_str, loc=3 if params[0] < 0 else 4, prop={"fontsize":4})
        ax.add_artist(anchor)

        # Legend
        ax.legend()
        return tg
        
    def hall_measurement_fit(self, thickness=None, field_range = (None, None),
                             axes=None, display=False, fit_kwargs={}):
        """ Uses object Field, Rxx and Rxy to fit the following hall parameters:
        - Hall Density N_hall (m^{-3} or m^{-2})
        - Hall Mobility Mu_hall (m^{3}/Vs or m^{2}/Vs)
        - Zero-field Rxy (Ohms)
        To calculate Mu_hall, the Rxx value closest to Field=0 is used.
        If thickness is None, assumed measurement to be 2D instead of 3D.

        Parameters
        ----------
        thickness : float, optional
            The thickness of the sample. 
            If None, assumed to be a 2D hall measurement instead of 3D.
        field_range : tuple, optional
            , by default (None, None)
        axes : matplotlib.axes, optional
            Optional parameter to pass graphing axis.
        display : bool, optional
            Option to generate a fit graph or not, by default False. 
            Ignored if an axes object is passed.

        Returns
        -------
        _type_
            _description_
        """
        
        # Identify subset of data within field range to fit.
        lowerb, upperb = field_range
        if lowerb is not None or upperb is not None:
            # Both bounds
            if lowerb is not None and upperb is not None: 
                inds = np.where((upperb > self.field) & (self.field > lowerb))
            # Only lower bound
            elif lowerb is not None: 
                inds = np.where(self.field > lowerb)
            # Only upper bound
            elif upperb is not None: 
                inds = np.where(upperb >= self.field)
            # Acquire subset
            subset_f = self.field[inds]
            subset_rxy = self.rxy[inds]
            subset_rxx = self.rxx[inds]
        else: #unbounded!
            subset_f = self.field
            subset_rxy = self.rxy
            subset_rxx = self.rxx
        
        #Adjust endpoint graphing markers for future.
        field_endpoints = np.array([np.min(self.field), np.max(self.field)]) #domain to plot 
        fit_field = field_endpoints.copy()  # graphing markers for fit.
        if lowerb is not None and lowerb > fit_field[0]:
            fit_field[0] = lowerb
        if upperb is not None and upperb < fit_field[1]:
            fit_field[1] = upperb
        
        # Perform fit
        params, unc = self._hall_measurement_fit(
            field=subset_f, rxy=subset_rxy, rxx=subset_rxx, thickness=thickness, **fit_kwargs)
        
        # Graph if desired.
        if display or axes is not None: #either condition allows for plots to be generated.
            # Generate graph
            tg = self._hall_measurement_plot(params=params, unc=unc, thickness=thickness, axes=axes)
            
        # Don't return transport graph object, user can pass axis to control graphing.
        return params, unc
    
    def _HLN_plot(self, params, unc, fitdomain=None, axes=None):
        #Setup kwargs for Hall function
        fit_kwargs = {
            "alpha": params[0],
            "lphi": params[1]
        }
        if len(params)==3:
            fit_kwargs["offset"] = params[2]
            WALfn = localisation.WAL.HLN_offset
        elif len(params) == 2:
            WALfn = localisation.WAL.HLN
        else:
            raise AttributeError("Not enough parameters supplied to plot.")
        
        #Setup new figure if not passed.
        ax = axes if axes is not None else plt.subplots(1,1)[1]
        
         # Get raw data inside subset
        if fitdomain is not None:
            ssinds = np.where(np.bitwise_and(
                self.field > fitdomain[0], self.field < fitdomain[1]))  # subset field
            ssfield = self.field[ssinds]
        else:
            ssfield = self.field

        # If fitdomain not passed, default to regular field.
        if fitdomain is None:
            fitdomain = [np.min(self.field), np.max(self.field)]

        # Find minimum field value
        minind = np.where(np.abs(ssfield) == np.min(np.abs(ssfield)))
        sigmaxx_b0 = np.average(self.sigmaxx[minind]) #Before new fn.
        # remove zerofield offset within subset data.
        dsigmaxx = self.sigmaxx - sigmaxx_b0
            
        # Plot raw data according to subset offset. Only plot data less than 3x fitdomain[1].
        if np.max(np.abs(self.field)) < 3 * np.max(np.abs(fitdomain)):
            #no need to select subset of scattered data.
            scatfield = self.field
            scatdsig = dsigmaxx
        else:
            #display subset of scattered data.
            scatinds = np.where(np.abs(self.field) < 3 * np.max(np.abs(fitdomain)))
            scatfield = self.field[scatinds]
            scatdsig = dsigmaxx[scatinds]
            
        #Use class var for plotting
        hm = hallbar_measurement 
        ax.scatter(scatfield, scatdsig, label="Data", s=0.3, 
                    c=hm.COLS[hm.clr_i % len(hm.COLS)])
        hm.clr_i += 1
        
        # Plot Fit line outside display domain with dashes.
        dashinds = np.where(np.abs(scatfield) > np.max(np.abs(fitdomain)))
        if len(dashinds[0] > 0):
            dashdomain = [np.min(scatfield[dashinds]), np.max(scatfield[dashinds])]
            dashx = np.linspace(dashdomain[0],dashdomain[1],400)
            dashsigmaxx = WALfn(dashx, **fit_kwargs)
            ax.plot(dashx, dashsigmaxx, "--",
                    c=hm.COLS[hm.clr_i % len(hm.COLS)])
        
        # Plot Fit Line over display domain. Needs more points than just a line haha.
        x = np.linspace(fitdomain[0],fitdomain[1],400)
        fitsigmaxx = WALfn(x, **fit_kwargs)
        ax.plot(x, fitsigmaxx, label="Fit", 
                c=hm.COLS[hm.clr_i % len(hm.COLS)])

        # Plot Fit Domain Endpoints
        fit_sigmaxx = WALfn(fitdomain, **fit_kwargs)
        ax.scatter(fitdomain, fit_sigmaxx, marker='|', s=100, 
                    c=hm.COLS[hm.clr_i % len(hm.COLS)])
    
        hm.clr_i += 1
        # ax.scatter(fitdomain, fit_sigmaxx, marker='|', s=100, label="Fit Domain")

        # Labels
        tg = graphwrappers.transport_graph(ax)
        tg.xFieldT(0)
        tg.yConductivity(0, subscript="xx")
        ax.set_title("WAL Fit")

        # Display Fit Values on Graph:
        strvals = ["{:1.2E} $\pm$ {:1.2E}".format(
            Decimal(params[i]), Decimal(unc[i])) for i in range(len(params))]

        # dim = 2 if self.thickness is None else 3
        # strdim = str(dim)
        ann_str = r"$\alpha$: "+ strvals[0]+\
            "\n" + r"$\ell_{\phi}$: "+ strvals[1]+ r" m"+\
            ("" if len(params)==2 else "\n" +\
                r"$\sigma_{xx}^{offset}$ = " + strvals[2] + " S")

        anchor = AnchoredText(ann_str, loc=2 if params[0] < 0 else 3, prop={"fontsize":6})
        ax.add_artist(anchor)

        # Legend
        ax.legend()
        
        return
    
    def HLN_fit(self, field_range=(None, None), axes=None, display=False, 
                p0 = [], bounds=([-1, 0, -np.infty], [1, np.infty, np.infty])):
        
        param_names = ["alpha", "lphi", "offset"]
        assert len(p0) <= len(param_names)

        # Setup intial parameters for fitting.
        fit_kwargs = {}
        for key, val in zip(param_names, p0):
            fit_kwargs[key] = val
        
        # Identify subset of data within field range to fit.
        lowerb, upperb = field_range
        if lowerb is not None or upperb is not None:
            # Both bounds
            if lowerb is not None and upperb is not None:
                inds = np.where(np.bitwise_and(upperb >= self.field, self.field > lowerb))
            # Only lower bound
            elif lowerb is not None:
                inds = np.where(self.field > lowerb)
            # Only upper bound
            elif upperb is not None:
                inds = np.where(upperb >= self.field)
            # Acquire subset
            subset_f = self.field[inds]
            subset_sigmaxx = self.sigmaxx[inds]
        else:  # unbounded!
            subset_f = self.field
            subset_sigmaxx = self.sigmaxx
        
        # Adjust endpoint graphing markers for future.
        field_endpoints = np.array(
            [np.min(self.field), np.max(self.field)])  # domain to plot
        fit_field = field_endpoints.copy()  # graphing markers for fit.
        if lowerb is not None and lowerb > fit_field[0]:
            fit_field[0] = np.min(subset_f)
        if upperb is not None and upperb < fit_field[1]:
            fit_field[1] = np.max(subset_f)

        # Perform fit
        # params, unc = self._HLN_fit(
        #     field=subset_f, sigmaxx=subset_sigmaxx, **fit_kwargs)
        params, unc = localisation.WAL.fitting.HLN_fit(
            B=subset_f, sigmaxx=subset_sigmaxx, bounds=bounds, **fit_kwargs)


        # Graph if desired.
        # either condition allows for plots to be generated.
        if display or axes is not None:
            # Generate graph
            tg = self._HLN_plot(fitdomain=fit_field,
                params=params, unc=unc, axes=axes)

        # Don't return transport graph object, user can pass axis to control graphing.
        return params, unc
    
    def HLN_fit_iterative(self, axes=None, display=False, 
                          b_window=10, 
                          p0=[0.5,1e-7,0], bounds=([-1, 0, -np.infty], [1, np.infty, np.infty])):
        
        param_names = ["alpha", "lphi", "offset"]
        assert len(p0) <= len(param_names)

        # Setup intial parameters for fitting.
        fit_kwargs = {}
        for key, val in zip(param_names, p0):
            fit_kwargs[key] = val
        fit_kwargs["b_window"] = b_window
        fit_kwargs["bounds"] = bounds
        
        # Perform fit
        params, unc = localisation.WAL.fitting.HLN_fit_iterative(
            B=self.field, sigmaxx=self.sigmaxx, **fit_kwargs)

        # Graph if desired.
        # either condition allows for plots to be generated.
        if display or axes is not None:
            # Recalculate the final fitting field.
            Bphi = localisation.WAL.HLN_li_to_Hi(params[1])
            Binds = np.where(np.abs(self.field) < fit_kwargs["b_window"] * Bphi)
            fit_field = [np.min(self.field[Binds]), np.max(self.field[Binds])]
            # Generate graph
            tg = self._HLN_plot(fitdomain=fit_field,
                params=params, unc=unc, axes=axes)

        # Don't return transport graph object, user can pass axis to control graphing.
        return params, unc
    
    def HLN_fit_const_alpha_iterative(self, alpha, axes=None, display=False, b_window=10,
                                      p0=[], bounds=([0, -np.infty], [np.infty, np.infty])):

        param_names = ["lphi", "offset"]
        assert len(p0) <= len(param_names)
        
        # Setup intial parameters for fitting.
        fit_kwargs = {}
        for key, val in zip(param_names, p0):
            fit_kwargs[key] = val
        fit_kwargs["b_window"] = b_window
        
        # Perform fit
        params, unc = localisation.WAL.fitting.HLN_fit_const_alpha_iterative(
            B=self.field, sigmaxx=self.sigmaxx, alpha=alpha, bounds=bounds, **fit_kwargs)

        # Graph if desired.
        # either condition allows for plots to be generated.
        if display or axes is not None:
            # Recalculate the final fitting field.
            Bphi = localisation.WAL.HLN_li_to_Hi(params[1])
            Binds = np.where(np.abs(self.field) <
                             fit_kwargs["b_window"] * Bphi)
            fit_field = [np.min(self.field[Binds]), np.max(self.field[Binds])]
            # Insert alpha into plot params, to properly plot.
            params_plot = [alpha] + list(params)
            unc_plot = [0] + list(unc)
            # Generate graph
            tg = self._HLN_plot(fitdomain=fit_field,
                                       params=params_plot, unc=unc_plot, axes=axes)

        # Don't return transport graph object, user can pass axis to control graphing.
        return params, unc
        
    def HLN_LMR_fit_iterative(self, axes=None, display=False,
                        b_window=10, p0=[0.5, 1e-7], 
                        bounds=([-2, 0], [2, np.infty])):

        param_names = ["alpha", "lphi"]
        assert len(p0) <= len(param_names)

        # Setup intial parameters for fitting.
        fit_kwargs = {}
        for key, val in zip(param_names, p0):
            fit_kwargs[key] = val
        fit_kwargs["b_window"] = b_window
        fit_kwargs["bounds"] = bounds

        # Perform fit
        params, unc = localisation.WAL.fitting.HLN_LMR_fit_iterative(
            B=self.field, sigmaxx=self.sigmaxx, **fit_kwargs)

        # Graph if desired.
        # either condition allows for plots to be generated.
        if display or axes is not None:
            # Recalculate the final fitting field.
            Bphi = localisation.WAL.HLN_li_to_Hi(params[1])
            Binds = np.where(np.abs(self.field) <
                             fit_kwargs["b_window"] * Bphi)
            fit_field = [np.min(self.field[Binds]), np.max(self.field[Binds])]
            # Generate graph
            tg = self._HLN_LMR_plot(params=params, unc=unc, 
                                    axes=axes, fitdomain=fit_field)

        # Don't return transport graph object, user can pass axis to control graphing.
        return params, unc

    def _HLN_LMR_plot(self,params, unc,
                   fitdomain=None, axes=None):
        
        # Setup kwargs for WAL function
        WAL = localisation.WAL #library..
        WAL_LMRfn = WAL.HLN_LMR
        WALfn = WAL.HLN
        LMRfn = WAL.sigma_LMR
        alpha, lphi, grad, R0 = params
        
        # Plot two graphs; conductance (abs) and the WAL fitting.
        if axes:
            assert len(axes) >= 2 #needs two graphs.
            ax = axes
            fig = ax[0].get_figure()
        else:
            fig, ax = plt.subplots(2,1, figsize=(12,10))
        
        ### FIRST PART - LMR Reduction
        # Plot raw datapoints on each
        ax[0].scatter(self.field, self.sigmaxx, label="Data", s=0.1)
        
        # Plot fits on each
        x = np.linspace(np.min(self.field), np.max(self.field), 1000)
        ax[0].plot(x, WAL_LMRfn(x, *params), 
                   label="Fit WAL + LMR")
        diff_HLN_LMR = WALfn(x[-1], *params[0:2]) # WAL component at high field (-ve)
        ax[0].plot(x, LMRfn(x, *params[2:]) + diff_HLN_LMR, # add it
                   "--", label="Fit LMR")  # Plot LMR fit
                
        ### SECOND PART - WAL FITTING 
        # calculate data dsigmaxx by removing LMR of fit.
        dsigmaxx = self.sigmaxx - LMRfn(self.field, *params[2:])
        
        # Get raw data inside fit subset
        if fitdomain is None:
            #default to regular field
            fitdomain = [np.min(self.field), np.max(self.field)] 

        # Plot raw data according to subset offset. Only plot data less than 2x fitdomain[1].
        if np.max(np.abs(self.field)) < 2 * np.max(np.abs(fitdomain)):
            # no need to select subset of scattered data.
            scatfield = self.field
            scatsig = dsigmaxx
        else:
            # display subset of scattered data.
            scatinds = np.where(np.abs(self.field) < 2 *
                                np.max(np.abs(fitdomain)))
            scatfield = self.field[scatinds]
            scatsig = dsigmaxx[scatinds]
        hm = hallbar_measurement # Use class var for plotting
        ax[1].scatter(scatfield, scatsig, label="Data (- LMR)", s=0.3,
                   c=hm.COLS[hm.clr_i % len(hm.COLS)])
        hm.clr_i += 1

        # Plot Fit line outside display domain with dashes.
        dashinds = np.where(np.abs(scatfield) > np.max(np.abs(fitdomain)))
        if len(dashinds[0] > 0):
            dashdomain = [np.min(scatfield[dashinds]),
                          np.max(scatfield[dashinds])]
            # dashx = np.linspace(dashdomain[0], dashdomain[1], 400)
            dashx = [dashdomain[0], dashdomain[1]]
            dashsigmaxx = WALfn(dashx, *params[:2])
            ax[1].plot(dashx, dashsigmaxx, "--",
                    c=hm.COLS[hm.clr_i % len(hm.COLS)])

        # Plot Fit Line over display domain. Needs more points than just a line haha.
        x = np.linspace(fitdomain[0], fitdomain[1], 400)
        fitsigmaxx = WALfn(x, *params[:2])
        ax[1].plot(x, fitsigmaxx, label="Fit (WAL)",
                c=hm.COLS[hm.clr_i % len(hm.COLS)])

        # Plot Fit Domain Endpoints
        fit_sigmaxx = WALfn(fitdomain, *params[:2])
        ax[1].scatter(fitdomain, fit_sigmaxx, marker='|', s=100,
                   c=hm.COLS[hm.clr_i % len(hm.COLS)])

        hm.clr_i += 1
        ax[0].set_xlabel("Field (T)")
        ax[1].set_xlabel("Field (T)")
        ax[0].set_ylabel(r"$\sigma_{xx}$ (S)")
        ax[1].set_ylabel(r"$\sigma_{xx}$ (S)")
        ax[1].set_title("WAL Fit")

        # Display Fit Values on Graph:
        strvals = ["{:1.2E} $\pm$ {:1.2E}".format(
            Decimal(params[i]), Decimal(unc[i])) for i in range(len(params))]

        # dim = 2 if self.thickness is None else 3
        # strdim = str(dim)
        ann_str = r"$\alpha$: " + strvals[0] +\
            "\n" + r"$\ell_{\phi}$: " + strvals[1] + r" m" +\
            "\n" + r"Grad = " + strvals[2] + r" $\Omega T^{-1}$" +\
            "\n" + r"$R0$ = " + strvals[3] + r" $\Omega$"

        anchor = AnchoredText(
            ann_str, loc=2 if params[0] < 0 else 3, prop={"fontsize": 6})
        ax[1].add_artist(anchor)

        # Legend
        ax[1].legend()
        
        return fig, ax


class hallbar_measurement_symmetrised(hallbar_measurement):
    """ Class to contain two sets of data, symmeteric and asymettric components.
        Uses same functions as hallbar_measurement class, but applies to appropriate 
        components of the symmetrised dataset instead.
    """
    
    def __init__(self, sym, asym) -> None:
        #Assertions
        assert(isinstance(sym, hallbar_measurement))
        assert(isinstance(asym, hallbar_measurement))
        assert(np.all(sym.all_vars().shape == asym.all_vars().shape))
        assert(np.all(sym.field == asym.field))
        
        #Copy 
        self.sym = sym.copy()
        self.asym = asym.copy()
        
        #Select class properties.
        self.field = self.sym.field # field values (independent variable) should be the exact same.
        self.rxx = self.sym.rxx # for a hall measurement, symmetric component is important
        self.rhoxx = self.sym.rhoxx 
        self.rxy = self.asym.rxy #for a hall measurement, asymmetric component is important.
        self.rhoxy = self.asym.rhoxy
        self.sigmaxx = self.sym.sigmaxx
        self.sigmaxy = self.asym.sigmaxy
        

    @override
    def copy(self):
        return hallbar_measurement_symmetrised(self.sym, self.asym)

    
    @classmethod
    @override
    def reset_clr_counter(cls):
        super(hallbar_measurement_symmetrised, cls).reset_clr_counter()
        cls.clr_i = 0


    
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
    