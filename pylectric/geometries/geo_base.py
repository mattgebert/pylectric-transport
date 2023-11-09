# Coding
from overrides import override
from abc import abstractmethod, ABCMeta
from typing import Type, Sequence, Self
#Methods
from pylectric.graphing import graphwrappers, journals
import pylectric
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import axes as mplaxes
import numpy as np
from enum import Enum
import pandas as pd

class transport_base(metaclass=ABCMeta):
    """ An abstract class to expand and bind graphing functions to geometric objects.
        All data is timeseries data, represented by rows for various variable columns.
        Data also should implement a sweep direction to plot.
        Graphical objects exist for scatter, plot, 
    """
    
    class sweep_enum(Enum):
        """Tracks sweeping direction for the class

        Parameters
        ----------
        Enum : enum.Enum
            Enumerate to track sweep direction in a dataset.
        """
        
        UNDEFINED = 0
        POSITIVE = 1 # +ve change in ind. var.
        NEGATIVE = 2 # -ve change in ind. var.
    
    def __init__(self) -> None:
        """Initalizes base transport class. 
        Requires implementation of sweep direction with independent variable(s).
        """
        # if not hasattr(self, "data"):
            # raise AttributeError("No data passed to constructor.")
        super().__init__()
        
        # Sweep Direction
        self._sweep_dir = None #initalizes None as might not match dimensions of data. 
        
        # Data Mask - Used to perform all operations on a subset of the data.
        self.mask = None
        
        # Data
        ## Independent Variables
        self._x = None
        self._xerrs = None
        self._xlabels = None
        
        ## Dependent Variables
        self._y = None
        self._yerrs = None
        self._ylabels = None
        
        ## Extra Variables
        self._z = None
        self._zerrs = None
        self._zlabels = None
        return

    def __copy__(self):
        """Creates a deep clone of base datasets.

        Returns
        -------
        transport_base
            _description_
        """
        clone = object.__new__(type(self))
        clone.sweep_dir = self.sweep_dir
        clone.mask = self.mask.copy() if self.mask is not None else None
        # X
        clone._x = self._x.copy() if self._x is not None else None
        clone._xerrs = self._xerrs.copy() if self._xerrs is not None else None
        clone._xlabels = self._xlabels.copy() if self._xlabels is not None else None
        # Y
        clone._y = self._y.copy() if self._y is not None else None
        clone._yerrs = self._yerrs.copy() if self._yerrs is not None else None
        clone._ylabels = self._ylabels.copy() if self._ylabels is not None else None
        # Z
        clone._z = self._z.copy() if self._y is not None else None
        clone._zerrs = self._zerrs.copy() if self._zerrs is not None else None
        clone._zlabels = self._zlabels.copy() if self._zlabels is not None else None
        return clone

    @property
    def sweep_dir(self) -> Self.sweep_enum | list[Self.sweep_enum]:
        return self._sweep_dir
        
    @sweep_dir.setter
    def sweep_dir(self, dir: Self.sweep_enum | bool | None | list[Self.sweep_enum] | list[bool | None]) -> None:
        if hasattr(dir, "__len__"): # Multiple sweep directions provided.
            x = self.x[0]
            if len(x.shape) == 2 and x.shape[1]==len(dir): # Check that provided sweep directions match number of columns of x .
                dirs = []
                for sweep in dir:
                    if sweep is transport_base.sweep_enum:
                        pass    
                    elif isinstance(sweep, bool):
                        sweep = transport_base.sweep_enum.POSITIVE if dir else transport_base.sweep_enum.NEGATIVE
                    elif sweep is None:
                        sweep = transport_base.sweep_enum.UNDEFINED
                    else:
                        raise AttributeError("Dir element is not sweep_enum, bool or None.")
                    dirs.append(sweep)
                self._sweep_dir = dirs
            elif len(x.shape) == 1 and len(dir) == 1: # Only one element in list and only one variable.
                dir = dir[0]
                if dir is transport_base.sweep_enum:
                    pass
                elif isinstance(dir, bool):
                    dir = transport_base.sweep_enum.POSITIVE if dir else transport_base.sweep_enum.NEGATIVE
                elif dir is None:
                    dir = transport_base.sweep_enum.UNDEFINED
                else:
                    raise AttributeError("Dir element is not sweep_enum, bool or None.")
                self._sweep_dir = dir
                return
            else:
                raise AttributeError("Number of sweep directions must match number of x columns.")
        elif len(self.x[0].shape) == 1: # Check that x only has one variable.
            if dir is transport_base.sweep_enum:
                pass
            elif isinstance(dir, bool):
                dir = transport_base.sweep_enum.POSITIVE if dir else transport_base.sweep_enum.NEGATIVE
            elif dir is None:
                dir = transport_base.sweep_enum.UNDEFINED
            else:
                raise AttributeError("Dir is not sweep_enum, bool or None.")
            self._sweep_dir = dir
            return
        else:
            raise AttributeError("Number of sweep directions must match number of x columns.")

    @property
    def x(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the independent variable(s).  

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return (self._x, self._xerrs)
        
    @abstractmethod
    @x.setter
    def x(self, vars: np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]) -> None:
        """Sets values for the independent variable(s). 
        Assignment automatically defines sweep direction, assuming data is a timeseries.
        X-errors are set to None if not provided in a tuple.

        Parameters
        ----------
        vars : np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]
            2D array with values for x, or a tuple with (x, xerr) or (x, xerr, xlabels).
            If labels is not provided, new x must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises if assignment doesn't match required format.
        """
        if isinstance(vars, np.ndarray):
            self._x = vars.copy()
            self._xerrs = None
            self.sweep_dir = self._determine_sweep_direction()
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._x = vars[0].copy()
            l = len(vars)
            if l == 1:
                return
            elif isinstance(vars[1], np.ndarray):
                self._xerrs = vars[1].copy()
                if l == 2:
                    return
                elif l == 3 and isinstance(vars[2], list):
                    self._xerrs = vars[1]
                    self._xlabels = vars[2]
                    return
        #         else:
        #             raise TypeError("X[2] doesn't match a list of strings, required for setting labels.")
        #     else:
        #         raise TypeError("X[1] doesn't match a np.ndarray, required for setting errors.")
        # else:
        #     raise TypeError("X doesn't match either a np.ndarray, or a tuple with X[0] as a np.ndarray.")
        raise TypeError("Set either with a np.ndarray or a tuple with (x, xerr) or (x, xerr, xlabels).")
    
    @property
    def independent_vars(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for x. Returns the independent variable(s).

        Returns:
            (numpy.ndarray, numpy.ndarray): A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.x()
    
    @independent_vars.setter
    def independent_vars(self, vars: np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Alias for x. Sets values for the independent variable(s). 
        Assignment automatically defines sweep direction, assuming data is a timeseries.
        X-errors are set to None if not provided in a tuple.

        Parameters
        ----------
        vars : np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for x, or a tuple with (x, xerr) or (x, xerr, xlabels).
            If labels is not provided, new x must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        self.x(vars=vars)
        return
    
    @abstractmethod
    @property
    def y(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the dependent variable(s).  

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return (self._y, self._yerrs)
    
    @abstractmethod
    @y.setter
    def y(self, vars: np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        
        """Sets values for the dependent variable(s).

        Parameters
        ----------
        vars : np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        if isinstance(vars, np.ndarray):
            self._y = vars.copy()
            self._yerrs = None
            return
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._y = vars[0].copy()
            l = len(vars)
            if l == 1:
                return
            elif isinstance(vars[1], np.ndarray):
                self._yerrs = vars[1].copy()
                if l == 2:
                    return
                elif l == 3 and isinstance(vars[2], list):
                    self._yerrs = vars[1]
                    self._ylabels = vars[2]
                    return
        #         else:
        #             raise TypeError("Y[2] doesn't match a list of strings, required for setting labels.")
        #     else:
        #         raise TypeError("Y[1] doesn't match a np.ndarray, required for setting errors.")
        # else:
        #     raise TypeError("Y doesn't match either a np.ndarray, or a tuple with Y[0] as a np.ndarray.")
        raise TypeError("Set either with a np.ndarray or a tuple with (y, yerr) or (y, yerr, ylabels).")
    
    @property
    def dependent_vars(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for y. Returns the dependent variables. These are the primary variables of the class.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.y()
    
    @dependent_vars.setter
    def dependent_vars(self, vars: np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Alias for y. Sets values for the dependent variable(s).

        Parameters
        ----------
        vars : np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        return self.y(vars=vars)

    @abstractmethod
    @property
    def z(self):
        """Returns the extra dependent variable(s), of lesser importance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        
        z = None
        zerr = None
        
        return (z, zerr)
    
    @abstractmethod
    @z.setter
    def z(self, vars: np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Sets values for the extra dependent variable(s).
        
        Parameters
        ----------
        vars : np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for z, or a tuple with (z, zerr) or (z, zerr, zlabels).
            If labels is not provided, new z must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        
        if isinstance(vars, np.ndarray):
            self._z = vars.copy()
            self._zerrs = None
            return
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._z = vars[0].copy()
            l = len(vars)
            if l == 1:
                return
            elif isinstance(vars[1], np.ndarray):
                self._zerrs = vars[1].copy()
                if l == 2:
                    return
                elif l == 3 and isinstance(vars[2], list):
                    self._zerrs = vars[1]
                    self._zlabels = vars[2]
                    return
        raise TypeError("Set either with a np.ndarray or a tuple with (z, zerr) or (z, zerr, zlabels).")

    @property
    def extra_vars(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for z. Returns the extra dependent variable(s), of lesser importance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.z
    
    @extra_vars.setter
    def extra_vars(self, vars: np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Alias for z. Sets values for the extra dependent variable(s).

        Parameters
        ----------
        vars : np.ndarray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        return self.z(vars)

    @property
    def data(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Returns independent and dependent variables, but not extra-dependent variables.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple with elements corresponding to values and uncertainty ((x,y), (xerr,yerr)).
            Each element is a tuple corresponding to independent and dependent variables.
            
        """
        x, xerr = self.x
        y, yerr = self.y
        return ((x,y), (xerr,yerr))
    
    @data.setter
    def data(self, vars: tuple[np.ndarray, np.ndarray] |
             tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] |
             tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]) -> None:
        """Sets independent and dependent variables, but not extra-dependent variables.

        Parameters
        ----------
        vars : tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]
            Tuple of values (X, Y). 
            X and Y can be np.ndarrays, or tuples with (x, xerr) or (x, xerr, xlabel)
            If labels is not provided, new x/y columns must match existing labels length.
        """
        if isinstance(vars, tuple) and len(vars) == 2:
            X,Y = vars
            # Use setters for X and Y.
            self.x(X)
            self.y(Y)
        else:
            raise TypeError("Data setting requires a tuple (X,Y) of length 2. X/Y can be a np.ndarray, or a tuple with (x, xerr) or (x, xerr, xlabels).")
        return
    

    
    @property
    def data_all(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """ Equivalent to independent variables, dependent variables and extra variables.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple with elements corresponding to values and uncertainty ((x,y), (xerr,yerr)).
            Each element is a tuple corresponding to independent and dependent variables.
            
        """
        x, xerr = self.x
        y, yerr = self.y
        z, zerr = self.z
        return ((x,y,z), (xerr,yerr,zerr))

    @data_all.setter
    def data_all(self, vars:tuple[np.ndarray, np.ndarray, np.ndarray] |
             tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] |
             tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]) -> None:
        """ Sets all data by calling setters for x,y and z.

        Parameters
        ----------
        vars : tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]
            

        Raises
        ------
        TypeError
            _description_
        """
        if isinstance(vars, tuple) and len(vars) == 3:
            X,Y,Z = vars
            # Use setters for X,Y,Z
            self.x(X)
            self.y(Y)
            self.z(Z)
        else:
            raise TypeError("Data setting requires a tuple (X,Y,Z) of length 3. X/Y/Z can be a np.ndarray, or a tuple with (x, xerr) or (x, xerr, xlabels).")
        return

    @property
    def all_vars(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """ Alias for data_all. Equivalent to independent variables, dependent variables and extra variables.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple with elements corresponding to values and uncertainty ((x,y), (xerr,yerr)).
            Each element is a tuple corresponding to independent and dependent variables.
            
        """
        return self.data_all()
    
    @all_vars.setter
    def all_vars(self, vars: tuple[np.ndarray, np.ndarray, np.ndarray] |
             tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] |
             tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]) -> None:
        return self.data_all(vars)
    
    @staticmethod
    def _determine_sweep_direction(timeseries: np.ndarray | list[np.ndarray]) -> Self.sweep_enum | list[Self.sweep_enum]:
        """Calculates the sweep direction for a given timeseries by identifying the majority trend through the given data.
        If timeseries increases > 90% of the time, POSITIVE sweep direction is assigned.
        If timeseries decreases > 90% of the time, NEGATIVE sweep direction is assigned.
        Otherwise, UNDEFINED sweep direction is assigned.

        Parameters
        ----------
        timeseries : np.ndarray | list[np.ndarray]
            A 1D or 2D array, or list of such arrays to determine sweep directions.

        Returns
        -------
        Self.sweep_enum | list[Self.sweep_enum]
            Enumerables for each ndarray or column.
            
        """
        if isinstance(timeseries, np.ndarray):
            if len(timeseries.shape) > 2:
                raise AttributeError("Method only accepts 1D or 2D arrays, where each column is processed for sweep direction.")
            else:
                if timeseries.shape == 1: #1D 
                    deltas = np.diff(timeseries[:,c])
                    if np.sum(deltas>0) / len(deltas) > 0.9:
                        return transport_base.sweep_enum.POSITIVE
                    elif np.sum(deltas<0) / len(deltas) > 0.9:
                        return transport_base.sweep_enum.NEGATIVE
                    else:
                        return transport_base.sweep_enum.UNDEFINED
                else:    
                    dirs = []
                    for c in range(timeseries.shape[1]):
                        # Process each 1D subarray.
                        dirs.append(transport_base._determine_sweep_direction(timeseries[:,c]))
                    return dirs
        elif isinstance(timeseries, list):
            dirs = []
            for element in timeseries:
                # Process each list element.
                dirs.append(transport_base._determine_sweep_direction(element))
            return dirs
        else:
            raise TypeError("Requires a list of np.ndarrays or a single np.ndarray.")
    
    @staticmethod
    def _1D_graph(data: tuple[np.ndarray, np.ndarray] | np.ndarray, 
                  ax: mplaxes.Axes | None = None,
                  graph_method: function = mplaxes.Axes.plot, 
                  **mpl_kwargs) -> pylectric.graphing.transport_graph:
        
        c1 = isinstance(data, tuple) and len(data)==2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)
        c2 = isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 2
        assert callable(graph_method)
        
        if c1 or c2:
            # Create figure/axis
            fig, ax = plt.subplots(1,1) if ax is None else (ax.get_figure(), ax)
            
            # Graphing wrapper
            tg = graphwrappers.transport_graph(ax)
            tg.defaults()
            fig.set_size_inches(
                w=journals.acsNanoLetters.maxwidth_2col, h=3)
            
            # Arrange data:
            if c1:
                x,y = data
            elif c2:
                x = data[:,0]
                y = data[:,1]
                
            # Plot data:
            graph_method(ax, x=x, y=y, **mpl_kwargs)
            
            # TODO: Uncomment...
            # tg.setDefaultTicks()
            
            return tg
        else:
            raise AttributeError("Data passed as an innapropriate format, requires np.ndarray(s) in the form (x,y) or data")

    @staticmethod
    def graph_scatter()
    
    @staticmethod
    def graph_errorbar()
    
    @staticmethod
    def graph_plot()


    @staticmethod
    def _plot_1Ddata(data, ax=None, **mpl_kwargs):
        return transport_base._graph_1Ddata(data, ax, scatter=False, **mpl_kwargs)

    @staticmethod
    def _scatter_1Ddata(data, ax=None, **mpl_kwargs):
        return transport_base._graph_1Ddata(data, ax, scatter=True, **mpl_kwargs)
    
    @staticmethod
    def _plot_2Ddata(data, ax=None, **mpl_kwargs):
        return transport_base._graph_2Ddata(data, ax, scatter=False, **mpl_kwargs)

    @staticmethod
    def _scatter_2Ddata(data, ax=None, **mpl_kwargs):
        return transport_base._graph_2Ddata(data, ax, scatter=True, **mpl_kwargs)

    @staticmethod
    def _plot_3Ddata(data, ax=None, **mpl_kwargs):
        return transport_base._graph_3Ddata(data, ax, scatter=False, **mpl_kwargs)
    @staticmethod
    def _scatter_3Ddata(data, ax=None, **mpl_kwargs):
        return transport_base._graph_3Ddata(data, ax, scatter=True, **mpl_kwargs)

    @staticmethod
    def _graph_1Ddata(data: tuple[np.ndarray, np.ndarray] | np.ndarray, ax: mpl.axes.Axes = None, scatter: bool = False, **mpl_kwargs):
        """Plots XY data in a 2D Numpy Array

        Args:
            data (Numpy.ndarray): _description_
            ax (matplotlib.axes): _description_. Defaults to None.

        Returns:
            pylectric.signals.graphing.transport_graph: _description_
        """
        assert isinstance(data, np.ndarray)
        
        if ax is None:
            # Create figure/axis
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        # Graphing wrapper
        tg = graphwrappers.transport_graph(ax)
        tg.defaults()
        fig.set_size_inches(
            w=journals.acsNanoLetters.maxwidth_2col, h=3)

        # Plot data:
        if scatter:
            ax.scatter(data[:, 0], data[:, 1], **mpl_kwargs)
        else:
            ax.plot(data[:, 0], data[:, 1], **mpl_kwargs)
        
        # TODO: Uncomment...
        # tg.setDefaultTicks()
        
        return tg

    @staticmethod
    def _graph_2Ddata(data, axes=None, scatter=False, **mpl_kwargs) -> graphwrappers.transport_graph:
        """Plots 2D data, that is multiple dependent variables against a single independent variable.

        Args:
            data (Numpy 2DArray): Column 0 corresponds to independent variable, columns 1 onward correspond to dependent variables.
            axes (matplotlib.axis.Axis, optional): Plots onto existing axis if provided. Defaults to None.
            

        MPL_kwargs:
            Generic matplotlib kwargs for plots.

        Raises:
            AttributeError: Mismatch in number of dependent variables and provided axes.

        Returns:
            _type_: _description_
        """
        #TODO - Refactor all parameters of functions for MPL objects to use kwargs instead - simplifying code MASSIVELY.
        
        # Requirements for axes object
        if axes is not None: #TODO: make this possible for axes arrays to be included from matplotlib.subplots [or isinstance(array)]
            assert isinstance(axes, plt.Axes) or (isinstance(axes, (list, tuple, np.ndarray)) and np.all([isinstance(ax, plt.Axes) for ax in axes]))
        # Requirements for Data
        assert isinstance(data, np.ndarray) 
        assert len(data.shape) == 2
        nd = data.shape[-1] - 1
        
        
        if axes is None:
            fig,axes = plt.subplots(nd, 1)
            # Graphing wrapper
            tg = graphwrappers.transport_graph(axes)
            fig.set_size_inches(
                w=journals.acsNanoLetters.maxwidth_2col, h=3*nd) #3 inches times number of plots.
        elif len(axes) != nd:
            raise AttributeError("Length of axes does not match data dimension.")
        else:
            tg = graphwrappers.transport_graph(axes)
            
        tg.defaults()
        axes = tg.ax
        
        for i in range(1, nd+1):
            if scatter:
                axes[i-1].scatter(data[:,0], data[:,i], **mpl_kwargs)
            else:
                axes[i-1].plot(data[:, 0], data[:, i], **mpl_kwargs)
            axes[i-1].legend()
            
        # TODO: Uncomment...
        tg.setDefaultTicks()
        # if label:
            # tg.set_legend_handle_size(15)
                
        return tg
    
    @staticmethod
    def _graph_3Ddata(data, axes=None, scatter=False, **mpl_kwargs) -> graphwrappers.transport_graph:
        #TODO: update to match _plot_2Ddata methods... NOT WORKING.
        assert isinstance(data, np.ndarray)
        dl = len(data.shape)
        assert dl == 2 or dl == 3 #if 3, assume first two columns are independent vars, no matter if in 2D or 3D shape.

        nd = data.shape[-1] - 2 #doesn't matter if 2D or 3D, last dimension length determines number of parameters.
        if axes is None:
            fig, axes = plt.subplots(nd, 1)
            fig.set_size_inches(
                w=journals.acsNanoLetters.maxwidth_2col, h=3*nd)
        
        # Graphing wrapper
        tg = graphwrappers.transport_graph(axes)
        
        if dl==3: 
            data = data.reshape([data.shape[0] * data.shape[1],nd])
        
        for i in range(2, nd+2):
            if scatter:
                axes[i-2].scatter(data[:, 0], data[:,1], data[:, i], **mpl_kwargs)
            else:
                axes[i-2].plot(data[:, 0], data[:,1], data[:, i], **mpl_kwargs)

        # TODO: Uncomment...
        # tg.setDefaultTicks()

        return tg
    
    @staticmethod
    def _data_compatability(A,B):
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        li = len(A.shape)
        if li > 1:
            assert A.shape[:-1] == B.shape[:-1]
        elif li == 1:
            assert A.shape[0] == B.shape[0]
        return
    
    @abstractmethod
    def plot_all_data(self, axes=None, scatter=False, **mpl_kwargs):
        """Generates a plot of all data attached to object.
        Subplots determined by independent variables (columns) and dependent / extra variables (rows).
        Requires overriding to label individual plots appropraitely.

        """
        vari = self.ind_vars()
        vard = self.dep_vars()
        vare = self.extra_vars()
        transport_base._data_compatability(vari, vard)
        if vare is not None:
            transport_base._data_compatability(vari, vare)
            data = np.c_[vari, vard, vare]
        else:
            data = np.c_[vari, vard]
            
        li = len(vari.shape)
        if li == 1:
            tg = transport_base._graph_2Ddata(data, axes, scatter, **mpl_kwargs) 
        elif li == 2 or li == 3:
            tg = transport_base._graph_3Ddata(data, axes, scatter, **mpl_kwargs)
        else:
            raise AttributeError("Too many independent variables")        
        return tg

    @abstractmethod
    def plot_dep_vars(self, axes=None, scatter=False, **mpl_kwargs):
        """Generates a plot of all data attached to object.
        Subplots determined by independent variables (columns) and dependent / extra variables (rows).
        Requires override to label plots appropriately.
        """
        vari = self.ind_vars()  # 1D array
        vard = self.dep_vars()  # 2D array

        assert isinstance(vari, np.ndarray)
        assert isinstance(vard, np.ndarray)
        if len(vari.shape) > 1:
            assert vari.shape[:-1] == vard.shape[:-1]
        elif len(vari.shape) == 1:
            assert vari.shape[0] == vard.shape[0]

        li = len(vari.shape)
        if li == 1:
            data = np.c_[vari, vard]
            tg = transport_base._graph_2Ddata(data, axes, scatter, **mpl_kwargs)
        elif li == 2 or li == 3:
            data = np.c_[vari, vard]
            tg = transport_base._graph_3Ddata(data, axes, scatter, **mpl_kwargs)
        else:
            raise AttributeError("Too many independent variables")

        # if 'label' in mpl_kwargs:
        #     for ax in tg.ax:
        #         lgnd = ax.legend()
        #         for handle in lgnd.legendHandles:
        #             handle.set_sizes([10])
        return tg
    
    @abstractmethod
    def __sub__(self, x):
        """ Abstract method for computing a difference in self.dep_vars() and self.extra_vars().
        This method itself will account for differences in datalengths, or mismatches in self.ind_vars() values between objects.
        Difference data is returned in the transport_base.all_vars() format, but needs to be assigned, therefore an abstract method.
        
        Args:
            x (transport_base): Another object whose data will be used to subtract against.
            
        Returns: 
            sub (Numpy NDarray): One numpy array of data (corresponding to the same format as transport_base.all_vars()) corresponding to subtraction.
        """
        assert isinstance(x, transport_base)
        # check that datalength of two arrays match.
        diff = self.ind_vars().shape[0] - x.ind_vars().shape[0]
        match = np.all(self.ind_vars() == x.ind_vars())
        ind_var_len = self.ind_vars().shape[1] if len(self.ind_vars().shape) > 1 else 1 #cover for single column variable case.
        if diff != 0 or not match:
            if not match:
                print("Independent values don't match, truncating data to largest matching independent set to perform subtraction...")
            
            # Axis sizes don't match size ensuring that values match along field.
            self_vars = self.all_vars()
            x_vars = x.all_vars()

            # Assuming data in 2D form.
            for i in range(ind_var_len): #for each independent variable (ie, number of columns)
                self_vars, x_vars = pylectric.signals.processing.trim_matching_fromsimilar(self_vars, x_vars, colN=i)  # iterate over data as many times as there are independent variables
            self_match = self_vars
            x_match = x_vars
        else:
            self_match = self.all_vars()
            x_match = x.all_vars()
        
        #Check independent variables actually match up as intended.
        assert np.all(self_match[:,0:ind_var_len] == x_match[:, 0:ind_var_len])
        
        #Copy independent variables, subtract all other variables.
        sub = self_match.copy()
        sub[:, ind_var_len:] = sub[:, ind_var_len:]  - x_match[:, ind_var_len:]
        return sub

    def sweep_arrow_location(self, i):
        if len(self.ind_vars().shape) == 1:
            #single x variable
            return pylectric.signals.feature_detection.find_arrow_location(xdata=self.ind_vars(),ydata=self.dep_vars()[:,i])

    def to_DataFrame(self):
        return pd.DataFrame(self.all_vars())

class transport_base_dataseries(transport_base):
    def __init__(self, dataseries) -> None:
        self.dataseries = {}
        for key, value in dataseries.items():
            assert isinstance(value, np.ndarray)
            self.dataseries[key] = value.copy()
        super().__init__()
    
    @override
    def extra_vars(self):
        return np.c_[*[self.dataseries[key] for key in self.dataseries]]
    
    @abstractmethod
    def plot_dataseries(self, key, ax=None, scatter=False, **mpl_kwargs):
        """Generates a single plot of a specific key in dataseries.
        Needs override with super call to label x axis for indipendent variables.
        """
        #TODO: Update for 3D version...
        assert key in self.dataseries
        indvars = self.ind_vars()
        value = self.dataseries[key]
        transport_base._data_compatability(indvars, value)
        data = np.c_[indvars, value]
        tg = transport_base._graph_2Ddata(data, ax, scatter, **mpl_kwargs)
        tg.ax[0].set_ylabel(key)
        return tg
    
    @abstractmethod
    def plot_dataseries_with_dep_vars(self, key, ax=None, scatter=False, **mpl_kwargs):
        """Generates a plot of all dependent data attached to object plus specific key in dataseries.
        Subplots determined by independent variables (columns) and dependent / extra variables (rows).
        Needs override with super call to label x & y axis for indipendent/dependent variables.
        """
        #TODO: Update for 3D version.
        if isinstance(key, (list, np.ndarray)):
            vare=None
            for k in key:
                assert k in self.dataseries
                vare = self.dataseries[k] if vare is None else np.c_[vare, self.dataseries[k]]
        else:
            assert key in self.dataseries
            vare = self.dataseries[key]
            
        vari = self.ind_vars()
        vard = self.dep_vars()
        transport_base._data_compatability(vari, vard)
        transport_base._data_compatability(vari, vare)
        data = np.c_[vari, vard, vare]
        tg = transport_base._graph_2Ddata(data, ax, scatter, **mpl_kwargs)
        i = vard.shape[-1]
        tg.ax[i].set_ylabel(key)
        return tg

    @override
    @abstractmethod
    def plot_all_data(self, axes=None, scatter=False, **mpl_kwargs):
        tg = super().plot_all_data(axes, scatter, **mpl_kwargs)
        dep_len = self.dep_vars().shape[-1]
        labels = list(self.dataseries)
        for i in range(len(self.dataseries)):
            print(i,labels[i])
            tg.ax[dep_len + i].set_ylabel(labels[i])
        return tg

    @abstractmethod
    def plot_all_dataseries(self, ax=None, scatter=False, **mpl_kwargs):
        """Generates a plot of all dataseries.
        Subplot for each dataseries (rows).
        Needs override with super call to label x axis for indipendent variables.
        """
        # TODO: Update for 3D version...
        indvars = self.ind_vars()
        ds_data = np.c_[*[self.dataseries[key] for key in self.dataseries]]
        transport_base._data_compatability(indvars, ds_data)
        data = np.c_[indvars, ds_data]
        tg = transport_base._graph_2Ddata(data, ax, scatter, **mpl_kwargs)
        for i in range(len(self.dataseries)):
            tg.ax[i].set_ylabel(list(self.dataseries)[i])
        return tg

    

# class graphable_ND_base(graphable_base):
#     def __init__(self) -> None:
#         super().__init__()
        
#     # @override(graphable_base)
#     @abstractmethod
#     def ind_vars(self):
#         """Returns the independent variables.

#         Returns:
#             List of Numpy array: A list of 1D data arrays, for each independent variable related to the object.
#         """
#         return None
    
#     @abstractmethod(graphable_base)
#     def dep_vars(self):
#         """Returns the dependent variables

#         Returns:
#             Numpy array: A (N+1) dimensional array of the dependent variables related to the object, where axis N represents the Nth indepentent variable.
#                 The (N+1)th dimension represents each dependent variable.
#         """
#         return None