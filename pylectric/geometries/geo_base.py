# Coding
from overrides import override
from abc import abstractmethod, ABCMeta
from typing import Type, Sequence, Self, Any
#Methods
import pylectric
from pylectric import graphing
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import axes as mplaxes
from matplotlib import figure as mplfigure
import numpy as np
import numpy.typing as npt
from enum import Enum
import pandas as pd

class measurement_base(metaclass=ABCMeta):
    """An abstract class to expand and bind functions to geometric objects.
        All data is timeseries data, represented by rows for various variable columns.
        Data also should implement a sweep direction.
        TODO: make all multivariable arrays 2D, even if only one variable, for data output consistency.
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
    
    @abstractmethod
    def __init__(self) -> None:
        """Initalizes base transport class. 
        Requires implementation of sweep direction with independent variable(s).
        """
        # if not hasattr(self, "data"):
            # raise AttributeError("No data passed to constructor.")
        super().__init__()
        
        # Sweep Direction
        self._sweep_dir = None #initalizes None as might not match dimensions of data. 
        
        # Data
        ## Independent Variables
        self._x = None
        self._xerr = None
        self._xlabels = None
        
        ## Dependent Variables
        self._y = None
        self._yerr = None
        self._ylabels = None
        
        ## Extra Variables
        self._z = None
        self._zerr = None
        self._zlabels = None
        
        # Parameters
        self.params = {} #empty list of parameters
        
        
        return
    
    def __copy__(self) -> Type[Self]:
        """Creates a deep clone of base datasets.

        Returns
        -------
        Type[Self]
            A new instance of the (sub-)class.
        """
        clone = object.__new__(type(self))
        # Sweep Dirs may be a list
        clone.sweep_dir = self.sweep_dir if not hasattr(self.sweep_dir, "__len__") else self.sweep_dir.copy()
        
        # X
        clone._x = self._x.copy() if self._x is not None else None
        clone._xerr = self._xerr.copy() if self._xerr is not None else None
        clone._xlabels = self._xlabels.copy() if self._xlabels is not None else None
        # Y
        clone._y = self._y.copy() if self._y is not None else None
        clone._yerr = self._yerr.copy() if self._yerr is not None else None
        clone._ylabels = self._ylabels.copy() if self._ylabels is not None else None
        # Z
        clone._z = self._z.copy() if self._y is not None else None
        clone._zerr = self._zerr.copy() if self._zerr is not None else None
        clone._zlabels = self._zlabels.copy() if self._zlabels is not None else None
        
        # Parameters:
        clone.params = {}
        for key,val in self.params:
            # Perform deep clone on each key,val pair if object has copy attr.
            keycopy = key.copy() if hasattr(key, "copy") else key
            valcopy = val.copy() if hasattr(val, "copy") else val
            clone[keycopy] = valcopy
            
        return clone


    @property
    def sweep_dir(self) -> Self.sweep_enum | list[Self.sweep_enum]:
        return self._sweep_dir
        
    @sweep_dir.setter
    def sweep_dir(self, dir: Self.sweep_enum | bool | None | list[Self.sweep_enum] | list[bool | None]) -> None:
        if hasattr(dir, "__len__"): # Multiple sweep directions provided.
            x = self.x
            if len(x.shape) == 2 and x.shape[1]==len(dir): # Check that provided sweep directions match number of columns of x .
                dirs = []
                for sweep in dir:
                    if sweep is measurement_base.sweep_enum:
                        pass    
                    elif isinstance(sweep, bool):
                        sweep = measurement_base.sweep_enum.POSITIVE if dir else measurement_base.sweep_enum.NEGATIVE
                    elif sweep is None:
                        sweep = measurement_base.sweep_enum.UNDEFINED
                    else:
                        raise AttributeError("Dir element is not sweep_enum, bool or None.")
                    dirs.append(sweep)
                self._sweep_dir = dirs
            elif len(x.shape) == 1 and len(dir) == 1: # Only one element in list and only one variable.
                dir = dir[0]
                if dir is measurement_base.sweep_enum:
                    pass
                elif isinstance(dir, bool):
                    dir = measurement_base.sweep_enum.POSITIVE if dir else measurement_base.sweep_enum.NEGATIVE
                elif dir is None:
                    dir = measurement_base.sweep_enum.UNDEFINED
                else:
                    raise AttributeError("Dir element is not sweep_enum, bool or None.")
                self._sweep_dir = dir
                return
            else:
                raise AttributeError("Number of sweep directions must match number of x columns.")
        elif len(self.x.shape) == 1: # Check that x only has one variable.
            if dir is measurement_base.sweep_enum:
                pass
            elif isinstance(dir, bool):
                dir = measurement_base.sweep_enum.POSITIVE if dir else measurement_base.sweep_enum.NEGATIVE
            elif dir is None:
                dir = measurement_base.sweep_enum.UNDEFINED
            else:
                raise AttributeError("Dir is not sweep_enum, bool or None.")
            self._sweep_dir = dir
            return
        else:
            raise AttributeError("Number of sweep directions must match number of x columns.")

    @staticmethod
    def _determine_sweep_direction(timeseries: npt.NDArray | list[npt.NDArray]) -> Self.sweep_enum | list[Self.sweep_enum]:
        """Calculates the sweep direction for a given timeseries by identifying the majority trend through the given data.
        If timeseries increases > 90% of the time, POSITIVE sweep direction is assigned.
        If timeseries decreases > 90% of the time, NEGATIVE sweep direction is assigned.
        Otherwise, UNDEFINED sweep direction is assigned.

        Parameters
        ----------
        timeseries : npt.NDArray | list[npt.NDArray]
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
                        return measurement_base.sweep_enum.POSITIVE
                    elif np.sum(deltas<0) / len(deltas) > 0.9:
                        return measurement_base.sweep_enum.NEGATIVE
                    else:
                        return measurement_base.sweep_enum.UNDEFINED
                else:    
                    dirs = []
                    for c in range(timeseries.shape[1]):
                        # Process each 1D subarray.
                        dirs.append(measurement_base._determine_sweep_direction(timeseries[:,c]))
                    return dirs
        elif isinstance(timeseries, list):
            dirs = []
            for element in timeseries:
                # Process each list element.
                dirs.append(measurement_base._determine_sweep_direction(element))
            return dirs
        else:
            raise TypeError("Requires a list of np.ndarrays or a single np.ndarray.")

    @property
    def x(self) -> npt.NDArray:
        """Returns the independent variable(s). 
        If mask_x is defined and has True values, corresponding x values are returned as np.nan.

        Returns
        -------
        npt.NDArray
            A 2D data array, with columns corresponding to the independent variables of x.
        """
        return self._x
        
    @x.setter
    def x(self, vars: npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list[str]]) -> None:
        """Sets values for the independent variable(s). 
        Assignment automatically defines sweep direction, assuming data is a timeseries.
        X-errors are set to None if not provided in a tuple.

        Parameters
        ----------
        vars : npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list[str]]
            2D array with values for x, or a tuple with (x, xerr) or (x, xerr, xlabels).
            If labels is not provided, new x must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises if assignment doesn't match required format.
        """
        newx = False # Perform additional operations if vars is valid format.
        xold = self.x
        xolderrs = self.xerr
        xoldlabels = self.xlabels
        # Check vars input and process accordingly:
        if isinstance(vars, np.ndarray):
            self._x = vars.copy()
            self.xerr = None
            newx=True
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._x = vars[0].copy()
            l = len(vars)
            if l == 1:
                newx=True
            elif isinstance(vars[1], np.ndarray):
                self.xerr = vars[1]
                if l == 2:
                    newx=True
                elif l == 3 and isinstance(vars[2], list):
                    self.xerr = vars[1]
                    self.xlabels = vars[2]
                    newx=True
        # Setup sweep direction, and enforce x mask on xerr (as fast as checking).
        if newx:
            self.sweep_dir = measurement_base._determine_sweep_direction(self._x)
            if isinstance(self._x, np.ma.MaskedArray) and isinstance(self._xerr, np.ma.MaskedArray):
                self._xerr.mask = self._x.mask
            return
        else:
            # Reset internal vars to original values.
            self._x = xold
            self._xerr = xolderrs
            self._xlabels = xoldlabels
            raise TypeError("Set either with a np.ndarray or a tuple with (x, xerr) or (x, xerr, xlabels).")
    
    @x.deleter
    def x(self) -> None:
        """Resets all variables x,y,z (and corresponding errors and labels) to None, as y and z are required to match the datalength of x.
        """
        self._x = None
        del self.xerr
        del self.xlabels
        del self.y #also calls del self.z
        return
    
    @property
    def y(self) -> npt.NDArray:
        """Returns the dependent variable(s). 
        If mask_y is defined and has True values, corresponding y values are returned as np.nan.

        Returns
        -------
        npt.NDArray
            A 2D data array, with columns corresponding to the independent variables of y.
        """
        return self._y
    
    @y.setter
    def y(self, vars: npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list[str]]) -> None:
        
        """Sets values for the dependent variable(s).

        Parameters
        ----------
        vars : npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list[str]]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        newy = False # Perform additional operations if vars is valid format.
        yold = self.y
        yolderrs = self.yerr
        yoldlabels = self.ylabels
        # Check vars input and process accordingly:
        if isinstance(vars, np.ndarray):
            self._y = vars.copy()
            self.yerr = None
            newy=True
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._y = vars[0].copy()
            l = len(vars)
            if l == 1:
                newy=True
            elif isinstance(vars[1], np.ndarray):
                self.yerr = vars[1]
                if l == 2:
                    newy=True
                elif l == 3 and isinstance(vars[2], list):
                    self.yerr = vars[1]
                    self.ylabels = vars[2]
                    newy=True
        # Setup sweep direction, and enforce y mask on yerr (as fast as checking).
        if newy:
            self.sweep_dir = measurement_base._determine_sweep_direction(self._y)
            if isinstance(self._y, np.ma.MaskedArray) and isinstance(self._yerr, np.ma.MaskedArray):
                self._yerr.mask = self._y.mask
            return
        else:
            # Reset internal vars to original values.
            self._y = yold
            self.yerr = yolderrs
            self.ylabels = yoldlabels
            raise TypeError("Set either with a np.ndarray or a tuple with (y, yerr) or (y, yerr, ylabels).")
    
    @y.deleter
    def y(self) -> None:
        """Resets y, yerr and ylabel variables to None. 
        Additionally resets z variables, as extra variables are (typically) only shown in conjunction with y.
        """
        self._y = None
        del self.yerr
        del self.ylabels
        del self.z
        return
    
    @property
    def z(self) -> npt.NDArray:
        """Returns the extra dependent variable(s), of lesser importance. 
        If mask_z is defined, data is masked.

        Returns
        -------
        npt.NDArray
            A 2D data array, with columns corresponding to the independent variables of x.
        """
        return self._z
    
    @z.setter
    def z(self, vars: npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list[str]]) -> None:
        """Sets values for the extra dependent variable(s).
        
        Parameters
        ----------
        vars : npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list[str]]
            2D array with values for z, or a tuple with (z, zerr) or (z, zerr, zlabels).
            If labels is not provided, new z must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        newz = False # Perform additional operations if vars is valid format.
        zold = self._z
        zolderrs = self._zerr
        zoldlabels = self._zlabels
        # Check vars input and process accordingly:
        if isinstance(vars, np.ndarray):
            self._z = vars.copy()
            self.zerr = None
            newz=True
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._z = vars[0].copy()
            l = len(vars)
            if l == 1:
                newz=True
            elif isinstance(vars[1], np.ndarray):
                self.zerr = vars[1]
                if l == 2:
                    newz=True
                elif l == 3 and isinstance(vars[2], list):
                    self.zerr = vars[1]
                    self.zlabels = vars[2]
                    newz=True
        # Setup sweep direction, and enforce z mask on zerr (as fast as checking).
        if newz:
            self.sweep_dir = measurement_base._determine_sweep_direction(self._z)
            if isinstance(self._z, np.ma.MaskedArray) and isinstance(self._zerr, np.ma.MaskedArray):
                self._zerr.mask = self._z.mask
            return
        else:
            self._z = zold
            self.zerr = zolderrs
            self.zlabels = zoldlabels
            raise TypeError("Set either with a np.ndarray or a tuple with (z, zerr) or (z, zerr, zlabels).")

    @z.deleter
    def z(self):
        """Resets z, zerr and zlabel variables to None.
        """
        self._z = None
        del self.zerr
        del self.zlabels
        return
    
    @property
    def data(self) -> tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]:
        """Returns independent and dependent variables and their errors, but not extra-dependent variables.

        Returns
        -------
        tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]
            A tuple with elements corresponding to values and uncertainty ((x,xerr), (y,yerr)).
            Each element is a np.ndarray corresponding to independent and dependent variables.
            
        """
        return (self.x, self.y)

    @data.setter
    def data(self, vars: tuple[npt.NDArray, npt.NDArray] |
             tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]] |
             tuple[tuple[npt.NDArray, npt.NDArray, list[str]], tuple[npt.NDArray, npt.NDArray, list[str]]]) -> None:
        """Sets independent and dependent variables, but not extra-dependent variables.

        Parameters
        ----------
        vars : tuple[npt.NDArray, npt.NDArray] | tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]] | tuple[tuple[npt.NDArray, npt.NDArray, list[str]], tuple[npt.NDArray, npt.NDArray, list[str]]]
            Tuple of values (X, Y). 
            X and Y can be np.ndarrays, or tuples with (x, xerr) or (x, xerr, xlabel)
            If labels is not provided, new x/y columns must match existing labels length.
        """
        if isinstance(vars, tuple) and len(vars) == 2:
            X,Y = vars
            # Use setters for X and Y.
            self.x = X
            self.y = Y
        else:
            raise TypeError("Data setting requires a tuple (X,Y) of length 2. X/Y can be a np.ndarray, or a tuple with (x, xerr) or (x, xerr, xlabels).")
        return
    
    @data.deleter
    def data(self) -> None:
        """Alias for x deleter. Resets all variables.
        """
        del self.x
        return
    
    @property
    def data_all(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Equivalent to independent variables, dependent variables and extra variables.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple with elements corresponding to values and uncertainty ((x,xerr), (y,yerr), (z,zerr)).
            
        """
        return (self.x, self.y, self.z)

    @data_all.setter
    def data_all(self, vars:tuple[np.ndarray, np.ndarray, np.ndarray] |
             tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] |
             tuple[tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
        """Sets all data by calling setters for x,y and z.

        Parameters
        ----------
        vars : tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]
            A tuple of length 3 corresponding to (x,y,z). Each index can be comprosied of x, (x, xerr) or (x, xerr, xlabels).

        Raises
        ------
        TypeError
            Raises if vars is not a tuple of length three.
        """
        if isinstance(vars, tuple) and len(vars) == 3:
            X,Y,Z = vars
            # Use setters for X,Y,Z
            self.x = X
            self.y = Y
            self.z = Z
        else:
            raise TypeError("Data setting requires a tuple (X,Y,Z) of length 3. X/Y/Z can be a np.ndarray, or a tuple with (x, xerr) or (x, xerr, xlabels).")
        return
    
    @data_all.deleter
    def data_all(self) -> None:
        """Alias for x deleter. Resets all data.
        """
        del self.x
        return
 
    def array(self) -> npt.NDArray:
        """Returns x,y variables in a single 2D numpy array.

        Returns
        -------
        npt.NDArray
            An array of variables x,y.
        """
        return np.c_[self.x, self.y]
 
    def array_all(self) -> npt.NDArray:
        """Returns all variables in a single 2D numpy array.

        Returns
        -------
        npt.NDArray
            An array of all variables of x,y,z.
        """
        return np.c_[self.x, self.y, self.z]

    @property
    def xerr(self) -> np.ndarray | None:
        """Errors of x variables.

        Returns
        -------
        np.ndarray | None
            A 2D array of uncertainties of each measurement in x, or None. 
            Columns correspond to different variables.
            Requires the same shape as x.
        """
        return self._xerr
    
    @xerr.setter
    def xerr(self, errs: npt.NDArray | None) -> None:
        """Sets new values for errors in x.

        Parameters
        ----------
        errs : npt.NDArray | None
            A 2D array of uncertainties of each measurement in x, or None. 
            Columns correspond to different variables.
            Requires the same shape as x.

        Raises
        ------
        AttributeError
            Raises an excpetion if the shapes of x and errs don't match.
        """
        if errs is None:
            self._xerr = None
            return
        if self._x.shape == errs.shape:
            self._xerr = errs.copy()
            return
        else:
            raise AttributeError("Errs doesn't match shape of x.")
    
    @xerr.deleter
    def xerr(self):
        """Resets xerr to None.
        """
        self._xerr = None
        return

    @property
    def yerr(self) -> np.ndarray | None:
        """Errors of y variables.

        Returns
        -------
        np.ndarray | None
            A 2D array of uncertainties of each measurement in y, or None. 
            Columns correspond to different variables.
            Requires the same shape as y.
        """
        return self._yerr
    
    @yerr.setter
    def yerr(self, errs: npt.NDArray | None) -> None:
        """Sets new values for errors in y.

        Parameters
        ----------
        errs : npt.NDArray
            A 2D array of uncertainties of each measurement in y, or None. 
            Columns correspond to different variables.
            Requires the same shape as y.

        Raises
        ------
        AttributeError
            Raises an excpetion if the shapes of y and errs don't match.
        """
        if errs is None:
            self._yerr = None
            return
        elif self._y.shape == errs.shape:
            self._yerr = errs.copy()
            return
        else:
            raise AttributeError("Errs doesn't match shape of y.")
        
    @yerr.deleter
    def yerr(self) -> None:
        """Resets yerr to None.
        """
        self._yerr = None
        
    @property
    def zerr(self) -> np.ndarray | None:
        """Errors of z variables.

        Returns
        -------
        np.ndarray | None
            A 2D array of uncertainties of each measurement in z, or None. 
            Columns correspond to different variables.
            Requires the same shape as z.
        """
        return self._zerr
    
    @zerr.setter
    def zerr(self, errs: npt.NDArray | None) -> None:
        """Sets new values for errors in z.

        Parameters
        ----------
        errs : npt.NDArray
            A 2D array of uncertainties of each measurement in z, or None. 
            Columns correspond to different variables.
            Requires the same shape as z.

        Raises
        ------
        AttributeError
            Raises an excpetion if the shapes of z and errs don't match.
        """
        if errs is None:
            self._zerr = None
            return
        if self._z.shape == errs.shape:
            self._zerr = errs.copy()
            return
        else:
            raise AttributeError("Errs doesn't match shape of z.")

    @zerr.deleter
    def zerr(self) -> None:
        """Resets zerr to None.
        """
        self._zerr = None
    
    def err_array(self) -> npt.NDArray:
        """Returns errors of variables in a single 2D numpy array.

        Returns
        -------
        npt.NDArray
            An array of errors of variables of x,y.
        """
        return np.c_[self.xerr, self.yerr]

    def err_array_all(self) -> npt.NDArray:
        """Returns all errors of variables in a single 2D numpy array.

        Returns
        -------
        npt.NDArray
            An array of all errors of variables of x,y,z.
        """
        return np.c_[self.xerr, self.yerr, self.zerr]

    @property
    def xlabels(self) -> list[str]:
        """List of xlabels, index matched to columns of x.
        
        Returns
        -------
        list[str]
            List of xlabel strings, index matched to columns of x.
        """
        return self._xlabels
    
    @xlabels.setter
    def xlabels(self, labels: list[str] | str | None) -> None:
        """Setter for xlabels, index matched to columns of x.

        Parameters
        ----------
        labels : list[str] | str | None
            List of xlabel strings or singular string, index matched to columns of x.

        Raises
        ------
        AttributeError
            Raised if the length of the provided list doesn't match the number of x columns. 
        """
        if isinstance(labels, str) and len(self._x.shape) == 1:
            # String provided
            self._xlabels = [labels]
        elif isinstance(labels, list) and len(self._x.shape) == 1 and len(labels) == 1:
            # Single label in list
            self._xlabels = labels.copy()
        elif isinstance(labels, list) and len(self._x.shape) == 2 and len(labels) == self._x.shape[1]:    
            # Multiple labels:
            self._xlabels = labels.copy()
        elif labels is None:
            self._xlabels = None
        else:
            raise AttributeError("Length of provided labels doesn't match number of x columns.")
    
    @xlabels.deleter
    def xlabels(self) -> None:
        """Resets xlabels to None.
        """
        self._xlabels = None
        return
    
    @property
    def ylabels(self) -> list[str]:
        """List of ylabels, index matched to columns of y.
        
        Returns
        -------
        list[str]
            List of ylabel strings, index matched to columns of y.
        """
        return self._ylabels
    
    @ylabels.setter
    def ylabels(self, labels: list[str] | str | None) -> None:
        """Setter for ylabels, index matched to columns of y.

        Parameters
        ----------
        labels : list[str] | str | None
            List of ylabel strings or singular string, index matched to columns of y.

        Raises
        ------
        AttributeError
            Raised if the length of the provided list doesn't match the number of y columns. 
        """
        if isinstance(labels, str) and len(self._y.shape) == 1:
            # String provided
            self._ylabels = [labels]
        elif isinstance(labels, list) and len(self._y.shape) == 1 and len(labels) == 1:
            # Single label in list
            self._ylabels = labels.copy()
        elif isinstance(labels, list) and len(self._y.shape) == 2 and len(labels) == self._y.shape[1]:    
            # Multiple labels:
            self._ylabels = labels.copy()
        elif labels is None:
            self._ylabels = None
        else:
            raise AttributeError("Length of provided labels doesn't match number of y columns.")

    @ylabels.deleter
    def ylabels(self) -> None:
        """Resets ylabels to None.
        """
        self._ylabels = None
        return

    @property
    def zlabels(self) -> list[str]:
        """List of zlabels, index matched to columns of z.
        
        Returns
        -------
        list[str]
            List of zlabel strings, index matched to columns of z.
        """
        return self._zlabels
    
    @zlabels.setter
    def zlabels(self, labels: list[str] | str | None) -> None:
        """Setter for zlabels, index matched to columns of z.

        Parameters
        ----------
        labels : list[str] | str | None
            List of zlabel strings or singular string, index matched to columns of z.

        Raises
        ------
        AttributeError
            Raised if the length of the provided list doesn't match the number of z columns. 
        """
        if isinstance(labels, str) and len(self._z.shape) == 1:
            # String provided
            self._zlabels = [labels]
        elif isinstance(labels, list) and len(self._z.shape) == 1 and len(labels) == 1:
            # Single label in list
            self._zlabels = labels.copy()
        elif isinstance(labels, list) and len(self._z.shape) == 2 and len(labels) == self._z.shape[1]:    
            # Multiple labels:
            self._zlabels = labels.copy()
        elif labels is None:
            self._zlabels = None
        else:
            raise AttributeError("Length of provided labels doesn't match number of z columns.")

    @zlabels.deleter
    def zlabels(self) -> None:
        """Resets zlabels to None.
        """
        self._zlabels = None
        return
    
    @property
    def labels(self) -> list[str] | None:
        """Returns x,y labels combined in a single, consecutive list.
        If x or y are not set (None), they are treated as empty lists.
        If both x and y are not set, None is returned.

        Returns
        -------
        list[str] | None
            List of strings describing each column for x,y variables.
        """
        xl = self.xlabels
        yl = self.ylabels
        if xl is None and yl is None:
            return None
        else:
            xl = xl if xl is None else []
            yl = yl if yl is None else []
            return xl + yl
      
    @labels.setter
    def labels(self, labels: list[str] | tuple(list[str], list[str])) -> None:
        if isinstance(labels, tuple) and len(labels)==2:
            xl, yl = labels
            self.xlabels = xl
            self.ylabels = yl
        elif (isinstance(labels, list)
              and self.x is not None and self.y is not None
              and len(labels) == self.x.shape[1] + self.y.shape[1]
            ):
            self.xlabels = labels[0:self.x.shape[1]]
            self.ylabels = labels[self.x.shape[1]:]
        else:
            raise AttributeError("List of strings does not match number of variables in x,y.")
        return
    
    @labels.deleter
    def labels(self) -> None:
        """Resets x,y labels to None. Calls x,y label deleters.
        """
        del self.xlabels
        del self.ylabels
        return 
    
    @property
    def labels_all(self) -> list[str]:
        """Returns all x,y,z labels in a single, consecutive list.
        If x, y or z are not set (None), they are treated as empty lists.
        If all x,y and z are not set, None is returned.

        Returns
        -------
        list[str]
            List of strings describing each column for x,y,z variables.
        """
        xl = self.xlabels
        yl = self.ylabels
        zl = self.zlabels
        if xl is None and yl is None and zl is None:
            return None
        else:
            xl = xl if xl is None else []
            yl = yl if yl is None else []
            zl = zl if zl is None else []
            return xl + yl + zl
            
    @labels_all.setter
    def labels_all(self, labels: list[str] | tuple(list[str], list[str], list[str])) -> None:
        if isinstance(labels, tuple) and len(labels)==3:
            xl, yl, zl = labels
            self.xlabels = xl
            self.ylabels = yl
            self.zlabels = zl
        elif (isinstance(labels, list)
              and self.x is not None and self.y is not None and self.z is not None
              and len(labels) == self.x.shape[1] + self.y.shape[1] + self.z.shape[1]
            ):
            self.xlabels = labels[0:self.x.shape[1]]
            self.ylabels = labels[self.x.shape[1]:self.x.shape[1] + self.y.shape[1]]
            self.zlabels = labels[self.x.shape[1] + self.y.shape[1]:]
        else:
            raise AttributeError("List of strings does not match number of variables in x,y,z.")    
    
    @labels_all.deleter
    def labels_all(self) -> None:
        """Resets all labels to None. Calls x,y,z label deleters.
        """
        del self.xlabels
        del self.ylabels
        del self.zlabels
        return
        
    @property
    def mask_x(self) -> np.ndarray | None:
        """Returns the mask for x. 
        False (True) implies datapoint is included (excluded).

        Returns
        -------
        np.ndarray[bool] | None
            Returns an array of mask values.
            If no mask, returns None.
        """
        if isinstance(self._x, np.ma.MaskedArray) and self._x.mask is not False:
            return self._x.mask
        else:
            return None
    
    @mask_x.setter
    def mask_x(self, mask: npt.NDArray | bool) -> None:
        """Sets the mask for x and xerr. 
        False (True) implies datapoint is included (excluded).

        Parameters
        ----------
        mask : npt.NDArray[bool] | bool
            Numpy array matching dimensions of existing x, with dtype bool.
            If bool is provided, all values are masked.

        Raises
        ------
        AttributeError
            Raises if the mask shape does not match the x shape.
            Raises if the mask is not boolean dtype.
        """
        # Bool provided. Scale mask to x shape.
        if self._x is not None and isinstance(mask, bool):
            mask = np.full_like(self._x, mask, dtype=bool)
        # Array provided. Check input mask matches shape of existing x.
        if self._x is not None and self._x.shape == mask.shape:
            # Check mask is boolean.
            if mask.dtype == bool:
                # Check if x, xerr are masked arrays, if not replace with one.
                if isinstance(self._x, np.ma.MaskedArray):
                    self._x.mask=mask
                else:
                    self._x = np.ma.MaskedArray(self._x, mask)
                if isinstance(self._xerr, np.ma.MaskedArray):
                    self._xerr.mask=mask
                else:
                    self._xerr = np.ma.MaskedArray(self._xerr, mask)
                return
            else:
                raise AttributeError("Provided mask is not boolean.")
        elif self._x is None:
            raise AttributeError("x values are not set.")
        else:
            raise AttributeError("Mask shape doesn't match existing x.")

    @mask_x.deleter
    def mask_x(self) -> None:
        """Removes Converts masked arrays back to regular numpy arrays.
        """
        if isinstance(self._x, np.ma.MaskedArray):
            self._x = np.array(self._x)
        if isinstance(self._x, np.ma.MaskedArray):
            self._xerr = np.array(self._xerr)
        return


    @property
    def mask_y(self) -> np.ndarray | None:
        """Returns the mask for y. 
        False (True) implies datapoint is included (excluded).

        Returns
        -------
        np.ndarray[bool] | None
            Returns an array of mask values.
            If no mask, returns None.
        """
        if isinstance(self._y, np.ma.MaskedArray) and self._y.mask is not False:
            return self._y.mask
        else:
            return None
    
    @mask_y.setter
    def mask_y(self, mask: npt.NDArray | bool) -> None:
        """Sets the mask for y and yerr. 
        False (True) implies datapoint is included (excluded).

        Parameters
        ----------
        mask : npt.NDArray[bool] | bool
            Numpy array matching dimensions of existing y, with dtype bool.
            If bool is provided, all values are masked.

        Raises
        ------
        AttributeError
            Raises if the mask shape does not match the y shape.
            Raises if the mask is not boolean dtype.
        """
        # Bool provided. Scale mask to y shape.
        if self._y is not None and isinstance(mask, bool):
            mask = np.full_like(self._y, mask, dtype=bool)
        # Array provided. Check input mask matches shape of existing y.
        if self._y is not None and self._y.shape == mask.shape:
            # Check mask is boolean.
            if mask.dtype == bool:
                # Check if y, yerr are masked arrays, if not replace with one.
                if isinstance(self._y, np.ma.MaskedArray):
                    self._y.mask=mask
                else:
                    self._y = np.ma.MaskedArray(self._y, mask)
                if isinstance(self._yerr, np.ma.MaskedArray):
                    self._yerr.mask=mask
                else:
                    self._yerr = np.ma.MaskedArray(self._yerr, mask)
                return
            else:
                raise AttributeError("Provided mask is not boolean.")
        elif self._y is None:
            raise AttributeError("y values are not set.")
        else:
            raise AttributeError("Mask shape doesn't match existing y.")

    @mask_y.deleter
    def mask_y(self) -> None:
        """Removes Converts masked arrays back to regular numpy arrays.
        """
        if isinstance(self._y, np.ma.MaskedArray):
            self._y = np.array(self._y)
        if isinstance(self._y, np.ma.MaskedArray):
            self._yerr = np.array(self._yerr)
        return


    @property
    def mask_z(self) -> np.ndarray | None:
        """Returns the mask for z. 
        False (True) implies datapoint is included (excluded).

        Returns
        -------
        np.ndarray[bool] | None
            Returns an array of mask values.
            If no mask, returns None.
        """
        if isinstance(self._z, np.ma.MaskedArray) and self._z.mask is not False:
            return self._z.mask
        else:
            return None
    
    @mask_z.setter
    def mask_z(self, mask: npt.NDArray | bool) -> None:
        """Sets the mask for z and zerr. 
        False (True) implies datapoint is included (excluded).

        Parameters
        ----------
        mask : npt.NDArray[bool] | bool
            Numpy array matching dimensions of existing z, with dtype bool.
            If bool is provided, all values are masked.

        Raises
        ------
        AttributeError
            Raises if the mask shape does not match the z shape.
            Raises if the mask is not boolean dtype.
        """
        # Bool provided. Scale mask to z shape.
        if self._z is not None and isinstance(mask, bool):
            mask = np.full_like(self._z, mask, dtype=bool)
        # Array provided. Check input mask matches shape of existing z.
        if self._z is not None and self._z.shape == mask.shape:
            # Check mask is boolean.
            if mask.dtype == bool:
                # Check if z, zerr are masked arrays, if not replace with one.
                if isinstance(self._z, np.ma.MaskedArray):
                    self._z.mask=mask
                else:
                    self._z = np.ma.MaskedArray(self._z, mask)
                if isinstance(self._zerr, np.ma.MaskedArray):
                    self._zerr.mask=mask
                else:
                    self._zerr = np.ma.MaskedArray(self._zerr, mask)
                return
            else:
                raise AttributeError("Provided mask is not boolean.")
        elif self._z is None:
            raise AttributeError("z values are not set.")
        else:
            raise AttributeError("Mask shape doesn't match existing z.")

    @mask_z.deleter
    def mask_z(self) -> None:
        """Removes Converts masked arrays back to regular numpy arrays.
        """
        if isinstance(self._z, np.ma.MaskedArray):
            self._z = np.array(self._z)
        if isinstance(self._z, np.ma.MaskedArray):
            self._zerr = np.array(self._zerr)
        return

    @property
    def vars_independent(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for x. Returns the independent variable(s).

        Returns:
            (numpy.ndarray, numpy.ndarray): A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.x()
    
    @vars_independent.setter
    def vars_independent(self, vars: npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Alias for x. Sets values for the independent variable(s). 
        Assignment automatically defines sweep direction, assuming data is a timeseries.
        X-errors are set to None if not provided in a tuple.

        Parameters
        ----------
        vars : npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for x, or a tuple with (x, xerr) or (x, xerr, xlabels).
            If labels is not provided, new x must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        self.x = vars.copy()
        return
    
    @vars_independent.deleter
    def vars_independent(self) -> None:
        """Alias for x deleter. Sets all variables x,y,z (and corresponding errors and labels) to None,
        as y and z are required to match the datalength of x.
        """
        del self.x
        return
        
    
    @property
    def vars_dependent(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for y. Returns the dependent variables. These are the primary variables of the class.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.y()
    
    
    @vars_dependent.setter
    def vars_dependent(self, vars: npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Alias for y. Sets values for the dependent variable(s).

        Parameters
        ----------
        vars : npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        return self.y(vars=vars)

    @vars_dependent.deleter
    def vars_dependent(self) -> None:
        """Alias for y deleter. Resets y, yerr and ylabel variables to None. 
        Additionally resets z variables, as extra variables are (typically) only shown in conjunction with y.
        """
        del self.y
        return

    @property
    def vars_extra(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for z. Returns the extra dependent variable(s), of lesser importance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.z
    
    @vars_extra.setter
    def vars_extra(self, vars: npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
        """Alias for z. Sets values for the extra dependent variable(s).

        Parameters
        ----------
        vars : npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        return self.z(vars)
    
    @vars_extra.deleter
    def vars_extra(self) -> None:
        """Alias for z deleter. Resets z, zerr and zlabel variables to None.
        """
        del self.z
        return
    
    @property
    def vars_all(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Alias for data_all. Equivalent to independent variables, dependent variables and extra variables.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple with elements corresponding to values and uncertainty ((x,y), (xerr,yerr)).
            Each element is a tuple corresponding to independent and dependent variables.
            
        """
        return self.data_all()
    
    @vars_all.setter
    def vars_all(self, vars: tuple[np.ndarray, np.ndarray, np.ndarray] |
             tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] |
             tuple[tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
        """Alias for data_all. Sets all data by calling setters for x,y and z.

        Parameters
        ----------
        vars : tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]
            A tuple of length 3 corresponding to (x,y,z). Each index can be comprosied of x, (x, xerr) or (x, xerr, xlabels).

        Raises
        ------
        TypeError
            Raises if vars is not a tuple of length three.
        """
        self.data_all = vars
        return
    
    @vars_all.deleter
    def vars_all(self) -> None:
        """Alias for x deleter. Resets all data.
        """
        del self.data_all
        return
    
    @abstractmethod
    def __sub__(self, x):
        """Abstract method for computing a difference in y and z.
        This method itself will account for differences in datalengths, or mismatches in x values between objects.
        Difference data is returned in the measurement_base.vars_all() format, but needs to be assigned, therefore an abstract method.
        
        Args:
            x (measurement_base): Another object whose data will be used to subtract against.
            
        Returns: 
            sub (Numpy NDarray): One numpy array of data (corresponding to the same format as measurement_base.vars_all()) corresponding to subtraction.
        """
        assert isinstance(x, measurement_base)
        # check that datalength of two arrays match.
        diff = self.ind_vars().shape[0] - x.ind_vars().shape[0]
        match = np.all(self.ind_vars() == x.ind_vars())
        ind_var_len = self.ind_vars().shape[1] if len(self.ind_vars().shape) > 1 else 1 #cover for single column variable case.
        if diff != 0 or not match:
            if not match:
                print("Independent values don't match, truncating data to largest matching independent set to perform subtraction...")
            
            # Axis sizes don't match size ensuring that values match along field.
            self_vars = self.vars_all()
            x_vars = x.vars_all()

            # Assuming data in 2D form.
            for i in range(ind_var_len): #for each independent variable (ie, number of columns)
                self_vars, x_vars = pylectric.signals.processing.trim_matching_fromsimilar(self_vars, x_vars, colN=i)  # iterate over data as many times as there are independent variables
            self_match = self_vars
            x_match = x_vars
        else:
            self_match = self.vars_all()
            x_match = x.vars_all()
        
        #Check independent variables actually match up as intended.
        assert np.all(self_match[:,0:ind_var_len] == x_match[:, 0:ind_var_len])
        
        #Copy independent variables, subtract all other variables.
        sub = self_match.copy()
        sub[:, ind_var_len:] = sub[:, ind_var_len:]  - x_match[:, ind_var_len:]
        return sub


    def to_DataFrame(self) -> pd.DataFrame:
        """Generates a Pandas DataFrame object, containing x,y data and using x,y labels if defined.

        Returns
        -------
        pd.DataFrame
            Variables x and y presented in a dataframe.
        """
        labels = self.labels
        return pd.DataFrame(self.array(), columns=labels if labels is not None else None)

    def to_DataFrame_all(self) -> pd.DataFrame:
        """Generates a Pandas DataFrame object, containing x,y,z data and using x,y,z labels if defined.

        Returns
        -------
        pd.DataFrame
            Variables x, y and z presented in a dataframe.
        """
        labels = self.labels_all
        return pd.DataFrame(self.array_all(), columns=labels if labels is not None else None)
    
    def to_DataFrame_x(self) -> pd.DataFrame:
        """Generates a Pandas DataFrame object, containing x data and errors and using labels if defined.
        
        Returns
        -------
        pd.DataFrame
            Variable x presented in a dataframe.
        """
        xlabels = self.xlabels
        xerr = self.xerr
        if xlabels is not None:
            labels = xlabels + [a + "_err" for a in xlabels] if xerr is not None else xlabels
        else:
            labels = None
        data = np.c_[self.x, xerr] if xerr is not None else self.x
        return pd.DataFrame(data, labels)
    
    def to_DataFrame_y(self) -> pd.DataFrame:
        """Generates a Pandas DataFrame object, containing y data and errors and using labels if defined.
        
        Returns
        -------
        pd.DataFrame
            Variable y presented in a dataframe.
        """
        ylabels = self.ylabels
        yerr = self.yerr
        if ylabels is not None:
            labels = ylabels + [a + "_err" for a in ylabels] if yerr is not None else ylabels
        else:
            labels = None
        data = np.c_[self.y, yerr] if yerr is not None else self.y
        return pd.DataFrame(data, labels)
    
    def to_DataFrame_z(self) -> pd.DataFrame:
        """Generates a Pandas DataFrame object, containing z data and errors and using labels if defined.
        
        Returns
        -------
        pd.DataFrame
            Variable z presented in a dataframe.
        """
        zlabels = self.zlabels
        zerr = self.zerr
        if zlabels is not None:
            labels = zlabels + [a + "_err" for a in zlabels] if zerr is not None else zlabels
        else:
            labels = None
        data = np.c_[self.y, zerr] if zerr is not None else self.z
        return pd.DataFrame(data, labels)
    
class measurement_graphable(measurement_base, metaclass=ABCMeta):
    """Expands measurement_base to add graphical binding functions.
    """
    
    ### Function TODO write
    # 1. Function to graph x,y data in all its permutations. 
    #    Static method, takes axes argument, and iterates along axis. 
    #    Also can apply labels. Requires labels to match provided x,y lengths.
    # 2. Function to take provided indexes, and plot a subset of the desired data.
    #    Static function, uses indexes, also applies to labels.
    # 3. Class function to use measurement data, labels, errors to generate such graphs.
    
    def _2D_graph(self,
                  xi: int | list[int] | None = None, 
                  yi: int | list[int] | None = None,
                  inc_z: bool = False,
                  labels: tuple[list[str], list[str]] = None,
                  ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] | None = None,
                  graph_method: function = mplaxes.Axes.scatter,
                  fig_size: tuple[float, float] | None = None,
                  **mpl_kwargs
                  ) -> Type[graphing.scalable_graph]:
        """_summary_
        
        
        Graphs and labels a x/y-axis of a list/array of axes given the indexes of x/y/z variables.
        Used in conjunction with _2D_graph, namely to be called afterward.

        Parameters
        ----------
        ax : mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes]
            Matplotlib axes, either a single or multiple in an array/list.
        xi : int | list[int] | None, optional
            A list or single index specifying which x variables to label, by default None (all x)
        yi : int | list[int] | None, optional
            A list or single index specifying which y variables to label, by default None (all y/z)
        inc_z : bool, optional
            Whether z variables are to be included in y indexes, by default False

        Raises
        ------
        AttributeError
            Raised in the event of a length mismatch, if ax is singular, but xlabels and ylabels are not singular.
        AttributeError
            Raised in the event of a length mismatch, if ax has less indexes than the pemutations of x/y.
        AttributeError
            Raised if ax is not a singular or list/array of matplotlib.axes.Axes.
        

        Parameters
        ----------
        xi : int | list[int] | None, optional
            _description_, by default None
        yi : int | list[int] | None, optional
            _description_, by default None
        inc_z : bool, optional
            _description_, by default False
        labels : tuple[list[str], list[str]], optional
            _description_, by default None
        ax : mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] | None, optional
            _description_, by default None
        graph_method : function, optional
            _description_, by default mplaxes.Axes.scatter
        fig_size : tuple[float, float] | None, optional
            _description_, by default None

        Returns
        -------
        Type[graphing.scalable_graph]
            _description_
        """
        
        ## Gather data and labels:
        if inc_z: # Check if zlabels/data are included.
            #labels
            if labels is not None:
                xl, yl = labels
            else:
                xl, yl, zl = self.labels_all
                yl = yl + zl if zl is not None else yl #pack z variables into y lists/arrays.
            #data
            x,y,z = self.data_all[0]
            y = np.c_[y,z] #pack z variables into y lists/arrays.
        else:
            #labels
            xl, yl = self.labels
            #data
            x,y = self.data[0]
        
        # If xi,yi provided, use indexing method.
        if xi is not None:
            x = self._2D_graph_index(x, xi)
            xl = self._2D_graph_index(xl, xi)
        if yi is not None: 
            y = self._2D_graph_index(y, yi)
            yl = self._2D_graph_index(yl, xi)
            
        # Generate graph
        ax = self._2D_graph_data(data=(x,y), labels = (xl, yl), ax=ax,
                                 fig_size=fig_size, graph_method=graph_method, **mpl_kwargs)
        
        # Wrap graph with scalable axis / helper label function.
        sg = self._2D_graph_wrapper(ax)
        
        return sg
        
    @staticmethod
    def _2D_graph_index(var: list | np.ndarray,
                        indexes: int | list[int]) -> list | np.ndarray:
        # Is a list? Useful for labels...
        if isinstance(var, list):
            if isinstance(indexes, list):
                var_subset = [var[indx] for indx in indexes] 
                # If single element, pull out of array.
                if len(var_subset) == 1:
                    var_subset = var_subset[0]
            else: #int
                var_subset = var[indexes]
        # An array - 2nd axis / columns are the index variables.
        elif isinstance(var, np.ndarray):
            if isinstance(indexes, list):
                var_subset = var[:,indexes] 
                # If single element, reduce dimension.
                if var_subset.shape[1] == 1:
                    var_subset = var_subset[:,0]
            else: #int
                var_subset = var[:,indexes]
        return var_subset
                
        
    
    @staticmethod
    def _2D_graph_data(data: tuple[np.ndarray, np.ndarray],
                  labels: tuple[list[str] | str, list[str] | str] | None = None,
                  ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] | None = None,
                  graph_method: function = mplaxes.Axes.scatter,
                  fig_size: tuple[float, float] | None = None,
                  **mpl_kwargs) -> mplaxes.Axes | npt.NDArray:
        """TODO: Double check this method works.
        """
        ## 1. Check format is valid requires (x,y) with numpy arrays.
        assert callable(graph_method)
        if not (isinstance(data, tuple) and len(data)==2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)):
            raise AttributeError("Data passed as an innapropriate format, requires np.ndarray(s) in the form (x,y)")
        
        ## 2. Unpack data.
        x,y = data
        # Check datalength matches for x and y.
        if x.shape[0] != y.shape[0]:
            raise AttributeError("Length of x data doesn't match length of y data.")
        # Get data shape, match to provided axes.
        xlen = x.shape[1] if len(x.shape) > 1 else 1
        ylen = y.shape[1] if len(y.shape) > 1 else 1
        
        ## 3. Check label and variable lengths match.
        if labels is not None:
            xl, yl = labels
            # x validity
            if isinstance(xl, list):
                assert len(xl) == xlen
            else:
                assert isinstance(xl, str) and xlen == 1
            # y validity
            if isinstance(yl, list):
                assert len(yl) == ylen
            else:
                assert isinstance(yl, str) and ylen == 1
        
        ## 3. Create plots if not provided.
        if fig_size is None:
            # If figsize not provided, set figsize to a standard multiple of the shape.
            fig_size = (xlen * 4, ylen * 3)    
        ax = plt.subplots(xlen,ylen, fig_size=fig_size)[1] if ax is None else ax
        
        ## 4. Label/graph depending on single axis or multiple plots.
        # Single Plot
        if isinstance(ax, mplaxes.Axes):
            # Labels
            if labels is not None:
                # Single axis requires singular labels
                if isinstance(xl, str) and isinstance(yl, str):
                    ax.set_xlabel(xl)
                    ax.set_ylabel(yl)
                else:
                    raise AttributeError("x/y/z labels are multiple valued, or xi/yi indexes are multiple valued, incompatible with a single Axes.")
            # Plot
            graph_method(ax, x=x, y=y, **mpl_kwargs)
        
        # Multiple plots
        elif ((isinstance(ax, list) or isinstance(ax, np.ndarray))
              and isinstance(xl, list) and isinstance(yl, list)):    
            # Check shape/length matches length of x/y lists.
            if ((isinstance(ax, np.ndarray) and #Array
                (np.prod(ax.shape) >= xlen * ylen)) #Ax Lengths > x,y var lengths
                or (isinstance(ax, list) # List
                and len(ax) >= xlen * ylen) #Ax Lengths > x,y var lengths
            ):
                if isinstance(ax, np.ndarray):
                    for i in range(xlen): #iterate over xy data/labels
                        for j in range(ylen):
                            k = i * len(yl) + j #convert label i,j to array item (if mismatch)
                            l,m = (k % ax.shape[1], round(k/ax.shape[1])) #array item to array indexes
                            # Data
                            graph_method(ax[l,m], x=x[:,i], y=y[:,j], **mpl_kwargs)
                            # Labels
                            if labels is not None:
                                # Set each x and y label iterating over 
                                ax[l,m].set_xlabel(xl[i])
                                ax[l,m].set_ylabel(yl[j])
                else: #list
                    for i in range(xlen):
                        for j in range(ylen):
                            # Data
                            graph_method(ax[i + j * ylen], x=x[:,i], y=y[:,j], **mpl_kwargs)
                            # labels:
                            if labels is not None:
                                ax[i + j * ylen].set_xlabel(xl[i])
                                ax[i + j * ylen].set_ylabel(yl[j])
            else:
                raise AttributeError("Number of axes is less than number of x & y(/z) combinations.")
        else:
            raise AttributeError("ax is not a matplotlib.axes.Axes or list/np.ndarray of such items.")
        return ax
    
    @staticmethod
    def _2D_graph_unc(data: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
                  labels: tuple[list[str] | str, list[str] | str] | None = None,
                  ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] | None = None,
                  graph_method: function = mplaxes.Axes.fill_between,
                  fig_size: tuple[float, float] | None = None,
                  **mpl_kwargs) -> mplaxes.Axes | npt.NDArray:
        """TODO: Double check this method works.
        """
        ## 1. Check format is valid requires (x,y) with numpy arrays.
        assert callable(graph_method)
        if not (isinstance(data, tuple) and len(data)==2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)):
            raise AttributeError("Data passed as an innapropriate format, requires np.ndarray(s) in the form (x,y)")
        
        ## 2. Unpack data.
        x,y = data
        y1,y2 = y
        # Check datalength matches for x and y.
        if x.shape[0] != y1.shape[0]:
            raise AttributeError("Length of x data doesn't match length of y data.")
        if y1.shape != y2.shape:
            raise AttributeError("y1 and y2 don't match shape.")
        # Get data shape, match to provided axes.
        xlen = x.shape[1] if len(x.shape) > 1 else 1
        ylen = y1.shape[1] if len(y1.shape) > 1 else 1
        
        ## 3. Check label and variable lengths match.
        if labels is not None:
            xl, yl = labels
            # x validity
            if isinstance(xl, list):
                assert len(xl) == xlen
            else:
                assert isinstance(xl, str) and xlen == 1
            # y validity
            if isinstance(yl, list):
                assert len(yl) == ylen
            else:
                assert isinstance(yl, str) and ylen == 1
        
        ## 3. Create plots if not provided.
        if fig_size is None:
            # If figsize not provided, set figsize to a standard multiple of the shape.
            fig_size = (xlen * 4, ylen * 3)    
        ax = plt.subplots(xlen,ylen, fig_size=fig_size)[1] if ax is None else ax
        
        ## 4. Label/graph depending on single axis or multiple plots.
        # Single Plot    
        if isinstance(ax, mplaxes.Axes):
            # Labels
            if labels is not None:
                # Single axis requires singular labels
                if isinstance(xl, str) and isinstance(yl, str):
                    ax.set_xlabel(xl)
                    ax.set_ylabel(yl)
                else:
                    raise AttributeError("x/y/z labels are multiple valued, or xi/yi indexes are multiple valued, incompatible with a single Axes.")
            # Plot
            graph_method(ax, x=x, y1=y1, y2=y2, **mpl_kwargs)
        
        # Multiple plots
        elif ((isinstance(ax, list) or isinstance(ax, np.ndarray))
              and isinstance(xl, list) and isinstance(yl, list)):    
            # Check shape/length matches length of x/y lists.
            if ((isinstance(ax, np.ndarray) and #Array
                (np.prod(ax.shape) >= xlen * ylen)) #Ax Lengths > x,y var lengths
                or (isinstance(ax, list) # List
                and len(ax) >= xlen * ylen) #Ax Lengths > x,y var lengths
            ):
                if isinstance(ax, np.ndarray):
                    for i in range(xlen): #iterate over xy data/labels
                        for j in range(ylen):
                            k = i * len(yl) + j #convert label i,j to array item (if mismatch)
                            l,m = (k % ax.shape[1], round(k/ax.shape[1])) #array item to array indexes
                            # Data
                            graph_method(ax[l,m], x=x[:,i], y1=y1[:,j], y2=y2[:,j], **mpl_kwargs)
                            # Labels
                            if labels is not None:
                                # Set each x and y label iterating over 
                                ax[l,m].set_xlabel(xl[i])
                                ax[l,m].set_ylabel(yl[j])
                else: #list
                    for i in range(xlen):
                        for j in range(ylen):
                            # Data
                            graph_method(ax[i + j * ylen], x=x[:,i], y1=y1[:,j], y2=y2[:,j], **mpl_kwargs)
                            # labels:
                            if labels is not None:
                                ax[i + j * ylen].set_xlabel(xl[i])
                                ax[i + j * ylen].set_ylabel(yl[j])
            else:
                raise AttributeError("Number of axes is less than number of x & y(/z) combinations.")
        else:
            raise AttributeError("ax is not a matplotlib.axes.Axes or list/np.ndarray of such items.")
        return ax
    
    
    @staticmethod
    def _2D_graph_wrapper(figax: mplaxes.Axes | npt.ArrayLike[mplaxes.Axes] | mplfigure.Figure) -> Type[graphing.scalable_graph]:
        """Wraps a figure, or a set of axes. Allows scaling and axis functions.
        This function should be used after calling _2D_graph, and can be overridden for different measurement 
        objects with well defined x/y axis, such as in electrical transport measurements.

        Parameters
        ----------
        figax : mplaxes.Axes | npt.ArrayLike[mplaxes.Axes] | mplfigure.Figure
            A single or multiple Axes, either passed as a figure or list or singular.

        Returns
        -------
        Type[graphing.scalable_graph]
            Wrapper object containing fig/ax.
        """
        return graphing.scalable_graph(figax)        
    
    def graph_plot(self, ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] = None, 
                   xi: int | list[int] | None = None, yi: int | list[int] | None = None,
                   inc_z: bool = False) -> Type[graphing.scalable_graph]:
        """Generates a matplotlib.pyplot.plot graph using object x,y data.
        Plots all x/y (and z if inc_z = True) combinations unless indexes are specified.

        Parameters
        ----------
        ax : mplaxes.Axes, optional
            , by default None
        xi : int | list[int] | None, optional
            _description_, by default None
        yi : int | list[int] | None, optional
            _description_, by default None
        inc_z : bool, optional
            _description_, by default False

        Returns
        -------
        Type[graphing.scalable_graph]
            _description_
        """
        x = self.x if xi is None else self.x[:,xi]
        if inc_z:
            y = np.c_[self.y, self.z] if yi is None else np.c_[self.y, self.z][:,yi]
        else:
            y = self.y if yi is None else self.y[:,yi]
        return self._2D_graph((x, y), ax=ax, graph_method=plt.plot, inc_z=inc_z)

    def graph_scatter(self, ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] = None, 
                      xi: int | list[int] | None = None, yi: int | list[int] | None = None,
                      inc_z: bool = False) -> Type[graphing.scalable_graph]:
        x = self.x if xi is None else self.x[:,xi]
        y = self.y if yi is None else self.y[:,yi]
        return self._2D_graph(data=(x, y), ax=ax, graph_method=plt.scatter, inc_z=inc_z)
    
    def graph_errorbar(self, ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] = None, 
                       xi: int | None = None, yi: int | None = None,
                       inc_z: bool = False) -> Type[graphing.scalable_graph]:
        x = self.x if xi is None else self.x[:,xi]
        y = self.y if yi is None else self.y[:,yi]
        xerr = self.xerr if xi is None else self.xerr[:,xi]
        yerr = self.yerr if yi is None else self.yerr[:,yi]
        return self._2D_graph((x, y), ax=ax, graph_method=plt.errorbar, **{"xerr":xerr, "yerr":yerr}, inc_z=inc_z)
    
    def graph_plot_error_fill(self, ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] = None, 
                              xi: int | None = None, yi: int | None = None,
                              inc_z: bool = False) -> Type[graphing.scalable_graph]:
        x, xerr = self.x
        y, yerr = self.y
        x = x if xi is None else x[:,xi]
        y = y if yi is None else y[:,yi]
        xerr = xerr if xi is None else xerr[:,xi]
        yerr = yerr if yi is None else yerr[:,yi]
        # use regular data for x/y plot
        sg = self._2D_graph((x, y), ax=ax, graph_method=plt.plot)
        ax = sg.ax
        # use modified data for filled uncertainty area
        yupper = y+yerr
        ylower = y-yerr
        self._2D_graph_unc((x, (ylower, yupper)), ax=ax)
        return sg
    
    def sweep_arrow_location(self, i):
        if len(self.ind_vars().shape) == 1:
            #single x variable
            return pylectric.signals.feature_detection.find_arrow_location(xdata=self.ind_vars(),ydata=self.dep_vars()[:,i])
    
    @staticmethod
    def _plot_2Ddata(data, ax=None, **mpl_kwargs):
        return measurement_base._graph_2Ddata(data, ax, scatter=False, **mpl_kwargs)

    @staticmethod
    def _scatter_2Ddata(data, ax=None, **mpl_kwargs):
        return measurement_base._graph_2Ddata(data, ax, scatter=True, **mpl_kwargs)

    @staticmethod
    def _plot_3Ddata(data, ax=None, **mpl_kwargs):
        return measurement_base._graph_3Ddata(data, ax, scatter=False, **mpl_kwargs)
    @staticmethod
    def _scatter_3Ddata(data, ax=None, **mpl_kwargs):
        return measurement_base._graph_3Ddata(data, ax, scatter=True, **mpl_kwargs)

    @staticmethod
    def _graph_2Ddata(data, axes=None, scatter=False, **mpl_kwargs) -> graphing.transport_graph:
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
            assert isinstance(axes, mplaxes.Axes) or (isinstance(axes, (list, tuple, np.ndarray)) and np.all([isinstance(ax, mplaxes.Axes) for ax in axes]))
        # Requirements for Data
        assert isinstance(data, np.ndarray) 
        assert len(data.shape) == 2
        nd = data.shape[-1] - 1
        
        
        if axes is None:
            fig,axes = plt.subplots(nd, 1)
            # Graphing wrapper
            tg = graphing.transport_graph(axes)
            fig.set_size_inches(
                w=journals.acsNanoLetters.maxwidth_2col, h=3*nd) #3 inches times number of plots.
        elif len(axes) != nd:
            raise AttributeError("Length of axes does not match data dimension.")
        else:
            tg = graphing.transport_graph(axes)
            
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
    
class set_base(metaclass=ABCMeta):
    """Allows binding of a series of measurement_base typed objects,
    and allows the projection of one variable or measurement as a secondary axis.
    Does not perform a deep copy on measurement_base objects, rather just references to them."""
    
    def __init__(self, instances: list[Type[measurement_base]]) -> None:
        super().__init__()
        self._measurements = instances        
        return
    
    def data(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Returns the concatenation of all sets of data in a tuple (X,Y) where X/Y are another tuple with elements (variables, uncertainties).

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple of concatenated datasets (variables, uncertainties). 
            Both variables and uncertainties are tuples of np.ndarrays, as (x,y) and (xerr, yerr) respectively.
        """
        datasets = [inst.data() for inst in self._measurements]
        X = [dset[0] for dset in datasets] # x, xerr
        Y = [dset[1] for dset in datasets] # y, yerr
        x = np.r_[*[dset[0] for dset in X]]
        y = np.r_[*[dset[0] for dset in Y]]
        xerr = np.r_[*[dset[1] for dset in X]]
        yerr = np.r_[*[dset[1] for dset in Y]]
        return ((x, xerr), (y, yerr))
    
    def data_all(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Returns the concatenation of all sets of data in a tuple (variables, uncertainties).

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple of concatenated datasets (variables, uncertainties). 
            Both variables and uncertainties are tuples of np.ndarrays, as (x,xerr) and (xerr, yerr) respectively.
        """
        datasets = [inst.data_all() for inst in self._measurements]
        X = [dset[0] for dset in datasets] # x, xerr
        Y = [dset[1] for dset in datasets] # y, yerr
        Z = [dset[2] for dset in datasets] # y, yerr
        x = np.r_[*[dset[0] for dset in X]]
        y = np.r_[*[dset[0] for dset in Y]]
        z = np.r_[*[dset[0] for dset in Z]]
        xerr = np.r_[*[dset[1] for dset in X]]
        yerr = np.r_[*[dset[1] for dset in Y]]
        zerr = np.r_[*[dset[1] for dset in Z]]
        return ((x, xerr), (y, yerr), (z, zerr))
    
    def data_by_param(self, param_name: str, func: function = None) -> list:
        param_list = []
        # Check param is in every dataset
        for i in range(len(self._measurements)):
            if param_name not in self._measurements[i].params:
                raise KeyError("Measurement index #", i, "does not contain'", param_name, "'in it's parameter set.")
            else:
                if func is not None:
                    param_list.append(func(self._measurements[i].params[param_name]))
                else:
                    param_list.append(self._measurements[i].params[param_name])
        return param_list
        
    def data_by_variable_index(self, i: int, func: function = None) -> tuple[np.ndarray, np.ndarray]:
        variable_list = []
        errors_list = []
        for j in range(len(self._measurements)):
            meas = self._measurements[j]
            xl = len(meas.x)
            yl = len(meas.y)
            zl = len(meas.z) if meas.z is not None else 0 #sometimes z not defined.
            # Get item
            if i >= xl + yl and i < xl + yl + zl:
                i -= xl + yl
                item = (meas.z[:,i], meas.zerr[:,i])
            elif i >= xl and i < xl + yl:
                i -= xl
                item = (meas.y[:,i], meas.yerr[:,i]) 
            elif i < xl:
                item = (meas.x[:,i], meas.xerr[:,i]) 
            else: # i >= xl + yl + zl
                raise IndexError("Variable index ", i, "is outside range of the total length of x,y,z variable lists with respective lengths (", str(xl), ",", str(yl), ",", zl ,").")
                
            # Process item with function
            if func is not None:
                if isinstance(func, callable):
                    item = (func(item[0]), func(item[1]))
                else:
                    raise AttributeError(func, " is not a np.ndarray operator.")    
            variable_list.append(item[0])
            errors_list.append(item[1])
        return np.array(variable_list), np.array(errors_list)
        
    def _duplicates(strlist: list[str]) -> list[str]:
        counter = {}
        duplicates = []
        for i in range(len(strlist)):
            if strlist[i] not in counter:
                counter[strlist[i]] = 1
            else:
                counter[strlist[i]] += 1
                if counter[strlist[i]] == 2:
                    duplicates.append(strlist[i])
        # duplicates = [s for s in counter.keys() if counter[s] > 1] #alternative
        return duplicates
    
    def _index(strlist: list[str], val: str) -> None | int:
        for i in range(len(strlist)):
            if strlist[i] == val:
                return i
        return None
        
    def data_by_variable_label(self, var_name: str, func: callable[[npt.NDArray, npt.NDArray], tuple(npt.NDArray, npt.NDArray)] = None) -> tuple[npt.NDArray, npt.NDArray]:
        variable_list = []
        errors_list = []
        for i in range(len(self._measurements)):
            meas = self._measurements[i]
            xlabels, ylabels, zlabels = meas.xlabels, meas.ylabels, meas.zlabels
            all_labels = xlabels + ylabels + zlabels
            xl, yl, zl = len(xlabels), len(ylabels), len(zlabels)
            
            # 1 Check for duplicates - then cannot identify by label
            duplicates = set_base._duplicates(all_labels)
            if len(duplicates) > 0:
                raise AttributeError("Duplicate labels across x,y(,z) variables: ", duplicates, 
                                     " - cannot isolate variable data.")
            else:    
                li = set_base._index(all_labels, var_name)
                if li is None:
                    raise KeyError(var_name, "is not found in x,y(,z) labels.")
                # Get item
                if li >= xl + yl and li < xl + yl + zl:
                    li -= xl + yl
                    item = (meas.z[:,li], meas.zerr[:,li]) 
                elif li >= xl and li < xl + yl:
                    li -= xl
                    item = (meas.y[:,li], meas.yerr[:,li]) 
                elif li < xl:
                    item = (meas.x[:,li], meas.xerr[:,li]) 
                else: # li > xl + yl + zl
                    raise IndexError("Index provided", li, "is outside the length of the label lists,", xl + yl + zl)
                
                # Process item with function
                if func is not None:
                    if isinstance(func, callable):
                        item = (func(item[0]), func(item[1]))
                    else:
                        raise AttributeError(func, " is not a np.ndarray operator.")    
                variable_list.append(item[0])
                errors_list.append(item[1])
        return np.array(variable_list), np.array(errors_list)
    
        
    def data_by_attribute(self, attr_name: str, attr_unc_name: str = None, func: function = None):
        variable_list = []
        errors_list = [] if attr_unc_name is not None else None
        for i in range(len(self._measurements)):
            meas = self._measurements[i]
            
            # Check for attribute
            if hasattr(meas, attr_name):
                item0 = getattr(meas, attr_name)
            else:
                raise AttributeError("Measurement object #", i, "does not have the attribute '", attr_name, "'.")
                
            item1 = None
            if attr_unc_name is not None and hasattr(meas, attr_unc_name):
                item1 = getattr(meas, attr_unc_name)
            elif attr_unc_name is not None:
                raise AttributeError("Measurement object #", i, "does not have the uncertainty attribute '", attr_unc_name, "'.")
            else:
                # attr_unc_name is None, does not want 
                pass
            
            # Process item with function
            if func is not None:
                if isinstance(func, callable):
                    item = (func(item[0]), func(item[1]))
                else:
                    raise AttributeError(func, " is not a np.ndarray operator.")    
            variable_list.append(item[0])
            errors_list.append(item[1])
        return np.array(variable_list), np.array(errors_list)
        
