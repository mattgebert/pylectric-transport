# Coding
from overrides import override
from abc import abstractmethod, ABCMeta
from typing import Type, Sequence, Self
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
    """ An abstract class to expand and bind functions to geometric objects.
        All data is timeseries data, represented by rows for various variable columns.
        Data also should implement a sweep direction.
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

    @abstractmethod
    def __copy__(self) -> Type[Self]:
        """ Creates a deep clone of base datasets.

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
        # Check vars input and process accordingly:
        if isinstance(vars, np.ndarray):
            self._x = vars.copy()
            self._xerrs = None
            newx=True
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._x = vars[0].copy()
            l = len(vars)
            if l == 1:
                newx=True
            elif isinstance(vars[1], np.ndarray):
                self._xerrs = vars[1].copy()
                if l == 2:
                    newx=True
                elif l == 3 and isinstance(vars[2], list):
                    self._xerrs = vars[1].copy()
                    self._xlabels = vars[2].copy()
                    newx=True
        # Setup sweep direction, and enforce x mask on xerrs (as fast as checking).
        if newx:
            self.sweep_dir = measurement_base._determine_sweep_direction(self._x)
            if isinstance(self._x, np.ma.MaskedArray) and isinstance(self._xerrs, np.ma.MaskedArray):
                self._xerrs.mask = self._x.mask
            return
        else:
            self._x = None
            self._xerrs = None
            raise TypeError("Set either with a np.ndarray or a tuple with (x, xerr) or (x, xerr, xlabels).")
    
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
        vars : npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list]
            2D array with values for y, or a tuple with (y, yerr) or (y, yerr, ylabels).
            If labels is not provided, new y must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        newy = False # Perform additional operations if vars is valid format.
        # Check vars input and process accordingly:
        if isinstance(vars, np.ndarray):
            self._y = vars.copy()
            self._yerrs = None
            newy=True
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._y = vars[0].copy()
            l = len(vars)
            if l == 1:
                newy=True
            elif isinstance(vars[1], np.ndarray):
                self._yerrs = vars[1].copy()
                if l == 2:
                    newy=True
                elif l == 3 and isinstance(vars[2], list):
                    self._yerrs = vars[1].copy()
                    self._ylabels = vars[2].copy()
                    newy=True
        # Setup sweep direction, and enforce y mask on yerrs (as fast as checking).
        if newy:
            self.sweep_dir = measurement_base._determine_sweep_direction(self._y)
            if isinstance(self._y, np.ma.MaskedArray) and isinstance(self._yerrs, np.ma.MaskedArray):
                self._yerrs.mask = self._y.mask
            return
        else:
            self._y = None
            self._yerrs = None
            raise TypeError("Set either with a np.ndarray or a tuple with (y, yerr) or (y, yerr, ylabels).")
    
    
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
        vars : npt.NDArray | tuple[npt.NDArray,npt.NDArray] | tuple[npt.NDArray, npt.NDArray, list]
            2D array with values for z, or a tuple with (z, zerr) or (z, zerr, zlabels).
            If labels is not provided, new z must match existing labels length.
        
        Raises
        ------
        TypeError
            Raises type error if assignment doesn't match required format.
        """
        newz = False # Perform additional operations if vars is valid format.
        # Check vars input and process accordingly:
        if isinstance(vars, np.ndarray):
            self._z = vars.copy()
            self._zerrs = None
            newz=True
        elif isinstance(vars, tuple) and isinstance(vars[0], np.ndarray):
            self._z = vars[0].copy()
            l = len(vars)
            if l == 1:
                newz=True
            elif isinstance(vars[1], np.ndarray):
                self._zerrs = vars[1].copy()
                if l == 2:
                    newz=True
                elif l == 3 and isinstance(vars[2], list):
                    self._zerrs = vars[1].copy()
                    self._zlabels = vars[2].copy()
                    newz=True
        # Setup sweep direction, and enforce z mask on zerrs (as fast as checking).
        if newz:
            self.sweep_dir = measurement_base._determine_sweep_direction(self._z)
            if isinstance(self._z, np.ma.MaskedArray) and isinstance(self._zerrs, np.ma.MaskedArray):
                self._zerrs.mask = self._z.mask
            return
        else:
            self._z = None
            self._zerrs = None
            raise TypeError("Set either with a np.ndarray or a tuple with (z, zerr) or (z, zerr, zlabels).")

    
    @property
    def data(self) -> tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]:
        """Returns independent and dependent variables and their errors, but not extra-dependent variables.

        Returns
        -------
        tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]
            A tuple with elements corresponding to values and uncertainty ((x,y), (xerr,yerr)).
            Each element is a tuple corresponding to independent and dependent variables.
            
        """
        x, xerr = self.x
        y, yerr = self.y
        return ((x,y), (xerr,yerr))

    @data.setter
    def data(self, vars: tuple[npt.NDArray, npt.NDArray] |
             tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]] |
             tuple[tuple[npt.NDArray, npt.NDArray, list[str]], tuple[npt.NDArray, npt.NDArray, list[str]]]) -> None:
        """Sets independent and dependent variables, but not extra-dependent variables.

        Parameters
        ----------
        vars : tuple[npt.NDArray, npt.NDArray] | tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]] | tuple[tuple[npt.NDArray, npt.NDArray, list], tuple[npt.NDArray, npt.NDArray, list[str]]]
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
    

    
    @property
    def data_all(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """ Equivalent to independent variables, dependent variables and extra variables.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            A tuple with elements corresponding to values and uncertainty ((x,y,z), (xerr,yerr,zerr)).
            
        """
        x, xerr = self.x, self.xerrs
        y, yerr = self.y, self.yerrs
        z, zerr = self.z, self.zerrs
        return ((x,y,z), (xerr,yerr,zerr))

    @data_all.setter
    def data_all(self, vars:tuple[np.ndarray, np.ndarray, np.ndarray] |
             tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] |
             tuple[tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
        """ Sets all data by calling setters for x,y and z.

        Parameters
        ----------
        vars : tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | tuple[tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list], tuple[np.ndarray, np.ndarray, list]]
            A tuple of length 3 corresponding to (x,y,z). Each index can be comprosied of x, (x, xerr) or (x, xerr, xlabels).

        Raises
        ------
        TypeError
            _description_
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
 
    def array(self) -> npt.NDArray:
        return 
 
    def full_array(self) -> npt.NDArray:
        """Returns all variables in a single 2D numpy array.

        Returns
        -------
        npt.NDArray
            An array of all variables of x,y,z.
        """
        return np.c_[self.x, self.y, self.z]

    @property
    def xerrs(self) -> np.ndarray | None:
        """ Errors of x variables.

        Returns
        -------
        np.ndarray | None
            A 2D array of uncertainties of each measurement in x, or None. 
            Columns correspond to different variables.
            Requires the same shape as x.
        """
        return self._xerrs
    
    @xerrs.setter
    def xerrs(self, errs: npt.NDArray) -> None:
        """ Sets new values for errors in x.

        Parameters
        ----------
        errs : npt.NDArray
            A 2D array of uncertainties of each measurement in x, or None. 
            Columns correspond to different variables.
            Requires the same shape as x.

        Raises
        ------
        AttributeError
            Raises an excpetion if the shapes of x and errs don't match.
        """
        if errs is None:
            self._zerrs = None
            return
        if self._x.shape == errs.shape:
            self._xerrs = errs.copy()
            return
        else:
            raise AttributeError("Errs doesn't match shape of x.")

    @property
    def yerrs(self) -> np.ndarray | None:
        """ Errors of y variables.

        Returns
        -------
        np.ndarray | None
            A 2D array of uncertainties of each measurement in y, or None. 
            Columns correspond to different variables.
            Requires the same shape as y.
        """
        return self._xerrs
    
    @yerrs.setter
    def yerrs(self, errs: npt.NDArray | None) -> None:
        """ Sets new values for errors in y.

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
            self._yerrs = None
            return
        elif self._y.shape == errs.shape:
            self._yerrs = errs.copy()
            return
        else:
            raise AttributeError("Errs doesn't match shape of y.")
        
    @property
    def zerrs(self) -> np.ndarray | None:
        """ Errors of z variables.

        Returns
        -------
        np.ndarray | None
            A 2D array of uncertainties of each measurement in z, or None. 
            Columns correspond to different variables.
            Requires the same shape as z.
        """
        return self._zerrs
    
    @zerrs.setter
    def zerrs(self, errs: npt.NDArray | None) -> None:
        """ Sets new values for errors in z.

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
            self._zerrs = None
            return
        if self._z.shape == errs.shape:
            self._zerrs = errs.copy()
            return
        else:
            raise AttributeError("Errs doesn't match shape of z.")

    def full_array_errs(self) -> npt.NDArray:
        """Returns all errors of variables in a single 2D numpy array.

        Returns
        -------
        npt.NDArray
            An array of all errors of variables of x,y,z.
        """
        return np.c_[self.xerrs, self.yerrs, self.zerrs]

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
    def xlabels(self, labels: list[str]) -> None:
        """Setter for xlabels, index matched to columns of x.

        Parameters
        ----------
        labels : list[str] | str
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
        else:
            raise AttributeError("Length of provided labels doesn't match number of x columns.")
    
    
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
    def ylabels(self, labels: list[str]) -> None:
        """Setter for ylabels, index matched to columns of y.

        Parameters
        ----------
        labels : list[str] | str
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
        else:
            raise AttributeError("Length of provided labels doesn't match number of y columns.")



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
    def zlabels(self, labels: list[str] | str) -> None:
        """Setter for zlabels, index matched to columns of z.

        Parameters
        ----------
        labels : list[str] | str
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
        else:
            raise AttributeError("Length of provided labels doesn't match number of z columns.")

    def labels(self) -> list[str]:
        """Returns x,y labels in a single, consecutive list.

        Returns
        -------
        list[str]
            List of strings describing each column for x,y variables.
        """
        return self.xlabels + self.ylabels

    def all_labels(self) -> list[str]:
        """Returns all x,y,z labels in a single, consecutive list.

        Returns
        -------
        list[str]
            List of strings describing each column for x,y,z variables.
        """
        return self.xlabels + self.ylabels + self.zlabels

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
        """Sets the mask for x and xerrs. 
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
                if isinstance(self._xerrs, np.ma.MaskedArray):
                    self._xerrs.mask=mask
                else:
                    self._xerrs = np.ma.MaskedArray(self._xerrs, mask)
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
            self._xerrs = np.array(self._xerrs)
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
        """Sets the mask for y and yerrs. 
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
                if isinstance(self._yerrs, np.ma.MaskedArray):
                    self._yerrs.mask=mask
                else:
                    self._yerrs = np.ma.MaskedArray(self._yerrs, mask)
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
            self._yerrs = np.array(self._yerrs)
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
        """Sets the mask for z and zerrs. 
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
                if isinstance(self._zerrs, np.ma.MaskedArray):
                    self._zerrs.mask=mask
                else:
                    self._zerrs = np.ma.MaskedArray(self._zerrs, mask)
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
            self._zerrs = np.array(self._zerrs)
        return



    @property
    def independent_vars(self) -> tuple[np.ndarray, np.ndarray]:
        """Alias for x. Returns the independent variable(s).

        Returns:
            (numpy.ndarray, numpy.ndarray): A tuple of two 2D data arrays corresponding to (values, errors). Errors may be None or NaN.
        """
        return self.x()
    
    @independent_vars.setter
    def independent_vars(self, vars: npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
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
        self.x = vars
        return
    
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
    def dependent_vars(self, vars: npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
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
    def extra_vars(self, vars: npt.NDArray | tuple[np.ndarray,np.ndarray] | tuple[np.ndarray, np.ndarray, list]) -> None:
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
             tuple[tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, np.ndarray, list[str]]]) -> None:
        return self.data_all(vars)
    
    
    @abstractmethod
    def __sub__(self, x):
        """ Abstract method for computing a difference in y and z.
        This method itself will account for differences in datalengths, or mismatches in x values between objects.
        Difference data is returned in the measurement_base.all_vars() format, but needs to be assigned, therefore an abstract method.
        
        Args:
            x (measurement_base): Another object whose data will be used to subtract against.
            
        Returns: 
            sub (Numpy NDarray): One numpy array of data (corresponding to the same format as measurement_base.all_vars()) corresponding to subtraction.
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
        labels = self.labels
        return pd.DataFrame(self.full_array(), columns=labels if labels is not None else None)

    def to_DataFrame_all(self):
        labels = self.labels
        return pd.DataFrame(self.full_array(), columns=labels if labels is not None else None)
    
    def to_DataFrame_x(self):
        
    
class graphable_measurement(measurement_base, metaclass=ABCMeta):
    """ Expands measurement_base to add graphical binding functions.
    
    """
    
    @staticmethod
    def _1D_wrapper(figax: mplaxes.Axes | npt.NDArray[] | mplfigure.Figure)
    
    @staticmethod
    def _1D_graph(data: tuple[np.ndarray, np.ndarray],
                  ax: mplaxes.Axes | list[mplaxes.Axes] | np.ndarray[mplaxes.Axes] | None = None,
                  graph_method: function = mplaxes.Axes.plot, 
                  fig_size: tuple[float, float] | None = None,
                  **mpl_kwargs) -> tuple[mplfigure.Figure, mplaxes.Axes | np.ndarray[mplaxes.Axes]]:
        assert callable(graph_method)
        # Check format is valid requires (x,y) with numpy arrays.
        if not (isinstance(data, tuple) and len(data)==2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)):
            raise AttributeError("Data passed as an innapropriate format, requires np.ndarray(s) in the form (x,y)")
        
        # Arrange data:
        x,y = data
        # Get data shape, match to provided axes.
        xlen = x.shape[1] if len(x.shape) > 1 else 1
        ylen = y.shape[1] if len(y.shape) > 1 else 1
        
        # 2D ax array/list
        if (hasattr(ax, "shape") and np.prod(ax.shape) == xlen * ylen
            ) or hasattr(ax, "__len__") and len(ax)  == xlen*ylen:
            # mutliple axes
            if hasattr(ax, "shape"):
                # If figsize not provided, set figsize to a standard multiple of the shape.
                if fig_size is None:
                    fig_size = (xlen * 4, ylen * 3)
                # Create figure/axis
                fig, ax = plt.subplots(xlen,ylen, fig_size=fig_size) if ax is None else (ax.get_figure(), ax)
                # Generate plots
                for i in range(xlen):
                    xi = x[:,i] if xlen > 1 else x
                    for j in range(ylen):
                        yi = y[:,j] if ylen > 1 else y
                        graph_method(ax[i, j], x=xi, y=yi, **mpl_kwargs)
            else:
                for i in range(xlen):
                    xi = x[:,i] if xlen > 1 else x
                    for j in range(ylen):
                        yi = y[:,j] if ylen > 1 else y
                        graph_method(ax[i*xlen + j], x=xi, y=yi, **mpl_kwargs)
        # Singular ax
        elif isinstance(ax, mplaxes.Axes) and xlen==1 and ylen==1:
            graph_method(ax, x=x, y=y, **mpl_kwargs)
        else:
            raise AttributeError("Number of provided axes doesn't match x*y variables.")
        return fig, ax
    
    def _graphwrapper(ax: mplaxes.Axes | ):
    
    
    def graph_plot(self, ax: mplaxes.Axes = None, xi: int | None = None, yi: int | None = None) -> graphing.transport_graph:
        x = self.x if xi is None else self.x[:,xi]
        y = self.y if yi is None else self.y[:,yi]
        return measurement_base._1D_graph((x, y), ax=ax, graph_method=plt.plot)

    def graph_scatter(self, ax: mplaxes.Axes = None, xi: int | None = None, yi: int | None = None) -> graphing.transport_graph:
        x = self.x if xi is None else self.x[:,xi]
        y = self.y if yi is None else self.y[:,yi]
        return measurement_base._1D_graph(data=(x, y), ax=ax, graph_method=plt.scatter)
    
    def graph_errorbar(self, ax: mplaxes.Axes = None, xi: int | None = None, yi: int | None = None) -> graphing.transport_graph:
        x, xerr = self.x
        y, yerr = self.y
        x = x if xi is None else x[:,xi]
        y = y if yi is None else y[:,yi]
        xerr = xerr if xi is None else xerr[:,xi]
        yerr = yerr if yi is None else yerr[:,yi]
        return measurement_base._1D_graph((x, y), ax=ax, graph_method=plt.errorbar)
    
    def graph_error_fill()
        return
    
    
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
    
    @staticmethod
    def _graph_3Ddata(data, axes=None, scatter=False, **mpl_kwargs) -> graphing.transport_graph:
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
        tg = graphing.transport_graph(axes)
        
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
        measurement_base._data_compatability(vari, vard)
        if vare is not None:
            measurement_base._data_compatability(vari, vare)
            data = np.c_[vari, vard, vare]
        else:
            data = np.c_[vari, vard]
            
        li = len(vari.shape)
        if li == 1:
            tg = measurement_base._graph_2Ddata(data, axes, scatter, **mpl_kwargs) 
        elif li == 2 or li == 3:
            tg = measurement_base._graph_3Ddata(data, axes, scatter, **mpl_kwargs)
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
            tg = measurement_base._graph_2Ddata(data, axes, scatter, **mpl_kwargs)
        elif li == 2 or li == 3:
            data = np.c_[vari, vard]
            tg = measurement_base._graph_3Ddata(data, axes, scatter, **mpl_kwargs)
        else:
            raise AttributeError("Too many independent variables")

        # if 'label' in mpl_kwargs:
        #     for ax in tg.ax:
        #         lgnd = ax.legend()
        #         for handle in lgnd.legendHandles:
        #             handle.set_sizes([10])
        return tg
    

class measurement_base_dataseries(measurement_base):
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
        measurement_base._data_compatability(indvars, value)
        data = np.c_[indvars, value]
        tg = measurement_base._graph_2Ddata(data, ax, scatter, **mpl_kwargs)
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
        measurement_base._data_compatability(vari, vard)
        measurement_base._data_compatability(vari, vare)
        data = np.c_[vari, vard, vare]
        tg = measurement_base._graph_2Ddata(data, ax, scatter, **mpl_kwargs)
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
        measurement_base._data_compatability(indvars, ds_data)
        data = np.c_[indvars, ds_data]
        tg = measurement_base._graph_2Ddata(data, ax, scatter, **mpl_kwargs)
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