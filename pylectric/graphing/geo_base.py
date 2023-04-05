# Coding
from overrides import override
from abc import abstractmethod, ABCMeta
#Methods
from pylectric.graphing import graphwrappers, journals
import pylectric
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from enum import Enum

class graphable_base(metaclass=ABCMeta):
    """Class to expand and bind graphing functions to geometric objects.

    """
    
    class sweep_enum(Enum):
        """Tracks sweeping direction for the class"""
        UNDEFINED = 0
        FORWARD = 1
        BACKWARD = 2
    
    def __init__(self) -> None:
        # if not hasattr(self, "data"):
            # raise AttributeError("No data passed to constructor.")
        super().__init__()
        self.sweep_dir = self.sweep_enum.UNDEFINED
        return None

    @abstractmethod
    def ind_vars(self):
        """Returns the independent variable.

        Returns:
            Numpy array: A 1D data array, for the independent variable related to the object.
        """
        xx=None
        return xx
    
    @abstractmethod
    def dep_vars(self):
        """Returns the dependent variables

        Returns:
            Numpy array: A 2D data array of the dependent variables related to the object. Must match length of ind_vars
        """
        yy = None
        return yy

    @abstractmethod
    def extra_vars(self):
        """Returns additional dependent variables

        Returns:
            Numpy array: A 2D data array of the dependent variables related to the object. Must match length of ind_vars
        """
        zz = None
        return zz

    def all_vars(self):
        return np.c_[self.ind_vars(), self.dep_vars(), self.extra_vars()]

    def _plot_1Ddata(data, ax=None, **mpl_kwargs):
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
        ax.scatter(data[:, 0], data[:, 1], **mpl_kwargs)
        
        # TODO: Uncomment...
        # tg.setDefaultTicks()
        
        return tg
    
    def _plot_2Ddata(data, axes=None, **mpl_kwargs) -> graphwrappers.transport_graph:
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
            assert isinstance(axes, plt.Axes) or (isinstance(axes, (list, tuple)) and np.all([isinstance(ax, plt.Axes) for ax in axes]))
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
            axes[i-1].scatter(data[:,0], data[:,i], **mpl_kwargs)
            axes[i-1].legend()
            
        # TODO: Uncomment...
        tg.setDefaultTicks()
        # if label:
            # tg.set_legend_handle_size(15)
                
        return tg
    
    def _plot_3Ddata(data, axes=None, **mpl_kwargs) -> graphwrappers.transport_graph:
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
            axes[i-2].scatter(data[:, 0], data[:,1], data[:, i], **mpl_kwargs)

        # TODO: Uncomment...
        # tg.setDefaultTicks()

        return tg
    
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
    def plot_all_data(self, axes=None, **mpl_kwargs):
        """Generates a plot of all data attached to object.
        Subplots determined by independent variables (columns) and dependent / extra variables (rows).
        Requires overriding to label individual plots appropraitely.

        """
        vari = self.ind_vars()
        vard = self.dep_vars()
        vare = self.extra_vars()
        graphable_base._data_compatability(vari, vard)
        if vare is not None:
            graphable_base._data_compatability(vari, vare)
            data = np.c_[vari, vard, vare]
        else:
            data = np.c_[vari, vard]
            
        li = len(vari.shape)
        if li == 1:
            tg = graphable_base._plot_2Ddata(data, axes, **mpl_kwargs) 
        elif li == 2 or li == 3:
            tg = graphable_base._plot_3Ddata(data, axes, **mpl_kwargs)
        else:
            raise AttributeError("Too many independent variables")        
        return tg

    @abstractmethod
    def plot_dep_vars(self, axes=None, **mpl_kwargs):
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
            tg = graphable_base._plot_2Ddata(data, axes, **mpl_kwargs)
        elif li == 2 or li == 3:
            data = np.c_[vari, vard]
            tg = graphable_base._plot_3Ddata(data, axes, **mpl_kwargs)
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
        Difference data is returned in the graphable_base.all_vars() format, but needs to be assigned, therefore an abstract method.
        
        Args:
            x (graphable_base): Another object whose data will be used to subtract against.
            
        Returns: 
            sub (Numpy NDarray): One numpy array of data (corresponding to the same format as graphable_base.all_vars()) corresponding to subtraction.
        """
        assert isinstance(x, graphable_base)
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
        print(self_match[:,0])
        print(x_match[:,0])
        assert np.all(self_match[:,0:ind_var_len] == x_match[:, 0:ind_var_len])
        
        #Copy independent variables, subtract all other variables.
        sub = self_match.copy()
        sub[:, ind_var_len:] = sub[:, ind_var_len:]  - x_match[:, ind_var_len:]
        return sub

    def sweep_arrow_location(self, i):
        if len(self.ind_vars().shape) == 1:
            #single x variable
            return pylectric.signals.feature_detection.find_arrow_location(xdata=self.ind_vars(),ydata=self.dep_vars()[:,i])

class graphable_base_dataseries(graphable_base):
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
    def plot_dataseries(self, key, ax=None, **mpl_kwargs):
        """Generates a single plot of a specific key in dataseries.
        Needs override with super call to label x axis for indipendent variables.
        """
        #TODO: Update for 3D version...
        assert key in self.dataseries
        indvars = self.ind_vars()
        value = self.dataseries[key]
        graphable_base._data_compatability(indvars, value)
        data = np.c_[indvars, value]
        tg = graphable_base._plot_2Ddata(data, ax, **mpl_kwargs)
        tg.ax[0].set_ylabel(key)
        return tg
    
    @abstractmethod
    def plot_dataseries_with_dep_vars(self, key, ax=None, **mpl_kwargs):
        """Generates a plot of all dependent data attached to object plus specific key in dataseries.
        Subplots determined by independent variables (columns) and dependent / extra variables (rows).
        Needs override with super call to label x & y axis for indipendent/dependent variables.
        """
        #TODO: Update for 3D version.
        assert key in self.dataseries
        vari = self.ind_vars()
        vard = self.dep_vars()
        vare = self.dataseries[key]
        graphable_base._data_compatability(vari, vard)
        graphable_base._data_compatability(vari, vare)
        data = np.c_[vari, vard, vare]
        tg = graphable_base._plot_2Ddata(data, ax, **mpl_kwargs)
        i = vard.shape[-1]
        tg.ax[i].set_ylabel(key)
        return tg

    @abstractmethod
    def plot_all_dataseries(self, ax=None, **mpl_kwargs):
        """Generates a plot of all dataseries.
        Subplot for each dataseries (rows).
        Needs override with super call to label x axis for indipendent variables.
        """
        # TODO: Update for 3D version...
        indvars = self.ind_vars()
        ds_data = np.c_[*[self.dataseries[key] for key in self.dataseries]]
        graphable_base._data_compatability(indvars, ds_data)
        data = np.c_[indvars, ds_data]
        tg = graphable_base._plot_2Ddata(data=data, ax=ax, **mpl_kwargs)
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