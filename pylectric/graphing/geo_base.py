# Coding
from overrides import override
from abc import abstractmethod, ABCMeta
#Methods
from pylectric.graphing import graphwrappers, journals
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class graphable_base(metaclass=ABCMeta):
    """Class to expand and bind graphing functions to geometric objects.

    """
    
    def __init__(self) -> None:
        # if not hasattr(self, "data"):
            # raise AttributeError("No data passed to constructor.")
        super().__init__()
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

    def _plot_1Ddata(data, ax=None, label=None):
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
        ax.scatter(data[:,0], data[:,1], label=label)
        
        # TODO: Uncomment...
        # tg.setDefaultTicks()
        
        return tg
    
    def _plot_2Ddata(data, axes=None, label=None):
        assert isinstance(data, np.ndarray)
        assert axes is None or isinstance(axes, plt.Axes) or (isinstance(
            axes, list) and isinstance(axes[0], plt.Axes))
        assert len(data.shape) == 2
        
        nd = data.shape[-1] - 1
        if axes is None:
            fig,axes = plt.subplots(nd, 1)
            # Graphing wrapper
            tg = graphwrappers.transport_graph(axes)
            fig.set_size_inches(
                w=journals.acsNanoLetters.maxwidth_2col, h=3*nd)
        elif len(axes) != nd:
            raise AttributeError("Length of axes does not match data dimension.")
        else:
            tg = graphwrappers.transport_graph(axes)
            
        tg.defaults()
        
        for i in range(1, nd+1):
            axes[i-1].scatter(data[:,0], data[:,i],label=label)
            
        # TODO: Uncomment...
        tg.setDefaultTicks()
                
        return tg
    
    def _plot_3Ddata(data, axes=None, label=None):
        #TODO: update to match _plot_2Ddata methods...
        assert isinstance(data, np.ndarray)
        dl = len(data.shape)
        assert dl == 2 or dl == 3 #assume first two columns are independent vars, no matter if in 2D or 3D shape.

        nd = data.shape[-1] - 2 #doesn't matter if 2D or 3D.
        if axes is None:
            fig, axes = plt.subplots(nd, 1)
            fig.set_size_inches(
                w=journals.acsNanoLetters.maxwidth_2col, h=3*nd)
        
        # Graphing wrapper
        tg = graphwrappers.transport_graph(axes)
        
        if dl ==3: 
            data = data.reshape([data.shape[0] * data.shape[1],nd])
        
        for i in range(2, nd+2):
            axes[i-2].scatter(data[:, 0], data[:,1], data[:, i], label=label)

        # TODO: Uncomment...
        # tg.setDefaultTicks()

        return tg
    
    def plot_all_data(self, axes=None, label = None):
        """Generates a plot of all data attached to object.
        Subplots determined by independent variables (columns) and dependent / extra variables (rows).

        """
        vari = self.ind_vars()  # 1D array    
        vard = self.dep_vars()  # 2D array
        vare = self.extra_vars()  # 2D array
        
        assert isinstance(vari, np.ndarray)
        assert isinstance(vard, np.ndarray)
        assert isinstance(vare, np.ndarray)
        if len(vari.shape) > 1:
            assert vari.shape[:-1] == vard.shape[:-1] 
            assert vari.shape[:-1] == vare.shape[:-1]
        elif len(vari.shape) == 1:
            assert vari.shape[0] == vard.shape[0]
            assert vari.shape[0] == vare.shape[0]
        
        li = len(vari.shape)
        if li == 1:
            data = np.c_[vari, vard, vare]
            tg = graphable_base._plot_2Ddata(data, axes, label) 
        elif li == 2 or li == 3:
            data = np.c_[vari, vard, vare]
            tg = graphable_base._plot_3Ddata(data, axes, label)
        elif li == 3:
            tg = graphable_base._plot_3Ddata(data, axes, label)
        else:
            raise AttributeError("Too many independent variables")
        
        if label:
            for legend in tg.fig.legends:
                if graphwrappers.transport_graph.use_pylectric_rcparams:
                    legend.delete()
            for ax in tg.ax:
                ax.legend()
    
        return tg

    # def plot_data(self, cols):
    #     tg = 
    
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