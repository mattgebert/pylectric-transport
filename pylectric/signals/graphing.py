from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

class acsJournals:
    # Parameters shared between single and double column figures.
    acsNanoLetters_Shared_Params = {
        'figure.dpi': 300,
        "font.family":
            'arial',
            # 'helvetica',
        'font.size': 7,  # 4.5 Minimum!
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.framealpha": 0,
        "axes.titlesize": 7,
        "axes.linewidth": 0.5,
    }

    acsNano_2Col_Params = acsNanoLetters_Shared_Params.copy()
    acsNano_2Col_Params.update({
        # 4.167in (300pt) to 7in (504pt) wide maximum, 9.167 in (660pt) high.
        "figure.figsize": [7, 3.33],
    })

    acsNano_1Col_Params = acsNanoLetters_Shared_Params.copy()
    acsNano_1Col_Params.update({
        # 3.33 in (240pt) wide, maximum 9.167 in (660pt) high.
        "figure.figsize": [3.33, 3.33],
    })


class electronic_graph():
    
    defaultParams = {

    }
    
    def __init__(self, *ax) -> None:
        """_summary_
        Args:
            ax (Axes): Multiple Matplotlib Axes object.
        """
        
        self.fig = None
        self.ax = []
        for obj in ax:
            if isinstance(obj, plt.Axes):
                self.ax.append(obj)
                if self.fig is None:
                    self.fig = obj.get_figure()
                else:
                    if self.fig != obj.get_figure():
                        raise AttributeError("Provided axis don't belong to the same figure")
        return
    
    def setDefaultParams(self, i=None):
        if self.fig:
            self.setDefaultTicks(i)
            self.fig.setParams(acsJournals.acsNano_1Col_Params)
            self.fig.setParams(electronic_graph.defaultParams)
        return

    def setParams(self, rcParams, i=None):
        self.fig.setParams(rcParams)
        return
    
    def _apply_fn_to_axes(self, fn, i=None, *args):
        """Expands function to be applied to single or multiple instances
        of axes depending on whether i is specified.

        Args:
            fn (function): Function that can be applied to a plt.Axes object.
            i (int, optional): Index for axes. Defaults to None (implying all axes).
        """
        #Typechecking
        assert type(fn) == function
        assert type(i) == int or i is None
        if i > len(self.ax):
            raise IndexError("Index i=" + str(i) + " outside ")
        # Multiple Axes
        if not i:
            for axis in self.ax:
                fn(axis, *args)
        # Single Axes
        else:
            fn(self.ax[i], *args)
    
    def _unit_prefix(order=0):
        
        #Return nothing if default order.
        if order == 0:
            return ""
        else:
            _prefix ={
                -21: "z",
                -18: "a",
                -15: "f",
                -12: "p",
                -9: "n",
                -6: r"$\mu$",
                -3: "m",
                -2: "c",
                -1: "d",
                3: "k",
                6: "M",
                9: "G",
                12: "T",
                15: "P",
                18: "E",
                21: "Z",
                24: "Y"
            }
            i = order % 3
            o = order - i
            if i != 0:
                return r"10$^{" + r"{:i}".format() + r"}$ " + _prefix[o]
            else:
                return _prefix[o]
    
    def _scaleTicks(order=0):
        #TODO: Implement tick scaling by factors such as natural numbers to avoid changing raw data.
        return
    
    def setDefaultTicks(self, i=None):
        defaultTicker = [ticker.MaxNLocator(5)]
        self._apply_fn_to_axes(plt.Axes.yaxis.set_major_locator, i=i, *defaultTicker)
        self._apply_fn_to_axes(plt.Axes.xaxis.set_major_locator, i=i, *defaultTicker)
        return
    
    ### y-Axis Labels:
    
    def yMobility(self, i=None, order=0):
        label = r"$\mu$ (" + electronic_graph._unit_prefix(order) + r"m$^2$/Vs)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
    
    def yMobilityCm(self, i=None, order=0):
        self.yMobility(i=i, order=-2)
    
    def yConductivity(self, i=None, order=0, subscript=""):
        
        label = r"$\sigma_{"+subscript + \
            r"}$ (" + electronic_graph._unit_prefix(order) + r"S)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yConductivityQuantum2(self, i=None, subscript=""):
        label = r"$\sigma_{"+subscript+r"}$ ($2e^2/h$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yConductivityQuantum2(self, i=None, subscript=""):
        label = r"$\sigma_{"+subscript+r"}$ ($e^2/h$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yResistivity(self, i=None, order=0, subscript=""):
        label = r"$\rho_{"+subscript + \
            r"}$ (" + electronic_graph._unit_prefix(order) + r"$\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yResistivityQuantum(self, i=None, subscript=""):
        label = r"$\rho_{"+subscript+r"}$ ($h/e^2$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yResistivityQuantum2(self, i=None,subscript=""):
        label = r"$\rho_{"+subscript+r"}$ ($h/2e^2$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yConductance(self, i=None, order=0, subscript=""):
        label = r"$G_{"+subscript + r"}$ (" + \
           electronic_graph._unit_prefix(order) + r"S)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    def yResistance(self, i=None, order=0, subscript=""):
        label = r"$R_{"+subscript + r"}$ (" + \
            electronic_graph._unit_prefix(order) + r"$\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
        
    
    ### x-Axis Labels:
        
    def xFieldT(self, i=None, order=0):
        label = r"$B$ (" + electronic_graph._unit_prefix(order) + r"T)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
    
    def xFieldOe(self, i=None, order=0):
        label = r"$B$ (" + electronic_graph._unit_prefix(order) + r"Oe)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])

    def xGateV(self, i=None, order=0):
        label = r"$V_g$ (" + electronic_graph._unit_prefix(order) + r"V)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i=i, *[label])
