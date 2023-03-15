import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from . import geo_base, journals

class transport_graph():
    
    use_pylectric_rcparams = False
    _exponent_to_prefix = {
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
    _prefix_to_exponents = {
        "z": -21,
        "a":-18,
        "f":-15,
        "p":-12,
        "n":-9,
        r"$\mu$":-6,
        "m":-3,
        "c":-2,
        "d":-1,
        "k":3,
        "M":6,
        "G":9,
        "T":12,
        "P":15,
        "E":18,
        "Z":21,
        "Y":24
    }
    
    def __init__(self, axfig) -> None:
        """_summary_
        Args:
            axfig (List of Axes | Single Figure): Multiple Matplotlib Axes objects or a single Matplotlib Figure.
        """
        if not (isinstance(axfig, list) or isinstance(axfig, np.ndarray)):
            axfig = [axfig,]
        
        self.fig = None
        self.ax = []
        self.scale = []  # Exponent of multiplicative factor for axes.
        for obj in axfig:
            if isinstance(obj, plt.Axes):
                self.ax.append(obj)
                #TODO: Check dimenstionality of axes.
                self.scale.append([0,0]) #exponent
                if self.fig is None:
                    self.fig = obj.get_figure()
                else:
                    if self.fig != obj.get_figure():
                        raise AttributeError("Provided axis don't belong to the same figure")
            elif isinstance(obj, plt.Figure) and len(axfig) == 1:
                self.fig = obj
                for axis in obj.axes:
                    self.ax.append(axis)
                    # TODO: Check dimensionality of axes... 
                    self.scale.append([0,0]) #exponent, assume [X,Y] scale.
            else:
                raise TypeError("'axfig' was not a list of plt.Axes objects or a single Figure object.")
        return
    
    def defaults(self, i=None):
        if self.fig and transport_graph.use_pylectric_rcparams:
            # self.setDefaultTicks(i)
            self.fig.tight_layout()
            mpl.rcParams.update(journals.journalBase.defaultParams)
        return
    
    def _unit_prefix(order=0):
        #Return nothing if default order.
        if order == 0:
            return ""
        else:
            i = order % 3
            o = order - i
            if i != 0:
                return r"10$^{" + r"{:i}".format() + r"}$ " + transport_graph._prefix[o]
            else:
                return transport_graph._prefix[o]
    
    def autoScaleAxes(self):
        """ Takes object axes, reduces units by factors of thousands, returns factored order.
        Assumes that raw data is in SI units.

        Returns:
            List of Tuples, consisting of two integers (xf, yf) : The factored order for each ax.xaxis and ax.yaxis respectively.
        """
        order = []
        for ax in self.axes:
            # TOOO: Change method by searching raw data, such as ax.get_children()[0].get_data() rather than possibly modified limits.
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            xo = np.floor(np.log10(xlim))
            yo = np.floor(np.log10(ylim))
            
            xf = xo - (xo % 3)
            yf = yo - (yo % 3)
            
            # Scale the tick labels
            transport_graph._scaleTicks(ax.xaxis, xf)
            transport_graph._scaleTicks(ax.yaxis, yf)
            
            # Scale the unit labels
            transport_graph._updateUnits(ax.xaxis, xf)
            transport_graph._updateUnits(ax.yaxis, yf)
            
            order.append((xf, yf))
        return order
    
    def scaleAxesByFactor(ax_xy,factor=1):
        return
    
    def scaleAxesByExponent(ax_xy, exponent=0):
        return
    
    def scaleAxesByNaturalNumbers(ax_xy, factor, unit_label):
        return
    
    def _matchAxesIndexs(self,ax_xy):
        """Takes an arbitrary X-Axis / Y-Axis axis handle and matches the figure axis index and x/y index.
        Args:
            ax_xy (matplotlib.Axes.XAxis | matplotlib.Axes.YAxis): Object matching xaxis or yaxis of matplotlib.figure.Axes object.
        Returns:
            (i,j): i is the axis index, j indicates an XAxis (0) or YAxis (1). If axis not matched, returns (None,None).
        """
        for i in range(len(self.ax)):
            ax = self.ax[i]
            if ax.xaxis == ax_xy:
                return i,0
            elif ax.yaxis == ax_xy:
                return i,1
            # TODO: Check for Z axis in mesh plots...
        #if no match
        return None,None
    
    def _scaleTicks(self,ax_xy, factor=1):
        """Scales tick scale by a set factor.

        Args:
            ax_xy (matplotlib.axis.XAxis | matplotlib.axis.YAxis): XAxis or YAxis Object.
            factor (int, optional): _description_. Defaults to 1.

        Raises:
            AttributeError: _description_
        """
        # TODO: Add support for Z axis.
        assert isinstance(ax_xy, plt.Axes)
        
        # Validate axis
        i,j = self._matchAxesIndexs(ax_xy) 
        if i == None:
            raise AttributeError("Passed X/YAxis doesn't any axes in the Figure.")
        
        # Update scale
        self.scale[i][j] *= factor
    
        # Update ticks    
        new_ticks = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x*factor))
        ax_xy.set_major_formatter(new_ticks)
        return
    
    def _geomToExpFactor(geom):
        exp = 0
        while geom < 1 or geom > 10:
            if geom < 1:
                geom = geom * 10
                exp -= 1
            elif geom > 10:
                geom = geom / 10
                exp += 1
        return geom, exp
    
    def _updateUnits(self,ax_xy):
        assert isinstance(ax_xy, plt.Axes)
        
        # Check if the axis is already scaled by some units.
        scale = self.scale[i][j]
        geom, exp = transport_graph._geomToExpFactor(scale)
        
        # Validate Axis
        i,j = self._matchAxesIndexs(ax_xy)
        if i == None:
            raise AttributeError("Passed X/YAxis doesn't any axes in the Figure.")
        
        #Get current label.
        current = ax_xy.get_label_text()
        if "(" in current and ")" in current:
            spl = current.split("(")
        else:
            #Assume to units, hence possibly modifications to scale?
            if geom != 1:
                newlabel = current + " ({:0.2} x10$^{:}$".format(geom, exp) + ")"
            else:
                newlabel = current + " (x10$^{:}$".format(geom, exp) + ")"
            ax_xy.set_label(newlabel)
            return
        
        # Check if clear single split.
        if len(spl) > 1:
            print("Warning: Axes label '" + current + "' contains mutliple unit parenthesis '('")
            raise Warning("Axes label '" + current +
                            "' contains mutliple unit parenthesis '('")
        
        # Assume units is contained in the last parenthesis only!
        pre = ""
        for a in spl[:-1]:
            pre += a
        spl = spl[-1].split(")")
        
        # Assume units ends with the next sequential close parenthesis.
        if len(spl) > 1:
            print("Warning: Axes label '" + current +
                "' contains mutliple unit parenthesis ')'")
            raise Warning("Axes label '" + current +
                        "' contains mutliple unit parenthesis ')'")
        units = spl[0]
        post = ""
        if len(spl) > 1: #condition needed to catch index error
            for a in spl[1:]:
                post += spl[i+1]
        
        # Check if unit is an existing prefix.
        if units[0] in transport_graph._prefix_to_exponents.keys():
            val = transport_graph._prefix_to_exponents[units[0]]
            if exp in transport_graph._prefix_to_exponents:
                if geom == 1:
                    newlabel = "(" + transport_graph._exponent_to_prefix[exp] + units[1:] + ")"
                else:
                    newlabel = " ({:0.2} ".format(geom) + transport_graph._exponent_to_prefix[exp] + units[1:] + ")"
            else:
                #Reduce exponent
                exp_rem = exp % 3
                exp_3s = exp - exp_rem
                if geom == 1:
                    newlabel = " (x10$^{:}$".form(exp_rem) + transport_graph._exponent_to_prefix[exp_3s] + units[1:] + ")"
                else:
                    newlabel = " ({:0.2} x10$^{:}$".format(geom, exp_rem) + transport_graph._exponent_to_prefix[exp_3s] + units[1:] + ")"
                
        # Check if units follow typical format
        elif units[1] == "." and units[4:8] == " x10":
            if geom == 1 and exp == 0:
                newlabel = ""
            elif exp == 1 and geom == 1:
                newlabel = " (x10)"
            elif geom == 1:
                newlabel = " (x10$^{:}$)".format(exp)
            elif exp == 0:
                newlabel = " ({:0.2})".format(geom)
            elif exp == 1:
                newlabel = " ({:0.2} x10)".format(geom)
            else:
                newlabel = " ({:0.2} x10$^{:}$)".format(geom, exp)
            ax_xy.set_label(newlabel)
        
        # Update the unit.
        
        return
        
    def _genUnitLabel(geom, exp):
        
        return
    
    def setDefaultTicks(self, i=None):
        # only execute if axis has actual values...
        defaultTickerX = ticker.MaxNLocator(5)
        for ax in self.ax:
            ax.xaxis.set_major_locator(defaultTickerX)
            defaultTickerY = ticker.MaxNLocator(5) #needs new object for every axis
            ax.yaxis.set_major_locator(defaultTickerY)
        return
    
    def _apply_fn_to_axes(self, fn, i=None, *args):
        """Expands function to be applied to single or multiple instances
        of axes depending on whether i is specified.

        Args:
            fn (function): Function that can be applied to a plt.Axes object.
            i (int, optional): Index for axes. Defaults to None (implying all axes).
        """
        # Typechecking
        assert callable(fn)
        assert type(i) == int or i is None

        # Multiple Axes
        if not i:
            for axis in self.ax:
                fn(axis, *args)
        # Single Axes
        else:
            if i > len(self.ax):
                raise IndexError("Index i=" + str(i) + " outside bounds.")
            fn(self.ax[i], *args)
    
    def _apply_fn_to_xaxes(self, fn, i=None, *args):
        """Expands function to be applied to single or multiple instances
        of axes.xaxis depending on whether i is specified.

        Args:
            fn (function): Function that can be applied to a plt.Axes object.
            i (int, optional): Index for axes. Defaults to None (implying all axes.xaxis).
        """
        # Typechecking
        assert callable(fn)
        assert type(i) == int or i is None

        # Multiple Axes
        if not i:
            for axis in self.ax:
                fn(axis.xaxis, *args)
        # Single Axes
        else:
            if i > len(self.ax):
                raise IndexError("Index i=" + str(i) + " outside bounds.")
            fn(self.ax[i], *args)
    
    def _apply_fn_to_yaxes(self, fn, i=None, *args):
        """Expands function to be applied to single or multiple instances
        of axes.yaxis depending on whether i is specified.

        Args:
            fn (function): Function that can be applied to a plt.Axes.yaxis object.
            i (int, optional): Index for axes. Defaults to None (implying all axes).
        """
        # Typechecking
        assert callable(fn)
        assert type(i) == int or i is None

        # Multiple Axes
        if not i:
            for axis in self.ax:
                fn(axis.yaxis, *args)
        # Single Axes
        else:
            if i > len(self.ax):
                raise IndexError("Index i=" + str(i) + " outside bounds.")
            fn(self.ax[i], *args)
    
    ####################################################################################
    #########################        Axes Labelling            #########################
    ####################################################################################
    # x-Axis Labels:

    def xFieldT(self, i=None, order=0):
        label = r"$B$ (" + transport_graph._unit_prefix(order) + r"T)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])

    def xFieldOe(self, i=None, order=0):
        label = r"$B$ (" + transport_graph._unit_prefix(order) + r"Oe)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])

    def xGateV(self, i=None, order=0):
        label = r"$V_g$ (" + transport_graph._unit_prefix(order) + r"V)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])
    
    ### y-Axis Labels:
    
    def yMobility(self, i=None, order=0):
        label = r"$\mu$ (" + transport_graph._unit_prefix(order) + r"m$^2$/Vs)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
    
    def yMobilityCm(self, i=None, order=0):
        self.yMobility(i=i, order=-2)
    
    def yConductivity(self, i=None, order=0, subscript=""):
        
        label = r"$\sigma_{"+subscript + \
            r"}$ (" + transport_graph._unit_prefix(order) + r"S)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yConductivityQuantum2(self, i=None, subscript=""):
        label = r"$\sigma_{"+subscript+r"}$ ($2e^2/h$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yConductivityQuantum2(self, i=None, subscript=""):
        label = r"$\sigma_{"+subscript+r"}$ ($e^2/h$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yResistivity(self, i=None, order=0, subscript=""):
        label = r"$\rho_{"+subscript + \
            r"}$ (" + transport_graph._unit_prefix(order) + r"$\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yResistivityQuantum(self, i=None, subscript=""):
        label = r"$\rho_{"+subscript+r"}$ ($h/e^2$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yResistivityQuantum2(self, i=None,subscript=""):
        label = r"$\rho_{"+subscript+r"}$ ($h/2e^2$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yConductance(self, i=None, order=0, subscript=""):
        label = r"$G_{"+subscript + r"}$ (" + \
           transport_graph._unit_prefix(order) + r"S)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    def yResistance(self, i=None, order=0, subscript=""):
        label = r"$R_{"+subscript + r"}$ (" + \
            transport_graph._unit_prefix(order) + r"$\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        
    
