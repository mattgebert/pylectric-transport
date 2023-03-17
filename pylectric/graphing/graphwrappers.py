import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from . import geo_base, journals

class scalable_graph():
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
        "a": -18,
        "f": -15,
        "p": -12,
        "n": -9,
        r"$\mu$": -6,
        "m": -3,
        "c": -2,
        "d": -1,
        "k": 3,
        "M": 6,
        "G": 9,
        "T": 12,
        "P": 15,
        "E": 18,
        "Z": 21,
        "Y": 24
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
        self.scale = []  # Multiplicative factor for axes.
        self.natscale = []  # Multiplicitive factor for natural units in axes.
        self.natlabel = []  # Label for natural units.
        self.prevlabel = []  # Labels prior to changing to natural units.
        for obj in axfig:
            if isinstance(obj, plt.Axes):
                self.ax.append(obj)
                # TODO: Check dimenstionality of axes.
                self.scale.append([1, 1])  # scale
                self.natscale.append([1, 1])
                self.natlabel.append([None, None])
                self.prevlabel.append([None, None])
                if self.fig is None:
                    self.fig = obj.get_figure()
                else:
                    if self.fig != obj.get_figure():
                        raise AttributeError(
                            "Provided axis don't belong to the same figure")
            elif isinstance(obj, plt.Figure) and len(axfig) == 1:
                self.fig = obj
                for axis in obj.axes:
                    self.ax.append(axis)
                    # TODO: Check dimensionality of axes...
                    self.scale.append([1, 1])  # scale, assume [X,Y] axis.
                    self.natscale.append([1, 1])
                    self.natlabel.append([None, None])
                    self.prevlabel.append([None, None])
            else:
                raise TypeError(
                    "'axfig' was not a list of plt.Axes objects or a single Figure object.")

    def _unit_prefix(order=0):
        # Return nothing if default order.
        if order == 0:
            return ""
        else:
            i = order % 3
            o = order - i
            if i != 0:
                return r"10$^{" + r"{:i}".format() + r"}$ " + transport_graph._prefix[o]
            else:
                return transport_graph._prefix[o]

    def setDefaultTicks(self, i=None):
        # only execute if axis has actual values...
        defaultTickerX = ticker.MaxNLocator(5)
        for ax in self.ax:
            ax.xaxis.set_major_locator(defaultTickerX)
            # needs new object for every axis
            defaultTickerY = ticker.MaxNLocator(5)
            ax.yaxis.set_major_locator(defaultTickerY)
        self.autoScaleAxes()
        return

    def autoScaleAxes(self):
        """ Takes object axes, reduces units by factors of thousands, returns factored order.
        Assumes that raw data is in SI units.
        """
        for ax in self.ax:
            # TOOO: Change method by searching raw data, such as ax.get_children()[0].get_data() rather than possibly modified limits.
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xo = np.floor(np.log10(np.max(np.abs(xlim))))
            yo = np.floor(np.log10(np.max(np.abs(ylim))))

            # Scale the tick labels
            if xo > 2 or xo < -2:
                self._scaleTicks(ax.xaxis, 10**xo)
                self._updateUnits(ax.xaxis)
            if yo > 2 or yo < -2:
                self._scaleTicks(ax.yaxis, 10**yo)
                self._updateUnits(ax.yaxis)
        return

    def scaleAxesByFactor(self, ax_xy, factor=1):
        # Relative scaling
        self._scaleTicks(ax_xy=ax_xy, factor=factor)
        return

    def scaleAxesByExponent(self, ax_xy, exponent=0):
        # Relative scaling
        self._scaleTicks(ax_xy=ax_xy, factor=10**exponent)
        return

    def setNaturalNumbersScale(self, ax_xy, factor, unit_label):
        # Relative scaling
        i, j = self._matchAxesIndexs(ax_xy)
        natlabel = self.natlabel[i, j]
        if natlabel != None:
            raise Warning("Natural unit scaling already exists for this axis.")
        else:
            self.natlabel[i, j] = unit_label
            self.prevlabel[i, j] = ax_xy.get_label_text()
        self.natscale[i, j] /= factor
        self._updateTicks(ax_xy)

    def removeScaling(self, ax_xy):
        i, j = self._matchAxesIndexs(ax_xy)
        self.scale[i][j] = 1
        self.removeNatScaling(ax_xy)
        return

    def removeNatScaling(self, ax_xy):
        i, j = self._matchAxesIndexs(ax_xy)
        self.natscale[i][j] = 1
        self.natlabel[i][j] = None
        if self.prevlabel[i, j] != None:
            ax_xy.set_label_text(self.prevlabel[i][j])
            self.prevlabel[i][j] = None
        self._updateTicks(ax_xy)
        self._updateUnits(ax_xy)
        return

    def _matchAxesIndexs(self, ax_xy):
        """Takes an arbitrary X-Axis / Y-Axis axis handle and matches the figure axis index and x/y index.
        Args:
            ax_xy (matplotlib.Axes.XAxis | matplotlib.Axes.YAxis): Object matching xaxis or yaxis of matplotlib.figure.Axes object.
        Returns:
            (i,j): i is the axis index, j indicates an XAxis (0) or YAxis (1). If axis not matched, returns (None,None).
        """
        for i in range(len(self.ax)):
            ax = self.ax[i]
            if ax.xaxis == ax_xy:
                return i, 0
            elif ax.yaxis == ax_xy:
                return i, 1
            # TODO: Check for Z axis in mesh plots...
        # if no match
        raise IndexError(
            "The provided X|Y axis is not contained in the this graph wrapper.")

    def _scaleTicks(self, ax_xy, factor=1):
        """Scales tick scale by removing a set factor.

        Args:
            ax_xy (matplotlib.axis.XAxis | matplotlib.axis.YAxis): XAxis or YAxis Object.
            factor (int, optional): _description_. Defaults to 1.

        Raises:
            AttributeError: _description_
        """
        # TODO: Add support for Z axis.
        assert isinstance(ax_xy, mpl.axis.Axis)

        # Validate axis
        i, j = self._matchAxesIndexs(ax_xy)
        if i == None:
            raise AttributeError(
                "Passed X/YAxis doesn't any axes in the Figure.")

        # Update scale
        self.scale[i][j] /= factor

        # Update ticks
        self._updateTicks(ax_xy)
        return

    def _updateTicks(self, ax_xy):
        # TODO: Add support for Z axis.
        assert isinstance(ax_xy, mpl.axis.Axis)

        # Validate axis
        i, j = self._matchAxesIndexs(ax_xy)
        if i == None:
            raise AttributeError(
                "Passed X/YAxis doesn't any axes in the Figure.")

        # Update ticks
        new_ticks = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x*self.scale[i][j]*self.natscale[i][j]))
        ax_xy.set_major_formatter(new_ticks)
        return

    def _geomToExpFactor(geom):
        exp = 0
        while abs(geom) < 1 or abs(geom) >= 10:
            if geom < 1:
                geom *= 10
                exp -= 1
            elif geom >= 10:
                geom /= 10
                exp += 1
        geom = np.round(geom, decimals=2)
        if geom == 10:
            geom /= 10
            exp += 1
            geom = np.round(geom, decimals=2)
        exp = int(exp)
        return geom, exp

    def _updateUnits(self, ax_xy):
        assert isinstance(ax_xy, mpl.axis.Axis)

        # Validate Axis
        i, j = self._matchAxesIndexs(ax_xy)
        if i == None:
            raise AttributeError(
                "Passed X/YAxis doesn't any axes in the Figure.")

        # Acquire the new scaling, convert factor to exponential format.
        # Note, scale value needs to be inverted for labels.
        scale = 1.0 / self.scale[i][j]
        geom, exp = transport_graph._geomToExpFactor(scale)

        # Get current label and check format. Override format regardless, but use current to determine current units.
        current = ax_xy.get_label_text()

        # FORMAT #0 - No Parenthesis for units (). Assume needs units update, just use numeric without prefixes.
        if not ("(" in current or ")" in current):
            if geom != 1:
                newlabel = current + \
                    " ({:.2f} x10$^{:}$".format(geom, exp) + ")"
            else:
                newlabel = current + " (x10$^{:}$".format(geom, exp) + ")"
            ax_xy.set_label(newlabel)
            return
        else:
            spl = current.split("(")
            # Assume to units, hence possibly modifications to scale?

        # Check if clear single split.
        if len(spl) > 2:
            raise Warning("Axes label '" + current +
                          "' contains mutliple unit parenthesis '('")

        # Assume units is contained in the last parenthesis only!
        pre = ""
        for a in spl[:-1]:
            pre += a
        spl = spl[-1].split(")")

        # Assume units ends with the next sequential close parenthesis.
        if len(spl) > 2:
            raise Warning("Axes label '" + current +
                          "' contains mutliple unit parenthesis ')'")
        units = spl[0]
        post = ""
        if len(spl) > 1:  # condition needed to catch index error
            for a in spl[1:]:
                post += a

        # If natural units are being used, do not consider the existing axes units, but keep label!
        if self.natlabel[i][j] != None and self.natscale[i][j] != 1:
            # Use natural labels.
            newlabel = pre+"(" + self._genFactorLabel(geom,
                                                      exp)+" "+self.natlabel[i][j]+")"

        # Check if units have some non-zero string, so they can use prefixes.
        elif len(units) > 0:
            # Find where the units are located in the units string.

            def removeExponent(str):
                # Used to isolate the exponent. ie x10^A for A.
                a = str.split(" ")  # Space after x10^A isolates to units.
                a = a[1:]  # Take everything after first space.
                b = ""
                for s in a:
                    b += s
                return b

            # FORMAT #1 scaling
            if len(units) > 10 and units[1] == "." and units[4:8] == " x10":
                # isolate the exponent string $^X $ where X is an integer before the units text.
                units2 = removeExponent(units[8:])
                # Create label based on existance of prefix
                if units2[0] in transport_graph._prefix_to_exponents.keys():
                    newlabel = pre + \
                        "(" + transport_graph._genUnitLabel(geom, exp) + \
                        units2[1:] + ")" + post
                else:
                    newlabel = pre + \
                        "(" + transport_graph._genUnitLabel(geom, exp) + \
                        units2[0:] + ")" + post
            # FORMAT #2 exp-scaling
            elif len(units) > 5 and units[0:3] == "x10":
                # isolate the exponent string $^X $ where X is an integer before the units text.
                units2 = removeExponent(units[3:])
                if units2[0] in transport_graph._prefix_to_exponents.keys():
                    newlabel = pre + \
                        "(" + transport_graph._genUnitLabel(geom, exp) + \
                        units2[1:] + ")" + post
                else:
                    newlabel = pre + \
                        "(" + transport_graph._genUnitLabel(geom, exp) + \
                        units2[0:] + ")" + post
            # FORMAT #3 geom-scaling
            elif len(units) > 5 and units[1] == "." and units[4] == " ":
                if units[5] in transport_graph._prefix_to_exponents.keys():
                    newlabel = pre + \
                        "(" + transport_graph._genUnitLabel(geom, exp) + \
                        units[6:] + ")" + post
                else:
                    newlabel = pre + \
                        "(" + transport_graph._genUnitLabel(geom, exp) + \
                        units[5:] + ")" + post
            # FORMAT #4 Prefix only.
            elif len(units) > 1 and units[0] in transport_graph._prefix_to_exponents.keys():
                newlabel = pre + \
                    "(" + transport_graph._genUnitLabel(geom, exp) + \
                    units[1:] + ")" + post
            # FORMAT #5 Single character.
            else:  # Len(units) > 0, no prefix.
                newlabel = pre + \
                    "(" + transport_graph._genUnitLabel(geom, exp) + \
                    units[:] + ")" + post
        else:
            raise Warning("Axes label '" + current +
                          "' has empty parenthesis.")
            return pre + "(" + transport_graph._genUnitLabel(geom, exp) + ")" + post
        ax_xy.set_label_text(newlabel)
        return

    def _genUnitLabel(geom, exp):
        """Generates a label for a unit, using prefixes. 
        Ie, (Ax10$^B$ CD) where A is a float, B is an integer, C is a prefix, D is the unit label.

        Args:
            geom (float): Multiplicitive scaling factor
            exp (int): The exponent of the factor's scale.

        Returns:
            str: A label 
        """
        exp_rem = exp % 3
        exp_3s = exp - exp_rem
        if exp_3s in transport_graph._exponent_to_prefix.keys():
            return transport_graph._genFactorLabel(geom, exp_rem) + " " + transport_graph._exponent_to_prefix[exp_3s]
        elif exp_3s == 0:
            label = transport_graph._genFactorLabel(geom, exp_rem)
            return label + " " if label != "" else ""  # don't return space...
        else:
            raise KeyError(str(exp_3s) + " does not have an assigned prefix.")

    def _genFactorLabel(geom, exp):
        """Generates a scientific label from Ax10$^B$ where A is a float and B is an integer.

        Args:
            geom (float): Multiplicitive scaling factor
            exp (int): The exponent of the factor's scale.

        Returns:
            str: The scne
        """
        assert isinstance(geom, int) or isinstance(geom, float)
        assert isinstance(exp, int)

        if geom == 1 and exp == 0:
            label = ""
        elif exp == 1 and geom == 1:
            label = "x10"
        elif geom == 1:
            label = "x10$^{:}$".format(exp)
        elif exp == 0:
            label = "{:.2f}".format(geom)
        elif exp == 1:
            label = "{:.2f}x10".format(geom)
        else:
            label = "{:.2f}x10$^{:}$".format(geom, exp)
        return label

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

    def _updateX_Ticks_Labels(self, i=None):
        if i:
            axis = self.ax[i].xaxis
            self._update_Ticks_Labels(axis)
        else:
            for i in range(len(self.ax)):
                self._update_Ticks_Labels(self.ax[i].xaxis)

    def _updateY_Ticks_Labels(self, i=None):
        if i:
            axis = self.ax[i].yaxis
            self._update_Ticks_Labels(axis)
        else:
            for i in range(len(self.ax)):
                self._update_Ticks_Labels(self.ax[i].yaxis)

    def _update_Ticks_Labels(self, ax_xy):
        self._updateTicks(ax_xy)
        self._updateUnits(ax_xy)

class transport_graph(scalable_graph):
    
    use_pylectric_rcparams = False
    
    
    def __init__(self, axfig) -> None:
        """_summary_
        Args:
            axfig (List of Axes | Single Figure): Multiple Matplotlib Axes objects or a single Matplotlib Figure.
        """
        super().__init__()
        return
    
    def defaults(self, i=None):
        if self.fig and transport_graph.use_pylectric_rcparams:
            # self.setDefaultTicks(i)
            self.fig.tight_layout()
            mpl.rcParams.update(journals.journalBase.defaultParams)
        return
    
    
    ####################################################################################
    #########################        Axes Labelling            #########################
    ####################################################################################
    # x-Axis Labels:

    def xFieldT(self, i=None, order=0):
        label = r"$B$ (" + transport_graph._unit_prefix(order) + r"T)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])
        self._updateX_Ticks_Labels(i)

    def xFieldInverseT(self, i=None, order=0):
        label = r"1/$B$ (" + transport_graph._unit_prefix(order) + r"T$^{-1}$)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])
        self._updateX_Ticks_Labels(i)

    def xFieldOe(self, i=None, order=0):
        label = r"$B$ (" + transport_graph._unit_prefix(order) + r"Oe)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])
        self._updateX_Ticks_Labels(i)
        
    def xFieldInverseOe(self, i=None, order=0):
        label = r"1/$B$ (" + transport_graph._unit_prefix(order) + r"Oe$^{-1}$)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])
        self._updateX_Ticks_Labels(i)

    def xGateV(self, i=None, order=0):
        label = r"$V_g$ (" + transport_graph._unit_prefix(order) + r"V)"
        self._apply_fn_to_axes(plt.Axes.set_xlabel, i, *[label])
        self._updateX_Ticks_Labels(i)
    
    
    
    ### y-Axis Labels:
    
    def yMobility(self, i=None, order=0):
        label = r"$\mu$ (" + transport_graph._unit_prefix(order) + r"m$^2$/Vs)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
    
    def yMobilityCm(self, i=None, order=0):
        self.yMobility(i=i, order=-2)
        self._updateY_Ticks_Labels(i)
    
    def yConductivity(self, i=None, order=0, subscript=""):
        label = r"$\sigma_{"+subscript + \
            r"}$ (" + transport_graph._unit_prefix(order) + r"S)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yConductivityQuantum2(self, i=None, subscript=""):
        #TODO: Update to use the implemented "natural number" scaling options.
        label = r"$\sigma_{"+subscript+r"}$ ($2e^2/h$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yConductivityQuantum2(self, i=None, subscript=""):
        # TODO: Update to use the implemented "natural number" scaling options.
        label = r"$\sigma_{"+subscript+r"}$ ($e^2/h$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yResistivity(self, i=None, order=0, subscript=""):
        label = r"$\rho_{"+subscript + \
            r"}$ (" + transport_graph._unit_prefix(order) + r"$\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yResistivityQuantum(self, i=None, subscript=""):
        #TODO: Update to use the implemented "natural number" scaling options.
        label = r"$\rho_{"+subscript+r"}$ ($h/e^2$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yResistivityQuantum2(self, i=None,subscript=""):
        #TODO: Update to use the implemented "natural number" scaling options.
        label = r"$\rho_{"+subscript+r"}$ ($h/2e^2$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yConductance(self, i=None, order=0, subscript=""):
        label = r"$G_{"+subscript + r"}$ (" + \
           transport_graph._unit_prefix(order) + r"S)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yResistance(self, i=None, order=0, subscript=""):
        label = r"$R_{"+subscript + r"}$ (" + \
            transport_graph._unit_prefix(order) + r"$\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
        
    def yMR_percentage(self, i=None, order=0, subscript=""):
        label = r"MR$_{" + subscript + r"}$ ($\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)
    
    def yMR_absolute(self, i=None, order=0, subscript=""):
        label = r"MR$_{" + subscript + r"}$ ($\Omega$)"
        self._apply_fn_to_axes(plt.Axes.set_ylabel, i, *[label])
        self._updateY_Ticks_Labels(i)

#TODO Update all y/x label functions to remove order, and also to consider if no subscript is provided.