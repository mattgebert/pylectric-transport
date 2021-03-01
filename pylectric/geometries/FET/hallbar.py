#Hall bar geometry device. Measurements have r_xx and/or r_xy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pylectric.analysis import mobility
from scipy import optimize as opt

class Meas_GatedResistance():
    """Class to handle a single sweep of raw data. Measures resistance against gate voltage to infer properties."""

    def __init__(self, data, Cg, L, W, D = None):
        """ Takes numpy array of the form array[rows, columns]
            Requires columns to be of the form: [V_gate (Volts), Resistance (Ohms)]
            Also takes Length L, Width W, and Thickness D to infer scaled material properties.
            If 2D materials, thickness can be left as None, and units don't matter.
            If 3D, thickness should in cm units (assuming L & W cancel out).
        """

        #1. Cols = [V_gate, Resistance]
        self.raw_data = np.array(data, np.longfloat).copy()
        self.conductivity_data = self.raw_data.copy()[:,0:2]
        self.conductivity_data[:,1] = np.reciprocal(self.conductivity_data[:,1] * W / L) if D is None else np.reciprocal(self.conductivity_data[:,1] * W * D / L)
        self.L = L
        self.W = W
        self.D = D
        self.Cg = Cg
        self.is2D = (D is None)

    def mobility_dtm_2D(self, graph = True, ax=None):
        """ Calculates the direct transconductance method mobility.
            Requires the gate capacitance (Farads).
            Returns Mobility and a graph if parameter is True.
            Mobility units are cm$^2$V$^{-1}$s${-1}$"""

        #Return Analysis gate voltage
        mu_dtm_2D = mobility.mobility_gated_dtm(self.conductivity_data.copy(), self.Cg)

        #Graphing
        if graph:
            #Check if graph to paste onto:
            if ax: #exists
                fig = ax.get_figure()
            else: #create new graph
                fig, (ax) = plt.subplots(1,1)
            #Plot
            ax.scatter(mu_dtm_2D[:,0], mu_dtm_2D[:,1], s=0.5)
            ax.set_xlabel("Gate voltage (V)")
            ax.set_ylabel("Mobility (cm$^2$V$^{-1}$s${-1}$)")
            return (mu_dtm_2D, fig)
        else:
            return mu_dtm_2D

    def conductivity_min(self):
        """ Finds the minimum value of conductivity and returns the index.
        """
        min = np.min(self.conductivity_data[:,1])
        index = np.where(self.conductivity_data[:,1] == min)[0][0]
        return index

    def discrete_sample_voltages(self, gate_voltages=None, center_voltage = 0, tollerance = 0.01):
        """
            Acquires a subset of resitivities, corresponding to gate voltages.
            Useful for tracking temperature behaviour over multiple samples.
            Matches requested voltage to within tollerance, from a central voltage.
        """
        #Default sampling if no list provided.
        if gate_voltages is None:
            gate_voltages = [-50,-40,-30,-20,-15,-10,-7.5,-5,-3,-2,-1,0,1,2,3,5,7.5,10,15,20,30,40,50]

        #Resistsances and gate voltages
        rvg_sampled = []

        #Find
        for gv in gate_voltages:
            vdelta = np.abs(self.conductivity_data[:,0] - center_voltage - gv)
            v_i = np.where(vdelta == np.min(vdelta))[0][0]
            if vdelta[v_i] < tollerance:
                #Offset center voltage, and take reciporical of conductivity for resistivity.
                rvg_sampled.append([self.conductivity_data[v_i,0]-center_voltage, 1.0/self.conductivity_data[v_i,1]])
            else:
                rvg_sampled.append([gv, np.nan]) #Add a NAN value to the array because value out of range.

        rvg_sampled = np.array(rvg_sampled)
        return rvg_sampled

    def __scatterVG(data, ax = None, s=1, c=None):
        if ax is None:
            fig, (ax1) = plt.subplots(1,1)
        else:
            ax1 = ax
        ax1.set_title("Electronic transport")
        ax1.set_xlabel("Gate Voltage (V)")
        ax1.tick_params(direction="in")

        if c is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        ax1.scatter(data[:,0], data[:,1], label=label, s=s, c=c)

        #Setup Axis Labels and Legends
        # handles, labels = ax1.get_legend_handles_labels()
        # fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
        return ax1

    ################### PLOTTING PARAMETERS ########################
    def plot_R_vG(self, ax = None, c = None, s=1, label=""):
        """Plots the raw resistance data versus gate voltage"""
        # Plot Resistance
        ax1 = self.__scatterVG(self.raw_data[:,0:2], ax=ax, s=s, c=c)
        # Generate Label
        ax1.set_ylabel("Resistance ($\Omega$)")
        return ax1

    def plot_Rho_vG(self, ax = None, c = None, s=1, label=""):
        """Plots the scaled resitivity data versus gate voltage"""
        # Calculate resistivity
        Rho = np.reciprocal(self.conductivity_data[:,1])
        # Plot restivitiy
        ax1 = self.__scatterVG(np.c_[self.conductivity_data[:,0], Rho], ax=ax, s=s, c=c)
        # Generate 2D/3D Label
        ax1.set_ylabel("Resisitivity ($\Omega$)") if self.is2D else ax1.set_ylabel("Resisitivity ($\Omega$ cm)")
        return ax1

    def plot_C_vG(self, ax = None, c1 = None):
        """Plots the raw conductance data versus gate voltage"""
        # Calculate conductance
        conductance = np.reciprocal(self.raw_data[:,1])
        # Plot conductance
        ax1 = self.__scatterVG(np.c_[self.raw_data[:,0], conductance], ax=ax, s=s, c=c)
        # Generate Label
        ax1.set_ylabel("Conductivity (x$10^{-3}$ S cm$^{1}$)")
        return

    def plot_Sigma_vG(self, ax = None, c1 = None):
        """Plots the scaled conductivity data versus gate voltage"""
        ax1 = self.__scatterVG(self.conductivity_data[:,0:2], ax=ax, s=s, c=c)
        ax1.set_ylabel("Conductivity ($\Omega$)") if self.is2D else ax1.set_ylabel("Conductivity (x$10^{-3}$ S cm$^{1}$)")
        return

class Meas_Temp_GatedResistance():
    """ Class to handle temperature indexed multiple sweeps of gated data.
        Measures resistance against gate voltage and temperature to infer properties."""
    def __init__(self, temps, vg, resistivity):
        """ Takes 1D arrays of temperature and gate voltage,
            and a 2D array (temp, voltage) of corresponding resitivities.
        """
        self.temps = np.array(temps)
        self.vg = np.array(vg)
        self.resistivity = np.array(resistivity)
        #Check dimensions matchup.
        if not (self.temps.shape[0] == self.resistivity.shape[0] and self.vg.shape[0] == self.resistivity.shape[1]):
            raise ValueError("Dimension mismatch: Temperature and gate voltage arrays didn't match dimensions of resisitivty array.")
        return

    def plot_Rho_vT(self, function, params, labels=None, ax=None, c=None, s=1):
        """ Generic scatter plot for resistance vs gate voltage.
            Colours length has to match length of voltages.
        """
        if c is not None and (len(self.vg) != len(c) or len(self.vg) != len(labels)):
            raise(AttributeError("There is a mismatch betweeen the number of colours (" + len(c) +"), voltages (" + len(self.vg) + "), and labels (" + str(len(labels)) + ")."))
        for i in range(len(self.vg)):
            vg = self.vg[i]
            if lable is None:
                ax.scatter(self.temps, self.resistivity[:,i], s=1, c = c[i])
            else:
                ax.scatter(self.temps, self.resistivity[:,i], s=1, c = c[i], label=labels[i])
        return

    def global_RTVg_fit(self, fit_function, params, boundsU=None, boundsL=None):
        """ Public function for globally fitting to the RTVg data.
            Fits to initial provided temperature and gate voltage dependent resistivity.
        """
        return Meas_Temp_GatedResistance.__fit(temps=self.temps, vg=self.vg, data=self.resistivity, fit_function=fit_function, x0=params, boundsU=boundsU, boundsL=boundsL)

    def global_RTVg_plot(self, function, params, ax=None, c=None, linewidth=1, labels=None, points=100, style=''):
        """ Generic plotting function for parameter set of function.
            Similar to __fit, requires
        """
        if ax is None:
            fig, (ax1) = plt.subplots(1,1)
        else:
            ax1 = ax
        ax1.set_title("Electronic transport")
        ax1.set_xlabel("Temperature (K)")
        ax1.tick_params(direction="in")

        max_t = np.max(self.temps)
        min_t = np.min(self.temps)
        fit_temps = np.linspace(min_t, max_t, points) #Arbitrary 200 plot points to look smooth.

        #Reshape inputs: First index is temp, second index is vg
        T = np.array([np.array(fit_temps) for i in range(len(self.vg))], dtype=np.longfloat) #Resize temp list for each vg.
        VG = np.array([np.array(self.vg) for i in range(len(fit_temps))], dtype=np.longfloat).T #Resize VG list for each temp.

        #Reshape inputs into 1D arrays:
        T_1D = np.reshape(T, (-1))
        VG_1D = np.reshape(VG, (-1))
        X = (T_1D, VG_1D)
        #Calculate function output
        param_resistivity = function(X, *params)

        #Plot result
        if c is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(self.vg)):
            param_resistivity_subset = param_resistivity[i*len(fit_temps):(i+1)*len(fit_temps)]
            if labels is None:
                ax1.plot(fit_temps, param_resistivity_subset, style, linewidth=linewidth, label=str(self.vg[i]), c=c[i])
            else:
                ax1.plot(fit_temps, param_resistivity_subset, style, linewidth=linewidth, label=labels[i], c=c[i])
        return

    def __fit(temps, vg, data, fit_function, x0, boundsU=None, boundsL=None):
        """ Private method to generate a fit to RVT data, not object specific.
            Requires 1D arrays of temperature, gate voltage, and a 2D matching array of data.
            This function reshapes into a 1D array for T, VG and Data, for the curve fit method to run.
        """

        #Reshape inputs: First index is temp, second index is vg
        T = np.array([np.array(temps) for i in range(len(vg))], dtype=np.longfloat) #Resize temp list for each vg.
        VG = np.array([np.array(vg) for i in range(len(temps))], dtype=np.longfloat).T #Resize VG list for each temp.
        #Reshape inputs into 1D arrays:
        T_1D = np.reshape(T, (-1))
        VG_1D = np.reshape(VG, (-1))
        data_1D = np.reshape(data.T, (-1))


        fitdata = data.copy().astype(float)
        params, covar = opt.curve_fit(fit_function, xdata=(T_1D, VG_1D), ydata=np.array(data_1D,dtype=np.longfloat), p0=x0 ,bounds=(boundsL, boundsU))

        # fitObj = RvT_data.fitParamsRvT1(params, vg, temps)
        # fitObj.fitted = True
        # fitObj.fitparams = params
        # fitObj.fitcovar = covar
        return params, covar





# class Meas_Temp_GatedResistance():
    # def __init__()


# #
# # class ClassName():
# #     """docstring for ."""
# #
# #     def __init__(self, arg):
# #         super(, self).__init__()
# #         self.arg = arg
# #
#
#
# a = np.array([[1,2,3],[5,8,13]])
# a.shape
# np.diff(a).shape
