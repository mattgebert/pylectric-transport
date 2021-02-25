#Hall bar geometry device. Measurements have r_xx and/or r_xy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pylectric.analysis import mobility

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
        self.raw_data = np.array(data).copy()
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
            Acquires a subset of gate voltages, useful for tracking temperature
            dependent behaviour over multiple samples.
            Matches requested voltages to within tollerance, from a central voltage.
        """
        #Default sampling if no list provided.
        if gate_voltages is None:
            gate_voltages = [-50,-40,-30,-20,-15,-10,-7.5,-5,-3,-2,-1,0,1,2,3,5,7.5,10,15,20,30,40,50]

        #Resistsances and gate voltages
        rvg_sampled = []

        #Find
        for gv in gate_voltages:
            vdelta = np.abs(self.raw_data[:,0] - center_voltage - gv)
            v_i = np.where(vdelta == np.min(vdelta))[0][0]
            if vdelta[v_i] < tollerance:
                rvg_sampled.append([self.raw_data[v_i,0]-center_voltage, self.raw_data[v_i,1]])
            else:
                rvg_sampled.append([gv, np.nan]) #Add a NAN value to the array because value out of range.

        rvg_sampled = np.array(rvg_sampled)
        return rvg_sampled

    def _scatterVG(data, ax = None, s=1, c=None):
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
        ax1 = self._scatterVG(self.raw_data[:,0:2], ax=ax, s=s, c=c)
        # Generate Label
        ax1.set_ylabel("Resistance ($\Omega$)")
        return ax1

    def plot_Rho_vG(self, ax = None, c = None, s=1, label=""):
        """Plots the scaled resitivity data versus gate voltage"""
        # Calculate resistivity
        Rho = np.reciprocal(self.conductivity_data[:,1])
        # Plot restivitiy
        ax1 = self._scatterVG(np.c_[self.conductivity_data[:,0], Rho], ax=ax, s=s, c=c)
        # Generate 2D/3D Label
        ax1.set_ylabel("Resisitivity ($\Omega$)") if self.is2D else ax1.set_ylabel("Resisitivity ($\Omega$ cm)")
        return ax1

    def plot_C_vG(self, ax = None, c1 = None):
        """Plots the raw conductance data versus gate voltage"""
        # Calculate conductance
        conductance = np.reciprocal(self.raw_data[:,1])
        # Plot conductance
        ax1 = self._scatterVG(np.c_[self.raw_data[:,0], conductance], ax=ax, s=s, c=c)
        # Generate Label
        ax1.set_ylabel("Conductivity (x$10^{-3}$ S cm$^{1}$)")
        return

    def plot_Sigma_vG(self, ax = None, c1 = None):
        """Plots the scaled conductivity data versus gate voltage"""
        ax1 = self._scatterVG(self.conductivity_data[:,0:2], ax=ax, s=s, c=c)
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

    def global_fit_RTVg():
        return

    def _fit(temp, vg, data, params = (3, 1, 2, 50), R0s_guess = None): #Da, A1, B1, R0
        #Global Fit Variables
        Da = params[0]
        a1 = params[1]
        B1 = params[2]

        #Reshape inputs: First index is temp, second index is vg
        T = np.array([np.array(temps) for i in range(len(vg))], dtype=float) #Resize temp list for each vg.
        VG = np.array([np.array(vg) for i in range(len(temps))], dtype=float).T #Resize VG list for each temp.
        #Reshape inputs into 1D arrays:
        T_1D = np.reshape(T, (-1))
        VG_1D = np.reshape(VG, (-1))
        data_1D = np.reshape(data.T, (-1))

        #Independent Fit Variables:
        R = []
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            #Each Vg has an offset resistance R0:
            if R0s_guess is not None and len(R0s_guess) == len(vg):
                R.append(R0s_guess[i])
            else:
                R.append(params[3])
            Rlower.append(0)
            Rupper.append(20000)
        R = tuple(R)
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        defaultBoundsL = [0.1,0.1,0.1] + list(Rlower)
        defaultBoundsU = [1e6, np.inf, 25] + list(Rupper)

        x0 = [Da, a1, B1]
        x0 += list(R)
        x0 = tuple(x0)
        fitdata = data.copy().astype(float)
        params, covar = opt.curve_fit(RvT_data.rho_T_1D, xdata=(T_1D, VG_1D), ydata=np.array(data_1D,dtype=float), p0=x0 ,bounds=(defaultBoundsL, defaultBoundsU))

        fitObj = RvT_data.fitParamsRvT1(params, vg, temps)
        fitObj.fitted = True
        fitObj.fitparams = params
        fitObj.fitcovar = covar
        return fitObj





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
