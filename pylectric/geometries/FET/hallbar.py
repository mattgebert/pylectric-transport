#Hall bar geometry device. Measurements have r_xx and/or r_xy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pylectric.analysis import mobility

class Meas_GatedResistance():
    """Class to handle a single sweep of raw data. Measures resistance against gate voltage to infer properties."""

    def __init__(self, data, L, W, Cg):
        """Takes numpy array of the form array[rows, columns]
           Requires columns to be of the form: [V_gate (Volts), Resistance (Ohms)]"""

        #1. Cols = [V_gate, Resistance]
        self.raw_data = np.array(data).copy()
        self.conductivity_data = self.raw_data.copy()
        self.conductivity_data[:,1] = np.reciprocal(self.conductivity_data[:,1] * W / L)
        self.L = L
        self.W = W
        self.Cg = Cg

    def mobility_dtm(self, graph = True, ax=None):
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
