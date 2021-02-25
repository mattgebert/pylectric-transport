### This module is to deal with the import of Resistance (R) as a function of Gate Voltage (V_g) data.
### This can be used to help import data. Alternatively import yourself into Numpy arrays.

import numpy as np
import matplotlib.pyplot as plt

class RVG_file():
    """Object for loading and reading RVG data"""
    def __init__(self, filepath):

        #Read header information (1: Titles, 2: Units, 3: Extra information.)
        with open(filepath,"r") as file:
            self.HEADERS = file.readline().split("\t")
            self.UNITS = file.readline().split("\t")
            self.COMMENTS = file.readline().split("\t")

        #Read data file -> \newline&Rows=lines of data, \tab&Columns=variable
        self.RAW_DATA = np.genfromtxt(filepath, dtype=float, delimiter="\t", skip_header=3) #Raw data
        if self.HEADERS[-1] == "\n":
            self.RAW_DATA = self.RAW_DATA[:,:-1]
        return

    def plot_all(self,ivar_index = 0):
        """Plots all parameters against the independent variable (ivar_index)."""

        num_vars = self.RAW_DATA.shape[1] - 1 #Exclude independent
        #Generate a nice display grid depending on number of variables.
        ii,jj = (1,1)
        while ii*jj < num_vars:
            if ii > 3*jj or ii>=4:
                jj+=1
            else:
                ii+=1
        #Make grid:
        plt.rcParams.update({'font.size': 3, "figure.figsize" : [ii,jj], 'figure.dpi':300})
        fig, (axes) = plt.subplots(jj,ii)
        axes = axes.reshape(-1) #squish list
        for i in range(num_vars):
            j = i
            if i >= ivar_index:
                j += 1 #Push index ahead.
            ax = axes[i]

            #Scale data below minimum 0.001 for pyplot.
            factor = 0
            while np.all(self.RAW_DATA[:,j] < 0.001):
                self.RAW_DATA[:,j] = self.RAW_DATA[:,j]*10
                factor += 1

            #Plot data:
            ax.scatter(self.RAW_DATA[:,ivar_index], self.RAW_DATA[:,j], s=0.5)
            ax.set_xlabel(str(self.HEADERS[ivar_index]) + " (" + str(self.UNITS[ivar_index]) + ")")
            if factor > 0:
                ax.set_ylabel(str(self.HEADERS[j]) + " (x$10^{-" + str(factor) + "}$ " + str(self.UNITS[j]) + ")")
            else: #No factor
                ax.set_ylabel(str(self.HEADERS[j]) + " (" + str(self.UNITS[j]) + ")")
            ax.tick_params(direction="in")
        plt.tight_layout()
        return fig

    def split_dataset_by_voltage_turning_point(self, ivar_index=0):
        """Splits a dataset in half, where the return run of the data also exists.
            Note that this includes flipping the direction of the second data run.
        """
        maxima = np.where(self.RAW_DATA[:,ivar_index]==np.max(self.RAW_DATA[:,ivar_index]))
        if len(maxima) == 1:
            i = maxima[0][0]
            return (self.RAW_DATA[:i+1,:], self.RAW_DATA[:i:-1,:])
        else:
            minima = np.where(self.RAW_DATA[:,ivar_index]==np.min(self.RAW_DATA[:,ivar_index]))
            if len(minima) == 1:
                i = minima[0][0]
                return (self.RAW_DATA[i+1:0,:], self.RAW_DATA[i:,:])
            else:
                raise AttributeError("The dataset does not have a single turning point.")
                return None

# class RVG_measurement():
    # def __init__(self, data):
#         self.RAW_DATA = data.copy()
#
#         #Initialize Class Properties
#         self.HAS_TEMP_DATA = None #Flag for temperature
#         self.SINGLE_TURNING_POINT = None #Flag for where the sweep begins from relative to the max and min.
#         self.TEMP_MEAN = None #Average temperature
#         self.TEMP_VAR = None #Variance of temperaturee
#         self.max_U = None #Maximum resistivity in U sweep
#         self.max_D = None #Maximum resistivity in D sweep
#         self.max_U_i = None #Index for maximum resistance in U sweep
#         self.max_D_i = None #Index for maximum resistance in D sweep
#         self.DX = np.abs(self.RAW_DATA[0,0] - self.RAW_DATA[1,0]) #Step spacing
#
#         #Check the turning point and separate updown sweep data
#         self.RAW_DATA_U = None #Note resistance -> resistivity
#         self.RAW_DATA_D = None #Note resistance -> resistivity
#         self.isolate_sweep_data()
#         #get max resisitivities:
#         self.max_U = np.amax(self.RAW_DATA_U,0)
#         self.max_D = np.amax(self.RAW_DATA_D,0)
#         #get indexes
#         self.max_U_i = np.where(self.RAW_DATA_U == self.max_U[1])[0]
#         self.max_D_i = np.where(self.RAW_DATA_D == self.max_D[1])[0]
#
#         #Calculate the mean and variance of the temp
#         self.get_temp_dist()
#
#         #Find the resistance for trends at different gate voltages.
#         self.get_vg_simple_points()
#
#         return
