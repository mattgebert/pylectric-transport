#Hall bar geometry device. Measurements have r_xx and/or r_xy
import numpy as np
import matplotlib.pyplot as plt
import warnings
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
        #Note, Resistance = Resistivity * L / (W * D) -> Conductivity = 1/Resistance  * L / (W * D)
        self.conductivity_data[:,1] = np.reciprocal(self.conductivity_data[:,1] * W / L) if D is None else np.reciprocal(self.conductivity_data[:,1] * W * D / L)
        self.L = L
        self.W = W
        self.D = D
        self.Cg = Cg
        self.is2D = (D is None)

    def mobility_dtm_2D(self, graph = True, ax=None, graph_kwargs = None, vg_offset = 0):
        """ Calculates the direct transconductance method mobility.
            Requires the gate capacitance (Farads).
            Returns Mobility and a graph if parameter is True.
            Mobility units are cm$^2$V$^{-1}$s${-1}$"""

        #Return Analysis gate voltage
        mu_dtm_2D = mobility.mobility_gated_dtm(self.conductivity_data.copy(), self.Cg)

        #Graphing
        if graph:
            if graph_kwargs is None:
                graph_kwargs = {}
            #Check if graph to paste onto:
            if ax: #exists
                fig = ax.get_figure()
            else: #create new graph
                fig, (ax) = plt.subplots(1,1)
            #Plot
            ax.scatter(mu_dtm_2D[:,0] - vg_offset, mu_dtm_2D[:,1], **graph_kwargs)
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
        """ Acquires a subset of resitivities, corresponding to gate voltages.
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

    def discrete_interpolated_voltages(self, gate_voltages=None, center_voltage = 0):
        """ Acquires a subset of resitivities, corresponding to gate voltages.
            Useful for tracking temperature behaviour over multiple samples.
            Matches requested voltage to within tollerance, from a central voltage.
        """

        cv = center_voltage
        #Default sampling if no list provided.
        if gate_voltages is None:
            gate_voltages = [-50,-40,-30,-20,-15,-10,-7.5,-5,-3,-2,-1,0,1,2,3,5,7.5,10,15,20,30,40,50]

        #Resistsances and gate voltages
        rvg_sampled = []

        #Find
        for gv in gate_voltages:
            vraw = self.conductivity_data[:,0] - center_voltage - gv
            vdelta = np.abs(vraw)
            v_i = np.where(vdelta == np.min(vdelta))[0][0]
            if vdelta[v_i] == 0:
                #Offset center voltage, and take reciporical of conductivity for resistivity.
                rvg_sampled.append([self.conductivity_data[v_i,0]-center_voltage, 1.0/self.conductivity_data[v_i,1]])
            else:
                if not (v_i < 1 or v_i > len(self.conductivity_data)-2): #check endpoint condition
                    # Interpolate data if not endpoints:
                    B1 = self.conductivity_data[v_i,:]
                    if vdelta[v_i + 1] < vdelta[v_i - 1]: #smaller is better for interpolation, closer to dirac point.
                        B2 = self.conductivity_data[v_i + 1,:]
                    else:
                        B2 = self.conductivity_data[v_i - 1,:]
                    #re-arranged gv = (alpha * (B1-cv) + (1-alpha) * (B2-cv)), finding linear interpolation.
                    alpha = (gv - (B2[0] - cv)) / (B1[0] - B2[0])
                    # Saftey check for result consistency.
                    if alpha < 0 or alpha > 1:
                        raise(ValueError("Calculation of linear interpolation factor (alpha = " + str(alpha) + ") is outside (0,1)."))
                    #Interolate
                    inter_v = (alpha * (B1[0] - cv)) + ((1-alpha) * (B2[0] - cv))
                    inter_resistivity = 1.0/(alpha * (B1[1]) + (1-alpha) * (B2[1]))
                    #append
                    rvg_sampled.append([inter_v, inter_resistivity]) #Add a NAN value to the array because value out of range.
                else:
                    rvg_sampled.append([gv, np.nan]) #Add a NAN value to the array because value out of range.

        rvg_sampled = np.array(rvg_sampled)
        return rvg_sampled

    def global_CVg_fit(self, fit_function, params, boundsU=None, boundsL=None):
        """ Public function for globally fitting to the R Vg data.
            Fits to initial provided temperature and gate voltage dependent resistivity.
        """
        return Meas_GatedResistance.__fit(vg=self.conductivity_data[:,0], sigma=self.conductivity_data[:,1], fit_function=fit_function, x0=params, boundsU=boundsU, boundsL=boundsL)

    def __fit(fit_function, vg, sigma, x0, boundsU=None, boundsL=None):
        """ Private method to generate a fit to Sigma Vg data, not object specific.
            Requires 1D arrays of gate voltage and conductivity (sigma).
        """
        #Check that shapes match:
        vg = np.array(vg, dtype=np.longfloat)
        sigma = np.array(sigma, dtype=np.longfloat)

        #conditions
        if not (vg.shape == sigma.shape and len(vg.shape) == 1):
            raise ValueError("Either Vg or Sigma arrays do not match in shape, or are not one-dimensional.")

        params, covar = opt.curve_fit(fit_function, xdata=vg, ydata=sigma, p0=x0 ,bounds=(boundsL, boundsU))

        return params, covar

    ################### PLOTTING PARAMETERS ########################
    def __scatterVG(data, ax = None, s=1, c=None, label=None, style=None, vg_offset = 0, scatter=True, **kwargs):
        if ax is None:
            fig, (ax1) = plt.subplots(1,1)
        else:
            ax1 = ax
            ax1.set_title("Electronic transport")
            ax1.set_xlabel("Gate Voltage (V)")
            ax1.tick_params(direction="in")

        if c is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        if style != None:
            if scatter:
                ax1.scatter(data[:,0] - vg_offset, data[:,1], style, label=label, s=s, c=c, **kwargs)
            else:
                ax1.plot(data[:,0] - vg_offset, data[:,1], style, label=label, linewidth=s, c=c, **kwargs)
        else:
            if scatter:
                ax1.scatter(data[:,0] - vg_offset, data[:,1], label=label, s=s, c=c, **kwargs)
            else:
                ax1.plot(data[:,0] - vg_offset, data[:,1], label=label, linewidth=s, c=c, **kwargs)
        return ax1

    def plot_R_vG(self, ax = None, c = None, s=1, label="", vg_offset = 0, scatter=True, **kwargs):
        """Plots the raw resistance data versus gate voltage"""
        # Plot Resistance
        ax1 = Meas_GatedResistance.__scatterVG(self.raw_data[:,0:2], ax=ax, s=s, c=c, label=label, vg_offset=vg_offset, scatter=scatter, **kwargs)
        # Generate Label
        ax1.set_ylabel("Resistance ($\Omega$)")
        return ax1

    def plot_Rho_vG(self, ax = None, c = None, s=1, label="", style=None, vg_offset = 0, scatter=True, **kwargs):
        """Plots the scaled resitivity data versus gate voltage"""
        # Calculate resistivity
        Rho = np.reciprocal(self.conductivity_data[:,1])
        # Plot restivitiy
        ax1 = Meas_GatedResistance.__scatterVG(np.c_[self.conductivity_data[:,0], Rho], ax=ax, s=s, c=c, label=label, style=style, vg_offset=vg_offset, scatter=scatter, **kwargs)
        # Generate 2D/3D Label
        ax1.set_ylabel("Resisitivity ($\Omega$)") if self.is2D else ax1.set_ylabel("Resisitivity ($\Omega$ cm)")
        return ax1

    def plot_C_vG(self, ax = None, c = None, s=1, label="", vg_offset = 0, scatter=True, **kwargs):
        """Plots the raw conductance data versus gate voltage"""
        # Calculate conductance
        conductance = np.reciprocal(self.raw_data[:,1])
        # Plot conductance
        ax1 = Meas_GatedResistance.__scatterVG(np.c_[self.raw_data[:,0], conductance], ax=ax, s=s, c=c, label=label, vg_offset=vg_offset, scatter=scatter, **kwargs)
        # Generate Label
        ax1.set_ylabel("Conductivity (S)")
        return ax1

    def plot_Sigma_vG(self, ax = None, c = None, s=1, label="", vg_offset = 0, scatter=True, **kwargs):
        """Plots the scaled conductivity data versus gate voltage"""
        ax1 = Meas_GatedResistance.__scatterVG(self.conductivity_data[:,0:2], ax=ax, s=s, c=c, label=label, vg_offset=vg_offset, scatter=scatter, **kwargs)
        ax1.set_ylabel("Conductivity (S)") if self.is2D else ax1.set_ylabel("Conductivity (S cm$^{1}$)")
        return ax1

class Meas_Temp_GatedResistance():
    """ Class to handle temperature indexed multiple sweeps of gated data.
        Measures resistance against gate voltage and temperature to infer properties."""
    def __init__(self, temps, vg, resistivity, is2D = True):
        """ Takes 1D arrays of temperature and gate voltage,
            and a 2D array (temp, voltage) of corresponding resitivities.
        """
        self.temps = np.array(temps)
        self.vg = np.array(vg)
        self.resistivity = np.array(resistivity)
        self.ylabel = "Resistivity ($\Omega$)" if is2D else "Resistivity ($\Omega$m)"

        ### Check for null columns or rows in resistivity data.
        # Clone original resistances
        new_res = np.copy(self.resistivity)
        new_vg = np.copy(self.vg)
        new_temps = np.copy(self.temps)
        # Find temperatures which are all Vg are null
        slices = []
        for k in range(self.resistivity.shape[0]):
            if np.all(np.isnan(self.resistivity[k,:])==True):
                slices.append(k)
        if len(slices) > 0:
            warnings.warn("Warning: Rows corresponding to T = " + str(self.temps[slices]) + " only contain NaN values, and are being removed.")
            new_res = np.delete(new_res, slices, axis=0)
            new_temps = np.delete(new_temps, slices)

        # Find voltages which all temperatures are null
        slices2 = []
        for l in range(self.resistivity.shape[1]):
            if np.all(np.isnan(self.resistivity[:,l])==True):
                slices2.append(l)
        if len(slices2) > 0:
            warnings.warn("Warning: Columns corresponding to Vg = " + str(self.vg[slices2]) + " only contain NaN values, and are being removed.")
            new_res = np.delete(new_res, slices2, axis=1)
            new_vg = np.delete(new_vg, slices2)
        # Set arrays to new object.
        if len(slices) > 0 or len(slices2) > 0:
            self.vg = new_vg
            self.temps = new_temps
            self.resistivity = new_res

        #Check dimensions matchup.
        if not (self.temps.shape[0] == self.resistivity.shape[0] and self.vg.shape[0] == self.resistivity.shape[1]):
            raise ValueError("Dimension mismatch: Temperature and gate voltage arrays didn't match dimensions of resisitivty array.")
        return

    def plot_Rho_vT(self, ax=None, c=None, labels=None, singleLabel=None, offsets=None, hollow=False, **kwargs):
        """ Generic scatter plot for resistance vs gate voltage.
            Colours length has to match length of voltages.
        """
        if c is not None and (len(self.vg) > len(c) or (labels is not None and len(self.vg) != len(labels))):
            raise(AttributeError("There is a mismatch betweeen the number of colours (" + str(len(c)) +"), voltages (" + str(len(self.vg)) + "), and labels (" + str(len(labels)) + ")."))
        if offsets is not None:
            if len(offsets) != len(self.vg):
                raise(AttributeError("There is a mismatch betweeen the number of offsets (" + str(len(offsets)) + ") and the number of voltages (" + str(len(self.vg)) + ")"))
        else:
            offsets = [0 for vg in self.vg]
        if ax is None:
            fig, (ax1) = plt.subplots(1,1)
        else:
            ax1 = ax
        # kwargs
        if c is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(self.vg)):
            #Colour:
            c_i = c[i % len(c)]
            kwargs["edgecolors"]=c_i
            kwargs["c"]=c_i
            if hollow and "c" in kwargs:
                kwargs.pop("c")
            elif not hollow and "edgecolors" in kwargs:
                kwargs.pop("edgecolors")
            #Labels:
            if singleLabel is not None:
                if i == 0:
                    label=singleLabel
                else:
                    label=None
            else:
                if labels is None:
                    vg = self.vg[i]
                    label="%0.02f" % vg
                else:
                    label=labels[i]
            kwargs["label"]=label
            # Boom.
            ax1.scatter(self.temps, self.resistivity[:,i] - offsets[i], **kwargs)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel(self.ylabel)
        return ax1

    def global_RTVg_fit(self, fit_function, params, boundsU=None, boundsL=None):
        """ Public function for globally fitting to the RTVg data.
            Fits to initial provided temperature and gate voltage dependent resistivity.
        """
        return Meas_Temp_GatedResistance.__fit(temps=self.temps, vg=self.vg, data=self.resistivity, fit_function=fit_function, x0=params, boundsU=boundsU, boundsL=boundsL)

    def global_RTVg_plot(self, function, params, ax=None, c=None, linewidth=1, labels=None, points=100, style='', singleLabel=None, offsets=None, T_max=None):
        """ Generic plotting function for parameter set of function.
            Similar to __fit, requires
        """
        if ax is None:
            fig, (ax1) = plt.subplots(1,1)
        else:
            ax1 = ax

        #Check if `offset`s exist for each gate voltage. If not generate 0 values.
        if offsets is not None:
            if len(offsets) != len(self.vg):
                raise(AttributeError("There is a mismatch betweeen the number of offsets (" + str(len(offsets)) + ") and the number of voltages (" + str(len(self.vg)) + ")"))
        else:
            offsets = [0 for vg in self.vg]

        # ax1.set_title("Electronic transport")
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel(self.ylabel)
        ax1.tick_params(direction="in")

        if T_max is None:
            max_t = np.max(self.temps)
        else:
            max_t = T_max
        min_t = np.min(self.temps)
        fit_temps = np.linspace(min_t, max_t, points) #Arbitrary 100 plot points to look smooth.

        #Reshape inputs: First index is temp, second index is vg
        T = np.array([np.array(fit_temps) for i in range(len(self.vg))], dtype=np.longfloat) #Resize temp list for each vg.
        VG = np.array([np.array(self.vg) for i in range(len(fit_temps))], dtype=np.longfloat).T #Resize VG list for each temp.

        #Reshape inputs into 1D arrays:
        T_1D = np.reshape(T, (-1))
        VG_1D = np.reshape(VG, (-1))
        X = (T_1D, VG_1D)
        #Calculate function output
        param_resistivity = function(X, *params)

        self.last_vg = VG_1D
        self.last_T = T_1D
        self.last_res = param_resistivity

        #Plot result
        if c is None:
            c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(self.vg)):
            c_i = c[i % len(c)]
            param_resistivity_subset = param_resistivity[i*len(fit_temps):(i+1)*len(fit_temps)]
            if singleLabel is not None:
                if i==0:
                    ax1.plot(fit_temps, param_resistivity_subset - offsets[i], style, linewidth=linewidth, label=str(singleLabel), c=c_i)
                else:
                    ax1.plot(fit_temps, param_resistivity_subset - offsets[i], style, linewidth=linewidth, c=c_i)
            else:
                if labels is None:
                    ax1.plot(fit_temps, param_resistivity_subset - offsets[i], style, linewidth=linewidth, label=str(self.vg[i]), c=c_i)
                else:
                    ax1.plot(fit_temps, param_resistivity_subset - offsets[i], style, linewidth=linewidth, label=labels[i], c=c_i)
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

        #Define warning strings.
        tmsg = "Warning: Some temperatures were 'NaN' value; %0.0d datalines have been removed (of total %0.0d)."
        vmsg = "Warning: Some voltages were 'NaN' value; %0.0d datalines have been removed (of total %0.0d)."
        rmsg = "Warning: Some resistances were 'NaN' value; %0.0d datalines have been removed (of total %0.0d)."
        xmsg = "Warning: Some initial params were 'NaN' value; %0.0d param values have been set to initialize at 0 (of total %0.0d params)."

        #Find any NaN data in the reshaped input.
        nans = np.where(np.isnan(T_1D))[0]
        # Remove NaN data
        if len(nans) > 0:
            T_1D = np.delete(T_1D, nans)
            VG_1D = np.delete(VG_1D, nans)
            data_1D = np.delete(data_1D, nans)
            warnings.warn(tmsg % (len(nans), len(T_1D)))
        #Repeat
        nans = np.where(np.isnan(VG_1D))[0]
        if len(nans) > 0:
            T_1D = np.delete(T_1D, nans)
            VG_1D = np.delete(VG_1D, nans)
            data_1D = np.delete(data_1D, nans)
            warnings.warn(vmsg % (len(nans), len(VG_1D)))
        #Repeat
        nans = np.where(np.isnan(data_1D))[0]
        if len(nans) > 0:
            T_1D = np.delete(T_1D, nans)
            VG_1D = np.delete(VG_1D, nans)
            data_1D = np.delete(data_1D, nans)
            warnings.warn(rmsg % (len(nans), len(data_1D)))

        # Check if nans in x0 as a reuslt too:
        nans = np.where(np.isnan(x0))[0]
        if len(nans) > 0:
            warnings.warn(xmsg % (len(nans), len(x0)))
        x0 = np.nan_to_num(x0)

        #Now fit data!
        params, covar = opt.curve_fit(fit_function, xdata=(T_1D, VG_1D), ydata=np.array(data_1D,dtype=np.longfloat), p0=x0 ,bounds=(boundsL, boundsU))

        return params, covar
