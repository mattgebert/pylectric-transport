__version__ = "0.0.1" #initial dump

#Class interpret RvG datafile, intepret sweeps and plot data.
class RVG_data():
    def __init__(self, filepath, geo_factor=1):
        #Set geo factor for devices:  \rho = R * (Geo_factor) = R * (W / L))
        self.GEO_FACTOR = geo_factor

        #Initialize Class Properties
        self.HAS_TEMP_DATA = None #Flag for temperature
        self.SINGLE_TURNING_POINT = None #Flag for where the sweep begins from relative to the max and min.
        self.TEMP_MEAN = None #Average temperature
        self.TEMP_VAR = None #Variance of temperaturee
        self.max_U = None #Maximum resistivity in U sweep
        self.max_D = None #Maximum resistivity in D sweep
        self.max_U_i = None #Index for maximum resistance in U sweep
        self.max_D_i = None #Index for maximum resistance in D sweep

        #Read header information (1: Titles, 2: Units, 3: Extra information.)
        with open(filepath,"r") as file:
            self.HEADERS = file.readline().split("\t")
            self.UNITS = file.readline().split("\t")
            self.COMMENTS = file.readline().split("\t")

        #Read data file
        self.RAW_DATA = np.genfromtxt(filepath, dtype=float, delimiter="\t", skip_header=3) #Raw data
        self.DX = np.abs(self.RAW_DATA[0,0] - self.RAW_DATA[1,0]) #Step spacing

        #Check the turning point and separate updown sweep data
        self.RAW_DATA_U = None #Note resistance -> resistivity
        self.RAW_DATA_D = None #Note resistance -> resistivity
        self.isolate_sweep_data()
        #get max resisitivities:
        self.max_U = np.amax(self.RAW_DATA_U,0)
        self.max_D = np.amax(self.RAW_DATA_D,0)
        #get indexes
        self.max_U_i = np.where(self.RAW_DATA_U == self.max_U[1])[0]
        self.max_D_i = np.where(self.RAW_DATA_D == self.max_D[1])[0]

        #Calculate the mean and variance of the temp
        self.get_temp_dist()

        #Find the resistance for trends at different gate voltages.
        self.get_vg_simple_points()

        return

    def isolate_sweep_data(self):
        #Aquire max and min of all columns.
        self.max = np.amax(self.RAW_DATA,0)
        self.min = np.amin(self.RAW_DATA,0)
        #If single turning point, max and min gate values are reached total of 3 times.
        #If two turning points, then max and min gate values are reached total of 2 times.
        self.max_vg_i = np.where(self.RAW_DATA[:,0] == self.max[0])[0]
        self.min_vg_i = np.where(self.RAW_DATA[:,0] == self.min[0])[0]

        # Set SINGLE_TURNING_POINT variable appropriately.
        x = len(self.max_vg_i) + len(self.min_vg_i)
        if x == 2:
            self.SINGLE_TURNING_POINT = False
        elif x == 3:
            self.SINGLE_TURNING_POINT = True
        else:
            raise IOError("Could not interpret the ends of the filepath correctly.")

        # Use indexes to separate datasets (upward sweep and downward sweep)
        if self.SINGLE_TURNING_POINT:
            RAW_DATA_U = self.RAW_DATA[0:self.max_vg_i[0]+1]
            RAW_DATA_D = self.RAW_DATA[self.max_vg_i[0]:]
            #swap downward sweep listed data direction
            RAW_DATA_D = RAW_DATA_D[::-1]
        else:
            RAW_DATA_U = np.concatenate((self.RAW_DATA[self.min_vg_i[0]:], self.RAW_DATA[0:self.max_vg_i[0]+1]), axis=0)
            RAW_DATA_D = self.RAW_DATA[self.min_vg_i[0]+1 : self.max_vg_i[0]] #inherently swaps downward sweep too.

        #Use GEO_FACTOR to convert resistance to resistivity.
        RAW_DATA_U[:,1] = RAW_DATA_U[:,1] * self.GEO_FACTOR
        RAW_DATA_D[:,1] = RAW_DATA_D[:,1] * self.GEO_FACTOR
        #Write to class property.
        self.RAW_DATA_U = RAW_DATA_U
        self.RAW_DATA_D = RAW_DATA_D
        return

    def get_temp_dist(self):
        self.TEMP_VAR = np.var(self.RAW_DATA[:,4], axis=0)
        self.TEMP_MEAN = np.mean(self.RAW_DATA[:,4], axis=0)
        return

    def get_vg_simple_points(self):

        class vg_points():
            def __init__(self, raw_data_u, raw_data_d, DX):
                #find all other indexes for different gate voltages to track.
                self.GATE_VOLTAGES = [-50,-40,-30,-20,-15,-10,-7.5,-5,-3,-2,-1,0,1,2,3,5,7.5,10,15,20,30,40,50]
                #Resistsances
                self.rvg_u = []
                self.rvg_d = []

                #get max resistances
                max_U = np.amax(raw_data_u,0)
                max_D = np.amax(raw_data_d,0)
                #get indexes
                max_U_i = np.where(raw_data_u[:,1] == max_U[1])[0]
                max_D_i = np.where(raw_data_d[:,1] == max_D[1])[0]
                #take middle index if multiple
                max_U_i = max_U_i[int(np.floor(len(max_U_i)/2))]
                max_D_i = max_D_i[int(np.floor(len(max_D_i)/2))]

                for gv in self.GATE_VOLTAGES:
                    di = int(gv / DX)

                    if (max_U_i + di > len(raw_data_u)) or (max_U_i + di < 0):
                        print(str(gv) + " V outside of U bounds.")
                        self.rvg_u.append(np.nan)
                    else:
                        self.rvg_u.append(raw_data_u[max_U_i + di,1])

                    if (max_D_i + di > len(raw_data_d) or (max_D_i + di < 0)):
                        print(str(gv) + " V outside of D bounds.")
                        self.rvg_d.append(np.nan)
                    else:
                        self.rvg_d.append(raw_data_d[max_D_i + di,1])
                return

            def plotRvG(self):
                fig, (ax1) = plt.subplots(1,1)
                ax1.scatter(self.GATE_VOLTAGES, self.rvg_u, label="→", s=2)
                ax1.scatter(self.GATE_VOLTAGES, self.rvg_d, label="←", s=2)
                #Setup Axis Labels and Legends
                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.2,0.5), loc = "center")
                ax1.set_title("Electronic transport")
                ax1.set_xlabel("Gate Voltage (V)")
                # ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
                ax1.set_ylabel(r"Resitivity ($\Omega$)")
                ax1.tick_params(direction="in")


        self.sampled_vg_data = vg_points(self.RAW_DATA_U, self.RAW_DATA_D, self.DX)
        return self.sampled_vg_data

    def plotRvG(self):
        if self.RAW_DATA_U is not None:
            #Plot data
            fig, (ax1) = plt.subplots(1,1)
            ax1.scatter(self.RAW_DATA_U[:,0], self.RAW_DATA_U[:,1], label="→ " + "{:.2f} K".format(self.TEMP_MEAN), s=2)
            ax1.scatter(self.RAW_DATA_D[:,0], self.RAW_DATA_D[:,1], label="← " + "{:.2f} K".format(self.TEMP_MEAN), s=2)
            #Setup Axis Labels and Legends
            # handles, labels = ax1.get_legend_handles_labels()
            # fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
            ax1.set_title("Electronic transport")
            ax1.set_xlabel("Gate Voltage (V)")
            # ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
            ax1.set_ylabel(r"Resistivity ($\Omega$)")
            ax1.tick_params(direction="in")
            return ax1
        else:
            return None

    def plotCvG(self, ax = None, c1 = None, c2 = None, s=1, up=True, down = True):
        if self.RAW_DATA_U is not None and self.RAW_DATA_D is not None:
            #Plot data
            if ax is None:
                fig, (ax1) = plt.subplots(1,1)
            else:
                ax1 = ax
            if up:
                ax1.scatter(self.RAW_DATA_U[:,0], np.reciprocal(self.RAW_DATA_U[:,1]), label="→ " + "{:.0f} K".format(self.TEMP_MEAN), s=s, c=c1)
            if down:
                ax1.scatter(self.RAW_DATA_D[:,0], np.reciprocal(self.RAW_DATA_D[:,1]), label="← " + "{:.0f} K".format(self.TEMP_MEAN), s=s, c=c2)
            #Setup Axis Labels and Legends
            # handles, labels = ax1.get_legend_handles_labels()
            # fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
            ax1.set_title("Electronic transport")
            ax1.set_xlabel("Gate Voltage (V)")
            # ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
            ax1.set_ylabel("Conductivity (x$10^{-3}$ S)")
            ax1.tick_params(direction="in")

            scale_y = 1e-3
            ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
            ax1.yaxis.set_major_formatter(ticks_y)
            return ax1
        else:
            return None

    def fitDataU(self, p0=None):
        if p0 is None:
            return fitParamsRvG.fitRes(data=self.RAW_DATA_U[:,0:2])
        else:
            return fitParamsRvG.fitRes(data=self.RAW_DATA_U[:,0:2], p0=p0)

    def fitDataD(self, p0=None):
        if p0 is None:
            return fitParamsRvG.fitRes(data=self.RAW_DATA_D[:,0:2])
        else:
            return fitParamsRvG.fitRes(data=self.RAW_DATA_D[:,0:2], p0=p0)

    def fitDataU_gauss(self, p0=None):
        if p0 is None:
            return fitParamsRvG.fitRes_gauss(data=self.RAW_DATA_U[:,0:2])
        else:
            return fitParamsRvG.fitRes_gauss(data=self.RAW_DATA_U[:,0:2], p0=p0)

    def fitDataD_gauss(self, p0=None):
        if p0 is None:
            return fitParamsRvG.fitRes_gauss(data=self.RAW_DATA_D[:,0:2])
        else:
            return fitParamsRvG.fitRes_gauss(data=self.RAW_DATA_D[:,0:2], p0=p0)

### ----------------------------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------------------------------------------------------------------------------------- ###
#Create the fitting fiuctnion
class fitParamsRvG():
    def __init__(self, sigma_pud_e = 1e-4, mu_e = 1000.0, mu_h = 1000.0 , rho_s_e = 100.0, rho_s_h = 100.0, vg_dirac = 0, sigma_const = 0.0, pow = 2.85):
        self.sigma_pud_e = sigma_pud_e
        self.mu_e = mu_e
        self.vg_dirac = vg_dirac
        self.mu_h = mu_h
        self.rho_s_e = rho_s_e
        self.rho_s_h = rho_s_h
        self.sigma_const = sigma_const
        self.pow = pow
        self.fitted = False
        self.fitparams = None
        self.fitcovar = None

    def initCovar(self, P=None):
        # https://stats.stackexchange.com/questions/50830/can-i-convert-a-covariance-matrix-into-uncertainties-for-variables
        #For each element of the covariance matrix, you need to convert the diagonal element information into +- uncertainty.
        #This can be done by simply square rooting.
        if P is None:
            if not hasattr(P, 'shape'):
                raise AttributeError("Fitted parameters don't have covariance matrix, or not supplied to method.")
            else:
                errs = np.sqrt(np.diag(self.fitcovar))
                self.sigma_pud_e_err, self.mu_e_err, self.vg_dirac_err, self.mu_h_err, self.rho_s_e_err, self.rho_s_h_err, self.sigma_const_err, self.pow_err = errs
        else:
            errs = np.sqrt(np.diag(P))
            self.sigma_pud_e_err, self.mu_e_err, self.vg_dirac_err, self.mu_h_err, self.rho_s_e_err, self.rho_s_h_err, self.sigma_const_err, self.pow_err = errs
        return

    def fitFuncP(vg, params):
        return fitParamsRvG.fitFunc(vg, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])

    def fitFuncP_gauss(vg, params):
        return fitParamsRvG.fitFuncGauss(vg, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8])

    def fitFunc(vg , sigma_pud_e = 1e-4, mu_e = 1000.0, mu_h = 1000.0 , rho_s_e = 100.0, rho_s_h = 100.0, vg_dirac = 0, sigma_const = 0.0, pow = 2.85):

        #Define Constants
        EPSILON_0 = 8.85e-12    #natural permissitivity constant
        EPSILON_SiO2 = 3.8      #relative sio2 permissivity factor
        e = 1.6e-19             #elementary chrage
        t_ox = 2.85e-7          #oxide thickness

        # -- Calculate terms --
        #Gate capacitance
        Cg = EPSILON_0 * EPSILON_SiO2 / t_ox / 10000 #10000 is to change from metric untis into units of cm^2.
        #Field effect carrier density
        N_c = Cg / e * np.abs(vg-vg_dirac)
        #Interred hole puddle density due to electron fit.
        sigma_pud_h = 1/(rho_s_e - rho_s_h + 1/sigma_pud_e)
        #electron and hole conductivity
        sigma_h = 1 / (rho_s_h + np.power(1/(np.power((sigma_pud_h),pow) + np.power(N_c * e * mu_h,pow)),1/pow))
        sigma_e = 1 / (rho_s_e + np.power(1/(np.power((sigma_pud_e),pow) + np.power(N_c * e * mu_e,pow)),1/pow))
        #condition for fitting
        cond = [vg > vg_dirac, vg <= vg_dirac]
        #gate dependent conductivity
        sigma = sigma_const + np.select(cond, [sigma_e, sigma_h])
        return sigma

    def fitFuncGauss(vg , sigma_pud_e = 1e-4, mu_e = 1000.0, mu_h = 1000.0 , rho_s_e = 100.0, rho_s_h = 100.0, vg_dirac = 0, sigma_const = 0.0, pow = 2.85, gauss_w=2):

        #Define Constants
        EPSILON_0 = 8.85e-12    #natural permissitivity constant
        EPSILON_SiO2 = 3.8      #relative sio2 permissivity factor
        e = 1.6e-19             #elementary chrage
        t_ox = 2.85e-7          #oxide thickness

        #Setup gaussian dirac point.
        def gauss(x, p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        # -- Calculate terms --
        #Gate capacitance
        Cg = EPSILON_0 * EPSILON_SiO2 / t_ox / 10000 #10000 is to change from metric untis into units of cm^2.
        #Field effect carrier density
        N_c = Cg / e * np.abs(vg-vg_dirac)
        #gaussian nature of dirac point:
        gauss_e = gauss(vg,(1.0/sigma_pud_e, vg_dirac, gauss_w))
        #Interred hole puddle density due to electron fit.
        # sigma_pud_h = 1/(rho_s_e - rho_s_h + 1/sigma_pud_e)

        gauss_h = gauss(vg,(1.0/sigma_pud_h, vg_dirac, gauss_w))
        #electron and hole conductivity
        # sigma_h = 1 / (rho_s_h + np.power(1/(np.power((sigma_pud_h),pow) + np.power(N_c * e * mu_h,pow)),1/pow))
        # sigma_e = 1 / (rho_s_e + np.power(1/(np.power((sigma_pud_e),pow) + np.power(N_c * e * mu_e,pow)),1/pow))
        sigma_h = 1 / (rho_s_h + np.power(1/(np.power((gauss_h),pow) + np.power(N_c * e * mu_h,pow)),1/pow))
        sigma_e = 1 / (rho_s_e + np.power(1/(np.power((gauss_e),pow) + np.power(N_c * e * mu_e,pow)),1/pow))
        #condition for fitting
        cond = [vg > vg_dirac, vg <= vg_dirac]
        #gate dependent conductivity
        sigma = sigma_const + np.select(cond, [sigma_e, sigma_h])
        return sigma

    def plotFit(self, x1 = None, ax = None, label="", c=None, s=1):
        # Intialize parameters
        if self.fitted:
            return fitParamsRvG.plotParamsP(x1=x1, ax = ax, params = tuple(self.fitparams), label=label, c=c, s=s)
        else:
            return None

    def plotParams(self, x1, ax=None, label = "", c = None, s=1):
        return fitParamsRvG.plotParamsP(x1=x1, ax=ax, params=(self.sigma_pud_e, self.mu_e, self.mu_h, self.rho_s_e, self.rho_s_h, self.vg_dirac, self.sigma_const, self.pow), label = label, c=c, s=s)

    def plotParamsP(x1, ax = None, params = (3e-4, 1000.0, 1000.0, 100.0, 100.0, 0.0, 0.0, 2.85), label="", c=None, s=1):
        # Initialize figure and axis.
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
            #Clear existing legend:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # Generate data based on params
        if x1 is None:
            x1 = np.linspace(-80, 80, 161)
        y1 = fitParamsRvG.fitFuncP(x1, params)
        ax.plot(x1,y1,label=label, c=c, linewidth=s)

        #Setup Axis Labels and Legends
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
        ax.set_title("Electronic transport")
        ax.set_xlabel("Gate Voltage (V)")
        # ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
        ax.set_ylabel("Conductivity (x$10^{-3}$ S)")
        ax.tick_params(direction="in")
        scale_y = 1e-3
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        return ax

    def plotParamsP_gauss(x1, ax = None, params = (3e-4, 1000.0, 1000.0, 100.0, 100.0, 0.0, 0.0, 2.85, 2), label="", c=None, s=1):
        # Initialize figure and axis.
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
            #Clear existing legend:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # Generate data based on params
        if x1 is None:
            x1 = np.linspace(-80, 80, 161)
        y1 = fitParamsRvG.fitFuncP_gauss(x1, params)
        ax.plot(x1,y1,label=label, c=c, linewidth=s)

        #Setup Axis Labels and Legends
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05,0.5), loc = "center")
        ax.set_title("Electronic transport")
        ax.set_xlabel("Gate Voltage (V)")
        # ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
        ax.set_ylabel("Conductivity (x$10^{-3}$ S)")
        ax.tick_params(direction="in")
        scale_y = 1e-3
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        return ax

    def fitCond(data, p0=(3e-4, 1000.0, 1000.0, 100.0, 100.0, None, 0.0, 2.85)):

        #If None guess for Vg_dirac, then use min conductivity/max resistance.
        if p0[5] is None:
            #get min cond index
            minv = np.amin(data,0)
            mini = np.where(data == minv[1])[0]
            if len(mini) > 1: #Take middle element if multiple matches.
                mini = mini[int(np.floor(len(mini)/2.0))]
            #modify list
            lst = list(p0)
            lst[5] = float(data[mini,0]) #Max resistance
            p0 = tuple(lst)

        #Assuming data[:,1] is conductivity.
        #Assuming data[:,0] is gate voltage.
        # defaultBoundsL = [0,0,0,0,0,-100,0,2]
        # defaultBoundsU = [5e-3,1e5,1e5,1e4,1e4,100,5e-3,5]
        defaultBoundsL = [0,0,0,0,0,p0[5],0,2]
        defaultBoundsU = [5e-3,1e5,1e5,1e4,1e4,p0[5]+0.1,5e-3,3]

        fitdata = data[:,0:2].copy().astype(float)
        params, covar = opt.curve_fit(fitParamsRvG.fitFunc, fitdata[:,0], fitdata[:,1], p0=p0, bounds=(defaultBoundsL, defaultBoundsU))

        fitObj = fitParamsRvG(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
        fitObj.fitted = True
        fitObj.fitparams = params
        fitObj.fitcovar = covar
        fitObj.initCovar(P=covar)
        return fitObj

    def fitCond_gauss(data, p0=(3e-4, 1000.0, 1000.0, 100.0, 100.0, None, 0.0, 2.85, 2)):

        #If None guess for Vg_dirac, then use min conductivity/max resistance.
        if p0[5] is None:
            #get min cond index
            minv = np.amin(data,0)
            mini = np.where(data == minv[1])[0]
            if len(mini) > 1: #Take middle element if multiple matches.
                mini = mini[int(np.floor(len(mini)/2.0))]
            #modify list
            lst = list(p0)
            lst[5] = float(data[mini,0]) #Max resistance
            p0 = tuple(lst)

        #Assuming data[:,1] is conductivity.
        #Assuming data[:,0] is gate voltage.
        # defaultBoundsL = [0,0,0,0,0,-100,0,2,0]
        # defaultBoundsU = [5e-3,1e5,1e5,1e4,1e4,100,5e-3,5, 100]
        defaultBoundsL = [0,0,0,0,0,p0[5],0,2,0]
        defaultBoundsU = [5e-3,1e5,1e5,1e4,1e4,p0[5]+0.1,5e-3,5, 100]

        fitdata = data[:,0:2].copy().astype(float)
        params, covar = opt.curve_fit(fitParamsRvG.fitFunc, fitdata[:,0], fitdata[:,1], p0=p0, bounds=(defaultBoundsL, defaultBoundsU))

        # fitObj = fitParamsRvG(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
        # fitObj.fitted = True
        # fitObj.fitparams = params
        # fitObj.fitcovar = covar
        # fitObj.initCovar(P=covar)
        return params, covar

    def fitRes(data, p0=(3e-4, 3000.0, 3000.0, 100.0, 100.0, None, 0.0, 2.85)):
        cond_data = data.copy()
        cond_data[:,1] = np.reciprocal(data[:,1])
        return fitParamsRvG.fitCond(data=cond_data, p0=p0)


    def fitRes_gauss(data, p0=(3e-4, 3000.0, 3000.0, 100.0, 100.0, None, 0.0, 2.85, 2)):
        cond_data = data.copy()
        cond_data[:,1] = np.reciprocal(data[:,1])
        return fitParamsRvG.fitCond(data=cond_data, p0=p0)

class fitSet():
    def __init__(self,temps, set):
        self.temps = temps
        self.puddle_cond = []
        self.puddle_cond_err = []
        self.mob_e = []
        self.mob_e_err = []
        self.mob_h = []
        self.mob_h_err = []
        self.rho_s_e = []
        self.rho_s_e_err = []
        self.rho_s_h = []
        self.rho_s_h_err = []
        self.v_dirac = []
        self.v_dirac_err = []
        self.const_cond = []
        self.const_cond_err = []
        self.pow = []
        self.pow_err = []

        for fit in set:
            self.puddle_cond.append(fit.sigma_pud_e)
            self.mob_e.append(fit.mu_e)
            self.mob_h.append(fit.mu_h)
            self.rho_s_e.append(fit.rho_s_e)
            self.rho_s_h.append(fit.rho_s_h)
            self.v_dirac.append(fit.vg_dirac)
            self.const_cond.append(fit.sigma_const)
            self.pow.append(fit.pow)

            try:
                if hasattr(fit.fitcovar, "shape"):
                    # print(len(self.puddle_cond_err))
                    self.puddle_cond_err.append(fit.sigma_pud_e_err)
                    self.mob_e_err.append(fit.mu_e_err)
                    print(len(self.mob_e_err))
                    self.mob_h_err.append(fit.mu_h_err)
                    self.rho_s_e_err.append(fit.rho_s_e_err)
                    self.rho_s_h_err.append(fit.rho_s_h_err)
                    self.v_dirac_err.append(fit.vg_dirac_err)
                    self.const_cond_err.append(fit.sigma_const_err)
                    self.pow_err.append(fit.pow_err)
            except ValueError:
                pass

        self.ylabels = {"sigma_pud_e"   :   r"Conductivity (x$10^{-3}$ S)",
                        "mu_e"          :   r"Mobility (cm$^2$V$^{-1}$s${-1}$)",
                        "mu_h"          :   r"Mobility (cm$^2$V$^{-1}$s${-1}$)",
                        "rho_s_e"       :   r"$\rho_S$ ($\Omega$)",
                        "rho_s_h"       :   r"$\rho_S$ ($\Omega$)",
                        "vg_dirac"      :   r"Voltage (V)",
                        "sigma_const"   :   r"Conductivity (x$10^{-3}$ S)",
                        "pow"           :   r"Power Index"
                        }

        return


    def plotPvT(self, param_data, param_err=None, label= "", ax=None, plot_errors=True):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if not param_err or not plot_errors:
            ax.plot(self.temps, param_data, 'o-', label=label)
        else:
            ax.errorbar(x=self.temps, y=param_data, yerr=param_err, fmt='o-', capsize=5, label=label)
        ax.set_xlabel("Temperature (K)")
        ax.tick_params(direction="in")
        return ax

    def plotMu(self, ax = None, label="", plot_errors=True):
        ax1 = self.plotPvT(param_data=self.mob_e, param_err=self.mob_e_err, label="Electron " + label, ax=ax, plot_errors=plot_errors)
        self.plotPvT(self.mob_h, self.mob_h_err, label="Hole " + label, ax=ax1, plot_errors=plot_errors)
        ax1.set_ylabel(self.ylabels["mu_e"])
        return ax1

    def plotRhoS(self, ax = None, label="", plot_errors=True):
        ax1 = self.plotPvT(self.rho_s_e, self.rho_s_e_err, label="Electron " + label, ax=ax, plot_errors=plot_errors)
        self.plotPvT(self.rho_s_h, self.rho_s_h_err, label="Hole " + label, ax=ax1, plot_errors=plot_errors)
        ax1.set_ylabel(self.ylabels["rho_s_e"])
        return ax1

    def plotVDirac(self, ax = None, label="", plot_errors=True):
        if np.all(np.array(self.v_dirac_err) > 50):
            ax1 = self.plotPvT(self.v_dirac, label="Dirac " + label, ax=ax, plot_errors=plot_errors)
            ax1.set_ylabel(self.ylabels["vg_dirac"])
        else:
            ax1 = self.plotPvT(self.v_dirac, self.v_dirac_err, label="Dirac " + label, ax=ax, plot_errors=plot_errors)
            ax1.set_ylabel(self.ylabels["vg_dirac"])
        return ax1

    def plotSigmaConst(self, ax = None, label="", plot_errors=True):
        ax1 = self.plotPvT(self.const_cond, self.const_cond_err, label=r"$\sigma_{0}$" + label, ax=ax, plot_errors=plot_errors)
        ax1.set_ylabel(self.ylabels["sigma_const"])
        scale_y = 1e-3
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
        ax1.yaxis.set_major_formatter(ticks_y)
        return ax1

    def plotSigmaPud(self, ax = None, label="", plot_errors=True):
        ax1 = self.plotPvT(self.puddle_cond, self.puddle_cond_err, label=r"$\sigma_{pud}$" + label, ax=ax, plot_errors=plot_errors)
        ax1.set_ylabel(self.ylabels["sigma_pud_e"])
        scale_y = 1e-3
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/scale_y))
        ax1.yaxis.set_major_formatter(ticks_y)
        return ax1

    def plotPower(self, ax= None, label="", plot_errors=True):
        ax1 = self.plotPvT(self.pow, self.pow_err, label=r"$\alpha$" + label, ax=ax, plot_errors=plot_errors)
        ax1.set_ylabel(self.ylabels["pow"])
        return ax1

### ----------------------------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------------------------------------------------------------------------------------- ###
class RvT_data():
    def __init__(self,RVG_set):
        self.gate_voltages = RVG_set[0].sampled_vg_data.GATE_VOLTAGES.copy()
        self.resistancesU = np.zeros((len(RVG_set), len(self.gate_voltages))) #Data ordered by 1. temps, 2. gate voltages.
        self.resistancesD = np.zeros((len(RVG_set), len(self.gate_voltages)))
        self.temps = np.zeros(len(RVG_set))
        self.fitted = False
        self.fitParams = None
        self.fitFunc = None

        #Gather data at each gate voltage for each temperature:
        for i in range(len(RVG_set)):
            rvg_obj = RVG_set[i]
            self.resistancesU[i, :] = rvg_obj.sampled_vg_data.rvg_u[:].copy()
            self.resistancesD[i, :] = rvg_obj.sampled_vg_data.rvg_d[:].copy()
            self.temps[i] = rvg_obj.TEMP_MEAN
        return

    #Graphing
    def graphU(self):
        return self.graph(self.resistancesU)

    def graphD(self):
        return self.graph(self.resistancesD)

    def graph(self, resistances, gate_voltages=None, temps=None):
        if gate_voltages is None:
            gate_voltages = self.gate_voltages
        if temps is None:
            temps = self.temps

        fig, (ax1) = plt.subplots(1,1)

        # cmap = cm.get_cmap("inferno")
        # cmap = cm.get_cmap("viridis")
        # cmap = cm.get_cmap("plasma")
        cmap = cm.get_cmap("coolwarm")
        dims = [1j * a for a in np.shape(resistances)]
        m1, m2 = np.mgrid[0:1:dims[1], 1:1:dims[0]]

        c = cmap(m1)

        for i in range(len(gate_voltages)):
            cmat = np.ones(len(resistances[:,i])) * gate_voltages[i]
            ax1.scatter(temps, resistances[:,i], label="{:0.3g}".format(gate_voltages[i]),s=20,c=c[i])

        handles, labels = ax1.get_legend_handles_labels()
        # fig.set_size_inches(8, 8)
        fig.legend(handles, labels, title="Legend", bbox_to_anchor=(1.2,0.85))#, loc = "best")
        ax1.set_title("Phonon dependent electronic transport")
        ax1.set_xlabel("Temperature (K)")
        # ax1.set_ylabel("Conductivity (cm$^2V^{-1}s^{-1}$)")
        ax1.set_ylabel(r"Resitivity ($\Omega$)")
        ax1.tick_params(direction="in")
        return ax1

    #Fitting functions
    def rho_A(temp, Da = 3.0):
        kB = 1.38e-23 #m^2 kg s^-2 K^-1
        rho_s = 7.6e-7 #kg/m^2
        vf = 1e6 #m/s
        vs=2.1e4 #m/s
        e = 1.60217662e-19 #C
        h = 6.62607004e-34

        return (h / np.power(e,2)) * (np.power(np.pi,2) * np.power(Da * e, 2) * kB * temp) / (2 * np.power(h,2) * rho_s * np.power(vs * vf, 2))

    def rho_B1(temp, vg, params = (1,2)): #Params: (A1, B1)
        e = 1.60217662e-19 #C
        kB = 1.38e-23 #m^2 kg s^-2 K^-1
        h = 6.62607004e-34

        a1,B1 = params

        expFactor = (np.reciprocal(np.exp(e * 59e-3 / kB * np.reciprocal(temp)) - 1) + 6.5 * (np.reciprocal(np.exp(e * 155e-3 / kB * np.reciprocal(temp)) - 1)))
        c1 = B1 * h / np.power(e,2)
        c2 = np.power(np.abs(vg + 0.001), -a1)
        coefFactor = c1 * c2
        return expFactor * coefFactor

    def rho_B2(temp, vg, params = (1,2,120e-3)): #Params: (A1, B1, E0)
        e = 1.60217662e-19 #C
        kB = 1.38e-23 #m^2 kg s^-2 K^-1
        h = 6.62607004e-34

        a1,B1,E0 = params

        expFactor = (np.reciprocal(np.exp(e * E0 / kB * np.reciprocal(temp)) - 1))
        c1 = (B1 * h / np.power(e,2))
        c2 = np.power(np.abs(vg + 0.001), -a1)
        coefFactor = c1 * c2
        return expFactor * coefFactor

    def rho_T_1D(X, *p):
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        Da, a1, B1, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps

        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        for i in range(0,vg_steps):
            #Define indexes of 2D data along 1D dimension
            i1=int(0+i*temp_steps)
            i2=int((i+1)*temp_steps)
            #Calculate each set of indexes
            retVal[i1:i2] = R0[i] + RvT_data.rho_A(temp[i1:i2], Da) + RvT_data.rho_B1(temp[i1:i2],vg[i1:i2],(a1,B1))
        return retVal

    def rho_T(X, Da, a1, B1, R0):
        temp, vg = X
        return R0 + RvT_data.rho_A(temp, Da) + RvT_data.rho_B1(temp,vg,(a1,B1))

    # Fitting
    class fitParamsRvT1():
        def __init__(self, p, vg, temps):
            self.Da, self.a1, self.B1, *R = p
            self.R0 = list(R)
            self.vg = vg #Accompany the set of R0 values.
            self.temps = temps
            self.fitted = False
            self.fitparams = None
            self.fitcovar = None
            return

        def plotParams(self, ax=None, temps = None, s=1, c=None):
            if ax is None:
                fig, (ax1) = plt.subplots(1,1)
            else:
                ax1 = ax
            if temps is None:
                temps = np.linspace(start=10,stop=400, num=390)
            for i in range(len(self.vg)): #for each gate voltages:
                voltage = self.vg[i]
                if c is None:
                    ax1.scatter(temps, RvT_data.rho_T((temps,voltage), Da=self.Da, a1=self.a1, B1=self.B1, R0=self.R0[i]), label=str(voltage), s=s)
                else:
                    ax1.scatter(temps, RvT_data.rho_T((temps,voltage), Da=self.Da, a1=self.a1, B1=self.B1, R0=self.R0[i]), label=str(voltage), s=s, c=c[i])

            ax1.set_title("Phonon dependent electronic transport")
            ax1.set_xlabel("Temperature (K)")
            ax1.set_ylabel(r"Resitivity ($\Omega$)")
            ax1.tick_params(direction="in")
            return ax1

    def global_fit_RvT(temp, vg, data, params = (3, 1, 2, 50), R0s_guess = None): #Da, A1, B1, R0
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
