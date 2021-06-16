__version__ = "0.1.0" #Re-worked temperature and gated behaviour.

from scipy.signal import savgol_filter, argrelextrema
import numpy as np
import math
from pylectric.geometries.FET import hallbar
from pylectric.geometries.FET.hallbar import Meas_GatedResistance, Meas_Temp_GatedResistance

class Graphene_Gated():
    def mobility_dtm_peaks(data, order=10):
        """Uses the "shoulders" of the DTM curve near the minimum conductivity
        (corresponding to a minima in the DTM curve) to find mobility values for
        electrons and holes in graphene.
        Data should be a 1D array of mobilities corresponding to low to high gate
        voltages that span the Dirac point. The default order 10 parameter is good
        for gate voltage increments of about 0.1 V.
        Return a tuple (i_holes, i_elec) that has the index of mobility peaks
        corresponding to holes and electrons respectively. """
        #Find local mobility maxima:
        maxima=argrelextrema(data, np.greater, order=order)[0]
        #Get minima location:
        minPos = argrelextrema(data,np.less,order=10)[0]
        minI = np.where(data==np.min(data[minPos]))[0][0]
        #Get next maxima index after and before minima.
        above_I = np.where(maxima > minI)[0]
        if above_I.shape[0] == 0:
            #Above value not found. Check lower ValueError
            below_I = np.where(maxima < minI)[0]
            if below_I.shape[0] == 0:
                return (None, None)
            else:
                below_I = below_I[-1]
                return (maxima[below_I], None)
        else:
            above_I = above_I[0]
            if above_I == 0:
                return (None, maxima[above_I])
            else:
                return (maxima[above_I-1], maxima[above_I])


    def sigma_graphene(vg, *p0):
        """ Models impurity dominated conductivity in graphene.
            Parameters include:
            1. sigma_pud_e  -   Puddling charge that gives an effective minimum conductivity.
            2. mu_e         -   Electron (+ve Vg) gate mobility.
            3. mu_h         -   Hole (-ve Vg) gate mobility.
            4. rho_s_e      -   Short range scattering (electron-electron interactions) at high carrier density.
            5. rho_s_h      -   Short range scattering (hole-hole interactions) at high carrier density.
            6. vg_dirac     -   Dirac point (volts).
            7. pow          -   A power index describing the curvature from carrier
                                dominated transport to puddling regime.
        """

        #Expand parameters
        sigma_pud_e, mu_e, mu_h, rho_s_e, rho_s_h, vg_dirac, pow = p0

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
        #carrier condition for fitting (electrons, or holes)
        carrier_cond = [vg > vg_dirac, vg <= vg_dirac]
        #gate dependent conductivity
        sigma = np.select(carrier_cond, [sigma_e, sigma_h])
        return sigma

    def fit_sigma_graphene(MGR_Obj, windows = 3):
        """ Fits conductivity data for graphene of a MGR_Obj.
            Dirac point should be within the data range of the window,
            as fitting is to @sigma_graphene asymmetric parameters. This is
            lightly enforced by using bounds on input dataset to within "windows" x
            voltage window.
        """
        if not type(MGR_Obj) is Meas_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_GatedResistance. Is intead: " + str(MGR_Obj.__class__())))
            return

        ###Initial Values
        #Get initial values for minimum conductivity and gate voltage.
        min_sigma_i = np.where(MGR_Obj.conductivity_data[:,1] == np.min(MGR_Obj.conductivity_data[:,1]))[0][0] #Find min index for conductivity
        #Fit Variables:
        vg_dirac = MGR_Obj.conductivity_data[min_sigma_i,0]
        sigma_pud_e = MGR_Obj.conductivity_data[min_sigma_i,1]
        #Default params:
        mu_e, mu_h = (1000,1000)
        rho_s_e, rho_s_h = (50,50)
        pow = 2.85
        #Pack initial values.
        x0 = (sigma_pud_e, mu_e, mu_h, rho_s_e, rho_s_h, vg_dirac, pow)

        ###Bounds
        #Find max and min voltages
        min_v_i = np.where(MGR_Obj.conductivity_data[:,0] == np.min(MGR_Obj.conductivity_data[:,0]))[0][0] #Find min index for gate voltage
        max_v_i = np.where(MGR_Obj.conductivity_data[:,0] == np.max(MGR_Obj.conductivity_data[:,0]))[0][0] #Find max index for gate voltage
        v_window = windows * (MGR_Obj.conductivity_data[max_v_i:,0] - MGR_Obj.conductivity_data[min_v_i, 0])
        #Pack bounds
        #                (sigma_pud_e, mu_e, mu_h, rho_s_e, rho_s_h, vg_dirac, pow)
        defaultBoundsL = [1e-10,    1,      1,      0,      0,       vg_dirac - v_window,    1.5]
        defaultBoundsU = [ 1e-3,    1e7,    1e7,    1e5,    1e5,    vg_dirac + v_window,    4.0]

        return MGR_Obj.global_RVg_fit(Graphene_Gated.sigma_graphene, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    def fit_min_cond_quadratic(MGR_Obj, factor=2):
        """ Fits a quadratic to the minimum conductivity curvature to find the dirac point voltage.
            Uses data that is multiplicatively within the minimum conductivity by "factor".
            Returns:
                - Dirac Point (Volts)
                - Polynomial object.
                - gate voltages for fitting.
        """
        if not type(MGR_Obj) is Meas_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_GatedResistance. Is intead: " + str(MGR_Obj.__class__())))

        ###Initial Values
        #Get initial values for minimum conductivity and gate voltage.
        min_sigma_i = np.where(MGR_Obj.conductivity_data[:,1] == np.min(MGR_Obj.conductivity_data[:,1]))[0][0] #Find min index for conductivity
        #Fit Variables:
        vg_dirac = MGR_Obj.conductivity_data[min_sigma_i,0]
        sigma_pud_e = MGR_Obj.conductivity_data[min_sigma_i,1]

        ## Find data window within range:
        # Get values greater than
        gt_factor_min_indexes = np.where(MGR_Obj.conductivity_data[:,1] > 2*np.min(MGR_Obj.conductivity_data[:,1]))[0]
        # take first below min_index
        search = np.where(gt_factor_min_indexes <= min_sigma_i)[0]
        if len(search) > 0:
            i1 = search[-1] #Last index will be closest to beginning of the range.
            sigma_i1 = gt_factor_min_indexes[i1]
            sigma_i2 = gt_factor_min_indexes[i1 + 1] #next index will be closest after
            # Fit to data range:
            subset = MGR_Obj.conductivity_data[sigma_i1:sigma_i2,:]
        else:
            # No point less than the minimum. But maybe something after!
            sigma_i2 = gt_factor_min_indexes[0] #next index will be closest after
            subset = MGR_Obj.conductivity_data[0:sigma_i2,:]
        quad = np.polynomial.polynomial.Polynomial.fit(x=subset[:,0],y=subset[:,1], deg=2).convert(domain=[-1,1])

        # Calculate
        c,b,a = quad.coef.tolist()
        x_tp = - b / (2*a) #turning point, in otherwords dirac point voltage.
        return x_tp, quad, subset[:,0]

    def carrier_density(gate_voltage, gate_capacitance):
        """ Calculates the gated carrier density in graphene.
            Assumes units are:
                Voltage:        Volts
                Capacitance:    Farads / m^2
        """
        e = 1.602e-19 #Coloumbs
        return 1/e * gate_capacitance * gate_voltage



    #
    # def n_imp(mobility, k_eff = 4):
    #     """
    #     Requires field effect mobility, and effective local dielectric constant (kappa) to calculate.
    #     Assuming mobility in units of cm^2/Vs.
    #     Default dielectric constant is 4, close to the value of SiO2.
    #     According to theory by Adam et al. https://www.pnas.org/content/104/47/18392
    #
    #     """
    #     e = 1.602e-19
    #     hb = 1.05457e-34
    #     h = hb * 2 * math.pi
    #
    #     def Rs(kappa):
    #         gamma = 0.000067431
    #         return math.pow(e,2)/(hb * gamma * kappa)
    #
    #     def G(x):
    #         return np.reciprocal(np.power(x,2)) * (math.pi/4 + 3*x - (3 * math.pi * np.power(x,2)/2)
    #             + (x * (3 * np.power(x,2) -2) / np.sqrt(np.power(x,2) -1)) * np.acos(np.reciprocal(x)))
    #
    #     #Using \mu = \sigma / n / e
    #     n_imp = e / h * np.reciprocal(mobility / 1e4) * (2 / G(2 * Rs(k_eff)))
    #
    #     return n_imp
    #
    #
    # def n_star(minimum_cond, n_imp):
    #      return


class Graphene_Phonons():
    ############################################################################
    ### Raw temperature functions
    ############################################################################

    #KEEP
    def rho_LA(temp, Da = 18.0):
        """ Models longitudinal acoustic phonon intrinsic in graphene.
            Reffered to as rho_A in https://www.nature.com/articles/nnano.2008.58
        """
        # Constants
        kB      = 1.38e-23 #m^2 kg s^-2 K^-1
        rho_s   = 7.6e-7 #kg/m^2
        vf      = 1e6 #m/s
        vs      = 2.1e4 #m/s
        e       = 1.60217662e-19 #C
        h       = 6.62607004e-34
        # Value
        return (h / np.power(e,2)) * (np.power(np.pi,2) * np.power(Da * e, 2) * kB * temp) / (2 * np.power(h,2) * rho_s * np.power(vs * vf, 2))

    #KEEP
    def rho_ROP_SiO2(temp, vg, params = (1,2), energies=None, couplings=None): #Params: (A1, B1)
        """ Models contribution of remote surface optical phonons,
            caused by SiO2 in proximity to graphene.
            Reffered to as rho_B1 in https://www.nature.com/articles/nnano.2008.58.
        """
        # Constants
        e = 1.60217662e-19 #C
        kB = 1.38e-23 #m^2 kg s^-2 K^-1
        # kB = 8.617333262145e-5 #eV K^-1
        h = 6.62607004e-34

        a1,B1 = params

        expFactor = 0
        if energies is not None and couplings is not None:
            if len(energies) != len(couplings):
                raise IndexError("The size of energies and couplings do not match.")
            else:
                for i in range(len(energies)):
                    expFactor += couplings[i] * np.reciprocal(np.exp(e * energies[i] / kB * np.reciprocal(temp)) - 1)
        else:
            # SiO2 phonon modes and couplings.
            e0,e1 = (59e-3, 155e-3)
            g0,g1 = (1.75e-3, 9.396e-3)
            expFactor = (g0 * np.reciprocal(np.exp(e * e0 / kB * np.reciprocal(temp)) - 1) +
                         g1 * np.reciprocal(np.exp(e * e1 / kB * np.reciprocal(temp)) - 1))
        c1 = B1 * h / np.power(e,2)
        c2 = np.power(np.abs(vg), -a1)
        coefFactor = c1 * c2
        return expFactor * coefFactor

    #KEEP
    def rho_ROP_SiO2_Ga2O3(temp, vg, params = (1,2), energies=None, couplings=None): #Params: (A1, B1)
        """ Models contribution of remote surface optical phonons,
            caused between SiO2 and Ga2O3 in proximity to graphene.
        """
        # Constants
        e = 1.60217662e-19 #C
        kB = 1.38e-23 #m^2 kg s^-2 K^-1
        # kB = 8.617333262145e-5 #eV K^-1
        h = 6.62607004e-34

        a1,B1 = params

        expFactor = 0
        if energies is not None and couplings is not None:
            if len(energies) != len(couplings):
                raise IndexError("The size of energies and couplings do not match.")
            else:
                for i in range(len(energies)):
                    expFactor += couplings[i] * np.reciprocal(np.exp(e * energies[i] / kB * np.reciprocal(temp)) - 1)
        else:
            # SiO2 phonon modes and couplings.
            e0,e1,e2 = (56.45e-3, 146.94e-3, 95.15e-3) #06 onward
            g0,g1,g2 = (9.75e-4, 2.02e-3, 8.12e-3) #06 onward
            expFactor = (g0 * np.reciprocal(np.exp(e * e0 / kB * np.reciprocal(temp)) - 1) +
                         g1 * np.reciprocal(np.exp(e * e1 / kB * np.reciprocal(temp)) - 1) +
                         g2 * np.reciprocal(np.exp(e * e2 / kB * np.reciprocal(temp)) - 1))

        c1 = B1 * h / np.power(e,2)
        c2 = np.power(np.abs(vg), -a1)
        coefFactor = c1 * c2
        return expFactor * coefFactor

    #KEEP
    def rho_ROP_Generic(temp, vg, params = (1,2,120e-3)): #Params: (A1, B1, E0)
        """ Models contribution of remote surface optical phonons,
            caused an abitrary single optical mode in proximity to graphene.
            Reffered to as rho_B2 in https://www.nature.com/articles/nnano.2008.58.
        """
        e = 1.60217662e-19 #C
        kB = 1.38e-23 #m^2 kg s^-2 K^-1
        # kB = 8.617333262145e-5 #eV K^-1
        h = 6.62607004e-34

        a1,B1,E0 = params

        expFactor = (np.reciprocal(np.exp(e * E0 / kB * np.reciprocal(temp)) - 1))
        c1 = (B1 * h / np.power(e,2))
        c2 = np.power(np.abs(vg), -a1)
        coefFactor = c1 * c2
        return expFactor * coefFactor

    ############################################################################
    ### Combination temperature functions
    ############################################################################

    #rho0
    def rho_Gr_on_Dielectric(X, *p):
        """ Fitting function for Graphene on/inbetween arbitrary dielectrics.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                Da, the deformation potential.
                a1, the power index coupling of remote phonons to gate voltage.
                B1, the coupling magnitude of remote phonons
                E0, the activation energy of the remote phonons
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        Da, a1, B1, E0, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps

        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da) + Graphene_Phonons.rho_ROP_Generic(temp[i1:i2],vg[i1:i2],(a1,B1,E0))

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da) + Graphene_Phonons.rho_ROP_Generic(temp[i1:i2],vg[i1:i2],(a1,B1,E0))
        return retVal

    def rho_ROP_Gr_on_Dielectric(X, *p):
        """ Fitting function for Graphene on/inbetween arbitrary dielectrics.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                a1, the power index coupling of remote phonons to carrier density.
                B1, the coupling magnitude of remote phonons
                E0, the activation energy of the remote phonons
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        a1, B1, E0, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps


        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_ROP_Generic(temp[i1:i2],vg[i1:i2],(a1,B1,E0))

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_ROP_Generic(temp[i1:i2],vg[i1:i2],(a1,B1,E0))
        return retVal

    #rho2
    def rho_Gr_on_SiO2(X, *p):
        """ Fitting function for Graphene on SiO2.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                Da, the deformation potential.
                a1, the power index coupling of remote phonons to gate voltage.
                B1, the coupling magnitude of remote phonons
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        Da, a1, B1, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps



        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da) + Graphene_Phonons.rho_ROP_SiO2(temp[i1:i2],vg[i1:i2],(a1,B1))

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da) + Graphene_Phonons.rho_ROP_SiO2(temp[i1:i2],vg[i1:i2],(a1,B1))
        return retVal

    #rho3
    def rho_ROP_Gr_on_SiO2(X, *p):
        """ Fitting function for Graphene on SiO2.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                a1, the power index coupling of remote phonons to gate voltage.
                B1, the coupling magnitude of remote phonons
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        a1, B1, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps

        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_ROP_SiO2(temp[i1:i2],vg[i1:i2],(a1,B1))

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_ROP_SiO2(temp[i1:i2],vg[i1:i2],(a1,B1))
        return retVal

    #rho4
    def rho_Gr_between_SiO2_Ga2O3(X, *p):
        """ Fitting function for Graphene inbetween arbitrary dielectrics.
            Resistance for linear acoustic and remote optical phonons.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                Da, the deformation potential.
                a1, the power index coupling of remote phonons to gate voltage.
                B1, the coupling magnitude of remote phonons
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        Da, a1, B1, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps


        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da) + Graphene_Phonons.rho_ROP_SiO2_Ga2O3(temp[i1:i2],vg[i1:i2],(a1,B1))

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da) + Graphene_Phonons.rho_ROP_SiO2_Ga2O3(temp[i1:i2],vg[i1:i2],(a1,B1))
        return retVal

    #rho5
    def rho_ROP_Gr_between_SiO2_Ga2O3(X, *p):
        """ Fitting function for remote optical phonons in graphene, inbetween arbitrary dielectrics.
            Note: Reistance from linear acoustic phonon contributions should be removed prior.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                a1, the power of the carrier density dependence
                B1, the coupling magnitude of remote phonons
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        a1, B1, *R0 = p
        # B1, *R0 = p
        # a1 = 18.0
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps

        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_ROP_SiO2_Ga2O3(temp[i1:i2],vg[i1:i2],(a1,B1))

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_ROP_SiO2_Ga2O3(temp[i1:i2],vg[i1:i2],(a1,B1))
        return retVal

    #rho6
    def rho_Gr_LA(X, *p):
        """ Fitting function for low temperature Graphene longitudinal acoustic phonons.
            X should be a tuple of 1D arrays of temperatures and gate voltages.
            p should be a tuple of parameters corresponding to:
                Da, the deformation potential.
                *R0, a list of initial resistances (ie, at T=0 K) for each gate voltage.
                Note R0 must be the same as the number of gate voltages.
        """
        #Expand 1D temp and vg lists from tuple.
        temp, vg = X
        #Expand parameters to count amount of gate voltages.
        Da, *R0 = p
        #Determine steps for gate voltages and temperatures in 1D array | one gate voltage per resistance parameter.
        vg_steps = len(R0)
        temp_steps = len(temp)/vg_steps


        #Setup new matrix for returning generated values.
        retVal = np.zeros(temp.shape)
        # Corrected function checks for unique values in vg (the steps) and identifies positions.
        vg_elements = np.unique(vg)
        for i in range(0,vg_steps):
            indexes = np.where(vg == vg_elements[i])[0]
            #Define the indexes of 2D data along 1D dimension.
            i1 = indexes[0]
            i2 = indexes[-1]
            #Calculate for each set of indexes
            retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da)

        # #Setup new matrix for returning generated values.
        # retVal = np.zeros(temp.shape)
        # for i in range(0,vg_steps):
        #     #Define indexes of 2D data along 1D dimension
        #     i1=int(0+i*temp_steps)
        #     i2=int((i+1)*temp_steps)
        #     #Calculate each set of indexes
        #     retVal[i1:i2] = R0[i] + Graphene_Phonons.rho_LA(temp[i1:i2], Da)
        return retVal

    ############################################################################
    ### Fitting temperature functions
    ############################################################################

    #fit0
    def fit_Gr_on_Dielectric(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of phonon modes affecting graphene given some RVT data fitting.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:]

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        Da, a1, B1, E0 = (18, 1, 5, 120e-3) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        defaultBoundsL = [1e-2,1e-2,1e-2, 1e-4] + list(Rlower)
        defaultBoundsU = [1e6, 5, 1e5, 1e1] + list(Rupper)

        x0 = [Da, a1, B1, E0]
        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_Gr_on_Dielectric, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    #fit1
    def fit_ROP_Gr_on_Dielectric(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of phonon modes affecting graphene given some RVT data fitting.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:]

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        a1, B1, E0 = (1, 5, 120e-3) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        defaultBoundsL = [0.001, 0.1, 1e-3] + list(Rlower)
        defaultBoundsU = [4, 1e5, np.inf] + list(Rupper)
        x0 = [a1, B1, E0]

        # B1, E0 = (5, 120e-3) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        # defaultBoundsL = [0.1, 1e-3] + list(Rlower)
        # defaultBoundsU = [1e5, np.inf] + list(Rupper)
        # x0 = [B1, E0]

        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_ROP_Gr_on_Dielectric, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    #fit2
    def fit_Gr_on_SiO2(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of graphene phonon modes given some RVT data fitting.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:] #Each Vg has an offset resistance R0:

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        Da, a1, B1 = (18, 1, 5) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        defaultBoundsL = [0.01,0.01,0.01] + list(Rlower)
        defaultBoundsU = [1e6, np.inf, 1e5] + list(Rupper)

        x0 = [Da, a1, B1]
        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_Gr_on_SiO2, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

        #fit3
    def fit_ROP_Gr_on_Sio2(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of phonon modes affecting graphene given some RVT data fitting.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:]

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        a1, B1, = (1, 18.0,) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        defaultBoundsL = [0.001, 0.01] + list(Rlower)
        defaultBoundsU = [4, 1e5] + list(Rupper)

        x0 = [a1, B1]
        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_ROP_Gr_on_SiO2, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    #fit4
    def fit_Gr_between_Sio2_Ga2O3(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of phonon modes affecting graphene given some RVT data fitting.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:]

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        Da, a1, B1 = (18, 1, 5) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        defaultBoundsL = [0.1,0.1,0.1] + list(Rlower)
        defaultBoundsU = [1e6, np.inf, 1e5] + list(Rupper)

        x0 = [Da, a1, B1]
        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_Gr_between_SiO2_Ga2O3, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    #fit5
    def fit_ROP_Gr_between_Sio2_Ga2O3(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of phonon modes affecting graphene given some RVT data fitting.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:]

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)
        #Bounds
        a1, B1, = (1, 5,) #Guesses for deformation potential, gate voltage power index and gate voltage coupling.
        defaultBoundsL = [0.001,0.01] + list(Rlower)
        defaultBoundsU = [4, 1e5] + list(Rupper)

        x0 = [a1, B1]
        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_ROP_Gr_between_SiO2_Ga2O3, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    #fit6
    def fit_Gr_LA(MTGR_Obj):
        """ Takes a Meas_Temp_GatedResistance object from pylectric.geometries.fet.hallbar
            Allows the calculation of graphene's longitudinal acoustic phonon mode
            given some RVT data fitting. Data should only include linear portion
            before coupled optical modes begin to contribute.
        """
        if not type(MTGR_Obj) is Meas_Temp_GatedResistance:
            raise(AttributeError("Passed object is not an instance of pylectric.geometries.FET.hallbar.Meas_Temp_GatedResistance. Is intead: " + str(MTGR_Obj.__class__())))
            return

        vg = MTGR_Obj.vg
        #Get initial values for R0 for each gate voltage.
        min_temp_i = np.where(MTGR_Obj.temps == np.min(MTGR_Obj.temps))[0][0] #Find min index for temperature
        initialR0s = MTGR_Obj.resistivity[min_temp_i,:] #Each Vg has an offset resistance R0:

        #Independent Fit Variables:
        Rlower = [] #Bounds
        Rupper = [] #Bounds
        for i in range(len(vg)):
            Rlower.append(0)
            Rupper.append(20000)
        R = initialR0s.tolist()
        Rupper = tuple(Rupper)
        Rlower = tuple(Rlower)

        #Bounds
        Da = (18) #Guesses for deformation potential
        defaultBoundsL = [0.01] + list(Rlower)
        defaultBoundsU = [1e6] + list(Rupper)

        x0 = [Da]
        x0 += list(R)
        x0 = tuple(x0)

        return MTGR_Obj.global_RTVg_fit(Graphene_Phonons.rho_Gr_LA, params=x0, boundsU=defaultBoundsU, boundsL=defaultBoundsL)

    ############################################################################
    ### Literature reference fits
    ############################################################################
    # Constants
    # e = 1.60217662e-19 #C
    # h = 6.62607004e-34

    #### Chen et al. 2008 ######################################################
    def Chen_Gr_LA(X):
        T,Vg = X
        Da = 18.0
        return Graphene_Phonons.rho_Gr_LA(T, Da=18.0)

    def Chen_Gr_ROP_SiO2(X):
        T,Vg = X
        # Parameters
        a1 = 1.04
        b1 = 0.607
        en = [0.059,0.155]
        gn = [1,6.5]
        params = (a1,b1)
        return Graphene_Phonons.rho_ROP_SiO2(T,Vg,params, energies=en, couplings=gn)

    def Chen_Gr_ROP_Generic(X):
        T,Vg = X
        # Parameters
        a1 = 1.02
        b1 = 3.26
        en = 0.104
        params = (a1,b1,en)
        return Graphene_Phonons.rho_ROP_Generic(T,Vg,params)

    def Chen_Gr_SiO2(X):
        T,Vg = X
        return Graphene_Phonons.Chen_Gr_LA(T) + Graphene_Phonons.Chen_Gr_ROP_SiO2(T,Vg)

    def Chen_Gr_Generic(X):
        T,Vg = X
        return Graphene_Phonons.Chen_Gr_LA(T) + Graphene_Phonons.Chen_Gr_ROP_Generic(T,Vg,params)

    #### Zou et al. 2012 ######################################################
    # Zou doesn't provide enough information for C2, assuming C1 dominates.
    # def Zou_Gr_ROP_SiO2(X):
    #     T,Vg = X
    #     C1, C2 = (5e2, 7e2)
    #     N = (2 * (1e12)) #Density at 27 V
    #
    #     a1 = 1.0
    #     b1 = 1.0
    #
    #     return return Graphene_Phonons.rho_ROP_SiO2(T,Vg,params)
