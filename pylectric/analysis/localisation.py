# Libraries & Imports
import numpy as np
from scipy.special import digamma
import scipy.optimize as so
import warnings

import matplotlib.pyplot as plt

#Constants
from scipy.constants import elementary_charge as e
from scipy.constants import Planck as h
from scipy.constants import pi
from scipy.constants import hbar


class WAL():
    """ Class to contain various literature Weak Anti-Localisation phenomena.
    """
    def HLN(B, alpha, lphi):
        """Formula for strong weak-anti-localisation, according to HLN.
            Here, B_SO >> B_Phi (the spin orbit characteristic field is much stronger 
            than the phase coherence characteristic field). See Eqn. (18) from
            https://academic.oup.com/ptp/article/63/2/707/1888502
        
        alpha   Prefactor (usually 0.5 for TI)
        lphi    Phase coherence length (in meters)
            
        B is a 1D vector containing Field.
        """
        
        # First factors
        prefactor = alpha * np.power(e,2) / pi / h
        Bphi = h / (2 * pi) / (4 * e * np.power(lphi,2))
        BB = Bphi/np.abs(B)
        # Secondary factors
        dig = digamma(0.5 + BB) #Digamma = ~ln(x) - 1/2x
        log = np.log(BB)
        # Check where zero values exist in B and numerically correct for resulting NaNs:
        if hasattr(B, "__len__"):
            infinities = np.where(np.isinf(BB))[0]
            dig[infinities] = 0
            log[infinities] = 0
        else:
            if np.isinf(BB):
                dig = 0
                log = 0

        return -prefactor*(dig-log)
    
    def HLN_offset(B, alpha, lphi, offset):
        """Same as HLN, but wiht an additional constant offset parameter.
        
        sig_off Conductivity offset (in Seimens)
        
        """
        return WAL.HLN(B, alpha, lphi) - offset
    
    def sigma_LMR(B, grad, R0):
        """Calculates an equivalent conductance of a linear magnetoresistance 
        + constant resisatnce at a magnetic field B. Can be subtracted from 
        high-field data (i.e. no WAL) prior to WAL analysis.

        Parameters
        ----------
        B : Any
            Set of field values.
        grad : float
            Gradient of the linear-magneto resistance.
        R0 : float
            Constant resistance offset

        Returns
        -------
        Any
            Set of conductances.
        """
        return 1 / (grad * np.abs(B) + R0)
    
    def HLN_LMR(B, alpha, lphi, grad, R0):
        """Addition of WAL.HLN and WAL.sigma_LMR to characterise conductivity.

        Parameters
        ----------
        B : Any
            Set of field values.
        alpha : float
            Prefactor (usually 0.5 for TI)
        lphi : float
            Phase coherence length (in meters)
        grad : float
            Gradient of the linear-magneto resistance.
        R0 : float
            Constant resistance offset

        Returns
        -------
        Any
            Set of corresponding conductance values.
        """
        return WAL.HLN(B, alpha, lphi) + WAL.sigma_LMR(B, grad, R0)
    
    
    class fitting():
        """ Bundled fitting functions for WAL formulas. """    
    
        def HLN_fit(B, sigmaxx, alpha=0.5, lphi=100e-9, offset=0, bounds = None):
            """Fits weak-antilocalisation contributions to the simplified HLN formula, where
            spinorbit scattering strength B_SO >> B_Phi phase coherence strength. 
            Uses WAL.HLN to perform fit calculation.

            Parameters
            ----------
            B : np.ndarray
                List of field values in Tesla.
            sigmaxx : np.ndarray
                List of sigma_xx values (fit performs sigma_xx@B=0 subtraction).
            alpha : float, optional
                Prefactor usually describing material system, 
                by default 0.5 for topological insulators. 
                Default bounds [-1,1]
            lphi : float, optional
                Phase coherence length, by default 100e-9 meters = 100 nm.
                Default bounds [0,1e-6]
            offset : float, optional
                Constant sigma offset, uncaptured by subtracting sigma(B=0). 
                By default 0. Default bounds [-np.infty, np.infty].
                
            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                Returns a tuple containing a list of parameters (alpha, lphi, offset) 
                and their respective uncertainties.
            """
            assert(np.all(B.shape == sigmaxx.shape))
            
            # Check if rxx has zero field value.
            # if not (np.any(B > 0) and np.any(B < 0)):
            #    warnings.warn("B-Field domain doesn't cover B=0.")
            
            # get closest value to 0.
            min_i = np.where(np.abs(B) == np.min(np.abs(B)))
            sigmaxx_b0 = np.average(sigmaxx[min_i])
            dsigmaxx = sigmaxx - sigmaxx_b0 #get change as close to zero field as possible.
            
            #As finding dsigmaxx is not perfect for noisy data, use a constant offset value.
            
            p0=(alpha, #0.5 for TIs
                lphi,  # 100 nm
                offset) # 0 S 
        
            if bounds is None:
                bounds = ([-1, 0, -np.infty], #lower
                          [1, 1e-6, np.infty]) #upper
        
            params, covar = so.curve_fit(WAL.HLN_offset, B, dsigmaxx, p0=p0, bounds=bounds)
            unc = np.sqrt(np.diag(covar))
            
            return params, unc
    
        def HLN_fit_iterative(B, sigmaxx, alpha=0.5, lphi=100e-9, #offset=0,
                                    b_window=10, b_tol=0.1, iter_max = 50,
                                    bounds = None, b0_lim = 0.025,
                                    sub_hf_linear = False, hf_lim = 1.5):
            """Fits weak-antilocalisation contributions to the simplified HLN formula, where
            spinorbit scattering strength B_SO >> B_Phi phase coherence strength. 
            Uses WAL.HLN to perform fit calculation.
            Additionally truncates fit data iteratively until B < b_window*Bphi convergences

            Parameters
            ----------
            B : np.ndarray
                List of field values.
            sigmaxx : np.ndarray
                List of sigma_xx values (fit performs sigma_xx@B=0 subtraction).
            alpha : float, optional
                Prefactor usually describing material system, 
                by default 0.5 for topological insulators. 
                Default bounds [-1,1]
            lphi : float, optional
                Phase coherence length, by default 100e-9 meters = 100 nm.
                Default bounds [0,1e-6]
            # offset : float, optional
            #     Constant sigma offset, uncaptured by subtracting sigma(B=0). 
            #     By default 0. Default bounds [-np.infty, np.infty].
            b_window : float, optional
                Allowed field domain, such that B < b_window*B_phi. By default 10.
                Imporant because WAL is a dominant effect at small field.
            b_tol : float, optional
                Allowed tollerance (%) between B_phi and B/b_window. If outside tolerance,
                iterates fit again with updated domain.
            b0_lim : float, optional
                Upper limit on absolute field values that sigma_xx@b0 can be
                calculated from. Usually averages 5 nearest points (eitherside) to Bmin.
                Default as 0.025 T.
            sub_hf_linear : bool, optional
                Option to subtract high-field linear component from sigma_xx. 
                Uses hf_tol to determine linear estimate.
            hf_lim : float, optional
                Minimum threshold value for field (T) by which to subtract a linear
                component. Has to be a magnitude. Default as 1.5 T (B > 1.5 T). 

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                Returns a tuple containing a list of parameters (alpha, lphi, offset) 
                and their respective uncertainties.
            """
            assert (np.all(B.shape == sigmaxx.shape))
            assert b_tol < 1 and b_tol > 0

            # get closest value to 0.
            min_i = np.where(np.abs(B) == np.min(np.abs(B))) #could be multiple values.
            # minB = np.average(sigmaxx[min_i]) #value not needed...
            # Use 5 closest values (eitherside) to avoid significant \sigma0 error.
            # Ensure 5 closest values at field value less than 0.025 T. 
            npi = np.linspace(-5,5,11, dtype=int) + min_i[0][0] #nearby possible indexes of first index.
            npi = npi[np.bitwise_and(npi >= 0, npi < len(B))] #correct for out of range
            npb = B[npi] #get nearby possible field values
            npx = sigmaxx[npi] #get nearby possible sigmaxx values
            nbi = np.where(npb < b0_lim) #nearby field values (valid)
            sigmaxx_b0 = np.average(npx[nbi]) #average field value within 5 datapoints
            
            # If removing hf_linear, fit to 1/(R0 + B*grad)
            if sub_hf_linear:
                assert hf_lim > 0 #a magnitude.
                #assume 
                hfi = np.where(np.abs(B) > hf_lim)
                hfb = B[hfi]
                hfs = sigmaxx[hfi]
                p0=((hfs[0]-hfs[-1])/(hfb[0]-hfb[-1]), #estimate gradient
                    1/sigmaxx_b0) #use estimate for zero field conductance.
                hfparams, hfcovar = so.curve_fit(WAL.sigma_LMR, hfb, hfs)
                hfunc = np.sqrt(np.diag(hfcovar))
                
                fig,ax = plt.subplots(1,1)
                ax.scatter(B, sigmaxx, label="Data", s=0.01)
                x = np.linspace(np.min(B), np.max(B), 20)
                ax.plot(x, WAL.sigma_LMR(x, *hfparams), "--", label="Fit")
                
                # Remove remaining offset from zero field sigmaxx.
                dSxxB0 = (sigmaxx_b0 - 1/hfparams[1]) #difference between WAL and LMR.
                # Fit value for R0 doesn't work for HLN. Need true sigma_xx zero field.
                y = WAL.sigma_LMR(x, *hfparams) + dSxxB0 # Add value to rise
                ax.plot(x, y, "--", label="Fit (-SigmaB0)")
                display(fig)
                
                # Additional subtraction to remove non-zero (zer-field) conductance.
                dsigmaxx = sigmaxx - WAL.sigma_LMR(B, *hfparams) - dSxxB0 
            else:
                # get change from (as close to) zero field (as possible).
                dsigmaxx = sigmaxx - sigmaxx_b0

            # p0 = (alpha,  # 0.5 for TIs
            #     lphi,  # 100nm
            #     offset) # 0 S
            p0 = (alpha,  # 0.5 for TIs
                lphi)  # 100nm

            if bounds is None:
                bounds = ([-2, 0],  # lower
                          [2, 1e-6])  # upper
                # bounds = ([-1, 0, -np.infty],  # lower
                #           [1, 1e-6, np.infty])  # upper

            # Perform initial fit 
            Lp = lphi #store pre-fit Lphi.
            Bphi = WAL.HLN_li_to_Hi(lphi=lphi)
            # subset_inds = np.where(np.abs(B) < b_window * Bphi)
            # subset_B = B[subset_inds]
            # subset_dsigma = dsigmaxx[subset_inds]
            ## using entire range.
            subset_B = B
            subset_dsigma = dsigmaxx
            # params, covar = so.curve_fit(WAL.HLN_offset,
            #                              subset_B, subset_dsigma, p0=p0, bounds=bounds)
            params, covar = so.curve_fit(WAL.HLN, 
                                        subset_B, subset_dsigma, p0=p0, bounds=bounds)
            
            # Calculate new B_lim
            fitted_Lphi = params[1]
            fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1]) #corresponding field of fitted Lphi.
            
            # Re-perform fit while: B_lim*(1-b_tol) < B_phi < (1+b_tol)*B_lim
            i = -1
            Bphi=0
            # print("i=" + str(i) + " Bphi: " + str(Bphi) + " Bphi_fit: " + str(fitted_Bphi))
            while not (Bphi*(1-b_tol) < fitted_Bphi and fitted_Bphi < (1+b_tol)*Bphi
                       ) and (i < iter_max):
                i+=1
                # Set new Bphi to fitted_Bphi
                movement = 0.25  # dampened movement towards fitted Bphi %
                Bphi = Bphi*(1-movement) + fitted_Bphi*(movement)
                Lp = Lp*(1-movement) + fitted_Lphi*(movement)
                # to avoid oscillation fitting...
                
                #Update subsets
                subset_inds = np.where(np.abs(B) < b_window * Bphi)
                subset_B = B[subset_inds]
                subset_dsigma = dsigmaxx[subset_inds]
                
                #Update p0
                p0 = [params[0], Lp]                                                                                                                                           
                
                #Perform fit again
                # params, covar = so.curve_fit(
                #     WAL.HLN_offset, subset_B, subset_dsigma, p0=p0, bounds=bounds)
                params, covar = so.curve_fit(
                    WAL.HLN, subset_B, subset_dsigma, p0=p0, bounds=bounds)
                # corresponding field of fitted Lphi.
                fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1])
                fitted_Lphi = params[1]
                
                if i > iter_max:
                    raise RuntimeError("Failed to converge for Bphi (B < " +\
                        str(b_window)+ "* Bphi) after " + str(iter_max) + "iterations")
            
            unc = np.sqrt(np.diag(covar))

            if not sub_hf_linear:
                return params, unc
            else:
                return params, unc, (hfparams, hfunc, dSxxB0)

        def HLN_LMR_fit_iterative(B, sigmaxx, alpha=0.5, lphi=100e-9,  # offset=0,
                              b_window=10, b_tol=0.1, iter_max=50,
                              bounds=None, b0_lim=0.025):
            """Fits weak-antilocalisation contributions to the simplified HLN formula, where
            spinorbit scattering strength B_SO >> B_Phi phase coherence strength. 
            Uses WAL.HLN to perform fit calculation.
            Additionally truncates fit data iteratively until B < b_window*Bphi convergences

            Parameters
            ----------
            B : np.ndarray
                List of field values.
            sigmaxx : np.ndarray
                List of sigma_xx values (fit performs sigma_xx@B=0 subtraction).
            alpha : float, optional
                Prefactor usually describing material system, 
                by default 0.5 for topological insulators. 
                Default bounds [-2,2]
            lphi : float, optional
                Phase coherence length, by default 100e-9 meters = 100 nm.
                Default bounds [0,1e-6]
            b_window : float, optional
                Allowed field domain, such that B < b_window*B_phi. By default 10.
                Imporant because WAL is a dominant effect at small field.
            b_tol : float, optional
                Allowed tollerance (%) between B_phi and B/b_window. If outside tolerance,
                iterates fit again with updated domain.
            b0_lim : float, optional
                Upper limit on absolute field values that sigma_xx@b0 can be
                calculated from. Usually averages 5 nearest points (eitherside) to Bmin.
                Default as 0.025 T.
            sub_hf_linear : bool, optional
                Option to subtract high-field linear component from sigma_xx. 
                Uses hf_tol to determine linear estimate.
            hf_lim : float, optional
                Minimum threshold value for field (T) by which to subtract a linear
                component. Has to be a magnitude. Default as 1.5 T (B > 1.5 T). 

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                Returns a tuple containing a list of parameters (alpha, lphi, offset) 
                and their respective uncertainties.
            """
            assert (np.all(B.shape == sigmaxx.shape))
            assert b_tol < 1 and b_tol > 0

            # get closest value to 0.
            # could be multiple values.
            min_i = np.where(np.abs(B) == np.min(np.abs(B)))
            # minB = np.average(sigmaxx[min_i]) #value not needed...
            # Use 5 closest values (eitherside) to avoid significant \sigma0 error.
            # Ensure 5 closest values at field value less than 0.025 T.
            # nearby possible indexes of first index.
            npi = np.linspace(-5, 5, 11, dtype=int) + min_i[0][0]
            # correct for out of range
            npi = npi[np.bitwise_and(npi >= 0, npi < len(B))]
            npb = B[npi]  # get nearby possible field values
            npx = sigmaxx[npi]  # get nearby possible sigmaxx values
            nbi = np.where(npb < b0_lim)  # nearby field values (valid)
            # average field value within 5 datapoints
            sigmaxx_b0 = np.average(npx[nbi])

            # estimate the LMR gradient
            hf_lim = 1.2 # data >1.2 T
            hfi = np.where(np.abs(B) > hf_lim)
            hfb = B[hfi]
            hfs = sigmaxx[hfi]
            
            pLMR = ((1/hfs[0]-1/hfs[-1])/(hfb[0]-hfb[-1]),  # estimate gradient
                    1/sigmaxx_b0)  # use estimate for zero field conductance.
            boundsLMR = [
                [-np.inf, -np.inf], #lower
                [np.inf, np.inf] #upper
            ]
            
            pHLN = (alpha,  # 0.5 for TIs
                  lphi)  # 100nm
            boundsHLN = [
                [-2, 0], #lower
                [2, 1e-6] #upper
            ] if bounds is None else bounds

            p0HLN_LMR = np.r_[pHLN,pLMR]
            bHLN_LMR = np.c_[boundsHLN, boundsLMR]

            # Perform initial fit
            Lp = lphi  # store pre-fit Lphi.
            Bphi = WAL.HLN_li_to_Hi(lphi=lphi)
            
            # using entire range.
            subset_B = B
            subset_sigma = sigmaxx
            print("Init", p0HLN_LMR)
            params, covar = so.curve_fit(WAL.HLN_LMR,
                    subset_B, subset_sigma, p0=p0HLN_LMR, bounds=bHLN_LMR)

            print("Fit", params)
            # # Calculate new B_lim
            # fitted_Lphi = params[1]
            # # corresponding field of fitted Lphi.
            # fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1])

            # # Re-perform fit while: B_lim*(1-b_tol) < B_phi < (1+b_tol)*B_lim
            # i = -1
            # Bphi = 0
            # # print("i=" + str(i) + " Bphi: " + str(Bphi) + " Bphi_fit: " + str(fitted_Bphi))
            # print(i, params)
            # while not (Bphi*(1-b_tol) < fitted_Bphi and fitted_Bphi < (1+b_tol)*Bphi
            #            ) and (i < iter_max):
            #     i += 1
            #     # Set new Bphi to fitted_Bphi
            #     movement = 0.25  # dampened movement towards fitted Bphi %
            #     Bphi = Bphi*(1-movement) + fitted_Bphi*(movement)
            #     Lp = Lp*(1-movement) + fitted_Lphi*(movement)
            #     # to avoid oscillation fitting...

            #     # Update subsets
            #     subset_inds = np.where(np.abs(B) < b_window * Bphi)
            #     subset_B = B[subset_inds]
            #     subset_sigma = sigmaxx[subset_inds]

            #     # Update p0
            #     pHLN = (params[0], Lp)
            #     pLMR = (params[2], params[3])
            #     p0HLN_LMR = np.r_[pHLN, pLMR] 

            #     # Perform fit again
            #     # params, covar = so.curve_fit(
            #     #     WAL.HLN_offset, subset_B, subset_dsigma, p0=p0, bounds=bounds)
            #     params, covar = so.curve_fit(
            #         WAL.HLN_LMR, subset_B, subset_sigma, p0=p0HLN_LMR, bounds=bHLN_LMR)
            #     print(i, params)
            #     # corresponding field of fitted Lphi.
            #     fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1])
            #     fitted_Lphi = params[1]

            #     if i > iter_max:
            #         raise RuntimeError("Failed to converge for Bphi (B < " +
            #                            str(b_window) + "* Bphi) after " + str(iter_max) + "iterations")

            unc = np.sqrt(np.diag(covar))

            return params, unc



        def HLN_fit_const_alpha_iterative(B, sigmaxx, alpha, lphi=100e-9, offset=0,
                                     b_window=10, b_tol=0.1, iter_max=50, bounds=None):
            """Fits weak-antilocalisation contributions to the simplified HLN formula, where
            spinorbit scattering strength B_SO >> B_Phi phase coherence strength. 
            Uses WAL.HLN to perform fit calculation, though uses constant alpha.
            Additionally truncates fit data iteratively until B < b_window*Bphi convergences

            Parameters
            ----------
            B : np.ndarray
                List of field values.
            sigmaxx : np.ndarray
                List of sigma_xx values (fit performs sigma_xx@B=0 subtraction).
            alpha : float
                Prefactor usually describing material system, 
                normally 0.5 for topological insulators.
            lphi : float, optional
                Phase coherence length, by default 100e-9 meters = 100 nm.
                Default bounds [0,1e-6]
            offset : float, optional
                Constant sigma offset, uncaptured by subtracting sigma(B=0). 
                By default 0. Default bounds [-np.infty, np.infty].
            b_window : float, optional
                Allowed field domain, such that B < b_window*B_phi. By default 10.
                Imporant because WAL is a dominant effect at small field.
            b_tol : float, optional
                Allowed tollerance (%) between B_phi and B/b_window. If outside tolerance,
                iterates fit again with updated domain.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                Returns a tuple containing a list of parameters (lphi, offset) and their 
                respective uncertainties.
            """
            assert (np.all(B.shape == sigmaxx.shape))
            assert b_tol < 1 and b_tol > 0

            # get closest value to 0.
            min_i = np.where(np.abs(B) == np.min(np.abs(B)))
            sigmaxx_b0 = np.average(sigmaxx[min_i])

            # get change from (as close to) zero field (as possible).
            dsigmaxx = sigmaxx - sigmaxx_b0

            p0 = (lphi,  # 100nm
                  offset)  # 0 S

            if bounds is None:
                bounds = ([0, -np.infty],  # lower
                          [1e-6, np.infty])  # upper

            # Perform initial fit
            Bphi = WAL.HLN_li_to_Hi(lphi=lphi)
            # subset_inds = np.where(np.abs(B) < b_window * Bphi)
            # subset_B = B[subset_inds]
            # subset_dsigma = dsigmaxx[subset_inds]
            # using entire range.
            subset_B = B
            subset_dsigma = dsigmaxx
                
            def const_alpha(B, lph, off):
                return WAL.HLN_offset(B=B, lphi=lph, offset=off, alpha=alpha)
            
            params, covar = so.curve_fit(const_alpha,
                                         subset_B, subset_dsigma, p0=p0, bounds=bounds)

            # Calculate new B_lim
            # corresponding field of fitted Lphi.
            fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1])

            # Re-perform fit while: B_lim*(1-b_tol) < B_phi < (1+b_tol)*B_lim
            i = -1
            Bphi = 0
            # print("i=" + str(i) + " Bphi: " + str(Bphi) + " Bphi_fit: " + str(fitted_Bphi))
            while not (Bphi*(1-b_tol) < fitted_Bphi and fitted_Bphi < (1+b_tol)*Bphi) and (i < iter_max):
                i += 1
                # Set new Bphi to fitted_Bphi
                movement = 0.25  # movement towards fitted Bphi %
                Bphi = Bphi*(1-movement) + fitted_Bphi*(movement)
                # to avoid oscillation fitting...

                # Update subsets
                subset_inds = np.where(np.abs(B) < b_window * Bphi)
                subset_B = B[subset_inds]
                subset_dsigma = dsigmaxx[subset_inds]

                # Perform fit again
                params, covar = so.curve_fit(
                    const_alpha, subset_B, subset_dsigma, p0=p0, bounds=bounds)
                # corresponding field of fitted Lphi.
                fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1])

                if i > iter_max:
                    raise RuntimeError("Failed to converge for Bphi (B < " +
                                       str(b_window) + "* Bphi) after " + str(iter_max) + "iterations")

                # print("i=" + str(i) + " Bphi: " + str(Bphi) + " Bphi_fit: " + str(fitted_Bphi))

            unc = np.sqrt(np.diag(covar))

            return params, unc
    def HLN_full(B, li, lso, l0, ls):
        """Full formula for strong weak-anti-localisation, according to HLN.
            See https://academic.oup.com/ptp/article/63/2/707/1888502
            Also see the following for the field interpretation of the relaxation times.
            https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.48.1046
            
            Note digamma(z) ~= ln(z) - 1/2z
            Hence you may see the form 
        
        Here p0 is a list parameters:
            alpha   Prefactor (usually 0.5 for TI)
            lphi    Phase coherence length (in meters)
            
        B is a 1D vector containing Field.
        """
        
        # Unpack parameter values.
        # alpha, lphi = p0 #packed values.
        
        # First factors
        prefactor = np.power(e, 2) / pi / h
        
        
        # Convert li to Hi
        # Hphi = WAL.HLN_li_to_Hi(lphi) # phase coherence == spin impurity scattering?
        # He = WAL.HLN_li_to_Hi(le) # mean free path == potential scattering + inelastic scattering?
        # Hso = WAL.HLN_li_to_Hi(lso) # spin orbit scattering
        
        
        Hi = WAL.HLN_li_to_Hi(li) #inelastic scattering
        Hso = WAL.HLN_li_to_Hi(lso) #spin orbit scattering
        H0 = WAL.HLN_li_to_Hi(l0) #potential scattering
        Hs = WAL.HLN_li_to_Hi(ls) #magnetic spin/impurity scattering
        
        # Convert to eqn factors
        H1 = H0 + Hso + Hs
        H2 = 4/3*Hso + 2/3 * Hs + Hi
        H3 = 2*Hs + Hi
        H4 = 2/3 * Hs + 4/3 * Hso + Hi
        
        # Diagamma components
        dig1 = digamma(0.5 + H1 / B)
        dig2 = digamma(0.5 + H2 / B)
        dig3 = digamma(0.5 + H3 / B)
        dig4 = digamma(0.5 + H4 / B)

        # Check where zero values exist in B and numerically correct for resulting NaNs:
        infinities = np.where(np.isinf(B))[0]
        dig1[infinities] = 0
        dig2[infinities] = 0
        dig3[infinities] = 0
        dig4[infinities] = 0
        
        #Equation
        result = - prefactor * (dig1 - dig2 + 0.5*dig3 - 0.5*dig4)
        
        return result
    
    def HLN_li_to_Hi(lphi):
        """Conversion of a characteristic length to field. 
        This could be spin-orbit, potential, magnetic or inelastic,
        which correspond to 

        Parameters
        ----------
        lphi : float, numpy.ndarray
            Value(s) of the characteristic length (m)

        Returns
        -------
        Bphi
            Values(s) of the chatacteristic field (T)
        """
        
        Bi = hbar / (4 * e * lphi**2)
        return Bi
    
    def HLN_Hi_to_li(Hphi):
        """Inverse of HLN_li_to_Hi"""
        return np.sqrt(Hphi * (4 * e) / hbar)
        
    def HLN_li_to_ti(lphi, D):
        """Conversion of the characteristic length to relaxation time"""
        tphi = lphi**2 / D
        return tphi
    
    def HLN_ti_to_li(tphi, D):
        """Inverse of HLN_lphi_to_tphi"""
        return np.sqrt(D*tphi)
    
    def diffusion_constant(Rxx, thickness, N):
        """Diffusion constant calculation according to G. Bergman
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.48.1046

        Parameters
        ----------
        Rxx : float
            Longitudinal resistance (ohms) at 0 field.
        thickness : float
            Sample thickness (m)
        N : float
            Density of states (/m^3)

        Returns
        -------
        _type_
            Diffusion constant in units m^2/s
        """
        return 1/(Rxx * thickness * N* e**2)
    
    
class WL():
    pass