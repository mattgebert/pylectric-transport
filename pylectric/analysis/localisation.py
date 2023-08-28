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
        sig_off Conductivity offset (in Seimens)
            
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
        infinities = np.where(np.isinf(BB))[0]
        dig[infinities] = 0
        log[infinities] = 0

        return -prefactor*(dig-log)
    
    def HLN_offset(B, alpha, lphi, offset):
        """Same as HLN, but wiht an additional constant offset parameter."""
        return WAL.HLN(B, alpha, lphi) - offset
    
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
    
        def HLN_fit_iterative(B, sigmaxx, alpha=0.5, lphi=100e-9, offset=0,
                                    b_window=10, b_tol=0.1, iter_max = 50,
                                    bounds = None):
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
                Returns a tuple containing a list of parameters (alpha, lphi, offset) and their 
                respective uncertainties.
            """
            assert (np.all(B.shape == sigmaxx.shape))
            assert b_tol < 1 and b_tol > 0

            # get closest value to 0.
            min_i = np.where(np.abs(B) == np.min(np.abs(B)))
            sigmaxx_b0 = np.average(sigmaxx[min_i])

            # get change from (as close to) zero field (as possible).
            dsigmaxx = sigmaxx - sigmaxx_b0

            p0 = (alpha,  # 0.5 for TIs
                lphi,  # 100nm
                offset) # 0 S

            if bounds is None:
                bounds = ([-1, 0, -np.infty],  # lower
                          [1, 1e-6, np.infty])  # upper

            # Perform initial fit 
            Bphi = WAL.HLN_li_to_Hi(lphi=lphi)
            # subset_inds = np.where(np.abs(B) < b_window * Bphi)
            # subset_B = B[subset_inds]
            # subset_dsigma = dsigmaxx[subset_inds]
            ## using entire range.
            subset_B = B
            subset_dsigma = dsigmaxx
            params, covar = so.curve_fit(WAL.HLN_offset, 
                                        subset_B, subset_dsigma, p0=p0, bounds=bounds)
            
            # Calculate new B_lim
            fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1]) #corresponding field of fitted Lphi.
            
            # Re-perform fit while: B_lim*(1-b_tol) < B_phi < (1+b_tol)*B_lim
            i = -1
            Bphi=0
            # print("i=" + str(i) + " Bphi: " + str(Bphi) + " Bphi_fit: " + str(fitted_Bphi))
            while not (Bphi*(1-b_tol) < fitted_Bphi and fitted_Bphi < (1+b_tol)*Bphi) and (i < iter_max):
                i+=1
                # Set new Bphi to fitted_Bphi
                movement = 0.25  # movement towards fitted Bphi %
                Bphi = Bphi*(1-movement) + fitted_Bphi*(movement)
                # to avoid oscillation fitting...
                
                #Update subsets
                subset_inds = np.where(np.abs(B) < b_window * Bphi)
                subset_B = B[subset_inds]
                subset_dsigma = dsigmaxx[subset_inds]
                
                #Perform fit again
                params, covar = so.curve_fit(
                    WAL.HLN_offset, subset_B, subset_dsigma, p0=p0, bounds=bounds)
                # corresponding field of fitted Lphi.
                fitted_Bphi = WAL.HLN_li_to_Hi(lphi=params[1])
                
                if i > iter_max:
                    raise RuntimeError("Failed to converge for Bphi (B < " +\
                        str(b_window)+ "* Bphi) after " + str(iter_max) + "iterations")
                    
                # print("i=" + str(i) + " Bphi: " + str(Bphi) + " Bphi_fit: " + str(fitted_Bphi))
            
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
        
    def HLN_lphi_to_tphi(lphi, D):
        """Conversion of the characteristic length to relaxation time"""
        tphi = lphi**2 / D
        return tphi
    
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