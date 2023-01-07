# Libraries & Imports
import numpy as np
from scipy.special import digamma

#Constants
from scipy.constants import elementary_charge as e
from scipy.constants import Planck as h
from scipy.constants import pi
from scipy.constants import hbar


class WAL():
    """ Class to contain various literature Weak Anti-Localisation phenomena.
    """
    def HLN_WAL_Simple(B, *p0):
        """Formula for strong weak-anti-localisation, according to HLN.
            Here, B_SO >> B_Phi (the spin orbit characteristic field is much stronger than the phase coherence characteristic field)
            See https://academic.oup.com/ptp/article/63/2/707/1888502
        
        Here p0 is a list parameters:
            alpha   Prefactor (usually 0.5 for TI)
            lphi    Phase coherence length (in meters)
            
        B is a 1D vector containing Field.
        """
        
        # Unpack parameter values.
        alpha, lphi = p0 #packed values.
        
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
    
    def HLN_WAL_Full(B, *p0):
        """Full formula for strong weak-anti-localisation, according to HLN.
            See https://academic.oup.com/ptp/article/63/2/707/1888502
        
        Here p0 is a list parameters:
            alpha   Prefactor (usually 0.5 for TI)
            lphi    Phase coherence length (in meters)
            
        B is a 1D vector containing Field.
        """
        
        # Unpack parameter values.
        alpha, lphi = p0 #packed values.
        
        # First factors
        prefactor = np.power(e,2) / pi / h
        
        
            
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
    
    
class WL():
    pass