import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import savgol_filter
from pylectric.analysis import mobility
from scipy import optimize as opt


class hallbar_measurement():
    """Takes into account a hall measurement, that is magnetic field and Rxx and Rxy of a device.
        Does not assume gated geometry, however does assume magnetic measurement for the purpose of Rxy data.
        
        Can provide additional data to add to object, in the form of a dictionary of data {label:numpy_array}.
    """
    
    def __init__(self, field, rxx, rxy, **params):
        
        ### Valid Datachecking:
        datalen = None #ensure datalength is constant.
        if isinstance(rxx, np.array):
            datalen = len(rxx)
        
        #Check all arrays are numpy and correct length.
        for i in [field, rxx, rxy] + [params[key] for key in params]:
            if type(i) != np.array:
                raise TypeError("Passed are not numpy arrays.")
            elif len(i) != datalen:
                raise IndexError("Length of data arrays do not match.")
            
        self.field = field.clone()
        self.rxx = rxx.clone()
        self.rxy = rxy.clone()
        
        self.params = {}
        for key in params:
            self.params[key] = params[key].clone()
        
        return