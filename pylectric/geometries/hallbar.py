import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import savgol_filter
from scipy import optimize as opt
from pylectric.analysis import mobility
import pylectric

class hallbar_measurement():
    """Takes into account a hall measurement, that is magnetic field and Rxx and Rxy of a device.
        Does not assume gated geometry, however does assume magnetic measurement for the purpose of Rxy data.
        
        Can provide additional data to add to object, in the form of a dictionary of data {label:numpy_array}.
    """
    
    def __init__(self, field, rxx, rxy, dataseries = {}, geom = 1, **params):
        
        ### Valid Datachecking:
        datalen = None #ensure datalength is constant.
        if isinstance(rxx, np.ndarray):
            datalen = len(rxx)
        
        #Check all arrays are numpy and correct length.
        for i in [field, rxx, rxy] + [dataseries[key] for key in dataseries]:
            if not isinstance(i,np.ndarray):
                raise TypeError("Passed are not numpy arrays.")
            elif len(i) != datalen:
                raise IndexError("Length of data arrays do not match.")
            
        self.field = field.copy()
        self.rxx = rxx.copy()
        self.rxy = rxy.copy()
        self.geom = geom
        
        # Convert rxx, rxy to rhoxx, rhoxy, sigmaxx, sigmaxy
        
        self.rhoxx = rxx.copy() * geom
        self.rhoxy = rxy.copy() * geom
        self.sigmaxx = self.rhoxx / (self.rhoxx**2 + self.rhoxy**2)
        self.sigmaxy = -self.rhoxy / (self.rhoxx**2 + self.rhoxy**2)
        
        #Dataseries are other columns of data.
        self.dataseries = {}
        for key in dataseries:
            self.dataseries[key] = dataseries[key].copy()
        
        self.params = params
        
        return
    
    def symmterise(): # -> tuple[hallbar.hallbar_measurement, hallbar.hallbar_measurement]:
        
        #Symmetrise rho data then convert to sigma.
        
        
            
        return
    
    def clone(self):
        newobj = hallbar_measurement(field=self.field, rxx=self.rxx, rxy=self.rxy)
        newobj.dataseries = self.dataseries.copy()
        for key in newobj.dataseries:
            newobj.dataseries[key] = newobj.dataseries[key].copy()
        newobj.rhoxx = self.rhoxx.copy()
        newobj.rhoxy = self.rhoxy.copy()
        newobj.sigmaxx = self.sigmaxx.copy()
        newobj.sigmaxy = self.sigmaxy.copy()
        newobj.geom = self.geom
        newobj.params = self.params.copy()
        return 
    
    def plots():
        fig, ax = plt.subplots()
        return 

    