# Function Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
# Custom Libraries
import pylectric
from pylectric.graphing import geo_base, graphwrappers
from pylectric.analysis import mobility
# Programming Syntax
import warnings
from overrides import override
from abc import abstractmethod
from enum import Enum

    
class fourprobe_measurement(geo_base.graphable_base_dataseries):
    """ Class for a 4-probe (aka. kelvin probe, four-terminal sensing) measurement.
        This class is specific for the source dependence (Current, Frequency, Gate) on the Rxx of a device.
        In particular, this class is useful for testing the validity of a device in performance limits, such as heating, reactance and leakage.
        Only one parameter can be used.
    """
    
    class iv_types(Enum):
        """Enumerate object to source specify independent variable category"""
        UNDEFINED = 0
        CURRENT = 1
        VOLTAGE = 2
        FREQUENCY = 3
        
        @classmethod
        def match_str_to_enum(cls, str):
            """Checks the contents of a string and checks if it matches one of the category types.
            Otherwise returns UNDEFINED.

            Args:
                str (string): String to check for pre-defined label.

            Returns:
                _type_: _description_
            """
            str = str.lower().replace("(","").replace(")","").split(" ")
            for part in str:
                if part in ["current", "cur", "i", "a"]:
                    return cls.CURRENT
                elif part in ["freq", "frequency", "hz", "f"]:
                    return cls.FREQUENCY
                elif part in ["v_g","v$_g$","$v_g$","gate","v","voltage"]:
                    return cls.VOLTAGE
            return cls.UNDEFINED

    iv_labels = {
        iv_types.CURRENT:"Current (A)",
        iv_types.VOLTAGE:"Voltage (V)",
        iv_types.FREQUENCY:"Frequency (Hz)",
        iv_types.UNDEFINED:None
    }
            
    def __init__(self, src, rxx, src_type, src_label="", dataseries={}, geom=1, params={}):
        """One of (Current, Frequency, Gate) need to be included as 'source'."""
        
        # Setup axes plotting default as logarithmic.
        self.set_logarithmic()
        
        # Valid Datachecking:
        datalen = None  # ensure datalength is constant.
        if isinstance(src, np.ndarray):
            datalen = len(src)
        else:
            raise TypeError("Passed are not numpy arrays.")
        
        # Check all arrays are numpy and correct length.
        for i in [rxx] + [dataseries[key] for key in dataseries]:
            if not isinstance(i, np.ndarray):
                raise TypeError("Passed are not numpy arrays.")
            elif len(i) != datalen:
                raise IndexError("Length of data arrays do not match.")

        # Label identification
        assert src_type in self.iv_types
        self.src_type = src_type
        self.src_label = src_label
                
        # Copy values
        self.src = src.copy()
        self.rxx = rxx.copy()
        self.geom = geom
        self.params = params
        # Convert rxx,rhoxx
        self._calculateTransport()

        # initialise super object
        super().__init__(dataseries=dataseries)
        return
    
    def set_logarithmic(self):
        self.graphing_scale = "log"  # matches matplotlib yscale arguments.
    
    def set_linear(self):
        self.graphing_scale = "linear" #matches matplotlib yscale arguments.
    
    @classmethod
    def fromcurrent(cls, current, rxx, dataseries={}, geom=1, current_label=iv_labels[iv_types.CURRENT]):
        return cls(src=current, rxx=rxx, src_type=cls.iv_types.CURRENT, dataseries=dataseries, geom=geom, src_label=current_label)

    @classmethod
    def fromvoltage(cls, gate_v, rxx, dataseries={}, geom=1, gate_label=iv_labels[iv_types.VOLTAGE]):
        return cls(src=gate_v, rxx=rxx, src_type=cls.iv_types.GATE, dataseries=dataseries, geom=geom, src_label=gate_label)
    
    @classmethod
    def fromfreq(cls, freq, rxx, dataseries={}, geom=1, freq_label=iv_labels[iv_types.FREQUENCY]):
        return cls(src=freq, rxx=rxx, src_type=cls.iv_types.FREQUENCY, dataseries=dataseries, geom=geom, src_label=freq_label)

    @override
    def ind_vars(self):
        return self.src

    def ind_vars_labels(self):
        return self.src_label
    
    def ind_vars_type(self):
        return self.src_type

    @override
    def dep_vars(self):
        return self.rxx
    
    @override
    def extra_vars(self):
        return np.c_[*[self.dataseries[key] for key in self.dataseries]]

    def _calculateTransport(self):
        self.rhoxx = self.rxx.copy() * self.geom
        return

    def copy(self):
        newobj = fourprobe_measurement(rxx=self.rxx.copy(),  src=self.src.copy(), src_type=self.src_type, src_label=self.src_label, dataseries={}, geom=self.geom)
        # dataseries items
        newobj.dataseries = {}
        for key,val in self.dataseries.items():
            newobj.dataseries[key] = val.copy()
        # params items
        newobj.params = {}
        for key,val in self.params.items():
            newobj.params[key] = val
        return newobj

    def __add__(self,x):
        assert isinstance(x, fourprobe_measurement)
        #check if type matchups.
        if not self.src_type == x.src_type:
            raise AttributeError("Measurement types do not match (" + str(self.src_type) + ", " + str(x.src_type) + ")")
        #Else:
        newobj = self.copy()
        newobj.src = np.r_[newobj.src, x.src]
        newobj.rxx = np.r_[newobj.rxx, x.rxx]
        newobj.rhoxx = np.r_[newobj.rhoxx, x.rhoxx]
        for key,val in newobj.dataseries.items():
            newobj.dataseries[key] = np.c_[val, x.dataseries[key]]
        paramoverlaps = 0
        for key,val in x.params.items():
            if not key in newobj.params.keys():
                newobj.params[key] = val
            else:
                paramoverlaps += 1
        if paramoverlaps > 0:
            print("Objects combined with " + str(paramoverlaps) + " param overlaps ignored from second item.")
        return newobj

    def _update_mpl_kwargs_xscale(self,**mpl_kwargs):
        return mpl_kwargs

    @override
    def plot_all_data(self, axes=None, **mpl_kwargs):
        mpl_kwargs = self._update_mpl_kwargs_xscale(**mpl_kwargs)
        tg = super().plot_all_data(axes, mpl_kwargs)
        return tg

    @override
    def plot_all_dataseries(self, ax=None, **mpl_kwargs):
        mpl_kwargs = self._update_mpl_kwargs_xscale(**mpl_kwargs)
        tg = super().plot_all_dataseries(ax, **mpl_kwargs)
        return tg
    
    @override
    def plot_dataseries(self, key, ax=None, **mpl_kwargs):
        mpl_kwargs = self._update_mpl_kwargs_xscale(**mpl_kwargs)
        tg = super().plot_dataseries(key, ax, **mpl_kwargs)
        return tg
    
    @override
    def plot_dataseries_with_dep_vars(self, key, ax=None, **mpl_kwargs):
        mpl_kwargs = self._update_mpl_kwargs_xscale(**mpl_kwargs)
        tg = super().plot_dataseries_with_dep_vars(key, ax, **mpl_kwargs)
        return tg
    
    @override
    def plot_dep_vars(self, axes=None, **mpl_kwargs):
        mpl_kwargs = self._update_mpl_kwargs_xscale(**mpl_kwargs)
        tg = super().plot_dep_vars(axes, **mpl_kwargs)
        return tg
        
    
    @override
    def __sub__(self, x):
        subdata = super().__sub__(x)
        newobj = self.copy()
        
        self.src = newobj[:, 0]
        self.rxx = newobj[:, 1]
        for key, i in zip(self.dataseries.keys(), range(subdata.shape[1] - 2)):
            newobj.dataseries[key] = subdata[:, 2+i]
        
        return 