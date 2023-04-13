# Function Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
# Custom Libraries
from pylectric.graphing import geo_base, graphwrappers
# Programming Syntax
import warnings
from overrides import override
from abc import abstractmethod
from enum import Enum

class unit_enum(Enum):
    """"""
    #TODO Unfinished class, needs to integrate with graphing.graphwrappers.transport_graph and geometries.fourprobe
    
    """Enumerate object to source specify variable categories"""
    UNDEFINED           =   0
    CURRENT             =   1
    VOLTAGE             =   2
    FREQUENCY           =   3
    RESISTANCE          =   4
    RESISTIVITY         =   5
    CONDUCTANCE         =   6
    CONDUCTIVIY         =   7
    RESISTANCE_QUANTUM  =   8
    CONDUCTANCE_QUANTUM =   9
    MAGNETIC_FIELD_T    =   10
    MAGNETIC_FIELD_Oe   =   11
    FLUX_QUANTUM        =   12
    
    """Labels corresponding to enumerates"""
    unit_labels = {
        UNDEFINED   :   r"Undefined",
        CURRENT     :   r"Current (A)",
        VOLTAGE     :   r"Voltage (V)",
        FREQUENCY   :   r"Frequency (Hz)",
        RESISTANCE  :   r"R ($\Omega$)",
        RESISTIVITY :   r"$\rho$ ($\Omega$)",
        CONDUCTANCE :   r"Conductance (S)",
        CONDUCTIVIY :   r"$\sigma$ (S)",
        # RESISTANCE
    }

    @classmethod
    def match_str_to_enum(cls, str):
        """Checks the contents of a string and checks if it matches one of the category types.
            Otherwise returns UNDEFINED.

            Args:
                str (string): String to check for pre-defined label.

            Returns:
                _type_: _description_
        """
        str = str.lower().replace("(", "").replace(")", "").split(" ")
        for part in str:
            if part in ["current", "cur", "i", "a"]:
                return cls.CURRENT
            elif part in ["freq", "frequency", "hz", "f"]:
                return cls.FREQUENCY
            elif part in ["v_g", "v$_g$", "$v_g$", "gate", "v", "voltage"]:
                return cls.VOLTAGE
        return cls.UNDEFINED

    