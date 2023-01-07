import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import savgol_filter
from pylectric.analysis import mobility
from scipy import optimize as opt


class kelvinprobe_measurement():
    """Takes into account a 4-probe (aka. kelvin probe) measurement, that is an Rxx of a device.
    """
    
    def __init__(self, rxx, rxy):
        