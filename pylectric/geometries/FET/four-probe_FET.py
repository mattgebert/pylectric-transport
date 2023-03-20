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


class fourprobe_measurement(geo_base.graphable_base):
    """ Takes into account a 4-probe (aka. kelvin probe, four-terminal sensing) measurement.
        This is an Rxx of a device.
    """
    
    def __init__(self):
        pass