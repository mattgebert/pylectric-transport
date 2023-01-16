import numpy as np

class voltageProbes:
    """Container for common voltage probe signal unit conversions.
    """
    def V2R_ConstCurrent(data, const_current):
        """Converts voltage probe resistance in volts to resistance in ohms. 
        Assumes a constant current source, which is true in the case of a voltage source with high series resistance compared to sample resistance.

        Args:
            data (Numpy Array): Array of voltage measurements of resistance (volts).
            const_current (float): The current through the device (amps).
        """
        assert isinstance(data, np.ndarray)
        assert isinstance(const_current, float) or isinstance(const_current, int)
            
        return data / const_current
    
    def V2R_VarCurrent(data, v_source, series_res):
        """Converts voltage probe resistance in volts to resistance in ohms. 
        Assumes a variable current source, in the case where the sample resistance is comparable to the output impedance of the current source.
        Note, due to contact resistance and un-measured parts of the sample, this method will yield slightly incorrect data.

        Args:
            data (Numpy Array): Array of voltage measurements of resistance (volts).
            v_source (float): Voltage source
            series_res (float): Series resistance to sample.
        """
        
        assert isinstance(data, np.ndarray)
        assert isinstance(v_source, float) or isinstance(v_source, int)
        
        v_res = v_source - data
        i_res = v_res / series_res
        
        return data / i_res
    
    
class preamplifier:
    
    def removeConstGain(data, gain) -> np.ndarray:
        assert isinstance(data, np.ndarray)
        
        return data / gain



# class currentDrain():
    
#     pass