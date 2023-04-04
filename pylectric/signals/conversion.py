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
        assert isinstance(const_current, (float, int))
            
        return data / const_current
    
    def V2R_VarCurrent(data, v_source, series_res):
        """Converts voltage probe resistance in volts to resistance in ohms. 
        Assumes a variable current source, in the case where the sample resistance is comparable to the output impedance of the current source.
        Note, due to contact resistance and un-measured parts of the sample, this method will yield slightly incorrect data.

        Args:
            data (np.ndarray): Array of voltage measurements of resistance (volts).
            v_source (float): Voltage source
            series_res (float): Series resistance to sample.
            
        Returns:
            np.ndarray: List of the source current magnitude.
        """
        
        assert isinstance(data, np.ndarray)
        assert isinstance(v_source, (float, int, np.ndarray))
        if isinstance(v_source, np.ndarray):
            assert data.shape[0] == v_source.shape[0]
        
        v_res = v_source - data
        i_res = v_res / series_res
        resistance = data / i_res
        
        return resistance

class source:
    """Container for common voltage/current source unit conversions"""
    def V2C(data, circuit_resistance):
        """Converts variable source voltage to current. Circuit resistance can vary with voltage or be a constant value.

        Args:
            data (np.ndarray): Array of voltage source magnitude.
            circuit_resistance (float | np.ndarray): Either a constant value of the complete circuit resistance or a varying array.

        Returns:
            np.ndarray: List of the source current magnitude.
        """
        
        assert isinstance(data, np.ndarray)
        assert isinstance(circuit_resistance, (float, int, np.ndarray))
        if isinstance(circuit_resistance, np.ndarray):
            assert data.shape[0] == circuit_resistance.shape[0]
        
        current = data / circuit_resistance
        return current
        
    def C2V(data, circuit_resistance):

        assert isinstance(data, np.ndarray)
        assert isinstance(circuit_resistance, (float, int, np.ndarray))
        if isinstance(circuit_resistance, np.ndarray):
            assert data.shape[0] == circuit_resistance.shape[0]

        voltage = data * circuit_resistance
        return voltage
  
    
class preamplifier:
    
    def removeConstGain(data, gain) -> np.ndarray:
        assert isinstance(data, np.ndarray)
        
        return data / gain



# class currentDrain():
    
#     pass