
import numpy as np
import scipy.constants as sc
import scipy.optimize as so


    

class hall():
    '''Signals obtained from a basic hall measurement on a 3D sample.
    '''        
    
    def hall_voltage(field, hall_density, current, thickness, vxy0=0):
        """Returns the voltage of a 3D hall measurement at non-zero field.

        Args:
            field (float): Magnetic field.
            hall_density (float): Sheet density of carrier.
            r0 (float): Zero-field Vxy voltage.
        """

        # Vh = I*B / (q * ns) ns=sheet density
        # Vh = I * B / (q * n * thickness)
        vxy_hall = current * field / (hall_density * sc.e * thickness)

        return vxy0 + vxy_hall
    
    def hall_resistance(field, hall_density, thickness, rxy0=0):
        """Returns the resistance of a 3D hall measurement at non-zero field.

        Args:
            field (float): Magnetic field.
            hall_density (float): Sheet density of carrier.
            thickness (float): Sheet thickness.
            rxy0 (float, optional): Zero-field Rxy resistance.
        """
        
        # Vh = I*B / (q * ns), ns=sheet density
        # Vh = I * B / (q * n * thickness)
        # Rxy = Vh / I
        # Rxy = B / q * n * thickness
        
        rxy_hall = field / (hall_density * sc.e * thickness)
        
        return rxy0 + rxy_hall
    
    def hall_resistivity(field, hall_density, width, length, rhoxy0=0):
        """Returns the sheet resistivity Rs of a 3D hall measurement at non-zero field.

        Args:
            field (float): Magnetic field.
            hall_density (float): Sheet density of carrier.
            thickness (float): Sheet thickness.
            rxy0 (float, optional): Zero-field Rxy resistance.
        """
        #QUESTION FOR MICHAEL & SEMONTI
        
        #Rxy = rhoxy * L / W / thickness
        #-->rhoxy = B / (q * n) * (W / L)
        rxy_hall = field / (hall_density * sc.e) * (width / length)

        return rhoxy0 + rxy_hall
    
    
    def hall_measurement_fit(field, rxy, rxx, thickness):
        """Returns a tuple of three parameters after fitting a hall resistance contribution.
            - Hall carrier, whether the device is electron or hole dominated.
            - Hall density, a measurement of the device carrier density
            - Hall mobility, a measurment of the carrier mobility.

        Args:
            field (_type_): The magnetic field applied.
            rxy (_type_): The corresponding measured Hall resistance (Rxy).
            rxx (_type_): The zero field longituidnal resistance. If an array is passed, will assume value at minimum field.
        """
        
        # 1 Get Rxy0 Value
        b0 = np.where(np.abs(field) == np.min(np.abs(field))) # could be multiple values at zero field
        rxy_b0 = np.mean(rxy[b0]) # average across all (as close to) zero-field values provided.
        
        # 2 Get Rxx0 Value
        if isinstance(rxx, np.ndarray) and (field.shape == rxx.shape):
            rxx_b0 = np.mean(rxx[b0])
        elif isinstance(rxx, float):
            rxx_b0 = rxx
        else:
            raise TypeError("Rxx is required to be either a singular value or array corresponding to `field`.")
        
        # 3 Educated guess for initial density:
        imax = np.argmax(rxy)
        imin = np.argmin(rxy)
        grad = (rxy[imax] - rxy[imin]) / (field[imax] - field[imin])
        # n ~= dB / (q * thickness * dRxy)
        n_hall0 = 1/grad/sc.e/thickness   # default material density, /m^2
        
        # 4 Package:
        p0 = (n_hall0, thickness, rxy_b0)
        
        # 4 Setup fitting function for constant thickness parameter
        def fn(field, hall_density, rxy0):
            return hall.hall_resistance(field, hall_density, thickness, rxy0)
        
        # 5 Fit Hall resistance to line
        params, covar = so.curve_fit(fn, field, rxy, p0)
        
        # 6 Convert parameters to measurable quantities based on sample.
        unc = np.sqrt(np.diag(covar))
        
        # 7 Calculate parameters:
        n_hall = params[0]
        n_hall_unc = unc[0] #std
        mu_hall = 1/(sc.e * n_hall * rxx_b0)
        unc_mu_hall = np.sqrt(np.square(- mu_hall / n_hall * unc[0]))
        rxy0 = params[1]
        unc_rxy0 = unc[1]

        # 8 Package and return
        return ([n_hall, mu_hall, rxy0], [n_hall_unc, unc_mu_hall, unc_rxy0])

class hall2D():
    """Hall measurements for 2D materials.
    Resistivity is characteristically different in 2D systems, which exist on surface or in single layers.
    Therefore thickness/depth  doesn't typically contribute to the geometric considerations, unless comparing to 
    bulk counterparts. 
    In 3D, Resistance = L / (W * T) * Resitivitiy, where resitivity is measured in Ohm meters.
    In 2D, this becomes Resistance = L / (W) * Resitivitiy, where resistivity is measured in Ohms.
    """
    
    def hall_voltage(field, hall_density, current, vxy0=0):
        """Returns the voltage of a 2D hall measurement at non-zero field.

        Args:
            field (float): Magnetic field.
            hall_density (float): Sheet density of carrier.
            r0 (float): Zero-field Vxy voltage.
        """

        # Vh = I*B / (q * n)
        vxy_hall = current * field / (hall_density * sc.e)

        return vxy0 + vxy_hall

    def hall_resistance(field, hall_density, rxy0=0):
        """Returns the resistance of a 2D hall measurement at non-zero field.

        Args:
            field (float): Magnetic field.
            hall_density (float): Sheet density of carrier.
            rxy0 (float, optional): Zero-field Rxy resistance.
        """

        # Vh = I*B / (q * n), n = carrier density 
        # Vh = I * B / (q * n)
        # Rxy = Vh / I
        # Rxy = B / (q * n)

        rxy_hall = field / (hall_density * sc.e)

        return rxy0 + rxy_hall

    def hall_resistivity(field, hall_density, width, length, rhoxy0=0):
        """Returns the sheet resistivity Rs of a 3D hall measurement at non-zero field.

        Args:
            field (float): Magnetic field.
            hall_density (float): Sheet density of carrier.
            thickness (float): Sheet thickness.
            rxy0 (float, optional): Zero-field Rxy resistance.
        """

        # Rxy = rhoxy * L / W
        # -->rhoxy = B / (q * n) * (W / L)
        rhoxy_hall = field / (hall_density * sc.e) * (width / length)

        return rhoxy0 + rhoxy_hall

    def hall_mobility(hall_density, rxx_b0):
        mu_hall = 1/(sc.e * hall_density * rxx_b0)
        return mu_hall

    def hall_measurement_fit(field, rxy, rxx):
        """Returns a tuple of three parameters after fitting a hall resistance contribution.
            - Hall density, a measurement of the device carrier density
            - Hall mobility, a measurment of the carrier mobility.
            - Offset Rxy, signal value offset at zero field.

        Args:
            field (_type_): The magnetic field applied.
            rxy (_type_): The corresponding measured Hall resistance (Rxy).
            rxx (_type_): The zero field longituidnal resistance. If an array is passed, will assume value at minimum field.
        """

        # 1 Get Rxy0 Value
        # could be multiple values at zero field
        b0 = np.where(np.abs(field) == np.min(np.abs(field)))
        # average across all (as close to) zero-field values provided.
        rxy_b0 = np.mean(rxy[b0]) 

        # 2 Get Rxx0 Value
        if isinstance(rxx, np.ndarray) and field.shape == rxx.shape:
            rxx_b0 = np.mean(rxx[b0])
        elif isinstance(rxx, float):
            rxx_b0 = rxx
        else:
            raise TypeError(
                "Rxx is required to be either a singular value or array corresponding to `field`.")

        # 3 Educated guess for initial density:
        imax = np.argmax(rxy)
        imin = np.argmin(rxy)
        grad = (rxy[imax] - rxy[imin]) / (field[imax] - field[imin])
        # n ~= dB / (q * dRxy)
        n_hall0 = 1/grad/sc.e   # default material density, /m^2
        
        # 4 Package 
        p0 = (n_hall0, rxy_b0)

        # 5 Fit Hall resistance to line
        params, covar = so.curve_fit(hall2D.hall_resistance, field, rxy, p0)

        # 6 Convert parameters to measurable quantities based on sample.
        unc = np.sqrt(np.diag(covar))

        # 7 Calculate parameters:
        n_hall = params[0]
        n_hall_unc = unc[0]  # std
        mu_hall = 1/(sc.e * n_hall * rxx_b0)
        unc_mu_hall = np.sqrt(np.square(- mu_hall / n_hall * unc[0]))
        rxy0 = params[1]
        unc_rxy0 = unc[1]


        # 8 Package and return
        return ([n_hall, mu_hall, rxy0], [n_hall_unc, unc_mu_hall, unc_rxy0])
