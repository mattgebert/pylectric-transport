class SiO2_Properties():
    """Class for calculating properties of SiO2"""
    def capacitance(thickness, epsilon = None):
        """Calculates capacitance(F) of some thickness"""
        EPSILON_0 = 8.85e-12    #natural permissitivity constant
        EPSILON_SiO2 = 3.8      #relative sio2 permissivity factor
        e = 1.6e-19             #elementary chrage
        # t_ox = 2.85e-7          #oxide thickness

        #Can modify if desired.
        if epsilon != None:
            EPSILON_SiO2 = epsilon

        # -- Calculate terms --
        #Gate capacitance
        Cg = EPSILON_0 * EPSILON_SiO2 / thickness

        return Cg
