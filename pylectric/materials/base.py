class dielectric_material():
    """ Generic class to define properties and behaviour of a dielectric material.
    """

    def __init__(self, epsilon_r, t_di):
        """ Epsilon_r should be the relative permissitivity constant
            t_di should be the oxide thickenss in meters.
        """

        self.epsilon_r = epsilon_r
        self.t_di = t_di
        return


    def capacitance_per_cm2(self, t_di):
        """ Calculates capacitance(Farads) of some t_di, per cm^2.
        """
        return self.capacitance(t_di=t_di) * 1e-4

    def capacitance(self, t_di=None, area = None):
        """ Calculates capacitance(Farads) of some t_di, per unit meter.
            If area is specified then calculates exact capacitance.
        """
        return dielectric_material.__capacitance(
            t_di    =   (self.t_di if t_di is None else t_di),
            epsilon =   self.epsilon_r,
            area    =   area
        )

    def __capacitance(t_di, epsilon, area = None):
        """ Calculates capacitance(Farads) of some t_di, per unit area.
            If area (meters) is specified then calculates exact capacitance.
        """
        EPSILON_0 = 8.85e-12    #natural permissitivity constant

        # -- Calculate terms --
        #Gate capacitance
        if area is None:
            Cg = EPSILON_0 * epsilon / t_di
        else:
            Cg = EPSILON_0 * epsilon / t_di * area
        return Cg
