from .base import dielectric_material

class SiO2(dielectric_material):
    """Class for calculating properties of SiO2"""

    def __init__(self, t_ox):
        super().__init__(epsilon_r=4.2, t_di=t_ox)
        return
