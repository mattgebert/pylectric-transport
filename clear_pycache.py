import pathlib

[p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]
[p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]
[p.rmdir() for p in pathlib.Path('.').rglob('__pycache__ (1)')]

import math

math.pi
kB      = 1.38e-23 #m^2 kg s^-2 K^-1
rho_s   = 7.6e-7 #kg/m^2
vf      = 1e6 #m/s
vs      = 2.1e4 #m/s
e       = 1.60217662e-19 #C
h       = 6.62607004e-34


Da2 = 0.5 * e**2 / h * (2 * h**2 * rho_s * vs**2  * vf**2) / (math.pi**2 * kB)
math.sqrt(Da2)/e

0.07 * h/e**2 * 1e-2 / 150
dRhodT = 0.12
Da2 = dRhodT * (e**2 / h) * (2 * h**2 * rho_s * vs**2  * vf**2) / (math.pi**2 * kB)
math.sqrt(Da2)/e
