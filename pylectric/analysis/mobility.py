import numpy as np

def mobility_gated_dtm(data, Cg):
    """ Calculates the direct transconductance method mobility.
        Requires the gate capacitance (Farads).
        Data should have columns as [Gate Voltage, Conductivity].
        Returns Mobility, units are cm$^2$V$^{-1}$s${-1}$
    """

    dd = np.diff(data[:,:].copy(), axis=0)
    grad = np.abs(dd[:,1] / dd[:,0])
    mu_dtm = 1.0 / Cg * grad * 1.0E4

    #Pack into array to return
    mu_dtm_2D = np.zeros((len(mu_dtm), 2))
    mu_dtm_2D[:,1] = mu_dtm
    mu_dtm_2D[:,0] = data[:-1,0] + dd[:,0]

    return mu_dtm_2D
