import pylectric
import numpy as np
import math

def reduction(data, step, colN=0):
    """Averages data to fit into bins with width step, along the column colN"""
    
    # Get Max/Min
    nmax = np.amax(data[:, colN])
    nmin = np.amin(data[:, colN])
    print(nmin,nmax)
    
    # number of new points
    Nupper = math.ceil(nmax / step)
    Nlower = math.floor(nmin / step)
    N = Nupper - Nlower
    print(N, Nupper)
    # Setup new colN
    x = np.linspace(Nlower * step, Nupper * step, N+1)

    ### Generate averaged data from other indexes
    # Acquire 
    # data2 = data.copy()
    # data2 = np.delete(data2, colN, axis=1)
    # Setup new arrays
    new_data = np.zeros((N, len(data[0])))
    new_data_std = np.zeros((N, len(data[0])))
    print(new_data.shape)
    # Popuate with averaged data
    for i in range(N):
        x1 = x[i] - 0.5 * step
        x2 = x[i] + 0.5 * step
        ind = np.where(np.bitwise_and(data[:,colN]>=x1, data[:,colN] < x2))[0]
        # ind_data = data2[ind, :]
        ind_data = data[ind, :]
        new_data[i,:] = np.average(ind_data, axis=0)
        new_data_std[i,:] = np.std(ind_data, axis=0)
    return new_data, new_data_std

def symmetrise(data) -> tuple[np.ndarray, np.ndarray]:

    return
