import pylectric
import numpy as np
import math

def reduction(data, step, colN=0):
    """Averages data to fit into bins with width step, along the column colN"""
    assert isinstance(step, float) or isinstance(step, int)
    assert step>0
    assert colN >= 0
    
    # Get Max/Min
    nmax = np.amax(data[:, colN])
    nmin = np.amin(data[:, colN])
    
    # number of new points
    Nupper = math.ceil(nmax / step)
    Nlower = math.floor(nmin / step)
    N = Nupper - Nlower
    # Setup new colN
    x = np.linspace(Nlower * step, Nupper * step, N+1)
    if type(step) != int and step % 1 != 0:
        #Caculate step precision.
        decimals = - int(np.floor(np.log10(step)))
        #apply rounding to linspace.
    x = np.round(x,decimals=decimals) #correct any float imprecision. ie - linspace(-8.0, 8.01, 1602) does not give 0.01 steps appropriately. 

    ### Generate averaged data from other indexes
    # Acquire 
    # data2 = data.copy()
    # data2 = np.delete(data2, colN, axis=1)
    # Setup new arrays
    new_data = np.zeros((N+1, len(data[0])))
    new_data_std = np.zeros((N+1, len(data[0])))
    # Popuate with averaged data
    for i in range(N+1):
        x1 = x[i] - 0.5 * step
        x2 = x[i] + 0.5 * step
        ind = np.where(np.bitwise_and(data[:,colN]>=x1, data[:,colN] < x2))[0]
        # ind_data = data2[ind, :]
        ind_data = data[ind, :]
        new_data[i,:] = np.average(ind_data, axis=0)
        new_data_std[i,:] = np.std(ind_data, axis=0)
    return x, new_data, new_data_std

def symmetric_reduction(data, step, colN=0):
    """Generates same function as 'reduction', but additionally trims the data so that the reduced data is symmetric about colN.

    Args:
        data (_type_): _description_
        step (_type_): _description_
        colN (int, optional): _description_. Defaults to 0.

    Raises:
        AttributeError: _description_
        AttributeError: _description_
        AttributeError: _description_

    Returns:
        _type_: _description_
    """
    assert isinstance(data, np.ndarray)
    assert len(data) > 1
    assert len(data.shape) == 2
    assert isinstance(step, float) or isinstance(step, int)
    assert isinstance(colN, int)
    
    x, new_data, new_data_std = reduction(data,step, colN)
    new_data[:, colN] = x #override colN to ensure symmetric data. 
    
    print(x)
    
    assert len(x) > 1
    # Modify bounds of data to keep symettric about 0.
    dif = abs(x[0]) - abs(x[-1]) #endpoint difference
    if dif == 0 and x[0] == -x[-1]:
        pass #bounding and balancing are GOOD.
    elif dif == 0:
        raise AttributeError("Specified ColN starts and ends on the same signed value.")
    else: 
        if dif > 0:
            # if multiple values, take first value (furthest away from x[-1])
            srch = np.where(np.abs(x[:-1]) == np.abs(x[-1]))[0]
            if len(srch) > 0:
                ind = srch[0]
            else:
                raise AttributeError("No enpoint values matched.")
            x = x[ind:]
            new_data = new_data[ind:,:]
            new_data_std = new_data_std[ind:, :]
        else: #dif < 0
            # if multiple values, take last value (furthest away from x[0])
            srch = np.where(np.abs(x[1:]) == np.abs(x[0]))[0]
            if len(srch) > 0:
                ind = srch[-1]
            else:
                raise AttributeError("No enpoint values matched.")
            ind = 2 + srch[-1] #need to index forward
            x = x[:ind]
            new_data = new_data[:ind]
            new_data_std = new_data_std[:ind, :]
    return x, new_data, new_data_std

def symmetrise(data, colN=0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrises a dataset, retains original values for colN.
    If odd length, assumes symmetry about the midpoint.
    If even length, assumes symmetry between the middle two points.
    

    Args:
        data (NDarray): 2D array, indexed by rows, columns.
        colN (int, optional): _description_. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    assert isinstance(data, np.ndarray)
    assert type(colN)==int
    
    dlen = len(data)
    is_odd = dlen & 1
    if is_odd:
        dhalf = (dlen - 1) >> 1 
        #Includes mid point:
        sym = (data[dhalf:] + data[dhalf::-1]) / 2
        asym = (data[dhalf:] - data[dhalf::-1]) / 2 
        
        # Redistribute
        full_sym = np.r_[sym[::-1,], sym[1::]] #don't dpublicate midpoint
        full_asym = np.r_[-asym[::-1,], asym[1::]] #don't dpublicate midpoint

    else: #even, no midpoint
        dhalf = dlen >> 1
        sym = (data[dhalf:] + data[dhalf-1::-1]) / 2
        asym = (data[dhalf:] - data[dhalf-1::-1]) / 2
        
        # Redistribute
        full_sym = np.r_[sym[::-1,], sym[0::]] 
        full_asym = np.r_[-asym[::-1,], asym[0::]] 
        
    
    # Restore independent variable
    full_sym[:, colN] = data[:, colN]  
    full_asym[:, colN] = data[:, colN]
    
    return full_sym, full_asym

