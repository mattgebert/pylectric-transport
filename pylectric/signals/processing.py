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

def trim_symmetric(data, colN):
    """Ensures that data is symmetric about the midpoint of ColN of data.
    ColN can be a list of indexes.
    Returns (data, (trim_start, trim_mid, trim_end, trim_mid_i)) """
    # TODO https://en.wikipedia.org/wiki/Bloom_filter
    # https://stackoverflow.com/questions/497338/efficient-list-intersection-algorithm
    # Find the largest intersection of two lists
    
    #Quick fix solution:
    # 1. Reverse first list
    # 2. overlap list from endpoints and see how many elements match.
    
    #reverse
    forward = data[:,colN]
    backward = data[::-1, colN]
    
    #calculate overlaps
    max_overlap = 0 #biggest overlap vector
    max_overlap_i = None #distance from beginning of data[:,coln] where vector lies
    max_overlap_j = None #distance from end of data[:,coln] where vector lies.
    for len1 in range(len(forward)):
        for i in range(0,len(forward)-len1):
            for j in range(0,len(forward)-i-len1):
                print(i,j)
                overlap = np.all(forward[i:i+len1] == backward[j:j+len1]) #all have to be overlapping...
                if overlap and (len1 > max_overlap):
                    max_overlap = len1
                    max_overlap_i = i
                    max_overlap_j = j
    #return new data
    newdata = data.copy()
    newdata = np.r_[newdata[max_overlap_i:max_overlap_i+max_overlap, :], #Component from forward
                    newdata[len(forward)-max_overlap_j-max_overlap:len(forward)-max_overlap_j, :]] #component from backward
    trim_start = max_overlap_i
    trim_end = len(forward) - max_overlap_j
    trim_mid = len(newdata) - (trim_start + trim_end)
    trim_mid_i = trim_start + int(len(newdata)/2)
    return newdata, (trim_start, trim_mid, trim_end, trim_mid_i)
    

def trim_matching(data1, data2, colN = [0]):
    """Takes two arrays, and trims the rows to find the maximum overlap of values in colN"""
    # TODO: FIX NEEDS TESTS.
    
    #Combine data into one array, use symmetric matching to find largest overlap area
    assert data1.shape[1] == data2.shape[1] #columns must match, rows can be different lengths.
    cdata = np.r_[data1, data2[::-1]] #reverse data2 so it alignes symettrically about the centre for trim_symmetric.
    symdata, (data1_trim_start, trim_mid, data2_trim_end,
              trim_mid_i) = trim_symmetric(cdata, colN)
    
    data1_trim_end = (len(data1) - trim_mid_i)
    data1_trimlen = data1_trim_start + data1_trim_end
    data2_trim_start = (trim_mid - data1_trim_end)
    data2_trimlen = data2_trim_start + data2_trim_end
    
    print("Trim_matching removed", data1_trimlen,
          "from data1 and", data2_trimlen, "from data2.")
    midpoint = int(np.round(len(symdata)/2))
    data1_trim = symdata[:midpoint, :]
    data2_trim = symdata[midpoint:, :]
    data2_trim = data2_trim[::-1] #correct reverse
    
    return data1_trim, data2_trim

def trim_matching_fromsimilar(data1, data2, colN = [0]):
    """Takes two arrays, and trims the rows using the following assumptions:
    1. Step sizes in colN are the same
    2. One domain is a subset of the other
    3. Smaller size fits exactly in bigger size."""
    c = len(data1) > len(data2) #condition
    if len(data1) == len(data2):
        if np.all(data1[:,colN] == data2[:,colN]):
            return (data1, data2)
        else:
            raise AttributeError("data1 & data2 match size, but not elements.")
    else:
        d1 = data1 if c else data2  # d1 always longer list
        d2 = data2 if c else data1  # d2 alway shorter list
        trim_len = len(d1)-len(d2)
        for i in range(trim_len+1):
            j = trim_len - i
            s1 = d1[i:len(d1)-j,colN]
            if np.all(s1 == d2[:,colN]):
                d1 = d1.copy()[i:len(d1)-j,:]
                d2 = d2.copy()
                return (d1,d2) if c else (d2,d1)
        raise AttributeError("The two lists do not immediately match by trimming ends to match size.")
        
        

def symmetric_reduction(data, step, colN=0):
    """Generates same function as 'reduction', but additionally trims the data so that the reduced data is symmetric about the middle of colN.

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

def symmetrise(data, colN=[0], full_domain=False) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrises a dataset, retains original values for colN.
    If odd length, assumes symmetry about the midpoint.
    If even length, assumes symmetry between the middle two points.
    

    Args:
        data (NDarray):         2D array, indexed by rows, columns.
        colN (int, optional):   Data column by which to symmetrise. Defaults to 0.
        full_domain (bool):     Whether to return the full domain or not. Defaults to False.

    Returns:
        [sym, asym] (tuple[np.ndarray, np.ndarray]): Returns Symmetric and Asymmetric components of the domain.
    """
    assert isinstance(data, np.ndarray)
    assert isinstance(colN, int) or (isinstance(colN, (list, np.ndarray)) and np.all([isinstance(a) for a in colN]))
    
    dlen = len(data)
    is_odd = dlen & 1 #bitcheck if odd or even
    if is_odd:
        dhalf = (dlen - 1) >> 1 #bit reduce by 1 to get half value.
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
    sym[:, colN] = data[dhalf:, colN]
    asym[:, colN] = data[dhalf:, colN]
    
    if full_domain:
        return full_sym, full_asym
    else:
        return sym, asym


def normalise(data, colN = None) -> np.ndarray:
    """Normalizes each column specified in colN by its maximum absolute value.

    Args:
        data (np.ndarray): Input data array
        colN (list, optional): Columns by which to normalize. Defaults to None, implying all columns.

    Returns:
        np.ndarray: Normalized data array.
    """
    
    if not colN:
        imax = np.max(np.abs(data), axis=0) # 2D slice of maximum values along each var.
        norm = data / imax
    else:
        imax = np.max(np.abs(data[:,colN]), axis=0) #2D slice of maximum values along each var.
        norm = data[:, colN] / imax
    
    return norm