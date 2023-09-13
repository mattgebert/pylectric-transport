import pylectric
import numpy as np
from scipy import interpolate
import math
import warnings

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
        if len(ind) == 0:
            warnings.warn("No datapoints between {:0.2f} and {:0.2f} in the column of interest.".format(x1,x2))
        ind_data = data[ind, :]
        new_data[i,:] = np.average(ind_data, axis=0)
        new_data_std[i,:] = np.std(ind_data, axis=0)
    return x, new_data, new_data_std

def reduction_interpolate(data, step, colN=0, linear=True):
    """Interpolates all data to fit into points evenly spaced by step, along the column colN.
    Uses all points within the +- step/2 width, plus nearest neibours.
    Uses either linear interpolation or barycentric interpolation from NumPy.

    Parameters
    ----------
    data : _type_
        _description_
    step : _type_
        _description_
    colN : int, optional
        _description_, by default 0
    """
    assert isinstance(step, float) or isinstance(step, int)
    assert step > 0
    assert colN >= 0
    
    # Get Max/Min
    nmax = np.amax(data[:, colN])
    nmin = np.amin(data[:, colN])

    # number of new points
    # As using bins (ie, half width either side of x), only require rounding, not floor/ceiling.
    Nupper = int(np.round(nmax / step))
    Nlower = int(np.round(nmin / step))
    N = Nupper - Nlower
    # Setup new colN
    x = np.linspace(Nlower * step, Nupper * step, N+1)
    
    if type(step) != int and step % 1 != 0:
        # Caculate step precision.
        decimals = - int(np.floor(np.log10(step)))
        # apply rounding to linspace.
        # correct any float imprecision. ie - linspace(-8.0, 8.01, 1602) does not give 0.01 steps appropriately.
        x = np.round(x, decimals=decimals)

    # Generate interpolated data from other indexes
    # Setup new arrays
    new_data = np.zeros((N+1, len(data[0])))
    new_data_std = np.zeros((N+1, len(data[0])))
    
    # Popuate with interpolated data
    for i in range(N+1):
        xi = x[i]
        
        x1 = xi - 0.5 * step
        x2 = xi + 0.5 * step
        
        # domain inclusive - indexes within x1-x2 data range. Might be zero.
        dombits1 = data[:, colN] >= x1
        dombits2 = data[:, colN] < x2
        dombits = np.bitwise_and(dombits1, dombits2)
        dominc = np.where(dombits)
        # domain exclusive - indexes closest to x1-x2 data range. Inverse of dominc.
        # domexc = np.where(np.bitwise_not(dominc))
        domexc1 = np.where(np.bitwise_not(dombits1))
        domexc2 = np.where(np.bitwise_not(dombits2))
        
        #Check if at endpoints of bins, where there are no points beyond xi. 
        # Set alternative point to nearest neighbor outside range, and linearly interpolate.
        if len(domexc1[0]) == 0:
            if i==0: #no values less than 0th bin range.
                domexc1 = domexc2
            else:
                raise IndexError("No values less than index #",i,": xi=",xi,". This should never happen.")
            
        if len(domexc2[0]) == 0:
            if i==N: #no values less than 0th bin range.
                domexc2 = domexc1
            else:
                raise IndexError("No values greater than index #",i,": xi=",xi,". This should never happen.")
        
        # Find the nearest points outside the domain (x1,x2).
        x1closest = np.abs(data[domexc1][:,colN] - x1)
        x2closest = np.abs(data[domexc2][:,colN] - x2)
        min1 = np.where(np.min(x1closest) == x1closest)
        min2 = np.where(np.min(x2closest) == x2closest)
        
        # warnings.warn("Multiple 'closest' values to interpolation domain.", 
        #               RuntimeWarning, stacklevel=2)
        
        # Use multiple values to construct the linear/baryocentric interpolation.
        p1 = data[domexc1][min1]
        p2 = data[dominc]
        p3 = data[domexc2][min2]
        
        
        # dont add same datapoint twice. This can occur at xmin and xmax values.
        if np.array(min1).shape == np.array(min2).shape and np.all(p1[:,colN] == p3[:,colN]):
            yobs = np.r_[p1, p2]
            xobs = yobs[:,colN]
            
            
            
            #linearly interpolate here, as endpoints more sensitive to big error if using polynomial interpolation.
            for col in range(yobs.shape[-1]):
                f = interpolate.interp1d(xobs, yobs[:,col], fill_value="extrapolate" if (i==0 or i==N) else np.nan)
                yc = f(xi)
                
                # y = interpolate.pchip_interpolate(xobs,yobs,x[i]) #sets colN values to x[i], requires no duplicate x values...
                new_data[i, col] = yc
        else:
            # yobs = np.sort(np.r_[p1,p2,p3], axis=colN)
            yobs = np.r_[p1, p2, p3]
            xobs = yobs[:, colN]
            # xobs = np.r_[p1[:,colN],p2[:,colN],p3[:,colN]]
            # Interpolate new data.
            y = interpolate.barycentric_interpolate(xobs, yobs, xi)  # sets colN values to x[i] anyway haha.
            if np.all(np.isnan(y)):
                for col in range(yobs.shape[-1]):
                    f = interpolate.interp1d(xobs, yobs[:,col], fill_value="extrapolate" if (i==0 or i==N) else np.nan)
                    yc = f(xi)
                    y[col] = yc
                warnings.warn("Barycentric_interpolate failed for:\nxi="+str(xi)+",\nxobs="+str(
                    xobs)+",\nyobs="+str(yobs[:, 0:3])+".\nUsing linear interpolate instead. New values are"+str(y[0:3]))
                if np.all(np.isnan(y)):
                    warnings.warn("Linear_interpolate failed for: xi="+str(xi)+", as well. Setting value to nan.")
            new_data[i] = y
            
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
        
        

def symmetric_reduction(data, step, colN=0, sym_x = 0):
    """Generates same function as 'reduction', but additionally trims the data so that the reduced data is symmetric about the middle of colN.

    Args:
        data (_type_): _description_
        step (_type_): _description_
        colN (int, optional): _description_. Defaults to 0.
        sym_x (int/float, optional): Defaults to 0. 

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
    dif = (abs(x[0]- sym_x) - abs(x[-1]- sym_x)) #endpoint differences from sym point.
    
    if dif == 0 and (x[0]-sym_x) == -(x[-1]-sym_x):
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
        else: #dif <= 0
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


def symmetric_reduction_interpolate(data, step, colN=0, sym_x=0):
    """Generates same function as 'reduction_interpolate', 
    but additionally trims the data so that the reduced data is symmetric about the middle of colN.

    Args:
        data (_type_): _description_
        step (_type_): _description_
        colN (int, optional): _description_. Defaults to 0.
        sym_x (int/float, optional): Defaults to 0. 

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

    x, new_data, new_data_std = reduction_interpolate(data, step, colN)
    new_data[:, colN] = x  # override colN to ensure symmetric data.

    assert len(x) > 1
    # Modify bounds of data to keep symettric about 0.
    # endpoint differences from sym point.
    dif = (abs(x[0] - sym_x) - abs(x[-1] - sym_x))

    if dif == 0 and (x[0]-sym_x) == -(x[-1]-sym_x):
        pass  # bounding and balancing are GOOD.
    elif dif == 0:
        raise AttributeError(
            "Specified ColN starts and ends on the same signed value.")
    else:
        if dif > 0:
            # if multiple values, take first value (furthest away from x[-1])
            srch = np.where(np.abs(x[:-1]) == np.abs(x[-1]))[0]
            if len(srch) > 0:
                ind = srch[0]
            else:
                raise AttributeError("No enpoint values matched.")
            x = x[ind:]
            new_data = new_data[ind:, :]
            new_data_std = new_data_std[ind:, :]
        else:  # dif <= 0
            # if multiple values, take last value (furthest away from x[0])
            srch = np.where(np.abs(x[1:]) == np.abs(x[0]))[0]
            if len(srch) > 0:
                ind = srch[-1]
            else:
                raise AttributeError("No enpoint values matched.")
            ind = 2 + srch[-1]  # need to index forward
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
        return full_sym,full_asym
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


def split_dataset_by_turning_point(data, colN=0, max=True, endpoint_exclusion=0) -> tuple[np.ndarray, np.ndarray]:
    """Finds data splitpoint by considering the abs maximum/minimum value. 
       If more than 1 datapoint at same min/max value, inbetween values will be deleted.

    Args:
        data (_type_): Dataset to split
        colN (_type_, optional): Column of the turning point. Defaults to 0.
        max (bool, optional): Search order for maximum or minium. Defaults to True (finding a maximum).
        endpoint_exclusion (float, optional): A total percentage of the data to exclude from the search by trimming ends.

    Raises:
        AttributeError: If a singular turning point cannot be identified.

    Returns:
        (np.ndarray, np.ndarray)
    """
    i = colN
    
    assert endpoint_exclusion >= 0 and endpoint_exclusion <= 100 #limits for percentages.
    
    def srchMax(data,i):
        j = None
        maxima = np.where(data[:,i]==np.max(data[:,i]))
        if len(maxima[0]) == 1: #only 1 max
            j = maxima[0][0]
        elif len(maxima[0] > 1):
            # calculate index differences.
            d = np.diff(maxima[0])
            # if all next to each other, take middle.
            if np.all(np.abs(d) == 1):
                j = maxima[0][math.floor(len(maxima)/2)]  # middle index.
            else: 
                #if separated by two or more entries, just average.
                #raise warning too....                 
                warnings.warn("Finding extrema turning point: Two or more values identical thus unclear turning point. Have determined turning point by taking average of values.")
                j = int(np.round(np.average(maxima[0])))
        return j

    def srchMin(data,i):
        j = None
        minima = np.where(data[:,i]==np.min(data[:,i]))
        if len(minima[0]) == 1: #only 1 min
            j = minima[0][0]
        elif len(minima[0] > 1):
            # calculate index differences.
            d = np.diff(minima[0])
            # if all next to each other, take middle.
            if np.all(np.abs(d) == 1):
                j = minima[0][math.floor(len(minima)/2)]  # middle index.
            else:
                warnings.warn(
                    "Finding extrema turning point: Two or more values identical thus unclear turning point. Have determined turning point by taking average of values.")
                j = int(np.round(np.average(minima[0])))
        return j

    #setup data if trimming active:
    searchData = data
    frac=0
    if endpoint_exclusion != 0:
        dlen = len(data)
        frac = round(dlen * endpoint_exclusion / 100 / 2) #factor two due to cutting from each end.
        print(f"Trimming {frac} datapoints at each end to search for extrema.")
        searchData = data[frac:-frac,:]

    # Find TP in data
    if max:
        j = srchMax(searchData, i)
        if not j:
            j = srchMin(searchData, i)
    else:
        j = srchMin(searchData, i)
        if not j:
            j = srchMax(searchData, i)
            
    if not j:
        raise AttributeError("The dataset does not have a singular turning point.")    
    else:
        ds1 = data[:frac+j+1,:].copy() #inclusive of endpoint j.
        ds2 = data[frac+j:,:].copy()
        return ds1, ds2 