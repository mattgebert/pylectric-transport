import numpy as np
from scipy.signal import savgol_filter
from ..graphing.geo_base import graphable_base

def peak_arrow_location(xdata,ydata, direction ,peak_n=0, window_percent=5.0):
    """Finds a single location for an arrow to sit on a graph. 
    Ideally somewhere there is a flat gradient over a large domain range (ie, 5% minimum)"""
    assert direction in graphable_base.sweep_enum
    if len(xdata) > 6:
        xrange = np.abs(np.max(xdata) - np.min(xdata))
        yrange = np.abs(np.max(xdata) - np.min(xdata))
        
        #1 apply smoothing filter to data
        window = len(xdata) / 100.0 * window_percent # 5% smoothening window
        window = int(np.round(window)) if window > 5 else 5
        xsmooth = savgol_filter(xdata, window, 3) #windowsize, polynomial order.
        ysmooth = savgol_filter(ydata, window, 3)
        
        #2 Calculate gradient
        xdot = np.diff(xsmooth)
        ydot = np.diff(ysmooth)
        ydotx = ydot/xdot
        
        #3 Find gradient points that pass have a sign change.
        sign = np.sign(ydotx)
        signchange = (np.roll(sign,1) - sign != 0) & (np.logical_not(np.isnan(sign))) & (np.logical_not(np.isnan(np.roll(sign,1))))  #true if change in sign
        locs = np.where(signchange == True)[0] #indexes of changes in sign
        
            
        #4 use gradient point that corresponds to largest peak.
        yabs = np.abs(ysmooth) #abs to find max height
        peaks = yabs[locs] #yabs values at changes in sign
        
        #5 sort peaks from largest to smallest.
        inds = np.argsort(peaks)[::-1] #get indicies from sorting peaks, and reverse to make largest at the front.
        sorted_locs = locs[inds] #sorted locations from largest peak to smallest
        sorted_locs = sorted_locs[np.logical_not(np.isnan(sorted_locs))] #Remove any locations that are NaN.
        
        #6 Select peak
        if peak_n > len(sorted_locs):
            raise AttributeError("Identified peak number (" + str(len(sorted_locs)) + ") is smaller than peak_n (" + str(peak_n) + ").",)
        select_loc = sorted_locs[peak_n]
        
        #6 package to return
        if direction is graphable_base.sweep_enum.FORWARD:
            arrow_dx = xrange * 0.025  # 2.5% of x
        elif direction is graphable_base.sweep_enum.BACKWARD:
            arrow_dx = -xrange * 0.025  # 2.5% of x
        arrow_dy = 0
        arrow_x = xsmooth[select_loc] - arrow_dx/2
        arrow_y = ysmooth[select_loc]
        yoffset = abs(yrange*0.1)
        arrow_y += yoffset if arrow_y > 0 else -yoffset
        return arrow_x, arrow_y, arrow_dx, arrow_dy
    else:
        raise AttributeError("6 datapoints not enough to smooth, will not find arrow location.")