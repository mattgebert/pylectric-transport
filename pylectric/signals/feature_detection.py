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
        
        #4 Check if no peak match, then return near max value
        yabs = np.abs(ysmooth) #abs to find max height
        yabs = yabs[np.logical_not(np.isnan(yabs))]
        if len(locs) == 0:
            print("Peak not found, instead using value near maximum and 5% away from endpoint.")
            i = np.where(yabs == np.max(yabs))[0]
            upperbound = 0.9 * xrange + np.min(xdata)
            lowerbound = 0.1 * xrange + np.min(xdata)
            if xsmooth[i] >= upperbound:
                x = 0.9 * xrange + np.min(xdata)
            elif xsmooth[i] <= lowerbound:
                x = 0.1 * xrange + xsmooth[i]
            else:
                x = xsmooth[i]
            i = np.where(np.min(xsmooth - x) == xsmooth-x)[0][0]
            yoffset = abs(0.1*yrange)
            y = ysmooth[i]
            y += (yoffset if y > 0 else -yoffset) #create offset to seperate from data.
            dy = 0
            if direction is graphable_base.sweep_enum.FORWARD:
                dx = xrange * 0.025  # 2.5% of x
            elif direction is graphable_base.sweep_enum.BACKWARD:
                dx = -xrange * 0.025  # 2.5% of x
            return x, y, dx, dy
                
        #5 use gradient point that corresponds to largest peak.
        peaks = yabs[locs] #yabs values at changes in sign
        
        #6 sort peaks from largest to smallest.
        inds = np.argsort(peaks)[::-1] #get indicies from sorting peaks, and reverse to make largest at the front.
        sorted_locs = locs[inds] #sorted locations from largest peak to smallest
        sorted_locs = sorted_locs[np.logical_not(np.isnan(sorted_locs))] #Remove any locations that are NaN.
        
        #7 Select peak
        if peak_n > len(sorted_locs):
            raise AttributeError("Identified peak number (" + str(len(sorted_locs)) + ") is smaller than peak_n (" + str(peak_n) + ").",)
        select_loc = sorted_locs[peak_n]
        
        #8 package to return
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