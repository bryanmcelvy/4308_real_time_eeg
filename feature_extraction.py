import numpy as np
from scipy import signal
from pandas import DataFrame, Series

''' Feature Extraction Functions '''

def findArea(x):
    ''' This function finds the area under the wave for a given time series `x` of window size `W`'''
    area = 0
    
    x = np.array(x)
    W = len(x)
    
    for i in np.arange(W, dtype=int):
        area += abs(x[i])
    
    area /= W
    
    return area

def indicatorFunc(x0,x1):
    ''' This function represents a Boolean indicator function, returning 1 when the condition is TRUE and 0 when FALSE '''
    if x1-x0 < 0: return 1
    else: return 0

def findNormDecay(x):
    '''This function finds the chance corrected fraction of data has 
    a (+) or (-) derivative for a time series `x` with window size `W`
    '''
    D = 0 # normalized decay
    
    x = np.array(x)
    W = len(x)
    
    for i in np.linspace(start=1, stop=W-2, num=W-2, dtype=int):
        D += indicatorFunc(x[i], x[i+1])
    
    D = D / (W-1)
    D = abs(D - 0.5)
    return D

def findLineLength(x):
    ''' This function sums the distance between all consecutive readings within a time series `x` of window size `W` '''
    l = 0 # line length
    x = np.array(x)
    W = len(x)
    
    for i in np.arange(W-1, dtype=int):
        l += abs(x[i]-x[i-1])
    return l

def findMeanEnergy(x):
    ''' This function finds the mean energy of a time series `x` with window size `W` '''
    E = 0 # mean energy
    
    x = np.array(x)
    W = len(x)
    
    if W > 0:
        for i in np.linspace(start=0, stop=W-1, num=W, dtype=int):
            E += abs(x[i]**2)
        
        E /= W
        return E
    else: return 0

def findAvgPeakAmp(x):
    ''' This function finds the avg. amplitude of the `K` peaks present in time series `x` '''
    P_A = 0
    
    x = np.array(x)
    peaks, _ = signal.find_peaks(x=x, prominence=10, distance=1)
    K = len(peaks)
    
    if K > 0:
        for k in np.linspace(start=0, stop=K-1, num=K, dtype=int):
            P_A += x[peaks[k]]**2 # mean squared amplitude of K peaks
            
        P_A /= K
        if P_A == 0: return P_A
        else: return np.log10(P_A)
    else: return 0 # returns 0 if there are no peaks (K=0)

def findAvgValleyAmp(x):
    ''' This function finds the avg. amplitude of the `V` valleys present in time series `x` '''
    V_A = 0
    
    x = np.array(x)
    valleys, _ = signal.find_peaks(x=-x, prominence=10, distance=1)
    V = len(valleys)
    
    if V > 0:
        for v in np.linspace(start=0, stop=V-1, num=V, dtype=int):
            V_A += x[valleys[v]]**2
        
        V_A /= V
        if V_A == 0: return V_A
        else: return np.log10(V_A)
    else: return 0 # returns 0 if there are no valleys (V=0)

def findNormPeakNum(x):
    ''' This function finds the number of peaks normalized by the 
    avg. difference between readings within time series `x` with window size `W` '''
    N_P = 0 # peak num
    
    x = np.array(x)
    peaks, _ = signal.find_peaks(x=x, prominence=10, distance=1)
    W = len(x)
    K = len(peaks)
    if K > 0:
        for i in np.linspace(start=0, stop=W-2, num=W-1, dtype=int):
            N_P += abs(x[i+1]-x[i])
        
        N_P /= (W-1)
        N_P = K / N_P
        return N_P
    else: return 0 # returns 0 if there are no peaks (K=0)

def findPeakVariation(x):
    ''' This function finds the variation between peaks and valleys across time and electrical signal 
    for a time series `x`. The feature with the smallest length is used for comparisons.'''
    P_V = 0
    
    x = np.array(x)
    peaks, _ = signal.find_peaks(x=x, prominence=10, distance=1) # idx of peaks in x
    valleys, _ = signal.find_peaks(x=-x, prominence=10, distance=1) # idx of valleys in x
    K = len(peaks)
    V = len(valleys)
    
    if (K <= 1) or (V <= 1): return 0 # prevents divide-by-zero error
    else:
        n = K if K <= V else V # iteration variable; uses smallest feature if feature lengths are unequal
        
        # Find mean and st. dev. of the indicies (sample numbers)
        u_PV = 0 # idx mean
        for i in np.linspace(start=0, stop=n-1, num=n, dtype=int):
            u_PV += peaks[i] - valleys[i]
        u_PV /= n
        
        s_PV = 0 # idx std
        for i in np.linspace(start=0, stop=n-1, num=n, dtype=int):
            s_PV += (peaks[i] - valleys[i] - u_PV)**2
        s_PV = np.sqrt( s_PV / (n-1) )
        
        # Find mean and st. dev. of the readings (voltage values)
        u_xPV = 0 # voltage mean
        for i in np.linspace(start=0, stop=n-1, num=n, dtype=int):
            u_xPV += x[peaks[i]] - x[peaks[i]]
        u_xPV /= n
        
        s_xPV = 0 # voltage std
        for i in np.linspace(start=0, stop=n-1, num=n, dtype = int):
            s_xPV += (x[peaks[i]] - x[valleys[i]] - u_xPV)**2
        s_xPV = np.sqrt( s_xPV / (n-1) )
            
        # Calculate peak variation
        P_V = 1 / (s_PV * s_xPV) if (s_PV != 0 and s_xPV != 0) else 0
        
        return P_V

def findRootMeanSquare(x):
    ''' This function finds the sqrt of the mean of the squared data points from time series `x` with window size `W` '''
    rms = 0
    
    x = np.array(x)
    W = len(x)
    
    if W > 0:
        for i in range(W):
            rms += x[i]**2
            
        rms = np.sqrt(rms/W)
        return rms
    else: return 0 # returns 0 if there are no peaks (K=0)
    
''' Perform all functions '''
def extractFeatures(x):
    ''' This function returns the feature vector for the given time series data '''
    feature_vector = np.array([0 for _ in range(9)])
    feature_vector[0] = findArea(x)
    feature_vector[1] = findNormDecay(x)
    feature_vector[2] = findLineLength(x)
    feature_vector[3] = findMeanEnergy(x)
    feature_vector[4] = findAvgPeakAmp(x)
    feature_vector[5] = findAvgValleyAmp(x)
    feature_vector[6] = findNormPeakNum(x)
    feature_vector[7] = findPeakVariation(x)
    feature_vector[8] = findRootMeanSquare(x)
    return feature_vector