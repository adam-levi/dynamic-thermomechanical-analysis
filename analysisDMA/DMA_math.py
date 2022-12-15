import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

#linear interpolation/extrapolation
def findXPoint(xa,xb,ya,yb,yc):
    m = (xa - xb) / (ya - yb)
    xc = (yc - yb) * m + xb
    return xc

#will determine if there are data points that are outliers based on if they are outside the 25% to 75% percentile range
def detect_outlier(data):
    # find q1 and q3 values
    q1, q3 = np.percentile(sorted(data), [25, 75])
 
    # compute IRQ
    iqr = q3 - q1
 
    # find lower and upper bounds
    lower_bound = q1 - (1.0 * iqr)
    upper_bound = q3 + (1.0 * iqr)
 
    outliers = [x for x in data if x <= lower_bound or x >= upper_bound]
 
    return outliers

#remove outliers from df, after removing the outlier it will check if there are new outliers now that the data has lower variance
def remove_outliers(df, lower_index, upper_index):
    #defining data range in which to look for outliers to remove
    data_range = df['Displacement (µm)'][range(lower_index,upper_index)].values.tolist()
    outliers = detect_outlier(data_range)
    
    if len(outliers) > 0:
        #print(f"There are outliers at these displacement(s):  {outliers} ")
        for i in range(0,len(outliers)):
            df = df.drop(df[df['Displacement (µm)']==outliers[i]].index.values).reset_index()
            df = df.drop('level_0', axis = 1)
        #after removing outliers, check if there are new outliers now that the variance has decreased
        df, outliers = remove_outliers(df, lower_index, upper_index) 
    return df, outliers

#find_max uses find_peaks from scipy_signals to find local maxima. It will gradually decrease the prominence input until a peak is found
def find_max(array, prom):
    max_pts = find_peaks(array, distance = 50, prominence = prom)
    max_arr = np.asarray(max_pts[0])
    if len(max_arr) == 0:
        prom *= 0.99
        a, b = find_max(array, prom)
        return a, b
    else:
        return max_arr, prom
