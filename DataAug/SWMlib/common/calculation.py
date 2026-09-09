import numpy as np

def rrspread_cal(rri_array):
    """
    input ---
        rri_array: 1D array of RR interval
    output ---
        rr_spread_value: rr spread of the input rri 
    """
    maxrri = np.max(rri_array)
    minrri = np.min(rri_array)
    rr_spread_value = (maxrri - minrri) / (maxrri + minrri)
    
    return rr_spread_value

def normalization(x):
   x_max = np.max(x)
   x_min = np.min(x)
   x_norm = (x - x_min) / (x_max - x_min+1)  
   return x_norm