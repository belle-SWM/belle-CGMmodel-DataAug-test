import numpy as np
import math

def mean_filter(signals, window_size):
    
    """
    input ---
        signals: 1D array of signal
        width: the half size of the window of mean filter

    output ---
        new_signals: mean filtered signal
    """
    
    new_signals = []
    width=int(math.floor(window_size/2))
    length=len(signals)
    
    for i in range(length):

        i_start = i - width
        i_end = i + width + 1

        if i_start < 0:
            i_start = 0

        if i_end > length:
            i_end = length

        signal_i = np.nanmean(signals[i_start:i_end])

        new_signals.append(signal_i)

    return new_signals


def rri_filter(rris,lower_thr=245,upper_thr=2100,medianflag=True):
        """
        input ---
            rris: 1-D list containing rri values in the unit of millseconds
            lower_thr: filtered out the rri value which is smaller than lower_thr(with default value=245)
            upper_thr: filtered out the rri value which is greater than upper_thr(with default value=2100)
            medianflag: if set this flag=True, the rri value will be filtered out if it is out of the range defined by the median of elements in th input array

        output ---
            filtered_rri_array: the filtered rri array
        """
        filtered_rri_array=[]
        final_filtered_rri_array=[]
        
        for i in range(len(rris)):  
            if((rris[i]>=lower_thr) and (rris[i]<=upper_thr)):            
                filtered_rri_array.append(rris[i])
        
        if(medianflag): ##另外加上中值濾波   
            if(len(filtered_rri_array)>0):
                rri_median=np.median(filtered_rri_array)
                for i in range(len(filtered_rri_array)):
                    if((filtered_rri_array[i]>=(rri_median-0.55*rri_median)) and (filtered_rri_array[i]<=rri_median+0.62*rri_median)):
                        final_filtered_rri_array.append(filtered_rri_array[i])
            
            return final_filtered_rri_array
        else:
            return filtered_rri_array    