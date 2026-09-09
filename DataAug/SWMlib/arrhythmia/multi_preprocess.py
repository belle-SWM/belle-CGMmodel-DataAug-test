import numpy as np
from ..ecg.baseline import baseline_remove
from .irregular import irregular_detection


var_dict = {}  ## A global dictionary storing the variables passed from the initializer.


def init_worker(x, x_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = x
    var_dict['X_shape'] = x_shape


def segment_preprocess(i):
    # Simply computes the sum of the i-th row of the input matrix X
    x_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    ecg = x_np[i]
    # Remove Baseline and Pulse
    ecg_filt = baseline_remove(ecg)
    # Check Output Length
    if len(ecg_filt) < 2500:
        output = np.zeros(2500)
        output[:len(ecg_filt)] = ecg_filt
    else:
        output = ecg_filt[:2500]
    return output.tolist()


def ecg_analysis(idx): 

    x_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])   
    sig = x_np[idx]  
    results = irregular_detection(sig,idx)  ## Detect irregularity

    return results
