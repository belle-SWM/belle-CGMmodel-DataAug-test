import numpy as np
from ..ecg.baseline import baseline_remove
from ..ecg.rpeak import rpeak_detection

# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape


def get_Rloc(i):
    # Simply computes the sum of the i-th row of the input matrix X
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    ecg = X_np[i]

    # Remove Baseline and Pulse
    ecg_filt = baseline_remove(ecg)

    # Rescale signal for R peak detection
    scale = 1
    if 300 >= max(ecg_filt) > 150:
        scale = 2
    elif 150 >= max(ecg_filt):
        scale = 4
    else:
        scale = 1
    ecg_filt = ecg_filt * scale

    # R peak detection
    Rpeaks = rpeak_detection(ecg_filt)
    if len(Rpeaks) <= 1:
        return [0]
    Ridxs = Rpeaks[1:] + 2500*i
    
    return Ridxs.tolist()
