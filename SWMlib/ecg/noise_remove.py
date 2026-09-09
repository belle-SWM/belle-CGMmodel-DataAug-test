import numpy as np
from scipy.ndimage import median_filter

def remove_impulse(ecgs):
    ecgs_diff = np.diff(ecgs)
    j = 0
    for i, (ecg, ecg_diff) in enumerate(zip(ecgs[:-1], ecgs_diff)):
        if j == 0:
            if ecg_diff < -500:
                j = 1
                try:
                    while ecgs_diff[i + j] == 0:
                        j += 1
                except:
                    j -= 1

                if ecgs_diff[i + j] > 500:
                    for k in range(i + 1, i + j + 1): ecgs[k] = (ecgs[i] + ecgs[i + j + 1]) / 2

            elif ecg_diff > 700:
                j = 1
                try:
                    while ecgs_diff[i + j] == 0:
                        j += 1

                except:
                    j -= 1

                if ecgs_diff[i + j] < -700:
                    for k in range(i + 1, i + j + 1): ecgs[k] = (ecgs[i] + ecgs[i + j + 1]) / 2
        else:
            j = j - 1

    return ecgs
