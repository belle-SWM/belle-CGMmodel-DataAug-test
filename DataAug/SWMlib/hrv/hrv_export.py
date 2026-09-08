# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:24:12 2023

@author: SWM-Jared
"""

import numpy as np
import hrvanalysis as hrv
from scipy.stats import skew
from scipy.stats import kurtosis

def hrv_feature_export_(rris):
    """
    input ---
        rris: 1-D numpy array contains rri values in the unit of millseconds

    output ---
        hrv_features_duration = {
        'vlf': vlf,
        'lf': lf,
        'hf': hf,
        'vlf_max_freq': vlf_freq,
        'lf_max_freq': lf_freq,
        'hf_max_freq': hf_freq,
        'vlf_max_power': vlf_max,
        'lf_max_power': lf_max,
        'hf_max_power': hf_max,
        'vlf_kurtosis': vlf_kurtosis,
        'lf_kurtosis': lf_kurtosis,
        'hf_kurtosis': hf_kurtosis,
        'vlf_skew': vlf_skew,
        'lf_skew': lf_skew,
        'hf_skew': hf_skew,
        'lf_hf_ratio': lf_hf_ratio,
        'lfnu': lfnu,
        'hfnu': hfnu,
        'total_power': total_power,
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_nni': median_nni,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr
    } 

    """
        
    rr_intervals_without_outliers = hrv.remove_outliers(rr_intervals=rris, low_rri=300, high_rri=2000, verbose=False)
    interpolated_rr_intervals = hrv.interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
    nn_intervals_list = hrv.remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="karlsson", verbose=False)
    interpolated_nn_intervals = hrv.interpolate_nan_values(rr_intervals=nn_intervals_list)
    
    nnis = [x for x in interpolated_nn_intervals if str(x) != 'nan']
    
    freq, psd = hrv.extract_features._get_freq_psd_from_nn_intervals(nn_intervals=nnis, method="welch", sampling_frequency=4, interpolation_method="linear", vlf_band=[0.003, 0.04], hf_band=[0.15, 0.40])
    
    vlf_indexes = np.logical_and(freq >= 0.003, freq < 0.04)
    lf_indexes = np.logical_and(freq >= 0.04, freq < 0.15)
    hf_indexes = np.logical_and(freq >= 0.15, freq < 0.40)
    
    vlf_kurtosis = kurtosis(psd[vlf_indexes])
    vlf_skew = skew(psd[vlf_indexes])
    
    lf_kurtosis = kurtosis(psd[lf_indexes])
    lf_skew = skew(psd[lf_indexes])
    
    hf_kurtosis = kurtosis(psd[hf_indexes])
    hf_skew = skew(psd[hf_indexes])
    
    freq, psd = hrv.extract_features._get_freq_psd_from_nn_intervals(nn_intervals=nnis, method="lomb", sampling_frequency=4, interpolation_method="linear", vlf_band=[0.003, 0.04], hf_band=[0.15, 0.40])
    
    vlf_indexes = np.logical_and(freq >= 0.003, freq < 0.04)
    lf_indexes = np.logical_and(freq >= 0.04, freq < 0.15)
    hf_indexes = np.logical_and(freq >= 0.15, freq < 0.40)
    
    vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
    hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])
    
    vlf_max = np.nanmax(psd[vlf_indexes])
    lf_max = np.nanmax(psd[lf_indexes])
    hf_max = np.nanmax(psd[hf_indexes])
    
    vlf_freq = freq[psd.tolist().index(vlf_max)]
    lf_freq = freq[psd.tolist().index(lf_max)]
    hf_freq = freq[psd.tolist().index(hf_max)]
    
    total_power = vlf + lf + hf

    lf_hf_ratio = lf / hf
    lfnu = (lf / (lf + hf)) * 100
    hfnu = (hf / (lf + hf)) * 100
    
    hrv_features_duration = {
        'vlf': vlf,
        'lf': lf,
        'hf': hf,
        'vlf_max_freq': vlf_freq,
        'lf_max_freq': lf_freq,
        'hf_max_freq': hf_freq,
        'vlf_max_power': vlf_max,
        'lf_max_power': lf_max,
        'hf_max_power': hf_max,
        'vlf_kurtosis': vlf_kurtosis,
        'lf_kurtosis': lf_kurtosis,
        'hf_kurtosis': hf_kurtosis,
        'vlf_skew': vlf_skew,
        'lf_skew': lf_skew,
        'hf_skew': hf_skew,
        'lf_hf_ratio': lf_hf_ratio,
        'lfnu': lfnu,
        'hfnu': hfnu,
        'total_power': total_power
    }
    
    ### 時間
    
    nnis_time = []
    
    rris_no_out_time = [ int(rri) for rri in rris if rri > 300 and rri < 2000 ]
    
    if len(rris_no_out_time) > 20:
        
        for i in range( 1, len(rris_no_out_time) ):
            
            rr_spread = abs((rris_no_out_time[i-1]-rris_no_out_time[i])/rris_no_out_time[i-1])
            
            if rr_spread < 0.2:
                nnis_time.append(rris_no_out_time[i])
        
    else :
        return {}
    
    diff_nni = np.diff(nnis)
    length_int = len(nnis) - 1 if nnis else len(nnis)

    # Basic statistics
    mean_nni = np.mean(nnis)
    median_nni = np.median(nnis)
    range_nni = max(nnis) - min(nnis)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    cvsd = rmssd / mean_nni

    sdnn = np.std(nnis, ddof=1)  
    cvnni = sdnn / mean_nni

    heart_rate_list = np.divide(60000, nnis)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    diff_nni = np.diff(nnis_time)
    length_int = len(nnis_time) - 1 if nnis else len(nnis_time)

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int

    hrv_time_domain_features = {
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_nni': median_nni,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }

    hrv_features_duration.update(hrv_time_domain_features)
    
    return hrv_features_duration

if __name__ == '__main__':

    ridxs_list = []

    hrvs_list = []

    for ridxs in ridxs_list:

        if len(ridxs) > 1:

            rris = np.diff(ridxs) * 4

            try :
                hrv_feature_dict = hrv_feature_export_(rris)

                if hrv_feature_dict:
                    hrv_feature_array = np.array(list(hrv_feature_dict.values()))
                
                else :
                    hrv_feature_array = np.empty(35)
                    hrv_feature_array[:] = np.nan
            
            except :
                hrv_feature_array = np.empty(35)
                hrv_feature_array[:] = np.nan

        else : 
            hrv_feature_array = np.empty(35)
            hrv_feature_array[:] = np.nan

        hrvs_list.append(hrv_feature_array)

    hrvs_list = np.array(hrvs_list)