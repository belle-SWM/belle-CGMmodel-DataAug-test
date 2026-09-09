# -*- coding: utf-8 -*-

import os
import numpy as np
import xgboost as xgb
import neurokit2 as nk

import warnings

warnings.filterwarnings("ignore")


def irr_info_export_(ridxs):

    if 150 <= len(ridxs) <= 1000:

        f_ridxs = np.array(ridxs) * 4
        f_rris = np.diff(f_ridxs)

        if np.nanmin(f_rris) > 300 and np.nanmax(f_rris) < 2000:

            f_rris_p = f_rris[:-2]
            f_rris_c = f_rris[1:-1]
            f_rris_n = f_rris[2:]

            irr_ratio = 0

            for rri_p, rri_c, rri_n in zip(f_rris_p, f_rris_c, f_rris_n):

                rri_avg = (rri_p + rri_n) / 2
                irr_ratio = abs(rri_c - rri_avg) / rri_avg

                if irr_ratio > 0.2:
                    irr_ratio += 1

            return irr_ratio/len(f_rris_p)*100, 1

        else:
            pass

    else:
        pass

    return np.nan, np.nan


def moving_average(input_list, N):

    output_list = []
    for i in range(len(input_list)):
        start_index = max(0, i - N)
        end_index = min(len(input_list), i + N + 1)
        relevant_values = [x for x in input_list[start_index:end_index] if not np.isnan(x)]
        if relevant_values:
            avg = np.nanmean(relevant_values)
        else:
            avg = np.nan
        output_list.append(avg)
    return output_list


def sleep_stage_classification_xgboost_(ridxs, gender, age, model_d_p):

    f_ridxs = np.array(ridxs)

    ridxs_30sec = [[] for _ in range(int((f_ridxs[-1] // 7500) + 1))]

    ridxs_30sec_len = len(ridxs_30sec)

    for f_ridx in f_ridxs:
        ridxs_30sec[int(f_ridx // 7500)].append(f_ridx)

    results = {
        'TIME': [],
        'GENDER': [],
        'AGE': [],
        'ARR_RATIO': [],
        'GOOD': [],
        'HRV_MeanNN': [],
        'HRV_SDNN': [],
        'HRV_SDANN1': [],
        'HRV_SDNNI1': [],
        'HRV_SDANN2': [],
        'HRV_SDNNI2': [],
        'HRV_SDANN5': [],
        'HRV_SDNNI5': [],
        'HRV_RMSSD': [],
        'HRV_SDSD': [],
        'HRV_CVNN': [],
        'HRV_CVSD': [],
        'HRV_MedianNN': [],
        'HRV_MadNN': [],
        'HRV_MCVNN': [],
        'HRV_IQRNN': [],
        'HRV_SDRMSSD': [],
        'HRV_Prc20NN': [],
        'HRV_Prc80NN': [],
        'HRV_pNN50': [],
        'HRV_pNN20': [],
        'HRV_MinNN': [],
        'HRV_MaxNN': [],
        'HRV_HTI': [],
        'HRV_TINN': [],
        'HRV_ULF': [],
        'HRV_VLF': [],
        'HRV_LF': [],
        'HRV_HF': [],
        'HRV_VHF': [],
        'HRV_TP': [],
        'HRV_LFHF': [],
        'HRV_LFn': [],
        'HRV_HFn': [],
        'HRV_LnHF': [],
        'HRV_SD1': [],
        'HRV_SD2': [],
        'HRV_SD1SD2': [],
        'HRV_S': [],
        'HRV_CSI': [],
        'HRV_CVI': [],
        'HRV_CSI_Modified': [],
        'HRV_PIP': [],
        'HRV_IALS': [],
        'HRV_PSS': [],
        'HRV_PAS': [],
        'HRV_GI': [],
        'HRV_SI': [],
        'HRV_AI': [],
        'HRV_PI': [],
        'HRV_C1d': [],
        'HRV_C1a': [],
        'HRV_SD1d': [],
        'HRV_SD1a': [],
        'HRV_C2d': [],
        'HRV_C2a': [],
        'HRV_SD2d': [],
        'HRV_SD2a': [],
        'HRV_Cd': [],
        'HRV_Ca': [],
        'HRV_SDNNd': [],
        'HRV_SDNNa': [],
        'HRV_DFA_alpha1': [],
        'HRV_MFDFA_alpha1_Width': [],
        'HRV_MFDFA_alpha1_Peak': [],
        'HRV_MFDFA_alpha1_Mean': [],
        'HRV_MFDFA_alpha1_Max': [],
        'HRV_MFDFA_alpha1_Delta': [],
        'HRV_MFDFA_alpha1_Asymmetry': [],
        'HRV_MFDFA_alpha1_Fluctuation': [],
        'HRV_MFDFA_alpha1_Increment': [],
        'HRV_DFA_alpha2': [],
        'HRV_MFDFA_alpha2_Width': [],
        'HRV_MFDFA_alpha2_Peak': [],
        'HRV_MFDFA_alpha2_Mean': [],
        'HRV_MFDFA_alpha2_Max': [],
        'HRV_MFDFA_alpha2_Delta': [],
        'HRV_MFDFA_alpha2_Asymmetry': [],
        'HRV_MFDFA_alpha2_Fluctuation': [],
        'HRV_MFDFA_alpha2_Increment': [],
        'HRV_ApEn': [],
        'HRV_SampEn': [],
        'HRV_ShanEn': [],
        'HRV_FuzzyEn': [],
        'HRV_MSEn': [],
        'HRV_CMSEn': [],
        'HRV_RCMSEn': [],
        'HRV_CD': [],
        'HRV_HFD': [],
        'HRV_KFD': [],
        'HRV_LZC': []
    }

    for j in range(ridxs_30sec_len):

        stage_index_s = j - 5 if j > 5 else 0
        stage_index_e = j + 5 if j < ridxs_30sec_len - 5 else ridxs_30sec_len

        ridxs = [ridx - stage_index_s*7500 for ridxs_i in ridxs_30sec[stage_index_s:stage_index_e] for ridx in ridxs_i]

        f_arr_ratio, f_good = irr_info_export_(ridxs)

        result = {
            'TIME': j,
            'GENDER': 0 if gender == 'F' else 1,
            'AGE': int(age),
            'ARR_RATIO': f_arr_ratio,
            'GOOD': f_good,
            'HRV_MeanNN': np.nan,
            'HRV_SDNN': np.nan,
            'HRV_SDANN1': np.nan,
            'HRV_SDNNI1': np.nan,
            'HRV_SDANN2': np.nan,
            'HRV_SDNNI2': np.nan,
            'HRV_SDANN5': np.nan,
            'HRV_SDNNI5': np.nan,
            'HRV_RMSSD': np.nan,
            'HRV_SDSD': np.nan,
            'HRV_CVNN': np.nan,
            'HRV_CVSD': np.nan,
            'HRV_MedianNN': np.nan,
            'HRV_MadNN': np.nan,
            'HRV_MCVNN': np.nan,
            'HRV_IQRNN': np.nan,
            'HRV_SDRMSSD': np.nan,
            'HRV_Prc20NN': np.nan,
            'HRV_Prc80NN': np.nan,
            'HRV_pNN50': np.nan,
            'HRV_pNN20': np.nan,
            'HRV_MinNN': np.nan,
            'HRV_MaxNN': np.nan,
            'HRV_HTI': np.nan,
            'HRV_TINN': np.nan,
            'HRV_ULF': np.nan,
            'HRV_VLF': np.nan,
            'HRV_LF': np.nan,
            'HRV_HF': np.nan,
            'HRV_VHF': np.nan,
            'HRV_TP': np.nan,
            'HRV_LFHF': np.nan,
            'HRV_LFn': np.nan,
            'HRV_HFn': np.nan,
            'HRV_LnHF': np.nan,
            'HRV_SD1': np.nan,
            'HRV_SD2': np.nan,
            'HRV_SD1SD2': np.nan,
            'HRV_S': np.nan,
            'HRV_CSI': np.nan,
            'HRV_CVI': np.nan,
            'HRV_CSI_Modified': np.nan,
            'HRV_PIP': np.nan,
            'HRV_IALS': np.nan,
            'HRV_PSS': np.nan,
            'HRV_PAS': np.nan,
            'HRV_GI': np.nan,
            'HRV_SI': np.nan,
            'HRV_AI': np.nan,
            'HRV_PI': np.nan,
            'HRV_C1d': np.nan,
            'HRV_C1a': np.nan,
            'HRV_SD1d': np.nan,
            'HRV_SD1a': np.nan,
            'HRV_C2d': np.nan,
            'HRV_C2a': np.nan,
            'HRV_SD2d': np.nan,
            'HRV_SD2a': np.nan,
            'HRV_Cd': np.nan,
            'HRV_Ca': np.nan,
            'HRV_SDNNd': np.nan,
            'HRV_SDNNa': np.nan,
            'HRV_DFA_alpha1': np.nan,
            'HRV_MFDFA_alpha1_Width': np.nan,
            'HRV_MFDFA_alpha1_Peak': np.nan,
            'HRV_MFDFA_alpha1_Mean': np.nan,
            'HRV_MFDFA_alpha1_Max': np.nan,
            'HRV_MFDFA_alpha1_Delta': np.nan,
            'HRV_MFDFA_alpha1_Asymmetry': np.nan,
            'HRV_MFDFA_alpha1_Fluctuation': np.nan,
            'HRV_MFDFA_alpha1_Increment': np.nan,
            'HRV_DFA_alpha2': np.nan,
            'HRV_MFDFA_alpha2_Width': np.nan,
            'HRV_MFDFA_alpha2_Peak': np.nan,
            'HRV_MFDFA_alpha2_Mean': np.nan,
            'HRV_MFDFA_alpha2_Max': np.nan,
            'HRV_MFDFA_alpha2_Delta': np.nan,
            'HRV_MFDFA_alpha2_Asymmetry': np.nan,
            'HRV_MFDFA_alpha2_Fluctuation': np.nan,
            'HRV_MFDFA_alpha2_Increment': np.nan,
            'HRV_ApEn': np.nan,
            'HRV_SampEn': np.nan,
            'HRV_ShanEn': np.nan,
            'HRV_FuzzyEn': np.nan,
            'HRV_MSEn': np.nan,
            'HRV_CMSEn': np.nan,
            'HRV_RCMSEn': np.nan,
            'HRV_CD': np.nan,
            'HRV_HFD': np.nan,
            'HRV_KFD': np.nan,
            'HRV_LZC': np.nan
        }

        try:

            if np.isnan(f_good):
                pass

            else:
                result_i = nk.hrv(ridxs, sampling_rate=250, show=False)

                for key in result_i.keys():
                    result[key] = result_i[key][0]

            for key in result.keys():
                results[key].append(result[key])

        except:

            pass

    new_feature_dict = {}

    new_feature_dict['TIME'] = results['TIME'] ##.tolist()
    ###new_feature_dict['STAGE'] = results['STAGE'] ##.tolist()
    new_feature_dict['GENDER'] = results['GENDER'] ##.tolist()
    new_feature_dict['AGE'] = results['AGE'] ##.tolist()

    ##sorted_list_without_nan = sorted([x for x in results['TIME'].tolist() if not np.isnan(x)])
    sorted_list_without_nan = sorted([x for x in results['TIME'] if not np.isnan(x)])
    percentiles = []

    n = len(sorted_list_without_nan)

    if n == 0:
        #new_feature_dict['TIME_PERCENTILE'] = ([np.nan] * len(results['TIME'].tolist())).tolist()
        new_feature_dict['TIME_PERCENTILE'] = ([np.nan] * len(results['TIME'])).tolist()
    else:

        ##for num in results['TIME'].tolist():
        for num in results['TIME']:

            if np.isnan(num):
                percentiles.append(np.nan)

            else:
                percentile = (sorted_list_without_nan.index(num) / (n - 1)) * 100
                percentiles.append(percentile)

        new_feature_dict['TIME_PERCENTILE'] = percentiles

    ##new_feature_dict['ARR_RATIO'] = results['ARR_RATIO'].tolist()
    ##new_feature_dict['GOOD'] = results['GOOD'].tolist()
    new_feature_dict['ARR_RATIO'] = results['ARR_RATIO']
    new_feature_dict['GOOD'] = results['GOOD']

    for key in results.keys():

        if 'HRV' in key:

            now_hrv_list = results[key]

            ##new_feature_dict[key] = now_hrv_list.tolist()
            new_feature_dict[key] = now_hrv_list

            sorted_list_without_nan = sorted([x for x in now_hrv_list if not np.isnan(x)])

            percentiles = []

            n = len(sorted_list_without_nan)

            if n == 0:
                new_feature_dict[key + '_PERCENTILE'] = [np.nan] * len(now_hrv_list)

            else:

                for num in now_hrv_list:

                    if np.isnan(num):
                        percentiles.append(np.nan)

                    else:
                        percentile = (sorted_list_without_nan.index(num) / (n - 1)) * 100
                        percentiles.append(percentile)

                new_feature_dict[key + '_PERCENTILE'] = percentiles

    dict_key_list = list(new_feature_dict.keys())

    for key in dict_key_list:

        if 'HRV' in key:
            now_hrv_list = new_feature_dict[key]

            new_hrv_list_1 = moving_average(now_hrv_list, 1)
            new_hrv_list_2 = moving_average(now_hrv_list, 2)

            new_feature_dict[key + '_MEAN_1'] = new_hrv_list_1
            new_feature_dict[key + '_MEAN_2'] = new_hrv_list_2

    new_feature_dict = new_feature_dict.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
    new_feature_dict = new_feature_dict.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    new_feature_dict = new_feature_dict[new_feature_dict['ARR_RATIO'] <= 0.2]

    float64_cols = new_feature_dict.select_dtypes(include=['float64']).columns
    new_feature_dict[float64_cols] = new_feature_dict[float64_cols].astype('float32')

    X_data = new_feature_dict.iloc[:, 3:]
    y_data = new_feature_dict.iloc[:, 2]

    dtest = xgb.DMatrix(X_data, label=y_data)

    if gender == 0:

        if age < 62:

            model_f_p = os.path.join(model_d_p, 'model_0.json')

            model = xgb.Booster()
            model.load_model(model_f_p)

        else:

            model_f_p = os.path.join(model_d_p, 'model_1.json')

            model = xgb.Booster()
            model.load_model(model_f_p)

    else:

        if age < 62:

            model_f_p = os.path.join(model_d_p, 'model_2.json')

            model = xgb.Booster()
            model.load_model(model_f_p)

        else:

            model_f_p = os.path.join(model_d_p, 'model_3.json')

            model = xgb.Booster()
            model.load_model(model_f_p)

    y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))  ####stage 預測結果

    return y_pred


if __name__ == "__main__":

    ridxs = []
    gender = 0
    age = 0
    model_d_p = r'./temp'

    stages = sleep_stage_classification_xgboost_(ridxs, gender, age, model_d_p)
