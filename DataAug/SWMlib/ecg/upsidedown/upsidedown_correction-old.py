"""
Online voting system including Algorithm and Pytorch model for the ECG Upside-Down detection.

Project: Upside-down ECG signal detection
Create time: Feb. 22, 2023
Author: Benjamin, Liu

"""

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from .parameters_setting import args
from .. import rpeak, noise_remove, quality_check
from ..baseline import baseline_remove


def _read_basedata(dir_ecg_txt):
    # Read ECG signal (.txt')
    with open(dir_ecg_txt) as f:
        basedata = f.readlines()
        f.close()
    basedata_list = np.zeros(len(basedata))
    for index, line in enumerate(basedata):
        basedata_list[index] = float(line[:-1])
    return basedata_list


def _find_outlier_peak(in_signal, std_range=0.9, sample_rate=250, plot_fig=False, fig_info=[]):
    mean = np.nanmean(in_signal)
    std = np.nanstd(in_signal)
    pos_peak_list_ = []
    neg_peak_list_ = []

    for i, point in enumerate(in_signal):
        pos_outlier = point > mean + std_range * std  # 正離群值
        neg_outlier = point < mean - std_range * std  # 負離群值

        # 正離群值
        if pos_outlier:
            try:
                is_pos_peak = (in_signal[i] - in_signal[i - 1]) >= 0 and (in_signal[i + 1] - in_signal[i]) <= 0
            except IndexError:
                is_pos_peak = False
            if is_pos_peak:
                pos_peak_list_.append(i)

        # 負離群值
        if neg_outlier:
            try:
                is_neg_peak = (in_signal[i] - in_signal[i - 1]) <= 0 and (in_signal[i + 1] - in_signal[i]) >= 0
            except IndexError:
                is_neg_peak = False
            if is_neg_peak:
                neg_peak_list_.append(i)

    """
    T peak 發生在 R peak 之後約 55~75 pt

    我們希望 + 離群值可以抓到 R 與 T
           - 離群值頂多抓到 S
    這樣離群值數量 + 大於 -  就可以判斷波形沒有顛倒  

    校正: 避免雜訊干擾 讓離群值峰值數量激增 解決方法是設立一範圍只保留此範圍的最大峰值
    """
    # 取樣頻率造成的點數範圍修正
    fs_ratio = sample_rate / 250

    # 正峰值位置校正
    pos_peak_list = pos_peak_list_.copy()
    finished = 0
    temp_pos = 0  # 暫存上次檢查到的位置
    while not finished and len(pos_peak_list) > 1:
        for p in range(temp_pos, len(pos_peak_list) - 1):
            val1 = in_signal[pos_peak_list[p]]
            val2 = in_signal[pos_peak_list[p + 1]]
            temp_pos = p
            if pos_peak_list[p + 1] - pos_peak_list[p] < int(40 * fs_ratio):
                # 保留數值較大者
                if val1 < val2:
                    pos_peak_list.remove(pos_peak_list[p])
                    break
                else:
                    pos_peak_list.remove(pos_peak_list[p + 1])
                    break
        if temp_pos >= len(pos_peak_list) - 2:
            finished = 1

    # 負峰值位置校正
    neg_peak_list = neg_peak_list_.copy()
    finished = 0
    temp_neg = 0  # 暫存上次檢查到的位置
    while not finished and len(neg_peak_list) > 1:
        for p in range(temp_neg, len(neg_peak_list) - 1):
            val1 = in_signal[neg_peak_list[p]]
            val2 = in_signal[neg_peak_list[p + 1]]
            temp_neg = p
            if neg_peak_list[p + 1] - neg_peak_list[p] < int(40 * fs_ratio):
                # 保留數值較小者
                if val1 < val2:
                    neg_peak_list.remove(neg_peak_list[p + 1])
                    break
                else:
                    neg_peak_list.remove(neg_peak_list[p])
                    break
        if temp_neg >= len(neg_peak_list) - 2:
            finished = 1

    in_signal = np.array(in_signal)

    # 畫圖
    if plot_fig:
        plt.figure()
        plt.title(f'{fig_info[0]}'
                  f'\nPositive peak:{len(pos_peak_list)}, Negative peak:{len(neg_peak_list)}'
                  f'\nSignal area: {fig_info[1]}')
        plt.plot((0, len(in_signal)), (mean + std_range * std, mean + std_range * std), color='lightgray')  # 離群值線
        plt.plot((0, len(in_signal)), (mean - std_range * std, mean - std_range * std), color='lightgray')  # 離群值線
        plt.plot((0, len(in_signal)), (0, 0), color='gray')  # 基線
        plt.plot(in_signal)
        plt.scatter(neg_peak_list, in_signal[neg_peak_list], color='darkorange', s=11)
        plt.scatter(pos_peak_list, in_signal[pos_peak_list], color='red', s=11)
        plt.show()

    return len(pos_peak_list), len(neg_peak_list)


def _do_check_outlier(check_signal, file_name='UserName', slice_index=0):
    """   門檻調整 容易判成顛倒  """

    # 圖表名稱
    ecg_slice_name = file_name + f'_{slice_index}'

    """
    檢查方法一: 訊號積分  面積小於或等於0  >>  判為顛倒
    """
    # 訊號積分
    area = np.trapz(check_signal)

    """
    檢查方法二: 訊號離群值數值為負的峰值個數  若大於 正的峰值個數 >> 判斷為顛倒
    """
    num_pos, num_neg = _find_outlier_peak(in_signal=check_signal,
                                         plot_fig=False,
                                         fig_info=[ecg_slice_name, area])

    outlier_sum = num_pos - num_neg

    #  檢查方法二: 訊號離群值數值為負的峰值個數  若大於 正的峰值個數 >> 判斷為顛倒(return 1)
    if num_neg > num_pos:
        return 1, outlier_sum, area

    # 檢查方法一: 訊號積分 面積小於或等於0  >>  判為顛倒(return 1)
    elif area <= 0:
        return 1, outlier_sum, area

    # 其餘判為訊號正常(return 0)
    else:
        return 0, outlier_sum, area


def _scale_norm(old_signal):
    if max(old_signal) == min(old_signal):
        return old_signal
    else:
        return (old_signal - min(old_signal)) / (max(old_signal) - min(old_signal))


def _load_checkpoint(model_date, model_name):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CheckPoint', model_date, model_name)
    
    if not os.path.isfile(filepath):  # 檢查檔案是否存在
        return None
    if args.Using_GPU:
        return torch.load(filepath)
    else:
        return torch.load(filepath, map_location=torch.device('cpu'))


def _cnn_testing_loop(check_signal, cnn_model, file_name='UserName', slice_index=0):
    # switch to evaluate mode
    cnn_model.eval()

    # 震幅尺度標準化 壓縮至0~1
    ecg_10s = _scale_norm(check_signal)

    # Normalization
    ecg_norm = (ecg_10s - args.CNN_norm_mean) / args.CNN_norm_std
    ecg_norm = ecg_norm.astype('float32')

    # Convert to tensor
    ecg_norm = ecg_norm.reshape(1, 1, -1)
    ecg_tensor = torch.from_numpy(ecg_norm)

    # Switch to GPU
    if args.Using_GPU:
        ecg_tensor = ecg_tensor.cuda()

    # Compute gradient
    ecg_tensor = torch.autograd.Variable(ecg_tensor, requires_grad=False)

    # compute output
    with torch.no_grad():
        output = cnn_model(ecg_tensor)
    output_prob = np.round(float(output.cpu().detach().numpy()[0]), 2)  # probility
    output_class = int(np.round(output.cpu().detach().numpy()[0], 0))  # class

    # print(f'{file_name}_{slice_index}- Class:{output_class}, Prob:{output_prob}')
    return output_class


def load_cnn():
    model = torch.nn.DataParallel(args.ModelFrame())

    # 使用 GPU
    if args.Using_GPU:
        model.cuda()
        cudnn.benchmark = True

    # 載入模型
    state = _load_checkpoint(model_date=args.Load_ModelDate, model_name=args.Load_ModelName + '.pth')
    if state:
        
        try:
            model.module.load_state_dict(state)
        except:
            model.load_state_dict(state)
    else:
        raise FileNotFoundError('Could not read checkpoint!')
    return model


def checking_ecg_upsidedown_v1(input_ecg, cnn_model, evaluate_quality=False):
    """
    :param input_ecg: 單個 srj檔的 ECG訊號
    :param evaluate_quality: 是否要開啟評估訊號品質機制
    :return: usd_flag(是否訊號顛倒的 flag)

    """
    # initial condition
    slice_count = 0  # 擷取訊號片段記數
    vote_result_arr = np.full(20, np.nan)
    slice_index_arr = np.full(20, np.nan)

    # 每個srj檔案前1~5分鐘,抽取20個訊號片段來判斷
    for slice_idx in range(17500, len(input_ecg), 2500):
        if slice_idx > 77500:  # 上限5分鐘
            break
        sig = input_ecg[slice_idx - 2500:slice_idx]
        sig = noise_remove.BaselineRemove(sig)
        if evaluate_quality:
            score_a = 0
            score_p = 0
            r_list = Rpeak.RPeakDetection(sig)
            score_a = quality_check.AreaRatio(sig, r_list)
            score_p = quality_check.PatternClustering(sig, r_list)

            if score_a * score_p > 98:
                if slice_count < 20:
                    # 故意顛倒訊號 (開發階段測試用)
                    # sig = sig * -1

                    # Algorithm checking Upside-Down
                    flag_algorithm, outlier_sum, area = _do_check_outlier(check_signal=sig, slice_index=slice_idx)

                    # CNN checking Upside-Down
                    if (-3 <= outlier_sum <= 3) or (-1000 < area < 1000):  # Algorithm low confidence conditions
                        flag_cnn = _cnn_testing_loop(check_signal=sig, cnn_model=cnn_model, slice_index=slice_idx)

                        # Write in vote array
                        vote_result_arr[slice_count] = flag_cnn
                        slice_index_arr[slice_count] = slice_idx
                    else:
                        # Write in vote array
                        vote_result_arr[slice_count] = flag_algorithm
                        slice_index_arr[slice_count] = slice_idx

                    # Each srj file cut 20 slice
                    slice_count += 1
                    if slice_count == 20:
                        break
        elif not evaluate_quality:
            if slice_count < 20:

                # Algorithm checking Upside-Down
                flag_algorithm, outlier_sum, area = _do_check_outlier(check_signal=sig, slice_index=slice_idx)

                # CNN checking Upside-Down
                if (-3 <= outlier_sum <= 3) or (-1000 < area < 1000):  # Algorithm low confidence conditions
                    flag_cnn = _cnn_testing_loop(check_signal=sig, cnn_model=cnn_model, slice_index=slice_idx)

                    # Write in vote array
                    vote_result_arr[slice_count] = flag_cnn
                    slice_index_arr[slice_count] = slice_idx
                else:
                    # Write in vote array
                    vote_result_arr[slice_count] = flag_algorithm
                    slice_index_arr[slice_count] = slice_idx

                # Each srj file cut 20 slice
                slice_count += 1
                if slice_count == 20:
                    break

    # Vote whether ECG is upside-down (usd_flag = -1:初始值, 0:訊號正常 ,1:訊號顛倒)
    try:
        usd_flag = int(round(np.nanmean(vote_result_arr), 0))
    except ValueError:
        usd_flag = -1

    return usd_flag


def _checking_ecg_upsidedown_v2(srj_ecg, evaluate_quality=False):
    
    """
    input --- 
        srj_ecg: 單個 srj檔的 ECG訊號
        evaluate_quality: 是否要開啟評估訊號品質機制
    output ---
        usd_flag(是否訊號顛倒的 flag)

    """
    measuring_mode='strap'
    method_type='vg'
    mode='revised'

    cnn_model = load_cnn()

    # initial condition
    slice_count = 0  # 擷取訊號片段記數
    vote_result_arr = np.full(30, np.nan)
    slice_index_arr = np.full(30, np.nan)

    # 每個srj檔案,抽取30個訊號片段來判斷
    for slice_idx in range(0, len(srj_ecg), 1):
        do_reverse = 0  # 有沒有故意反轉訊號(自動更新勿改)
        sig = srj_ecg[slice_idx]
        sig = baseline_remove(sig)
        # sig = sig*-1  # (開發階段測試用)

        if evaluate_quality:
            # 目前訊號表現
            score_a = 0
            score_p = 0
            ##r_list = rpeak.rpeak_detection(sig)
            r_list = rpeak.rpeak_detection(sig,measuring_mode=measuring_mode,method_type=method_type,mode=mode)
            score_a = quality_check.area_ratio(sig, r_list)
            score_p = quality_check.pattern_clustering(sig, r_list)

            # 目前訊號顛倒後表現
            sig2 = sig * -1
            score_a2 = 0
            score_p2 = 0
            r_list2 = rpeak.rpeak_detection(sig2,measuring_mode=measuring_mode,method_type=method_type,mode=mode)
            score_a2 = quality_check.area_ratio(sig2, r_list2)
            score_p2 = quality_check.pattern_clustering(sig2, r_list2)

            # 比較兩者誰 rpeak 打得比較好
            if len(r_list) <= len(r_list2) and score_a < score_a2 and score_p < score_p2:
                do_reverse = 1
                sig = sig2
                score_a = score_a2
                score_p = score_p2

            if score_a * score_p > 85:  #98
                if slice_count < 30:  # 未滿30個 "有效" 判斷結果

                    # Algorithm checking Upside-Down
                    flag_algorithm, outlier_sum, area = _do_check_outlier(check_signal=sig, slice_index=slice_idx)

                    # CNN checking Upside-Down
                    if (-3 <= outlier_sum <= 3) or (-1000 < area < 1000):  # Algorithm low confidence conditions
                        flag_cnn = _cnn_testing_loop(check_signal=sig, cnn_model=cnn_model, slice_index=slice_idx)
                        # print('CNN predict upside-down: ', flag_cnn)

                        # 若有故意反轉訊號,結果要顛倒
                        if do_reverse:
                            flag_cnn = int(not flag_cnn)

                        # Write in vote array
                        vote_result_arr[slice_count] = flag_cnn
                        slice_index_arr[slice_count] = slice_idx
                    else:
                        # 若有故意反轉訊號,結果要顛倒
                        if do_reverse:
                            flag_algorithm = int(not flag_algorithm)

                        # Write in vote array
                        vote_result_arr[slice_count] = flag_algorithm
                        slice_index_arr[slice_count] = slice_idx

                    # Each srj file cut 30 slice
                    slice_count += 1
                    if slice_count == 30:
                        break

        elif not evaluate_quality:
            if slice_count < 30:  # 未滿30個 "有效" 判斷結果

                # Algorithm checking Upside-Down
                flag_algorithm, outlier_sum, area = _do_check_outlier(check_signal=sig, slice_index=slice_idx)

                # CNN checking Upside-Down
                if (-3 <= outlier_sum <= 3) or (-1000 < area < 1000):  # Algorithm low confidence conditions
                    flag_cnn = _cnn_testing_loop(check_signal=sig, cnn_model=cnn_model, slice_index=slice_idx)

                    # Write in vote array
                    vote_result_arr[slice_count] = flag_cnn
                    slice_index_arr[slice_count] = slice_idx
                else:
                    # Write in vote array
                    vote_result_arr[slice_count] = flag_algorithm
                    slice_index_arr[slice_count] = slice_idx

                # Each srj file cut 30 slice
                slice_count += 1
                if slice_count == 30:
                    break

    # Vote whether ECG is upside-down (usd_flag = -1:初始值, 0:訊號正常 ,1:訊號顛倒)
    if sum(~np.isnan(vote_result_arr)) >= 6:  # 避免檔案過小造成少數flag做決定
        usd_flag = int(round(np.nanmean(vote_result_arr), 0))
    else:
        usd_flag = -1  # 有效預測樣本過少 不判斷顛倒

    #print('This file Upside-down vote num: ', sum(~np.isnan(vote_result_arr)))
    #print('Upside-down flag: ', usd_flag)

    return usd_flag


def upsidedown_correction(srj_ecgs, evaluate_quality=False):  ###主程式
    
    """
    input --- 
        srj_ecg: 單個 srj檔的 ECG訊號
        evaluate_quality: 是否要開啟評估訊號品質機制
    output ---
        srj_ecgs: 1D list of ECG which has been corrected(已經翻正的ECG訊號)

    """
    ###model = load_cnn()   ## Load CNN model for upside-down detection
    ##upsidedown_flag = [] ## Upside-down result list
    new_srj_ecgs = []        
    for srj_ecg in srj_ecgs:           
        usd_flag = _checking_ecg_upsidedown_v2(srj_ecg, evaluate_quality=evaluate_quality)
        if usd_flag == 1:
            for ecgs in srj_ecg:
                ecgs = np.array(ecgs) * -1
                new_srj_ecgs.append(ecgs.tolist())
        else:
            for ecgs in srj_ecg:
                new_srj_ecgs.append(ecgs)

    srj_ecgs = new_srj_ecgs
    
    return srj_ecgs

