# -*- coding: utf-8 -*-
"""
Created on 2024/04/12

@author: Dennis
"""
    
import numpy as np
from scipy import signal
from scipy.signal import resample
from neurokit2.signal import signal_fixpeaks, signal_formatpeaks
##from neurokit2.hrv.hrv_time import hrv_time
##from neurokit2.hrv.hrv_frequency import hrv_frequency
##from neurokit2.hrv.hrv_nonlinear import hrv_nonlinear
from neurokit2.signal.signal_power import signal_power
import neurokit2 as nk
import math
from .rpeak import rpeak_detection 


def _normalization(data,scale=10): ###副程式，使用於訊號正規化
    """
    input ---
        data: 1D numpy array for ECG signal 
        scale: 設定正規化到0~scale範圍
    output ---
        nor_data: normalized data
    """

    nor_data = np.array(data - np.min(data)) / (np.max(data) - np.min(data))*scale
    return nor_data


def quality_bassqi(ecg_cleaned, sampling_rate=250, window=1024, num_spectrum=[0, 60], dem_spectrum=[40, 60]):
    """
    calculate the relative power in the baseline
    
    inpupt ---
        ecg_cleaned: ECG signal which has been cleaned by nerokit's clean method
        sampling_rate: ECG sampling rate (default value=250)
        window: window size must be the power of 2 (default value=1024)
        num_spectrum: default value = [0, 60]
        dem_spectrum: default value = [40, 60] 
    
    output ---
        Score: the result of calculation   
    """
    
    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=True,
        window=window,
    )

    num_power = psd.iloc[0][0]
    dem_power = psd.iloc[0][1]

    score = 1 - dem_power / num_power
    return score

def r_stability_check(rpeaks_10s,max_rri_thr=500,min_rri_thr=75):  ##判定R波位置的穩定度

    """
    input ---
        rpeaks_10s: 1D numpy array contains R location index in 10-second ECG signal
        max_rri_thr: max rri threshold(unit: number of RRI points, default value=500)
        min_rri_thr: min rri threshold(unit: number of RRI points, default value=75)

    output ---
        r_stabiliy_score: Score between 0 - 100

    """

    r_stabiliy_score = 100

    if 5 <= len(rpeaks_10s) <= 33:

        rris_10s = np.diff(rpeaks_10s)

        if np.nanmax(rris_10s) > max_rri_thr or min_rri_thr > np.nanmin(rris_10s):

            r_stabiliy_score = 0

        else:

            for i in range(1, len(rris_10s) - 1):

                rris_10s_avg = (rris_10s[i + 1] + rris_10s[i - 1]) / 2
                r_now_score = (1 - abs(rris_10s[i] - rris_10s_avg) / rris_10s_avg) * 100

                if r_now_score < 0:
                    r_now_score = 0

                if r_stabiliy_score > r_now_score:
                    r_stabiliy_score = int(r_now_score)

    else:
        r_stabiliy_score = 0

    return r_stabiliy_score


def packet_loss_check(ecg, ridxs): ##檢查訊號是否有漏封包的狀況
    
    """
    input ---
         ecg: 1D array of ecg signal
         ridxs: 1D array of R peak index
    output ---
         score: 0 or 1 (1 means no packet loss in the input ECG signal)
         index: if score is 0, the index is the packet loss location 
    """

    check_len = 20
    score = 1   ##預設訊號沒有掉包的分數
    index = -1
    now_ecg = ecg[int(ridxs[0]):int(ridxs[-1])]
    for i in range(check_len-1, len(now_ecg)):
        if now_ecg[i] == now_ecg[i-1]:  # 現在這個點與前一個點振幅一樣
            flag = 1
            for j in range(1, check_len):  # 往查前check_len-1個點
                if (now_ecg[i] - now_ecg[i-j]) != 0:
                    flag = 0
                    break

            if flag == 1:
                score = 0
                index = i
                break

    return [score, index]


def pattern_clustering(ecg, ridxs, th=0.75): ##比對R波之間的相似度，越高表示R波偵測的情況更可靠
    
    """
    input ---
        ecg:  ECG訊號
        ridxs: R peak index list
        th: 門檻值

    output ---
        s: 訊號品質分數(0~100)
    """

    if 40 > len(ridxs) > 4:
    
        ecg = np.array(ecg, dtype='int32')
        ridxs = np.array(ridxs, dtype='int32')
        RRI = np.diff(ridxs)
        RRI = [rri for rri in RRI if 2000 >= 1000*rri/250 >= 250]
        if len(RRI) < 1:
            return 0
        rri_q1 = int(np.percentile(RRI, 25))
        ##rri_q1 = np.quantile(np.diff(ridxs), 0.25)
        
        before_r = int(0.334 * 0.8 * rri_q1)
        after_r = int(0.667 * 0.8 * rri_q1)
        
        ecgs_list = []   
        
        for rpeak in ridxs:
            n_rpeak_before = int(rpeak - before_r)
            n_rpeak_after = int(rpeak + after_r)
            if n_rpeak_before >= 0 and n_rpeak_after < len(ecg):
                ecgs_list.append(list(map(float,ecg[n_rpeak_before:n_rpeak_after])))
        
        if len(ecgs_list) <= 4:
            return 0
        
        ecgs_coeff = np.corrcoef(ecgs_list)

        relation= relation_cal(ecgs_coeff, th=th)
    
        relation = sorted([list(t) for t in set(tuple(element) for element in relation)], key=len, reverse=True)

        if(relation):

            s = len(relation[0]) / len(ecgs_list)

            return s*100
        else:
            return 0
    
    else:
        return 0


def relation_cal(corr, th):
    
    """
    Clustering patterns in groups by correlation coefficients.

    input ---
        corr (ndarray): square correlation coefficient matrix between patterns.

        th (float): threshold of correlation coefficient to determine whether two pattern are similar.

    output --- 
        relation (list): Indices (ndarray) of groups.
    """
    
    V = []
    for i in range(0, len(corr)):
        # % 相關係數大於th視為相似
        idx = np.argwhere(corr[:,i] > th)
        if len(idx)>0:
            V.append(idx[:,0])

    # 將V由大至小排列，目的:減少交集比對
    V.sort(key=len, reverse=True)

    # 整理分群關係
    Relation = []
    for i in range(0, len(V)):
        if len(Relation)==0:
            Relation.append(V[i])
        else:
            for j in range(0, len(Relation)):
                InterNum = len(np.intersect1d(V[i], Relation[j]))
                if InterNum>0:
                    Relation[j] = np.union1d(V[i], Relation[j])
                    break
                elif InterNum==0 and j==len(Relation)-1:
                    Relation.append(V[i])
                else:
                    continue

    # 確認群跟群之間沒有重複
    for i in range(0, len(Relation)):
        for j in range(0, len(Relation)):
            if i == j:
                continue

            C, ix, iy = np.intersect1d(Relation[i], Relation[j], return_indices=True)
            Px = len(C) / len(Relation[i])
            Py = len(C) / len(Relation[j])
            if Px > Py:
                np.delete(Relation[j], iy)
            else:
                np.delete(Relation[i], ix)

    Relation.sort(key=len, reverse=True)

    return Relation


def area_ratio(ecg, ridxs): ###檢查R波是否有漏偵測的狀況
    
    """
    Response the score based on areas of detected beats.
    
    Calculate areas of beats by given R peaks and uncovered areas.
    Check the ratio between uncovered areas and covered areas of beats.
    The score is higher if all areas are fully covered.
    
    input ---
      ecg: 1D array of ecg signal
      ridxs: 1D array of indices of R peaks in ecg

    output ---
      score between 0 ~ 1, the score of how detected beats cover areas in signal
    """
    
    # Calculate the median of RRI (unit:samples)
    RRIs = np.diff(ridxs)
    RRIs = [rri for rri in RRIs if 2000 >= 1000*rri/250 >= 250] # filter with ms
    if len(RRIs) < 1:
        return 0

    medRRI = np.median(RRIs)
    forward = int(0.4*medRRI)
    backward = int(0.6*medRRI)

    prevEndIdx = 0 #prevent from overlapping
    beatAreas = []
    missedAreas = []
    
    for i in range(0,len(ridxs)):
        # Set start point
        startIdx = int(ridxs[i] - forward)
        # Left Boundary
        if startIdx < 0:
            startIdx = int(0)
        # Prevent from overlapping
        if startIdx < prevEndIdx:
            startIdx = prevEndIdx

        # Set end point
        endIdx = int(ridxs[i] + backward)
        # Right boundary
        if endIdx >= len(ecg):
            endIdx = int(len(ecg))
        
        # Calculate Areas
        beat = np.absolute(ecg[startIdx:endIdx+1])
        beatArea = np.trapz(beat,dx=1/250)
        beatAreas.append(beatArea)
        if startIdx > prevEndIdx > 0:
            miss = np.absolute(ecg[prevEndIdx:startIdx+1])
            missArea = np.trapz(miss,dx=1/250)
            missedAreas.append(missArea)

        # Update last end point
        prevEndIdx = endIdx

    totalArea = np.trapz(np.absolute(ecg),dx=1/250)
    ideaArea = sum(beatAreas) + sum(missedAreas)
    R1 = ideaArea / totalArea
    ratio2 = max(missedAreas)/np.median(beatAreas) if len(missedAreas)>0 and len(beatAreas)>0 else 0
    R2 = 1-ratio2 if ratio2 < 1 else 0
    score = float(R1*R2)
    if(math. isinf(score)):
        score=0

    return score##   float(R1*R2)


def _frequency_analysis(data, fs=250): ####副程式
    noiseflag = 0
    fxx, pxx = signal.welch(data, fs)
    sumvalue = np.sum(np.abs(pxx[fxx > 20]))
    fft_sum = 0
    if sumvalue is not None:
        fft_sum = int(sumvalue)

    if fft_sum >= 50:
        noiseflag = 1

    return noiseflag, fft_sum


def check_high_freq_noise(sig, ridxs): ####檢查訊號中是否含有高頻雜訊
    
    """
    input ---
       sig: 1D numpy array of ECG signal
       ridxs: 1D array of index of R peak location
    
    output ---
       noiseflagArray: 1D numpy array of noise checking result of flags 
                       0 means the R-peak surrounding area without noise 
                       1 means the R-peak surrounding area is noisy 
    
    """

    sig = ((sig - np.min(sig)) / (np.max(sig) - np.min(sig))) * 100  ####正規化到0~100
    offset = 150
    noiseflag = 0
    noise_flag_array = np.zeros(len(ridxs))  
    for index, rpeak in enumerate(ridxs):
        datalength = offset * 2 + 1
        data = np.zeros(datalength)
        startindex = int(rpeak - offset)
        endindex = int(rpeak + offset)
        cutflag = 0
        if (startindex < 0):
            startindex = 0
            reminder = int(startindex - (rpeak - offset))
            data[0:reminder + 1] = sig[0]
            data[reminder + 1:datalength] = sig[startindex:endindex]
            cutflag = 1

        if (endindex > len(sig)):
            endindex = len(sig)
            reminder = int(rpeak + offset - endindex + 1)
            data[0:datalength - reminder] = sig[startindex:endindex]
            data[datalength - reminder:datalength] = sig[-1]
            cutflag = 1

        if (cutflag == 0):
            data = sig[startindex:endindex]           
            data=_normalization(data,scale=100) ####正規化到0~100

        noiseflag, fftsum = _frequency_analysis(data, fs=250)
        noise_flag_array[index] = noiseflag
       

    return noise_flag_array 


def _wavelet_g(w): ##副程式   

    return 4j * np.exp(1j * w / 2) * np.sin(w / 2)

def _wavelet_h(w): ##副程式   

    return np.exp(1j * w / 2) * np.cos(w / 2) ** 3

def _wavelet_filters(N, fs): ##副程式 

    M = N * 250 / fs
    w = np.arange(0, 2 * np.pi, 2 * np.pi / M)  # Frequency axis in radians

    ## Construct the filters at 250 Hz as specified in the paper
    Q = [_wavelet_g(w)]
    for k in range(2, 6):
        G = _wavelet_h(w)
        for l in range(1, k - 1):
            G *= _wavelet_h(2 ** l * w)
        Q += [_wavelet_g(2 ** (k - 1) * w) * G]

    # Resample the filters from 250 Hz to the desired sampling frequency
    for i in range(len(Q)):
        Q[i] = np.fft.fft(resample(np.fft.ifft(Q[i]), N))

    return Q

def _waveletdecomp(sig, Q): ##副程式       

    w = []

    ## Apply the filters in the frequency domain and return the result in the time domain
    for q in Q:
        w += [np.real(np.fft.ifft(np.fft.fft(sig) * q))]

    return w



def wavelet_detect_noise(ecg_baseline,qrs,fs=250): ##確認T波是否為蝙蝠頭
    
    """
    input ---
        ecg_baseline: ECG signal after baseline drafting removal
        qrs: 1D array of qrs indexes

    output ---
        score_list: 1D array of qulity checking result for each beat

    """

    sig = np.array(ecg_baseline)
    ## Number of samples in the signal ##Create the filters to apply the algorithme-a-trous
    Q = _wavelet_filters(sig.shape[0],fs)
    w = _waveletdecomp(sig, Q)
    rri = np.diff(qrs)
    rri = np.append(rri,int(2500-np.max(qrs)))
    w_num = 2
    score_list = []

    for j in range(0, len(qrs)):

        score = 1
        rri_thirdone = np.arange(int(qrs[j]) , int(qrs[j] + rri[j]-1))
        sig_t=np.abs(sig[rri_thirdone])
        sig_t2=np.abs(w[w_num][rri_thirdone])
        
        if(len(sig_t)>0 and len(sig_t2)>0):
            if(np.max(sig_t)==np.min(sig_t) or np.max(sig_t2)==np.min(sig_t2)):
                score=0
                score_list.append(score)
                continue
        else:
            score=0
            score_list.append(score)
            continue

        Nor_ecg = _normalization(sig_t) ###np.abs(sig[rri_thirdone]))
        Twave_check = _normalization(sig_t2)###np.abs(w[w_num][rri_thirdone]))
        Twave_check2 = list(Twave_check[1:])
        Twave_check2.append(Twave_check[0])
        if(np.isnan(np.min(sig_t)) or np.isnan(np.min(sig_t2))):
            score=0
            score_list.append(score)
            continue

        squared_error = np.sum(np.abs(np.array(Twave_check)-np.array(Twave_check2)))

        mse = np.mean(squared_error)
        slope, intercept = np.polyfit(list(Nor_ecg), list(Twave_check), 1)

        if  slope <= 0.6 and mse > 60:           
            score-=1
        elif 0.23 < slope <= 0.48 and mse > 50.3:           
            score -= 0.5
        elif 0.52 < slope <= 0.82 and mse > 52.5:            
            score -= 0.5
        elif -0.2 < slope <= 0.23 and mse > 38.1:
            score -= 0.5
       
        score_list.append(score)

    return score_list


def ecg_quality_check(ecgs_10s_impulse, rpeaks_10s_raw):

    ecgs_10s_vg = nk.ecg_clean(ecgs_10s_impulse, sampling_rate=250, method="vg")
    area_score_impulse = int(area_ratio(ecgs_10s_impulse, rpeaks_10s_raw) * 100)

    if area_score_impulse < 15:
        return "AV Block"

    else:

        bat_score_impulse = int(np.nanmean(wavelet_detect_noise(ecgs_10s_impulse, rpeaks_10s_raw, fs=250)) * 100)

        if bat_score_impulse < 40:
            return "Bat Head"

        else:

            '''
            signals_vg, info_vg = nk.ecg_peaks(ecgs_10s_vg, sampling_rate=250, method="neurokit", correct_artifacts=False, show=False)
            ridxs_vg = signals_vg["ECG_R_Peaks"]
            rpeaks_10s_vg = np.array([i for i, ridx in enumerate(ridxs_vg) if ridx == 1])
            '''
            rpeaks_10s_vg=rpeak_detection(ecgs_10s_vg)
            rpeaks_10s_vg=rpeaks_10s_vg[1:]
            pattern_score_vg = pattern_clustering(ecgs_10s_vg, rpeaks_10s_vg, th=0.85) 

            if pattern_score_vg < 75:
                return "Thumb"

            else:

                r_stabiliy_score_vg = r_stability_check(rpeaks_10s_vg)

                if r_stabiliy_score_vg < 80:
                    return "Irregular"

                else:
                    return "Normal"

    return "Error"


if __name__ == '__main__':
    #ecgs = scipy.io.loadmat('SWMIRB_546C0ED03BFC_1614844253584.mat')['ECG'].ravel()
    
    ID = str(1018)
    
    ecg_file = "ECG_" + ID + ".txt"

    with open(ecg_file) as f:
        ecgs = list(map(float, str(f.readlines())[2:-2].split(", ")))
        f.close()
    
    ### R Peak List
    
    rpeak_file = "R_" + ID + ".txt"
    
    with open(rpeak_file) as f:
        rpeaks = list(map(float, str(f.readlines())[2:-2].split(", ")))
        rpeaks = list(map(int, rpeaks))
        f.close()
    
    ecgs = np.array(ecgs)
    rpeaks = np.array(rpeaks)
    
    i_max = len(ecgs)//2500
    
    Score = []
    
    for i in range(i_max):
    
        ecgs_part = ecgs[i*2500:(i + 1)*2500]
    
        rpeaks_part = []
    
        for rpeak in rpeaks:
            if i*2500 <= rpeak < (i + 1)*2500:
                rpeaks_part.append(rpeak)
        
        rpeaks_part = np.array(rpeaks_part)-i*2500
        
        Score.append(pattern_clustering(ecgs_part, rpeaks_part, 0.8))
    
    print(Score)

