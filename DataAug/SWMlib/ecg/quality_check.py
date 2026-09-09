# -*- coding: utf-8 -*-
"""
Created on 2024/04/12

@author: Dennis
"""
    
import numpy as np
from scipy import signal
from scipy.signal import resample
##from neurokit2.signal import signal_fixpeaks, signal_formatpeaks
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
    range_val = np.max(data) - np.min(data)
    if range_val == 0:
        return data
    else:
        try:
            nor_data = (np.array(data - np.min(data)) / (np.max(data) - np.min(data)))*scale
            return nor_data
        except:
            return data
    

def _normalization_upper_lower(data,upper_lim,lower_lim): ###使用於訊號正規化，將資料設定在區間為upper_lim~lower_lim
    
    """
    input ---
        data: 1D numpy array for ECG signal 
        upper_lim: 設定正規化的最大上限
        lower_lim: 設定正規化的底限
    output ---
        nor_datas: normalized data
    """
   
    range_val = np.max(data)- np.min(data)   
    data_min = np.min(data)
    if range_val == 0:
        return data
    
    else:
        try:

            nor_data = ((upper_lim-lower_lim) * (np.array(data - data_min) / range_val))+lower_lim
            return nor_data
        
        except:
            return data
        

def _ginical(x):
    
    #gini係數計算
    """
    input:
    x必須是不含負數的numpy array

    output:
    gini:gini係數
    """
    x=np.sort(x)
    n=len(x)
    cumx=np.cumsum(x)
    sum_x = np.sum(x)

    if sum_x ==0:
        return 0
    
    gini = (n+1-(2*np.sum(cumx))/sum_x)/n

    return gini


def out_of_range_check(ecg, rpeaks_array=[], ginit =0.23):
    
    ##排除RRI中間訊號震幅飄動不一致的訊號    
    """
    input:
    ecg:是單一個10秒ECG signal
    rpeaks_array: 使用r peak detection得到之結果
    ginit:篩選之gini係數閾值，默認為0.23
    
    output:
    gooddataindex:
    0: 品質好與含有其他種壞的資料
    1: 有過大雜訊或是T,P波缺失的資料
    2: rpeak不足2的資料，
    3: 數據有因藍芽傳輸問題丟包的資料

    ecgnn1: normalized_ecg(-150~150)    
 
    """
    ecgnn=_normalization_upper_lower(ecg,300,0)  ##此正規化之後的範圍對R peak detection效果較好
    ecgnn1=_normalization_upper_lower(ecg,150,-150)  ##為了避免負值的尖點被正規化為0，造成後續取平方或絕對值無效
   
    if(len(rpeaks_array)==0):
        rpeaks_array=rpeak_detection(ecgnn, mode='original')
        rpeaks_array=rpeaks_array[1:]
        
    if len(rpeaks_array) < 2:
        return 2,ecgnn1

    rpeaks=np.array(rpeaks_array)  
    ##isloss = packet_loss_check(ecgnn,  15, rpeaks)#第一道過濾，濾出有漏封包的狀況
    isloss = packet_loss_check(ecgnn,  25, rpeaks)#第一道過濾，濾出有漏封包的狀況
    if isloss[0]!=1:
        return 3,ecgnn1
            
    gooddataindex=0         
    segmentmax=[]
        
    ##目前對前短後長有用，但相反未知，可是目前沒有相反的例子，暫且如此設計
    rri=np.diff(rpeaks)        
    medianw= np.median(rri)
    
    peaks_to_keep_indices = list(range(len(rpeaks)))
    for i in range(len(rri) - 1, 1, -1): ##一定要backward的操作
        if rri[i] < 0.7 * medianw:  ##0.7是初估之參數
            if i == (len(rri)-1):
                continue                       
            
            if rri[i+1]>rri[i-1]:              
                del peaks_to_keep_indices[i+1]
            #else:
                #rpeaks = np.delete(rpeaks, i+1)
        
    rpeaks = rpeaks[peaks_to_keep_indices]

    for i in range(len(rpeaks)-1): ##目前不處理r peak偵測在訊號的頭尾有漏打的情況(因為後續專案也是基於有偵測到的R peak去擷取特徵)
            
        start=rpeaks[i]+20
        end=rpeaks[i+1]-20
        segment=ecgnn1[start:end]

        if len(segment) ==0:
            continue
            
        delta = 0 
        segment_max=np.max(segment) ##rr中間片斷最大值
        segment_med = np.median(segment)
        segment_min = np.min(segment)
            
        diff_max_med = segment_max - segment_med
        diff_max_min = segment_med - segment_min

        if(diff_max_med >= diff_max_min):
            delta=diff_max_med
        else: 
            delta=diff_max_min
          

        segment_max2 = delta ** 2
        segmentmax.append(segment_max2)
        
              
    segmentmax=np.array(segmentmax)
    ginib=_ginical(segmentmax)

    if ginib> ginit:#gini係數範圍在0~1之間，越大代表離群值越大，或數值越分散，因此大於閾值者有較高機會有過大雜訊或是波段消失
        gooddataindex=1

    else:
        gooddataindex=0
    

    return gooddataindex,ecgnn1

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

def r_stability_check(rpeaks_10s,max_rri_thr=500,min_rri_thr=75): 

    """
    判定R波位置的穩定度

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


def packet_loss_check(ecg, check_len=20, ridxs=[]): 
    
    """
    檢查訊號是否有漏封包的狀況
    input ---
         ecg: 1D array of ecg signal
         ridxs: 1D array of R peak index
         check_len: how many lost points will be detected
    output ---
         score: 0 or 1 (1 means no packet loss in the input ECG signal)
         index: if score is 0, the index is the packet loss location 
    """

    score = 1   ##預設訊號沒有掉包的分數
    index = -1
    if(len(ridxs)==0):
        now_ecg=ecg
    else:
        now_ecg = ecg[int(ridxs[0]):int(ridxs[-1])]
    
    for i in range(check_len-1,len(now_ecg)):
        if now_ecg[i] == now_ecg[i-1]:  ## 現在這個點與前一個點振幅一樣
            flag = 1
            for j in range(1, check_len):  ## 往前查check_len-1個點
                if (now_ecg[i] - now_ecg[i-j]) != 0:
                    flag = 0
                    break

            if flag == 1:
                score = 0
                index = i
                break

    return [score, index]


def pattern_clustering(ecg, ridxs, th=0.75):
    
    """
    比對R波之間的相似度，越高表示R波偵測的情況更可靠

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


def area_ratio(ecg, ridxs): 
    
    """
    檢查R波是否有漏偵測的狀況
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


def _frequency_analysis(data, fs=250):
    noiseflag = 0
    fxx, pxx = signal.welch(data, fs)
    sumvalue = np.sum(np.abs(pxx[fxx > 20]))
    fft_sum = 0
    if sumvalue is not None:
        fft_sum = int(sumvalue)

    if fft_sum >= 50:
        noiseflag = 1

    return noiseflag, fft_sum


def check_high_freq_noise(sig, ridxs):
    
    """
    檢查訊號中是否含有高頻雜訊
    
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



def wavelet_detect_noise(ecg_baseline,qrs,fs=250): 
    
    """
    確認T波是否為蝙蝠頭
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


##------T broken wave detection and other distortion wave detection-----

def _window_product(rpeaks_array=None): ##副程式 
    
    """
    對一個ECG訊號產生對應的自適應window，該window用於後續的二值化與數連續的1群操作
    input: 
    rpeaks_array:該ECG signal的rpeak位置的陣列
    output: a adaptive window size
    """
    
    rpeaksdiff=[]
    end_index=len(rpeaks_array)-1

    for j in range(0,end_index):
        ##if j+1 == len(rpeaks_array):
        ##    break
        
        di = rpeaks_array[j+1] - rpeaks_array[j]
        rpeaksdiff.append(di)
    
    diffmedian=np.median(np.array(rpeaksdiff))
    
    if np.isnan(diffmedian) == 1:
        diffmedian = 0
    return int(diffmedian * 0.7)

def _zcr_diff(x): ##副程式 
    
    """
    計算zero-cross rate
    input:
    x:a part of ECG signal or the first derivative of ECG signal
    output: zero-cross rate, it means the oscillation frequency if signal
    """
    
    d=np.diff(x)
    d[np.abs(d) < 1e-4]==0.0
    s = np.sign(d)

    return np.mean(s[1:] * s[:-1] <0)

def _teager_energy(x): ##副程式 
    
    """
    計算訊號的teager_energy
    input:a part of ECG signal or the first derivative of ECG signal
    output: energy of oscillation in the part of ECG signal
    """
    
    ##x=np.asarray(x, dtype=float)
    
    if len(x) <3:
        return np.zeros_like(x)
    
    psi=np.zeros_like(x)
    psi[1:-1]=x[1:-1]**2 - x[:-2]*x[2:]
    
    return psi

def _teager_energy_mean(x): ##副程式 
    
    """
    計算一個訊號的平均teager_energy
    input:a part of ECG signal or the first derivative of ECG signal
    output: mean energy of oscillation in the part of ECG signal
    """

    psi=_teager_energy(x)
    return np.mean(np.abs(psi))

def _tk_zcr_hybrid_score(x): ##副程式 
    
    """
    取得teager_energy與zero-cross rate混成的訊號振盪程度指標
    input:a part of ECG signal or the first derivative of ECG signal
    output: score:a index for evaluating level of amplitude and oscillation frequency of ECG signal
    """
    
    e = _teager_energy_mean(x)
    tkeo_scale=np.max(np.abs(_teager_energy(x))) +1e-8
    e_norm=e/(tkeo_scale + 1e-8)
    z = _zcr_diff(x)
    score = e_norm*(1.0+2.0 *z)
    return score

def _sw_amplitude_qrs_amplifier(ecg): ##副程式 
    """
    判斷ECG訊號中需要放大的QRS波區段並放大
    input: 
    ecg: numpy array of length 2500 - the ECG signal or the first derivative of ECG signal you want to amplify
    output: 
    out: ECG signal which is amplified
    """
    ##ecg = np.asarray(ecg, dtype=float)
    n=2500 ##len(ecg)
    out = np.zeros(n) ##ecg.copy()
    end_index=n-10

    for start in range(0,end_index):
        end=start+11
        window = ecg[start:end]
        '''
        tkz = _tk_zcr_hybrid_score(window)
        
        if np.std(window) < 5:
            tkz=0
        '''
        if np.std(window) < 5:
            tkz=0
        else:
            tkz = _tk_zcr_hybrid_score(window)

        if tkz >0.8:
            out[start:end] = window * 2.6
        elif tkz <=0.8 and tkz >0.6:
            out[start:end] = window * 1.6
        elif tkz <=0.6 and tkz > 0.2:
            out[start:end] = window * 1.4
        
        
    return out

def _sliding_window_amplitude(ecg): ##副程式 
    
    """
    求ECG訊號的sliding window總和
    input: 
    ecg:ECG list of length of 2500 - the ECG signal or the first derivative of ECG signal which you want to compute sliding-window sums

    output:
    np.array(amp_sums_avg):array includes sums of each window
    """
    
    n=2500##  len(ecg)
    amp_sums_avg=[]
    end_index=n-8

    for start in range(0,end_index):
        end=start+9
        window = ecg[start:end]
        amp_sum_a=(np.sum(np.abs(window)) )
        amp_sums_avg.append(amp_sum_a)
    
    return np.array(amp_sums_avg)

def _qrs_single_hump(x,rectify ='abs'): ##副程式 
    
    """
    對一個ECG訊號平滑化
    input:ecg:the sums with overlapping sliding-window of the first derivative of ECG or other kinds of continuous data you want to smooth
    win_ms:window size for first smoothing
    rectify: way for smoothing
    
    output:
    mwi:smoothed ECG signal
    """

    ##x= np.asarray(ecg,float)

    if rectify =='abs':
        z=x
    elif rectify =='square':
        z = x *x
    else:
        raise ValueError("rectify must be 'abs' or 'square'")

    k = max(2,int(round(7/1000*250)))
    ker = np.ones(k,dtype=float) /k
    mwi = np.convolve(z, ker, mode='valid')

    return mwi

def _sliding_window_amplitude_binarycount(sig, window_size=45,controlarray=None): ##副程式 
    
    """
    對ECG signal的second derivative之sliding window總和在使用自適應window size的window下，取得該window平均後，
    對其平均大於某值的window內，其大於平均之倍數(閾值)的數值組成的群且該群之最大值減該閾值的值大於0時視為1，
    其他值為0的二值化，並在確定該window二值化後頭或尾不是1時，對於長度大於min_run_len的1群計數，且在該框內對應的放大後二階導數
    區域內保證其最大值減平均大於平均2倍時，採納該框的1群數量並用於判斷訊號是否有異常型態的波段
    input:
    sig:he sums with overlapping sliding-window of the first derivative of ECG or other kinds of continuous data
    window_size:range for computing averages over the sums of the first derivative of ECG
    controlarray: amplified first derivative of ECG signal,for double checking which numbers
    of contiguous runs of ones should be counted
    output:
    index: the index for judging data quality,0 is normal, 1 means T-distortion
    """

    ##sig=np.asarray(sig,dtype=float)
    n = 2500  ##len(sig)
    index=0 #0是正常的資料，1代表有問題
    end_index=n-window_size +1

    for start in range(0,end_index):  #連續型
        end = start + window_size
        window = sig[start:end]
        windowa = np.abs(window)
        amp_sum_a =( np.sum( windowa)//  window_size) ##len(window))

        if amp_sum_a <= 12:
            continue
        
        thr = amp_sum_a
        above =  windowa >thr
        idx = np.where(above)[0]
        run_start=idx[0]
        runs=[]

        end_index=len(idx)
        for i in range(1,end_index):
            if idx[i] != idx[i-1] +1:
                runs.append((run_start,idx[i-1]))
                run_start = idx[i]

        runs.append((run_start,idx[-1]))
        seg_out = np.zeros_like(window,dtype=int)

        for a,b in runs:
            local_max = np.max( windowa[a:b+1])
            if local_max - thr >=0:
                seg_out[a:b+1] = 1
            else:
                seg_out[a:b+1] = 0
                
        if seg_out[0] == 1 or seg_out[-1] == 1:
            continue
        
        idx_bin =np.where(seg_out ==1)[0]

        if idx_bin.size == 0:
            continue
        
        runs_bin=[]
        rs=idx_bin[0]

        end_index=len(idx_bin)
        for i in range(1,end_index):
            if idx_bin[i] != idx_bin[i-1] +1:
                runs_bin.append((rs,idx_bin[i-1]))
                rs = idx_bin[i]

        runs_bin.append((rs,idx_bin[-1]))
        L=len(window)
        valid_count=0

        for s,e in runs_bin:
            length = e-s
            if (length >= 7): ## and s > 0 and e < L-1:
                valid_count +=1

        cwindow = controlarray[start:end]
        cmax = np.max(cwindow)
        cmean = np.mean(cwindow)

        if cmax-cmean >= 2*cmean:
            if valid_count >=2:
                index =1
                break
            
    return index

def T_distortion_filter(ecg,rpeaks_array=None):
    """
    判斷哪些ECG signal有異常的波段形式
    input:location of ECG signal
    output: index for ECG signal quality
    """
    isTdistortion = 0
    ##ecg= _preprocessed_signal(ecgdata)
    ##ecg = ecg[:2500]#以上為讀取10秒資料
    
    windowsize = _window_product(rpeaks_array=rpeaks_array)

    if windowsize == 0:
        isTdistortion = 1

    ecgdg1=np.gradient(ecg)
    ecgdg2=np.gradient(ecgdg1)
    abs_ecg=np.abs(ecgdg2)#取得ecg原始二階導數
    abs_ecgqrsa = _sw_amplitude_qrs_amplifier(abs_ecg)#QRS放大
    ahampavg = _sliding_window_amplitude(abs_ecgqrsa)#對該訊號做sliding window總和(size = 9)
    abs_ecgsf = _qrs_single_hump(ahampavg,rectify ='abs')#平滑化

    if isTdistortion != 1:
        one_number = _sliding_window_amplitude_binarycount(abs_ecgsf,window_size=windowsize, controlarray=abs_ecgqrsa)
        if one_number != 0:
            isTdistortion = 1
        else:
            isTdistortion = 0
            
    return isTdistortion

##----------------------------------------------------------------------
def ecg_quality_check(ecgs_10s_impulse, rpeaks_10s_raw): ##old method

    ecgs_10s_vg = nk.ecg_clean(ecgs_10s_impulse, sampling_rate=250, method="vg")
    ##area_score_impulse = int(area_ratio(ecgs_10s_impulse, rpeaks_10s_raw) * 100)
    area_score_impulse_raw = area_ratio(ecgs_10s_impulse, rpeaks_10s_raw) * 100
    area_score_impulse = int(area_score_impulse_raw) if not math.isnan(area_score_impulse_raw) else 0

    
    if area_score_impulse < 15:
        return "AV Block"

    else:

        bat_score_impulse = int(np.nanmean(wavelet_detect_noise(ecgs_10s_impulse, rpeaks_10s_raw, fs=250)) * 100)

        if bat_score_impulse < 40:
            return "Bat Head"

        else:
            
            rpeaks_10s_vg=rpeak_detection(ecgs_10s_vg)          
            
            
            out_of_range_score=0            
            out_of_range_score=out_of_range_check(ecgs_10s_vg, rpeaks_10s_vg, ginit =0.23)                     
            
            if(out_of_range_score>0):
                return "Bad"
            else:           
                    
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


def ecg_quality_check_v2(ecgs_10s_impulse, rpeaks_10s_raw):

    if len(rpeaks_10s_raw)<5:
        return(-1, -1, "Bad")
    
    ecgs_10s_vg = nk.ecg_clean(ecgs_10s_impulse, sampling_rate=250, method="vg")

  

    rpeaks_10s_vg=rpeak_detection(ecgs_10s_vg)
    rpeaks_10s_vg=rpeaks_10s_vg[1:]
    
    if len(rpeaks_10s_vg)<5:
        return(-1, -1, "Bad")

   
    pattern_score_vg = pattern_clustering(ecgs_10s_vg, rpeaks_10s_vg, th=0.85) 
    r_stabiliy_score_vg = r_stability_check(rpeaks_10s_vg)

    if pattern_score_vg < 60:
        quality= "Bad"

    else:
        area_score_impulse = int(area_ratio(ecgs_10s_impulse, rpeaks_10s_raw) * 100)
        if area_score_impulse < 15:
            quality= "AV Block"

        else:
            bat_score_impulse = int(np.nanmean(wavelet_detect_noise(ecgs_10s_impulse, rpeaks_10s_raw, fs=250)) * 100)

            if bat_score_impulse < 40:
                quality= "Bat Head"

            elif pattern_score_vg < 75:
                quality= "Thumb"

            elif r_stabiliy_score_vg < 80:
                quality= "Irregular"
                    
            else:
                quality= "Normal"
                
    return (pattern_score_vg, r_stabiliy_score_vg, quality)
        


def ecg_quality_check_v3(ecgs_10s_vg, rpeaks_10s): ##20251206 dennis modified


    area_score_impulse_raw = area_ratio(ecgs_10s_vg, rpeaks_10s) * 100
    area_score_impulse = int(area_score_impulse_raw) if not math.isnan(area_score_impulse_raw) else 0

    
    if area_score_impulse < 15:
        return "Bad"
    else:        
        out_of_range_score=0            
        out_of_range_score,normalized_ecg=out_of_range_check(ecgs_10s_vg, rpeaks_10s, ginit =0.23) ##要修正                       
        if(out_of_range_score !=0):
            return "Bad"
        else:     
            pattern_score_vg = pattern_clustering(ecgs_10s_vg, rpeaks_10s, th=0.85) 
            if pattern_score_vg < 75:
                return "Bad"

            r_stabiliy_score_vg = r_stability_check(rpeaks_10s)

            if r_stabiliy_score_vg < 80:
                return "Irregular"
            else:
                ##ecgn= _normalization_upper_lower(ecgs_10s_vg, 150, -150)
                flag=T_distortion_filter(normalized_ecg,rpeaks_10s)

                if(flag==1):
                    return "Bad"
                else:
                    return "Normal"
                    

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

