'''
Version 3:
    1. A new mechanism to detect inverse T waves is included.
    2. Modified the mechanism of false R detection.
    3. Isolate the P wave detection from T wave detection.
Version 4:
    1. Speed enhanced: mean filter with pandas function.
    2. Bug fixed.
Version 5:
    1. Change the format of outputs to dictionary.
    2. Canceled plotting figures.
    3. Follow parameters from MIT-BIH test version.
'''

import numpy as np
from scipy import signal
import pandas as pd


def _findvally(x):
    """
    Return the first valley location if the magnitude is lower than 0.33*|R|.

    Parameter:
        x (ndarray): 1D ecg signal before or after R peak.

    Return:
        loc (int): The index of valley location in x.
    """

    # 去除x 的NaN
    nan_idx = np.argwhere(np.isnan(x)).flatten()[0]
    x = x[:nan_idx]
    if len(x) <= 0:
        return
    
    # 平移至零準位
    x = x - min(x)

    # 對 x 微分得到速度 vel
    vel = np.diff(x)
    
    # 定義 vel 的方向性 sgn
    sgn = np.ones(len(vel))
    sgn[vel <= 0] = -1
    sgn[np.isnan(vel)] = np.nan
    
    # 對 sgn 微分
    change = np.diff(sgn)
    
    # 方向由負轉正的地方為波谷
    idx = np.argwhere(change == 2)
    if len(idx)==0:
        return None
    else:
        # 找出第一個振幅低於0.33*Rpeak的波谷位置
        th = 0.33 * x[0]
        for k in range(0, len(idx)):
            loc = idx[k]
            if x[loc] < th:
                return int(loc + 1)

    
def _pattern_clustering(corr, th):
    '''
    Clustering patterns in groups by correlation coefficients.

    Parameters:
        corr (ndarray): square correlation coefficient matrix between patterns.

        th (float): threshold of correlation coefficient to determine whether two pattern are similar.

    Return: 
        Relation (list): Indices (ndarray) of groups.
    '''
    
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


def _mean_filter(X, P):
    '''
    Mean filter on signal X. This filter will ignore NaN.
    
    Parameters:
        X (ndarray): 1D signal.

        P (float): width of mean filter.

    Return:
        W (int): Even number of kernel size.

        MA (ndarray): 1D filtered signal.
    '''
    
    # 將濾波寬度轉成偶數
    W = np.ceil(P)
    if W%2 == 1:
        W += 1
    
    # 以 W 做平均濾波
    '''
    MA = np.full(X.shape, np.nan)
    for i in range(0, len(X)):
        if i-W/2 >= 0 and i+W/2 < len(X):
            Y = X[int(i-W/2) : int(i+W/2+1)]
            if sum(np.isnan(Y)) == 0:
                MA[i] = np.mean(Y)
    '''
    MA = pd.Series(X.flatten()).rolling(int(W+1),center=True).mean()
    MA = np.array(MA).reshape(X.shape)

    return W, MA


def _find_waves(X, P1, P2, target = None):
    '''
    Use two different width of mean filter to detect outstanding waves.

    Parameters:
        X (ndarray): 1D signal.
        
        P1 (float): narrower width of mean filter.

        P2 (float): wider width of mean filter.

        target (str): If the purpose is to find the T wave, then set target to "T".

    Return:
        peaks (ndarray): 2D array.
            1st column: the magnitude of peaks.
            2nd column: the index of peak in signal X
            3rd column: the standard deviation represents the level of wave
    '''
    # 以 P1 做平均濾波保留波的起伏
    W1, MApeak = _mean_filter(X, P1)

    # 以 P2 做平均濾波得到飄移基線
    W2, MAbase = _mean_filter(X, P2)

    # Block of interest: MApeak > MAbase表示波型有明顯起伏
    blocks = np.zeros(X.shape)
    for i in range(0, len(X)):
        if ~np.isnan(MApeak[i]) and ~np.isnan(MAbase[i]):
            if target == "T" and abs(MApeak[i]) > abs(MAbase[i]):
                blocks[i] = 200
            elif MApeak[i] > MAbase[i]:
                blocks[i] = 200

    # 查看各個blocks的連續性
    # 對blocks作微分，在blocks的開始和結束處才會有值
    Dif = np.diff(blocks)
    start_flag = 0
    peaks = []
    for i in range(0, len(Dif)):
        d = Dif[i]

        # block 的起點
        if d==200:
            start_flag = 1
            start_idx = i

        # block 的終點
        if d==-200 and start_flag:
            block_len = i - start_idx
           
            # block寬度大於 W1 才找峰值
            if block_len < W1:
                blocks[start_idx+1:i+1] = 0
            else:
                # 擷取該block對應的X值
                block = np.full(X.shape, np.nan)
                block[start_idx+1:i+1] = X[start_idx+1:i+1]

                # 尋找block中的峰值
                if target == "T":
                    block = abs(block)
                M = np.nanmax(block)
                Imax = np.nanargmax(block)
                
                # 計算兩個平均濾波的差:MAdiff
                MAdiff = MApeak[start_idx+1:i+1] - MAbase[start_idx+1:i+1]
                STD = np.std(MAdiff)
                
                peaks.append([M, Imax, STD])
                
            # reset
            start_flag = 0

    peaks = np.array(peaks)
    return peaks


def _interval_calc(Idx1, Idx2, Fs):
    '''
    Calculate intervals between each index of Idx1 and Idx2. Then calculate the averaged interval.

    Parameters:
        Idx1 (ndarray): 1D array of indices.

        Idx2 (ndarray): 1D array of indices.

        Fs (int): Sampling Rate.

    Return:
        avgInterval (float): Averaged interval in ms.
    '''
    
    intervals = np.full(Idx1.shape, np.nan)
    for i in range(0, len(Idx1)):
        x = Idx1[i]
        y = Idx2[i]
        if ~np.isnan(x) and ~np.isnan(y):
            intervals[i] = (y - x) / Fs
        
    if sum(~np.isnan(intervals))>0:
        avgInterval = round(1000 * np.nanmean(intervals))
    else:
        avgInterval = np.nan
    
    return avgInterval


def _interval_calc_perbeat(Idx1, Idx2, Fs):
        
    intervals = np.full(Idx1.shape, np.nan)
    for i in range(0, len(Idx1)):
        x = Idx1[i]
        y = Idx2[i]
        if ~np.isnan(x) and ~np.isnan(y):
            intervals[i] = (y - x) / Fs
        
        
    return intervals



def feature_gen(ecg, Fs, Ridx, FalseR=False): ###主要呼叫函式

    """

    input --- 
        ecg (ndarray): The 1st column is the value of ecg, and the 2nd column is Rpeak locations.
        Fs (int): The sampling rate of ecg.
        Ridx (ndarray): R peak location indices of ecg.      
        FalseR: the flag to enable false R detection, if not given, the default value is false

    output --- 
        Dict = {'Ridx': 1D numpy array which contains the modified R wave peak after false R detection
            'avgHR':(float) - Averaged Heart Rate (BPM)
            'avgPR':(float) - Averaged PR interval (ms)
            'avgQRS': (float) - Averaged QRS interval (ms)
            'QRSArray': 1D numpy array which contains QRS widths 
            'avgQT': (float) - Averaged QT interval (ms)
            'avgQTc':(float): Averaged Corrected QT interval (ms)
            'good_quality':False or True 
            }
      
    """

    ##### QRS detection ######    
    # 重新標記 Rpeak 位置
    ecg = np.array(ecg)    
        
    # butterworth bandpass filter design
    fs = 0.12 
    fc = 0.3
    b, a = signal.butter(2, [fs, fc], btype='bandpass', output='ba')

    # 尋找 Q 波和 S 波
    Sf = 50
    Sb = 99
    Beta = 0.17
    W1 = 20
    Qidx = np.full(Ridx.shape, np.nan)
    Sidx = np.full(Ridx.shape, np.nan)
    Beats= []
    false_R_flag = False
    for i in range(0, len(Ridx)):
        idx = int(Ridx[i])
        if idx-Sf >= 0 and idx+Sb+1 <= len(ecg):
            # extract a beat X from ecg
            X = ecg[idx-Sf : idx+Sb+1, 0]
            Beats.append(X)

            # bandpass filter on X
            Xf = signal.filtfilt(b, a, X)
            Xsq = Xf * Xf

            # Moving Average with W1 as window size to detect QRS segment
            MAqrs = np.zeros(len(Xsq))
            for j in range(0, len(Xsq)):
                if j-W1/2 >= 0 and j+W1/2 < len(Xsq):
                    window = Xsq[int(j-W1/2) : int(j+W1/2+1)]
                    MAqrs[j] = np.mean(window)

            # block of interest (QRS)
            z = np.mean(Xsq)
            thr1 = Beta * z
            block = np.zeros(len(MAqrs))
            block[MAqrs > thr1] = 1

            # 擷取出 block 對應的 X
            Xblock = np.full(len(X), np.nan)
            Xblock[block == 1] = X[block == 1]

            # 找Xblock前的第一個波谷為 Q
            XbeforeR = np.flip(Xblock[:Sf+1])
            Qloc = _findvally(XbeforeR)

            # 找Xblock後的第一個波谷為 S
            XafterR = Xblock[Sf:]
            Sloc = _findvally(XafterR)
            
            if(FalseR):
                # False R Detection
                if Qloc is None and Sloc is None:
                    false_R_flag = True
                    Ridx[i] = np.nan
        else:
            Qloc = None
            Sloc = None

        if Qloc:
            Qidx[i] = idx - Qloc
        if Sloc:
            Sidx[i] = idx + Sloc

    # 找出打錯的R，並把連帶影響的元素刪除
    Beats = np.array(Beats)
    false_R = np.argwhere(np.isnan(Ridx))
    if len(false_R)!=0:
        Ridx = np.delete(Ridx, false_R)
        Qidx = np.delete(Qidx, false_R)
        Sidx = np.delete(Sidx, false_R)
        Beats = np.delete(Beats, false_R, 0)
    
    ##### 計算 Beats 相似性並分群 ######
    
    # 計算相關係數
    th = 0.9
    if len(Beats) > 1:
        corr = np.corrcoef(Beats)
    else:
        corr = []

    # 利用相關係數分群
    Relation = _pattern_clustering(corr, th)

    # 標記不合群的 Beats
    igno_Ridx = []
    if len(Relation) > 1:
        Len = np.zeros(len(Relation))
        for i in range(0, len(Relation)):
            Len[i] = len(Relation[i])
        Per = Len / sum(Len)

        for c in np.argwhere(Per<0.1):
            ignore_list = Relation[int(c)]
            igno_Ridx.append(Ridx[ignore_list][0])
    
    
    # 計算 RRI & HR
    RRIs = np.full(Ridx.shape, np.nan)
    RRIs[1:] = np.diff(Ridx)
    avgRRI = np.nanmean(RRIs)

    # Paper參數 from:
    # Fast T Wave Detection Calibrated by Clinical Knowledge with Annotation of P and T Waves
    P1 = 0.07 * avgRRI
    P2 = 0.14 * avgRRI
    Dmin = 0.17
    #Dmax = 0.67
    Dmax = 0.5 # Based on MIT-BIH test
    RiTmin = np.ceil(Dmin * avgRRI)
    RiTmax = np.ceil(Dmax * avgRRI)

    ##### 尋找 T 波和 P 波 #####
    Tidx = np.full(Ridx.shape, np.nan)
    Pidx = np.full(Ridx.shape, np.nan)

    for i in range(0, len(Ridx)-1):
        idx1 = int(Ridx[i])
        idx2 = int(Ridx[i+1])

        '''
        if idx1 in igno_Ridx:
            continue
        '''

        # 避免漏打Rpeak造成誤判
        medRRI = np.nanmedian(RRIs)
        if idx2 - idx1 > 2*medRRI:
            continue

        # 擷取兩個 R 波之間的訊號
        XX = np.copy(ecg[idx1:idx2+1, 0])

        # 兩個R波太靠近，無法找T波和P波
        if len(XX) < P2:
            continue

        # 從當前S波到下一個Q波中尋找T波和P波
        # RS[T P]QR
        Sloc1 = Sidx[i] - Ridx[i]
        Qloc2 = Qidx[i+1] - Ridx[i+1]
        if ~np.isnan(Sloc1):
            XX[:int(Sloc1+1)] = np.nan
        if ~np.isnan(Qloc2):
            XX[int(-1+Qloc2):] = np.nan

        # P1 & P2 為尋找T波的兩個mean filter寬度，產生blocks of interest in Twave
        peaks = _find_waves(XX, P1, P2, "T")

        # 若T的位置落在paper定義的範圍內[RiTmin, RiTmax)，視為候選
        Tloc_candidate = []
        Tloc = []
        for j in peaks:
            Tloc = j[1]
            if RiTmax > Tloc > RiTmin:
                Tloc_candidate.append(j)
        Tloc_candidate = np.array(Tloc_candidate)

        if len(Tloc_candidate) > 0:
            # 從候選T中挑選強度最高的為Twave
            Imax = np.argmax(abs(Tloc_candidate[:,0]))
            Tloc = Tloc_candidate[Imax][1]
            Tidx[i] = Ridx[i] + Tloc

        if idx2+Sb <= len(ecg):
            # 在視窗的後1/3內尋找對應下個Rpeak的P波
            XX[:int(np.ceil(0.667*len(XX)))] = np.nan
            Ploc = []

            # 0.03*avgRRI, 0.06*avgRRI 為自定義的參數，參考P duration < 80ms
            Ppeaks = _find_waves(XX, 0.04*avgRRI, 0.08*avgRRI)
            if len(Ppeaks) > 0:
                # 將Ppeaks對應的wave起伏程度 / 對應R波的振幅 = Pscore , 以避免整個beat強度過小導致找不到Pwave
                if ~np.isnan(Qidx[i+1]) and ~np.isnan(Sidx[i+1]):
                    Qmag = abs(ecg[int(Qidx[i+1]), 0])
                    Rmag = abs(ecg[int(Ridx[i+1]), 0])
                    Smag = abs(ecg[int(Sidx[i+1]), 0])
                    maxMag = np.max([Qmag, Rmag, Smag])
                    Pscore = 100 * Ppeaks[:,2] / maxMag
                else:
                    Rmag = abs(ecg[int(Ridx[i+1]), 0])
                    Pscore = 100 * Ppeaks[:,2] / Rmag

                Ip = np.argmax(Pscore)
                Mp = np.max(Pscore)

                # 如果Pscore最大的位置為最後一個(最靠近下一個R波), 且大於自定義的閥值TH，視為Pwave
                th = 0.1 #0.2
                if Ip == len(Pscore)-1 and Mp > th:
                    Ploc = Ppeaks[Ip, 1]
                    # 避免T波跟P波重疊
                    if Ploc <= Tloc:
                        Ploc = []

            if Ploc:
                Pidx[i+1] = Ridx[i] + Ploc

    ##### OUTPUT ######
    avgHR = np.floor(60 * Fs / avgRRI)
    avgPR = _interval_calc(Pidx, Ridx, Fs)
    avgQRS = _interval_calc(Qidx, Sidx, Fs)
    QRSArray= _interval_calc_perbeat(Qidx, Sidx, Fs)  
    avgQT = _interval_calc(Qidx, Tidx, Fs)
    
    QTcs = np.full(Ridx.shape, np.nan)
    for i in range(0, len(Ridx)):
        Q = Qidx[i]
        T = Tidx[i]
        HR = 60 / (RRIs[i] / Fs)
        RR = 60 / HR
        if ~np.isnan(Q) and ~np.isnan(T):
            QT = (T - Q) / Fs
            if HR >= 60:
                QTc = QT / (RR ** 0.5)
            else:
                QTc = QT / (RR ** 0.33)

            QTcs[i] = QTc
    
    ###print(QTcs)        
    #if np.isnan(np.nanmean(QTcs)):
    if np.isnan(QTcs).all():
        avgQTc = np.nan
    else:
        avgQTc = round(1000 * np.nanmean(QTcs))

    Dict = {'Ridx':Ridx,
            'avgHR':avgHR,
            'avgPR':avgPR,
            'avgQRS':avgQRS,
            'QRSArray':QRSArray,   ###dennis add
            'avgQT':avgQT,
            'avgQTc':avgQTc,
            'good_quality':False if len(igno_Ridx)>0 else True
            }
    
    return Dict







