import numpy as np
from scipy import signal
from multiprocessing import Process,Lock
import multiprocessing
import math
from ..common.filters import mean_filter
import neurokit2 as nk
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd


def _find_max(indexs, values):
    value_max = -100000
    value_index_max = 0

    for index, value in zip(indexs, values):

        if value > value_max:
            value_max = value
            value_index_max = index

    return value_index_max, value_max


def rpeak_detection_bandpass(ecg, mode='original'): ###過去bandpass的方法找R波peak
    """
    input --- 
    ecg: 基線拉直後的ECG訊號
    mode: 'original': 一般模式(default value),
          'pvc': PVC偵測模式

    output ---
    1D numpy array(陣列第一個element是R波peak個數，第二element之後才是R波peak位置)
    -------
    """

    fs = 0.1  ##0.12
    fc = 0.2  ##0.3
    W1 = 27  ### ~=(35/360)*250,  35 is optimal parameter found by author in paper
    beta = 0.17
    length = len(ecg)
    ## --------butterworth band-pass filter-------------
    b, a = signal.butter(2, [fs, fc], 'bandpass')
    x = signal.filtfilt(b, a, ecg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # fit the result of Matlab

    ## -------square function----------
    y = np.multiply(x, x)

    SmoothedSig = np.zeros(length)
    SmoothedOffset = 10  ## 前後共看20的點
    for i in range(0, length):
        if i - SmoothedOffset >= 0 and i + SmoothedOffset < length:
            window = y[i - SmoothedOffset:i + SmoothedOffset + 1]
            SmoothedSig[i] = sum(window) / (2 * SmoothedOffset + 1)
        else:
            if i - SmoothedOffset < 0:
                window = y[:i + SmoothedOffset + 1]
            else:
                window = y[i - SmoothedOffset:]
            SmoothedSig[i] = sum(window) / len(window)
    meanValue = np.mean(SmoothedSig)
    Thr1 = meanValue * beta
    QRSLocationArray = np.where(SmoothedSig >= Thr1)[0]
    QRSCadidate_Count = len(QRSLocationArray)
    QRSLocationArray = np.append(QRSLocationArray, 0)

    ##-----------檢查波持續的寬度---------------
    Width = 0
    Thr2 = W1
    start_index = QRSLocationArray[0]
    end_index = 0
    RPeakArray = []
    RPeakHeightArray = []
    for i in range(1, QRSCadidate_Count + 1):  ##最後一筆資料後是0, 將其算入，如此最後一個R peak才會被算到
        if QRSLocationArray[i] - QRSLocationArray[i - 1] == 1:  ## 後減前只有差一，表示是連續的
            Width += 1
            end_index = QRSLocationArray[i]
        else:  ##沒有連續了，判斷之前連續了多少個點
            if Width >= Thr2:  ## 連續的點數通過門檻值，是R Peak，收集起來
                MaxIndex = np.argmax(ecg[start_index:end_index + 1])
                RPeakArray.append(start_index + MaxIndex)
                RPeakHeightArray.append(ecg[start_index + MaxIndex])  ## 收集濾波後的高度
            start_index = QRSLocationArray[i]
            Width = 0
    RPeakArray = np.asarray(RPeakArray)
    RPeakHeightArray = np.asarray(RPeakHeightArray)
    RPeak_Count = len(RPeakArray)

    # -----for PVC R peak detection----
    if mode == 'pvc':  # 針對PVC疾病偵測的模式
        if RPeak_Count == 0:
            DetectedRPeakArray = []
        else:
            DetectedRPeakArray = np.zeros(RPeak_Count + 1)
            DetectedRPeakArray[0] = RPeak_Count
            DetectedRPeakArray[1:] = RPeakArray

        return DetectedRPeakArray

    ##------------以下為heuristic方法，修正初步得到的R Peak------------------
    RPeakArray_Final = []
    if len(RPeakArray) > 0:
        sortedArray = np.sort(RPeakHeightArray)
        meanRPeakHeight = np.mean(sortedArray[RPeak_Count // 2:])  ## 取前二分之一較高的peak點的平均高度

        ##------------1. 修正R Peak, 太低的為雜訊過濾掉--------
        pass_idx = np.where(RPeakHeightArray >= meanRPeakHeight / 2.5)[0]
        RPeak_Count_Refined = len(pass_idx)
        RPeakArray_Refined = np.zeros((RPeak_Count_Refined, 3))  ## 1 放location, 2放高度，3放是否濾除的標示
        RPeakArray_Refined[:, 0] = RPeakArray[pass_idx]
        RPeakArray_Refined[:, 1] = RPeakHeightArray[pass_idx]

        ##----------2. 距離太近的點，排除之-----------
        ##計算R Peak點之間距離，若距離在62點(248ms)內，則進一步排除高度較低的點
        close_idx = np.where(np.diff(RPeakArray_Refined[:, 0]) <= 62)[0] + 1
        for k in close_idx:
            RPH_Diff = RPeakArray_Refined[k - 1][1] - RPeakArray_Refined[k][1]
            if RPH_Diff > 0:  ###前面較高
                RPeakArray_Refined[k][2] = 1  ###排除後面的
            else:
                RPeakArray_Refined[k - 1][2] = 1  ###排除前面的

        ##--------  3.一個一個檢查高度下降狀況，下降不夠快的排除之-------
        JumpOffset = 15  ##向外看15個點，看最低點高度是否小於此peak高度一半以下
        for k in range(0, RPeak_Count_Refined):
            nowLocation = int(RPeakArray_Refined[k][0])
            nowPeakHeight = ecg[nowLocation]
            minHeight = nowPeakHeight
            if nowPeakHeight > 150:  ##一般訊號
                SteepThr = nowPeakHeight / JumpOffset
            elif 150 >= nowPeakHeight > 50:  ##偏小的訊號
                SteepThr = 13
            else:  ##太小的訊號
                SteepThr = 7

            index = 0
            minIndex = 100000
            foundFlag = False
            for q in range(nowLocation + 1, nowLocation + JumpOffset + 1):  ##先檢查右邊
                if length > q >= 0:
                    index += 1
                    if minHeight > ecg[q]:
                        minHeight = ecg[q]
                        minIndex = index
                        Steep = (nowPeakHeight - minHeight) / minIndex
                        if Steep >= SteepThr:  ##確定有陡坡
                            foundFlag = True
                            break

            if foundFlag:  ##右邊有陡坡，再檢查左邊(使用陡坡比例)
                DeleteFlag = -1
                if nowPeakHeight > 150:  ##一般訊號
                    SteepThr = nowPeakHeight / JumpOffset
                elif 150 >= nowPeakHeight > 50:  ##訊號高度較小的
                    SteepThr = 7
                else:  ##訊號高度很小的
                    if nowPeakHeight >= meanRPeakHeight / 2.5:  ###高於平均值一半左右，直接過
                        SteepThr = 0
                        DeleteFlag = 0
                    else:
                        SteepThr = 50  ###太低的，直接封殺
                        DeleteFlag = 1

                foundFlag = False
                if DeleteFlag == 0:  ##50以下，高於平均，直接過
                    foundFlag = True
                elif DeleteFlag == 1:  ##50以下，又低於平均，直接封殺
                    foundFlag = False
                else:  ##50以上，要一個一個再檢查左邊下降狀況
                    minHeight = nowPeakHeight
                    index = 0
                    minIndex = 100000
                    foundFlag = False
                    for q in reversed(range(nowLocation - JumpOffset, nowLocation)):
                        if length > q >= 0:
                            index += 1
                            if minHeight > ecg[q]:
                                minHeight = ecg[q]
                                minIndex = index
                                Steep = (nowPeakHeight - minHeight) / minIndex
                                if Steep >= SteepThr:  ##找到陡坡了
                                    foundFlag = True
                                    break

                if not foundFlag:  ##沒有通過檢查
                    RPeakArray_Refined[k][2] = 1  ##設定排除之
            else:  ##右邊下降速度不夠快，排除之
                RPeakArray_Refined[k][2] = 1  ##設定排除之

            ##--------------最後收集沒有被過濾的candidate R Peak,作為最後結果輸出---------------
            if RPeakArray_Refined[k][2] == 0:
                RPeakArray_Final.append(RPeakArray_Refined[k][0])

    RPeak_Count_Final = len(RPeakArray_Final)
    if (RPeak_Count_Final == 0):
        DetectedRPeakArray = []
    else:
        DetectedRPeakArray = np.zeros(RPeak_Count_Final + 1)
        DetectedRPeakArray[0] = RPeak_Count_Final
        DetectedRPeakArray[1:] = RPeakArray_Final

    return DetectedRPeakArray


def rpeak_detection_patch(ecg): ###針對patch模式的ECG
    """
    input ---
        ecg: 基線拉直後的ECG訊號 
    
    output ---  
    1D numpy array(陣列第一個element是R波peak個數，第二element之後才是R波peak位置)
    -------
    """

    fs = 250
    lowcut = 15
    highcut = 37.5
    beta = 0.4
    Thr2 = 27
    ResponseThr = 50000
    JumpOffset = 15

    length = len(ecg)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    TwiceResponseThr = ResponseThr * 2

    ### 如果訊號過少 回傳空值
    if length < 12:
        return []

    ### 訊號前處理(平滑)
    ecg = mean_filter(ecg, 1)

    ### 訊號前處理(放大)
    ecg = np.array(ecg)
    ecg_max = max(ecg)

    if ecg_max <= 50:
        ecg = ecg * 4

    elif ecg_max <= 100:
        ecg = ecg * 3

    ### 帶通濾波並平方
    b, a = signal.butter(2, [low, high], 'bandpass')
    x = signal.filtfilt(b, a, ecg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    y = np.multiply(x, x)

    ### 平滑化並取平均
    SmoothedSig = mean_filter(y, 21)
    meanValue = np.nanmean(SmoothedSig)

    ### 篩選出足夠寬大的頻帶並找出尖峰
    Thr1 = meanValue * beta

    rpeaks_index = []
    rpeaks_height = []

    Width = 0

    for i, signal_value in enumerate(SmoothedSig):

        if signal_value > Thr1:

            Width += 1

        else:

            if Width >= Thr2:

                ### 篩選出前後3、6與9點有最大斜率的尖峰（貼片的Q或S波會出現過深的情況）
                slope_max = 0
                slope_index = 0

                for i_peak in range(i - Width, i):

                    if i_peak - 9 < 0:
                        left_slope_max = 0

                    else:
                        left_slope_max = max([ecg[i_peak] - ecg[i_peak + j] for j in range(-9, 0, 3)])

                    if i_peak + 9 >= length:
                        right_slope_max = 0

                    else:
                        right_slope_max = max([ecg[i_peak] - ecg[i_peak + j] for j in range(3, 12, 3)])

                    slope_max_temp = left_slope_max * 0.4 + right_slope_max * 0.6

                    if slope_max_temp > slope_max:
                        slope_max = slope_max_temp
                        slope_index = i_peak

                max_value_i, max_value = _find_max(range(slope_index - 3, slope_index + 4),
                                                   ecg[slope_index - 3:slope_index + 4])

                rpeaks_index.append(max_value_i)
                rpeaks_height.append(max_value)

            Width = 0

    ### 

    rpeaks_length = len(rpeaks_index)

    if rpeaks_length > 1:

        # 將高度全數轉正
        if np.nanmin(rpeaks_height) < 0:
            rpeaks_height = np.array(rpeaks_height)
            rpeaks_height = rpeaks_height - np.nanmin(ecg)

        # 取相對高度較高的
        rpeak_mean_height_thr = np.nanmean(rpeaks_height) / 2.5

        new_rpeaks_index = []
        new_rpeaks_height = []

        for rpeak_index, rpeak_height in zip(rpeaks_index, rpeaks_height):

            if rpeak_mean_height_thr <= rpeak_height and y[rpeak_index] <= TwiceResponseThr:
                new_rpeaks_index.append(rpeak_index)
                new_rpeaks_height.append(rpeak_height)

        ### 篩選過近的
        near_rpeaks_index = []
        near_rpeaks_height = []

        new_rpeaks_length = len(new_rpeaks_height)
        near_rpeaks_delete = np.zeros(new_rpeaks_length)
        rpeak_index_diff = np.diff(new_rpeaks_index)

        for i in range(new_rpeaks_length - 1):

            if abs(rpeak_index_diff[i]) < 45:

                if new_rpeaks_height[i] >= new_rpeaks_height[i + 1]:

                    near_rpeaks_delete[i + 1] = 1

                else:

                    near_rpeaks_delete[i] = 1

        for rpeak_index, rpeak_height, rpeak_delete in zip(new_rpeaks_index, new_rpeaks_height, near_rpeaks_delete):

            if rpeak_delete == 0:
                near_rpeaks_index.append(rpeak_index)
                near_rpeaks_height.append(rpeak_height)

        ### 篩選足夠尖的 前後看15個點 計算斜率夠大的次數
        sharp_rpeaks_index = []

        for rpeak_index, rpeak_height in zip(near_rpeaks_index, near_rpeaks_height):

            if rpeak_height <= 50:
                RightSteepThr = 7
                LeftSteepThr = 5

            elif 50 < rpeak_height <= 150:
                RightSteepThr = 13
                LeftSteepThr = 7

            else:
                RightSteepThr = (rpeak_height / JumpOffset) * 0.70
                LeftSteepThr = (rpeak_height / JumpOffset) * 0.70

            LeftEndIndex = rpeak_index - JumpOffset
            RightEndIndex = rpeak_index + JumpOffset

            if LeftEndIndex < 0:
                LeftEndIndex = 0

            if RightEndIndex >= length:
                RightEndIndex = length

            left_ecg_diff = np.diff(ecg[LeftEndIndex:rpeak_index])
            right_ecg_diff = np.diff(ecg[rpeak_index:RightEndIndex]) * -1

            SharpCount = 0

            for ecg_diff_i in right_ecg_diff:

                if ecg_diff_i > RightSteepThr:
                    SharpCount += 1

                if ecg_diff_i > 10 * RightSteepThr:
                    SharpCount += 2

            if SharpCount >= 3:

                if LeftSteepThr == 5:
                    sharp_rpeaks_index.append(rpeak_index)
                    continue

                SharpCount = 0

                for ecg_diff_i in left_ecg_diff:

                    if ecg_diff_i > LeftSteepThr:
                        SharpCount += 1

                    if ecg_diff_i > 10 * LeftSteepThr:
                        SharpCount += 2

                if SharpCount >= 2:
                    sharp_rpeaks_index.append(rpeak_index)

        sharp_rpeaks_index.insert(0, len(sharp_rpeaks_index))

        return np.array(sharp_rpeaks_index)

    else:

        return np.array([])

####------多執行續------------
def _rpeak_detection_func_mp(processnum, ecg, measuring_mode, method_type, mode, pindex, ns, lock):
    datalength = len(ecg)
    segmentnum = math.floor(datalength / 2500)  ###多少個完整10秒片段
    block_length = math.ceil(segmentnum / processnum) * 2500  ###每個processor分得資料長度,2500之倍數
    startindex = pindex * block_length + 2500
    endindex = startindex - 2500 + block_length

    if (endindex > len(ecg)):
        endindex = len(ecg)

    lastindex = 0
    for r in range(startindex, endindex + 1, 2500):
        lastindex = r
        nowEcg = np.array(ecg[r - 2500:r])        

        lock.acquire()
        ##Ridx = rpeak_detection(nowEcg)
        ##Ridx = _rpeak_detection_single(ecg_data=nowEcg,measuring_mode=measuring_mode,method_type=method_type,mode=mode) 
        Ridx= rpeak_detection(nowEcg,measuring_mode, method_type, mode)
        lock.release()

        if (len(Ridx) >= 2):
            ns.rpeakarray = np.concatenate((ns.rpeakarray, Ridx[1:] + (r - 2500)))

    if (endindex - lastindex) >= 1:  ##剩下片段
        nowEcg = np.array(ecg[lastindex + 1:endindex])        
        
        lock.acquire()
        ##Ridx = _rpeak_detection_single(ecg_data=nowEcg,measuring_mode=measuring_mode,method_type=method_type,mode=mode) 
        Ridx = rpeak_detection(nowEcg,measuring_mode, method_type, mode)
        lock.release()
        if (len(Ridx) >= 2):
            ns.rpeakarray = np.concatenate((ns.rpeakarray, Ridx[1:] + lastindex + 1))

'''
def rpeak_detection_mp(data, processnum,measuring_mode='strap', method_type='vg', mode='revised'):  ### R peak detection 多核心平行處理
    """
    input ---
        data: 基線拉直後的ECG訊號
        processnumber: 使用的CPU核心數
        measuring_mode: 配戴量測方式('strap' or 'patch')
        method_type: 偵測方法('vg' or 'bandpass')
        mode: 處理模式('original','revised', or 'pvc') 
              original為一般偵測模式
              reivsed 為偵測後會將位置修正到peak尖點，此為vg才有的模式
              注意若method_type為bandpass, 才會有pvc模式
    output ---
        1D numpy array(陣列第一個element是R波peak個數，第二element之後才是R波peak位置)
    """

    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.rpeakarray = np.array([], dtype="int32")
    lock = Lock()

    processes = [Process(target=_rpeak_detection_func_mp, args=(processnum,data,measuring_mode,method_type,mode,pindex,ns,lock)) for
                 pindex in range(processnum)]
    ## start all processes
    for process in processes:
        process.start()

    ## wait for all processes to complete
    for process in processes:
        process.join()

    if len(ns.rpeakarray) > 0:
        ns.rpeakarray = np.sort((ns.rpeakarray))

    return ns.rpeakarray
'''

def rpeak_detection_mp(ecg,processnum=1,measuring_mode='strap', method_type='vg', mode='revised'):  #### r peak detection 多核心平行處理
    
    """
    input ---
        data: 基線拉直後的ECG訊號
        processnumber: 使用的CPU核心數
        measuring_mode: 配戴量測方式('strap' or 'patch')
        method_type: 偵測方法('vg' or 'bandpass')
        mode: 處理模式('original','revised', or 'pvc') 
              original為一般偵測模式
              reivsed 為偵測後會將位置修正到peak尖點，此為vg才有的模式
              注意若method_type為bandpass, 才會有pvc模式
    output ---
        1D numpy array(陣列第一個element是R波peak個數，第二element之後才是R波peak位置)
    """

    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.rpeakarray = np.array([], dtype="int32")

    if(processnum==1):
        ##ns.rpeakarray=_rpeak_detection_single(ecg_data=ecg,measuring_mode=measuring_mode,method_type=method_type,mode=mode)
        ns.rpeakarray=rpeak_detection(ecg_data=ecg,measuring_mode=measuring_mode,method_type=method_type,mode=mode)
    else:
        lock = Lock()

        ##processes = [Process(target=_rpeak_detection_func_mp, args=(processnum,data,measuring_mode,method_type,mode,pindex,ns,lock)) for pindex in range(processnum)]
        processes=[]
        for pindex in range(processnum):
            processes.append(multiprocessing.Process(target=_rpeak_detection_func_mp,args=(processnum,ecg,measuring_mode,method_type,mode,pindex,ns,lock),daemon=False))

        ## start all processes
        for process in processes:
            process.start()

        ## wait for all processes to complete
        for process in processes:
            process.join()

        if len(ns.rpeakarray) > 0:
            ns.rpeakarray = np.sort((ns.rpeakarray))

    return ns.rpeakarray

def _missing_r_peaks(clear_ecg, nk_rpeak):
    rri = np.diff(nk_rpeak)
    average_rri = np.mean(rri)
    std_rri = np.std(rri)
    threshold = average_rri + std_rri
    miss_threshold = 0.5*threshold + 1.5*std_rri
    large_interval = np.where(rri > threshold)[0]

    missing_rpeaks = []

    for i in large_interval:
        start = nk_rpeak[i]
        end = nk_rpeak[i + 1]
        seg = clear_ecg[start:end]
        p, _ = find_peaks(seg, height=np.max(seg)*0.25)
        if len(p) > 0:
            max_peak_index = p[np.argmax(seg[p])]
            if (max_peak_index > miss_threshold) and ((end - start) - max_peak_index > miss_threshold):
                missing_rpeaks.append(start + max_peak_index)

    rev_rpeaks = sorted(nk_rpeak + missing_rpeaks)
    return rev_rpeaks


def _check_r_peak(ecg_data, rpeaks):
    updated_rpeaks = []
    gap = 30
    for rpeak in rpeaks:
        rpeak = int(rpeak)
        start = max(0, rpeak - gap)
        end = min(len(ecg_data), rpeak + gap)
        ecg_subset = ecg_data[start:end]

        tmp = max(ecg_subset)
        index = list(ecg_subset).index(tmp)
        peak_index = index + start
        if peak_index != rpeak:
            rpeak = peak_index
        updated_rpeaks.append(rpeak)

    return updated_rpeaks

def rpeak_detection(ecg_data,measuring_mode='strap',method_type='vg',mode='revised'): ### R peak detection 單核處理  
##def _rpeak_detection_single(ecg_data,measuring_mode='strap',method_type='vg',mode='revised'): ### R peak detection 單核處理
    
    """
    input ---
        ecg_data:ecg signal
        measuring_mode: 設定配戴方式('strap' or 'patch')
        method_type: 設定使用方法('vg' or 'bandpass')
        mode: 處理模式('original','revised', or 'pvc') 注意若method_type為bandpass, 才會有pvc模式
    output ---
        1D numpy array(陣列第一個element是R波peak個數，第二element之後才是R波peak位置)
    """
    
    rpeaks=np.array([])
    if(method_type=='vg' and mode=='pvc'):
        print('R peak detection automatically changes to bandpass method because vg method has no pvc detection mode!')
        method_type='bandpass'
  
    if(measuring_mode=='strap'):
        if(method_type=='vg'):
            clean_ecg = nk.ecg_clean(ecg_data, sampling_rate=250, method='nk')
            clean_ecg = nk.ecg_clean(clean_ecg, sampling_rate=250, method='vg')
            max_amp=max(clean_ecg)
            min_amp=min(clean_ecg)
            diff=max_amp-min_amp
            scale=1
            
            if(diff>0): ###不是直線，才可偵測R波
                
                if(max_amp>151):
                    scale=1
                elif(max_amp<=150 and max_amp>101):
                    scale=2      
                elif(max_amp<=100 and max_amp>51):
                    scale=3
                elif(max_amp<=50):                  
                    scale=6

                clean_ecg=clean_ecg*scale               
                try:                    
                    signals, info = nk.ecg_peaks(clean_ecg, sampling_rate=250, correct_artifacts=False, method='neurokit')
                    rpeaks = _check_r_peak(clean_ecg, info["ECG_R_Peaks"])
                except:
                    rpeaks=rpeak_detection_bandpass(ecg_data,mode)      
                    rpeaks=rpeaks[1:]
              
            else:
                rpeaks=np.array([])

            if(mode=='revised'):
                rpeaks = _missing_r_peaks(clean_ecg, rpeaks)
            
            rpeaks_final=np.zeros(len(rpeaks)+1,dtype=int)
            rpeaks_final[0]=len(rpeaks)
            rpeaks_final[1:]=rpeaks            
        else:  ###之前bandpass的方法
            rpeaks_final=rpeak_detection_bandpass(ecg_data,mode)
             
    else: ### patch mode
        rpeaks_final=rpeak_detection_patch(ecg_data)
    
    return rpeaks_final


def _phasor_trans(cleaned: np.ndarray, Rv: float):
    """
    input ---
        cleaned: ECG signal after cleaned
        Rv: the constant in phasor transform formula, smaller value will results more sensitive to wide peaks
    output ---
        pt: phasor transformed array
    """
    pt=np.array([math.atan(i/Rv) for i in cleaned])
    return(pt)

def _peak_widths_heights(cleaned:np.ndarray, peaks:np.ndarray, mode:str='slope', slope_c:float=0.9):
    """
    for each peak, find its left and right edge by selected mode, then return widths, height, and height/width ratio
    input --
        cleaned: ECG signal after cleaned
        peaks: detected peak array
        mode: 'slope' or not
        slope_c: slope constant for define peak edge
    output --
        widths: widths array of peak array
        heights: height array of peak array
        ratio: height/width ratio array of peak array
    """

    widths=np.zeros_like(peaks)+1
    heights_l=np.zeros_like(peaks, dtype=float)
    heights_r=np.zeros_like(peaks, dtype=float)

    ### peak ends at any slope not steep anymore
    if mode == 'slope':
        for i, peak in enumerate(peaks):
            ## find left edge
            x=peak-1
            while x>0:
                if (cleaned[x]-cleaned[x-1])>=(cleaned[x+1]-cleaned[x])*slope_c:
                    widths[i]+=1
                    x-=1

                else:
                    break
            
            heights_l[i]=cleaned[peak]-cleaned[x]
            
            ## find right edge
            x=peak+1
            while x <len(cleaned)-1:
                if (cleaned[x]-cleaned[x+1])>=(cleaned[x-1]-cleaned[x])*slope_c:
                    widths[i]+=1
                    x+=1
                else:
                    break

            heights_r[i]=cleaned[peak]-cleaned[x]

        heights=np.max([heights_l, heights_r], axis=0)

    ### peak ends at any slope start to raise
    else:
        for i, peak in enumerate(peaks):
            ## find left edge
            x=peak-1
            while x>0:
                if cleaned[x+1]>=cleaned[x]:
                    widths[i]+=1
                    x-=1

                else:
                    break

            heights_l[i]=np.round(cleaned[peak]-cleaned[x], 3)
            
            ## find right edge
            x=peak+1
            while x <len(cleaned):
                if cleaned[x-1]>=cleaned[x]:
                    widths[i]+=1
                    x+=1

                else:
                    break

            heights_r[i]=np.round(cleaned[peak]-cleaned[x], 3)

        heights=np.max([heights_l, heights_r], axis=0)
    
    ratio=np.round(heights/widths*100, 3)

    return ratio


def _do_pt_thres(cleaned, pt, pt_thres, sampling_rate, step=0.2):
    """
    input --
        cleaned: ECG signal after cleaned
        pt: phasor transformed array
        pt_thres: threshold for determining peak range
        sampling_rate: sampling rate of ECG signal
        step: pt_thres will decrease step each loop
    output --
        peaks: detected peak array
    """
    end=[]        
    length=len(cleaned)
    while (len(end)<=int(length/sampling_rate/1.5)) & (pt_thres>0.5):
        b=pt>=pt_thres
        start=np.where(np.diff([int(i) for i in b])==1)[0]
        end=np.where(np.diff([int(i) for i in b])==-1)[0]

        if len(end):
            if end[0]<start[0]:
                end=end[1:]

            if len(end)==0:
                break

        if len(start):
            if len(start)>len(end):
                start=start[:len(end)]

            if len(start)==0:
                break

        pt_thres-=step
    
    if len(end):
        ### peak is max of every (start, end) interval 
        peaks=np.array([np.argmax(cleaned[start[i]:end[i]]) for i in range(len(end))])+start
        return(peaks)
    else:
        return([])


def _do_pt_thres_r(cleaned, cleaned_r, pt_r, pt_thres, sampling_rate, step=0.2, mode:str='slope', slope_c:float=0.9):
    """
    input ---
        cleaned: ECG signal after cleaned
        cleaned_r: negative ECG signal after cleaned
        pt_r: negative phasor transformed array
        pt_thres: threshold for determining peak range
        sampling_rate: sampling rate of ECG signal
        step: pt_thres will decrease step each loop
        mode: 'slope' or not
        slope_c: slope constant for define peak edge
    output ---
        min_p: detected valley array
        max_p_r: peak array that reversed from vallet 
    """
    min_p = _do_pt_thres(cleaned_r, pt_r, pt_thres, sampling_rate, step)
    
    if len(min_p):
        max_p_r=_reversed_peaks(cleaned, min_p, mode, slope_c)
        return(min_p, max_p_r)
    
    else:
        return([], [])

        
def _reversed_peaks(cleaned, min_p, mode:str='slope', slope_c:float=0.9):
    """
    input ---
        cleaned: ECG signal after cleaned
        min_p: detected valley array
        mode: 'slope' or not
        slope_c: slope constant for define peak edge
    output ---
        max_p_r: peak array that reversed from vallet 
    """
    
    ### peak is at any slope not steep anymore
    if mode == 'slope':
        max_p_r=np.zeros_like(min_p)
        for i, p in enumerate(min_p):
            ## find peak left to valley 
            x=p-1
            while x>0:
                if (((cleaned[x]-cleaned[x-1])>=(cleaned[x+1]-cleaned[x])*slope_c) or (cleaned[x-1]-cleaned[x]>2)):
                    x-=1

                else:
                    max_p_r[i]=x
                    break

        return(max_p_r.astype('int'))
    
    ### peak is at any slope start to decrease
    else:
        max_p_r=np.zeros_like(min_p)
        for i, p in enumerate(min_p):
            ## find peak left to valley 
            x=p-1
            while x>0:
                if cleaned[x+1]>=cleaned[x]:
                    x-=1

                else:
                    max_p_r[i]=x
                    break

        return(max_p_r.astype('int'))
   
def _ostu_filtering(peaks, min_peak, max_peak, sorted_index, sorted_ratio):
    """
    input ---
        peaks: detected peak array
        min_peak: min numbers of peak
        max_peak: max numbers of peak
        sorted_index: argsort(ratio) where ratio is height/weight ratio array
        sorted_ratio: sort(ratio) where ratio is height/weight ratio array
    output ---
        Rpeaks: filtered Rpeaks array
    """
    start_idx=np.max([len(peaks)-max_peak, 0])
    end_idx=len(peaks)-min_peak
    std_array=np.zeros(end_idx-start_idx)

    for r in range(start_idx, end_idx):
        i=r-start_idx
        if r<=2:
            std_array[i]=np.std(sorted_ratio[r:], ddof=1)

        else:
            std_array[i]=np.std(sorted_ratio[:r], ddof=1)+np.std(sorted_ratio[r:], ddof=1)

    ratio_divide = np.argmin(std_array)+start_idx
    Rpeaks=np.sort(peaks[sorted_index[ratio_divide:]])
    return Rpeaks

def _standard_rri_filtering(peaks, length, min_peak, max_peak, sorted_index):
    """
    input ---
        peaks: detected peak array
        length: length of ECG array
        min_peak: min numbers of peak
        max_peak: max numbers of peak
        sorted_index: argsort(ratio) where ratio is height/weight ratio array
    output ---
        Rpeaks: filtered Rpeaks array
    """

    start_idx=np.max([len(peaks)-max_peak, 0])
    end_idx=len(peaks)-min_peak
    rri_dist=np.zeros(end_idx-start_idx)

    for r in range(start_idx, end_idx):
        i=r-start_idx
        g=np.sort(peaks[sorted_index[r:]])
        rri=np.diff(g)

        ## temporary standard rri in order to insert missing R
        standard_rri=length/len(g)

        ## consider as missed first R
        if g[0]>2*standard_rri:
            g=np.insert(g, 0, 0)

        ## consider as missed last R
        if length-g[-1]>2*standard_rri:
            g=np.insert(g, -1, length)

        ## actual standard rri
        standard_rri=(g[-1]-g[0])/(len(g)-1)
        rri_dist[i]=(np.sum(np.abs(rri-standard_rri))/len(g)/standard_rri)

    ratio_divide = np.argmin(rri_dist)+start_idx
    Rpeaks=np.sort(peaks[sorted_index[ratio_divide:]])
    return Rpeaks

def _filtering(filter, peaks, length, sorted_index, sorted_ratio, sampling_rate, ratio, cleaned):
    """
    input ---
        filter: selected filter type
        peaks: detected peak array
        length: length of ECG array
        sorted_index: argsort(ratio) where ratio is height/weight ratio array
        sorted_ratio: sort(ratio) where ratio is height/weight ratio array
        sampling_rate: sampling rate of ECG signal
        ratio: height/width ratio array of peak array
        cleaned: ECG signal after cleaned
    output ---
        Rpeaks: filtered Rpeaks array
    """
    min_peak=int(length//sampling_rate/60*40)
    max_peak=int(length//sampling_rate/60*240)

    if filter=='no' or len(peaks)<=min_peak:
        Rpeaks=peaks

    elif filter=='ostu':
        Rpeaks=_ostu_filtering(peaks, min_peak, max_peak, sorted_index, sorted_ratio)

    elif filter=='standard_rri':
        Rpeaks=_standard_rri_filtering(peaks, length, min_peak, max_peak, sorted_index)

    elif filter=='mix':
        Rpeaks=_ostu_filtering(peaks, min_peak, max_peak, sorted_index, sorted_ratio)
        if len(Rpeaks)==len(peaks):
            Rpeaks=_standard_rri_filtering(peaks, length, min_peak, max_peak, sorted_index)
    
    while sum(np.diff(Rpeaks)<sampling_rate//4):
        mask=np.ones_like(Rpeaks)

        ## all rri <1/4s =<0.25s
        for i in np.where(np.diff(Rpeaks)<sampling_rate//4)[0]:
            ## if two ratio show huge difference
            if abs(ratio[i]-ratio[i+1])/(ratio[i]+ratio[i+1])<0.2:
                ## compare their actual ECG value and keep the higher peak
                rrr=cleaned[Rpeaks[i:i+2]]

            else:
                ## else compare their ratio and keep the steeper peak
                rrr=ratio[i:i+2]

            mask[i+np.argmin(rrr)]=0

        Rpeaks=np.array([Rpeaks[i] for i in range(len(Rpeaks)) if mask[i]==1])
    
    return Rpeaks

def rpeak_detection_pt(signals, sampling_rate=250, crop_time=10, overlap_time=1, Rv=0.1, pt_thres=0.9, mode='slope', slope_c=0.9, filter='mix'):
    """
    input ---
        signals: raw ECG array
        sampling_rate: sampling rate of ECG signal
        crop_time: crop signal to be one with length of crop_time
        overlap_time: overlapping time for both margins of cropped segment
        Rv: the constant in phasor transform formula, smaller value will results more sensitive to wide peaks
        pt_thres: threshold for determining peak interval
        mode: used to check the starting and end point of R peak wave. Two modes: 'slope' or not
        slope_c: slope constant for mode 'slope'
        filter: selected filter type
    output ---
        allR: final Rpeaks array corresponding to raw ECG signals
    """
    cleaned=nk.ecg_clean(signals, sampling_rate=sampling_rate, method='neurokit')
    pt=_phasor_trans(cleaned, Rv)
    segment_num=int(np.floor(len(signals)/crop_time//sampling_rate))
    allR=[]

    for c in range(segment_num+1):
        (s, e) = (sampling_rate*crop_time*c, sampling_rate*crop_time*(c+1))
        if (c==segment_num):
            if (len(signals)>sampling_rate*crop_time*segment_num):
                e=len(signals)
            else:
                break
        
        ## 邊界放寬overlap_time秒，目的是偵測邊界上的Rpeak
        s_window=np.max([s-sampling_rate*overlap_time, 0])
        e_window=np.min([e+sampling_rate*overlap_time, len(signals)])
        length=int((e_window-s_window))
        
        seg_cleaned=cleaned[s_window:e_window]
        seg_pt=pt[s_window:e_window]
        max_p=_do_pt_thres(seg_cleaned, seg_pt, pt_thres, sampling_rate)
        
        seg_cleaned_r=-seg_cleaned
        seg_pt_r=-seg_pt
        (min_p, max_p_r)=_do_pt_thres_r(seg_cleaned, seg_cleaned_r, seg_pt_r, pt_thres, sampling_rate, mode=mode, slope_c=slope_c)

        ratio_r=_peak_widths_heights(seg_cleaned, max_p, mode=mode, slope_c=slope_c)
        ratio_l=_peak_widths_heights(seg_cleaned_r, min_p, mode=mode, slope_c=slope_c)

        peaks=np.sort(np.concatenate((max_p, max_p_r)))
        sorted_signal=np.argsort(np.concatenate((max_p_r, max_p)))
        ratio=np.concatenate((ratio_l, ratio_r))[sorted_signal]

        ## two too closed R peaks should be same peak
        while sum(np.diff(peaks)<5):
            mask=np.ones_like(peaks)
            for i in np.where(np.diff(peaks)<sampling_rate//4)[0]:
                if abs(ratio[i]-ratio[i+1])/(ratio[i]+ratio[i+1])<0.2:
                    rrr=seg_cleaned[peaks[i:i+2]]

                else:
                    rrr=ratio[i:i+2]

                mask[i+np.argmin(rrr)]=0

            peaks=np.array([peaks[i] for i in range(len(peaks)) if mask[i]==1])
            ratio=np.array([ratio[i] for i in range(len(ratio)) if mask[i]==1])
        
        sorted_ratio=np.sort(ratio)
        sorted_index=np.argsort(ratio)
        
        Rpeaks=_filtering(filter, peaks, length, sorted_index, sorted_ratio, sampling_rate, ratio, seg_cleaned)
          
        Rpeaks=Rpeaks[(Rpeaks>-s_window+s) * (Rpeaks<e-e_window+length)]
        allR.extend(list(Rpeaks+s_window))

    return(allR)   


def predictDataset():
    ## input path
    save_root=r'res'
    '''
    dataset='MIT-BIH'
    signal_path=r"MIT-BIH\Processed\MIT-BIH-A-*-Sample.txt"
    sample_path=r"MIT-BIH\Processed\MIT-BIH-A-*-Sample.txt"
    symbol_path=r"MIT-BIH\Processed\MIT-BIH-A-*-Symbol.txt"
    '''
    '''
    dataset='NST'
    signal_path=r"NST\Processed\MIT-BIH-A-*-Signal.txt"
    sample_path=r"NST\Processed\MIT-BIH-A-*-Sample.txt"
    symbol_path=r"NST\Processed\MIT-BIH-A-*-Symbol.txt"
    '''
    
    dataset='AHA'
    signal_path=r"AHA\Processed\ECG_1_*.txt"
    sample_path=r"AHA\Processed\*-Sample.txt"
    symbol_path=None

    # can change parameters
    filter='mix'    # ['no', 'ostu', 'standard_rri']
    crop_time=10
    peak_tolerance_sec=0.1
    Rv=0.1            # larger alpha will results insensitive to wide peaks
    pt_thres=np.pi/2-0.6    # higher threshold will results insensitive to wide peaks
    mode='slope'
    slope_c=0.9

    ## no need to change
    import os
    from glob import glob
    import json

    if dataset=='MIT-BIH':
        sampling_rate=360
        fn_slice_s=10
        fn_slice_e=-11
    elif dataset=='NST':
        sampling_rate=360
        fn_slice_s=10
        fn_slice_e=-11
    elif dataset=='AHA':
        sampling_rate=360
        fn_slice_s=6
        fn_slice_e=10

    print('Input database: ', dataset)
    print('Save result in: ', os.path.join(os.getcwd(), save_root))

    signal_files = sorted(glob(signal_path))
    if sample_path is not None:
        sample_files = sorted(glob(sample_path))
    if symbol_path is not None:
        symbol_files = sorted(glob(symbol_path))

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for idx in range(len(signal_files)):

        file=signal_files[idx]
        print('Processing', os.path.basename(file))
        signals=np.array(pd.read_csv(signal_files[idx], header=None)).ravel()
        if sample_path is not None:
            assert len(sample_files) == len(signal_files), 'not corresponding sample file'
            sample=np.array(pd.read_csv(sample_files[idx], header=None)).ravel()
        if symbol_path is not None:
            assert len(symbol_files) == len(signal_files), 'not corresponding symbol file'
            symbol=np.array(pd.read_csv(symbol_files[idx], header=None, quotechar="'")).ravel()
            ignore=np.array([i in ['Q', '+', 'x', '[', ']', 'p', '(N', '(P', '(B', '(VT', '(T', '(SVTA', '(IVR', '(NOD', '(AFIB', '(AFL', '(VFL', '(AB', '(PREX', '(BII', '(SBR', '|', '~', '"'] for i in symbol])
            sample=sample[~ignore]
        fn=os.path.basename(signal_files[idx])[fn_slice_s:fn_slice_e]
        p=rpeak_detection_pt(signals, sampling_rate, crop_time, Rv, pt_thres, mode, slope_c, 'mix')
        d={
            "dataset": dataset, 
            "signal_path": file, 
            "peak_tolerance_sec": peak_tolerance_sec, 
            "Rv": Rv, 
            "pt_thres": pt_thres, 
            "crop_time": crop_time,
            "find_peak_mode": mode, 
            "slope_constant": slope_c,
            "r_peaks": p
        }
        with open(os.path.join(save_root, fn+'.json'), mode='w') as f:
                f.write(json.dumps(d, indent=4, default=str))
    return 0