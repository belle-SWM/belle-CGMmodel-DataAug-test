import numpy as np
from scipy import signal
from multiprocessing import Process,Lock
import multiprocessing
import math
from ..common.filters import mean_filter
import neurokit2 as nk
from scipy.signal import find_peaks, butter, filtfilt


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

                signals, info = nk.ecg_peaks(clean_ecg, sampling_rate=250, correct_artifacts=False, method='neurokit')
                rpeaks = _check_r_peak(clean_ecg, info["ECG_R_Peaks"]) 
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