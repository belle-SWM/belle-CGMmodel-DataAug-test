
import os
import sys
import numpy as np
import pandas as pd
##import ujson as json
##from scipy import signal
###from matplotlib.patches import Ellipse
from datetime import date, datetime, timedelta
from multiprocessing import Process,Lock
import multiprocessing
###from scipy.ndimage import median_filter
import math
from ..ecg import baseline as BaselineRemove_Obj
from ..ecg import fiducial as Fiducial_Obj
from ..ecg import rpeak as Rpeak_Obj
from ..ecg import quality_check as Score_Obj


class PVCDetection:   
    def __init__(self): 
         self.PVCInformation=pd.DataFrame()            
    
    def _rri_filter_pvc(self, RRIArray): 
        BetweenRange_RRIArray = []
        FilteredRRIArray = []
        TempFilteredRRIArray = []
        BoolFilteredRRI = []
        for i in range(len(RRIArray)):
            if (RRIArray[i] >= 245 and RRIArray[i] <= 2000):
                TempFilteredRRIArray.append(RRIArray[i])
            else:
                TempFilteredRRIArray.append(0)

        for rri in RRIArray:
            if (rri >= 245 and rri <= 2000):
                BetweenRange_RRIArray.append(rri)

        medianRRI = np.median(BetweenRange_RRIArray)

        for rri in TempFilteredRRIArray:
            ##if(rri<=1.45*medianRRI and rri>=0.65*medianRRI):
            if (rri <= 1.65 * medianRRI and rri >= 0.45 * medianRRI):
                FilteredRRIArray.append(rri)
                BoolFilteredRRI.append(True)
            else:
                BoolFilteredRRI.append(False)

        return FilteredRRIArray, BoolFilteredRRI
    
    def _pvc_multiprocessing(self, processnumber, data, timearray, pindex, userid, ns, lock):
        fs = 250  ##ECG Sampling Rate
        block_length = math.ceil(len(data) / processnumber)
        startindex = pindex * block_length
        if (startindex < 5000):
            startindex = 5000

        endindex = startindex + block_length
        if (endindex >= len(data) - 2500):
            endindex = len(data) - 2500

        if ((startindex % 2500) != 0):  ###內縮殘餘片段，整體會有七個10秒片段不考慮
            startindex = (math.ceil(startindex / 2500) + 1) * 2500

        if ((endindex % 2500) != 0):  ###內縮殘餘片段，整體會有七個10秒片段不考慮
            endindex = math.ceil(endindex / 2500) * 2500

        for i in range(startindex, endindex, 2500):
            ecg_pre = data[i - 5000:i - 2500]  
            ecg = data[i - 2500:i]  
            ecg_next = data[i:i + 2500]  
            dt = timearray[int(i / 2500)]
            RawEcg = ecg
            ecgtemp = ecg_pre[2500 - 2 * fs:2500]  ###前後多抓2秒(避免PCV位於最前或最後被截除)
            ecgtemp.extend(ecg)
            ecgtemp.extend(ecg_next[0:2 * fs])  ###前後多抓2秒(避免PCV位於最前或最後被截除)
            ecg = ecgtemp
            ecg = np.array(ecg)
            MinValue = np.min(ecg)
            MaxValue = np.max(ecg)
            if (MinValue == MaxValue):
                continue

            lock.acquire()
            RpeakArray = Rpeak_Obj.rpeak_detection(ecg) ###, meausring_mode='strap',method_type='bandpass',mode='original')  
            lock.release()

            Ridx = RpeakArray[1:len(RpeakArray)]
            Ridx = np.where((Ridx >= i - 2500) & (Ridx < i))

            if len(Ridx) < 10 or len(Ridx) > 35 or Ridx[0] > 2 * fs or Ridx[-1] < 3500 - 2 * fs:
                continue

            RRIArray = np.diff(Ridx) * 1000 / 250
            RRIArray.astype("int32")

            score0 = Score_Obj.pattern_clustering(ecg, Ridx)
            score1 = Score_Obj.area_ratio(ecg, Ridx)
            QualtiyScore = score0 * score1

            if (QualtiyScore < 85):  ##訊號品質不好不分析
                continue

            ecg_norm = np.interp(ecg, (ecg.min(), ecg.max()), (0, 1)).reshape(len(ecg), 1)
            feature = Fiducial_Obj.feature_gen(ecg_norm, 250, Ridx)
            QRSArray = feature["QRSArray"]
            avgHR = feature["avgHR"]
            avgPR = feature["avgPR"]
            avgQRS = feature["avgQRS"]
            avgQT = feature["avgQT"]
            avgQTc = feature["avgQTc"]
            RRIArray = np.diff(Ridx) * 4
            for qrs_index in range(len(QRSArray)):
                QRSwidth = QRSArray[qrs_index]
                maxpreoffset = 50
                maxlastoffset = 50
                nowIndex = int(Ridx[qrs_index])
                preIndex = 0
                lastIndex = 0
                if (nowIndex < 50):
                    maxpreoffset = nowIndex

                if (nowIndex + 50 >= len(ecg)):
                    maxlastoffset = len(ecg) - nowIndex

                if (np.isnan(QRSwidth)):
                    for preoffset in range(maxpreoffset):
                        if (ecg[nowIndex - preoffset] == 0 or (
                                ecg[nowIndex - preoffset] >= 0 and ecg[nowIndex - preoffset - 1] <= 0)):
                            preIndex = nowIndex - preoffset
                            break

                    for lastoffset in range(maxlastoffset):
                        if (ecg[nowIndex + lastoffset] == 0 or (
                                ecg[nowIndex + lastoffset] >= 0 and ecg[nowIndex + lastoffset + 1] <= 0)):
                            lastIndex = nowIndex + lastoffset
                            break

                    nowWidth = (lastIndex - preIndex + 1) * 4 / 1000.0
                    if (nowWidth >= 0.114 and nowWidth <= 0.185):  ###寬但適當的QRS，回補回去
                        QRSArray[qrs_index] = nowWidth

            BoolArray_large = (QRSArray >= 0.114)
            BoolArray_small = (QRSArray <= 0.185)
            BoolArray = np.logical_and(BoolArray_large, BoolArray_small)
            PVC_CandidateIndexArray = np.where(BoolArray)[0]
            PVC_CandidateIndexArray = [index for index in PVC_CandidateIndexArray if
                                       (index < len(QRSArray) - 1 and (Ridx[index] > 500 and Ridx[index] < 3000))]
            if (len(PVC_CandidateIndexArray) == 0):
                continue

            FilteredRRI, BoolFilteredRRI = self._rri_filter_pvc(RRIArray)
            medianRRI = np.median(FilteredRRI)
            for qrs_index in PVC_CandidateIndexArray:
                RRI_Previous = RRIArray[qrs_index - 1]
                RRI_Next = RRIArray[qrs_index]

                meanLocRRI = (RRI_Previous + RRI_Next) / 2
                if (abs(meanLocRRI - medianRRI) >= 0.3 * medianRRI):  ###當下這拍前後RRI過短或過長
                    continue

                else:  ###-----Check Packet loss----
                    PacketLossFlag = False
                    CandidatePVC_QRS = np.array(ecg[int(Ridx[qrs_index]) - 20:int(Ridx[qrs_index]) + 20])
                    for m in range(5, len(CandidatePVC_QRS)):
                        FivePointSegment = CandidatePVC_QRS[m - 5:m]
                        if (sum(abs(np.diff(FivePointSegment))) == 0):
                            PacketLossFlag = True
                            break

                    if (PacketLossFlag):  ###漏封包，這拍不計算
                        continue

                    if ((RRI_Previous <= 0.88 * medianRRI) and (RRI_Next >= 1.12 * medianRRI) and (
                            RRI_Next / RRI_Previous) >= 1.20):
                        ecg_sec30 = ecg_pre  ###前後多抓10秒最後輸出ecg 30 secs片段
                        ecg_sec30.extend(RawEcg)
                        ecg_sec30.extend(ecg_next)  ###前後多抓10秒最後輸出ecg 30 secs片段
                        ecg_sec30_debasedline = BaselineRemove_Obj.baseline_remove(ecg_sec30,processnumber=1)
                        rpeak_indexarray = np.where((Ridx >= 500) & (Ridx < 3000))
                        rpeak_output = Ridx[rpeak_indexarray] - 500
                        rpeak_output = rpeak_output.astype(int)
                        newItem = pd.DataFrame([{"user_id": userid, "Measured_date": dt.strftime("%Y%m%d"),
                                                 "Measured_time": dt.strftime("%H%M%S.%f"), "HR": avgHR, "avgPR": avgPR,
                                                 "avgQRS": avgQRS, "avgQT": avgQT, "avgQTc": avgQTc, "Label": "PVC",
                                                 "Location": Ridx[qrs_index] - 500,
                                                 "ab-QRSWidth": QRSArray[qrs_index] * 1000,
                                                 "Ecg sec10": (int(ecg[500:3000])).tolist(),
                                                 "Ecg sec30": (int(ecg_sec30_debasedline)).tolist(),
                                                 "Score": QualtiyScore, "RPeaks": rpeak_output.tolist()}])
                        ns.PVCInformation = ns.PVCInformation.append(newItem, ignore_index=True)

    def _pvc_singleprocess(self, userid, data):
        fs = 250  ##ECG Sampling Rate
        PVCInformation = pd.DataFrame()
        TotalPVCCount = 0
        for i in range(1, len(data) - 1):
            ecg_pre = data[i - 1]["rows"]["ecgs"]
            ecg = data[i]["rows"]["ecgs"]
            ecg_next = data[i + 1]["rows"]["ecgs"]
            tt = data[i]["tt"]
            dt = datetime.fromtimestamp(int(tt) / 1000)
            RawEcg = ecg
            ecgtemp = ecg_pre[2500 - 2 * fs:2500]  ###前後多抓2秒(避免PCV位於最前或最後被截除)
            ecgtemp.extend(ecg)
            ecgtemp.extend(ecg_next[0:2 * fs])  ###前後多抓2秒(避免PCV位於最前或最後被截除)
            ecg = ecgtemp
            ecg = np.array(ecg)
            ecg = BaselineRemove_Obj.baseline_remove(ecg,processnumber=1) ##BaselineRemove(ecg)
            MinValue = np.min(ecg)
            MaxValue = np.max(ecg)
            if (MinValue == MaxValue):
                continue

            RpeakArray = Rpeak_Obj.rpeak_detection(ecg) ###, meausring_mode='strap',method_type='bandpass',mode='original') ##RPeakDetection(ecg, DetectionMode=1)
            Ridx = RpeakArray[1:len(RpeakArray)]

            if len(Ridx) < 10 or len(Ridx) > 35 or Ridx[1] > 2 * fs or Ridx[-1] < 3500 - 2 * fs:
                continue

            RRIArray = np.diff(Ridx) * 1000 / 250
            RRIArray.astype("int32")

            score0 = Score_Obj.pattern_clustering(ecg, Ridx)
            score1 = Score_Obj.area_ratio(ecg, Ridx)
            QualtiyScore = score0 * score1

            if (QualtiyScore < 85):  ##訊號品質不好不分析
                continue

            ecg_norm = np.interp(ecg, (ecg.min(), ecg.max()), (0, 1)).reshape(len(ecg), 1)
            feature = Fiducial_Obj.feature_gen(ecg_norm, 250, Ridx)
            QRSArray = feature["QRSArray"]
            avgHR = feature["avgHR"]
            avgPR = feature["avgPR"]
            avgQRS = feature["avgQRS"]
            avgQT = feature["avgQT"]
            avgQTc = feature["avgQTc"]
            RRIArray = np.diff(Ridx) * 4
            for qrs_index in range(len(QRSArray)):
                QRSwidth = QRSArray[qrs_index]
                maxpreoffset = 50
                maxlastoffset = 50
                nowIndex = int(Ridx[qrs_index])
                preIndex = 0
                lastIndex = 0
                if (nowIndex < 50):
                    maxpreoffset = nowIndex

                if (nowIndex + 50 >= len(ecg)):
                    maxlastoffset = len(ecg) - nowIndex

                if (np.isnan(QRSwidth)):
                    for preoffset in range(maxpreoffset):
                        if (ecg[nowIndex - preoffset] == 0 or (
                                ecg[nowIndex - preoffset] >= 0 and ecg[nowIndex - preoffset - 1] <= 0)):
                            preIndex = nowIndex - preoffset
                            break

                    for lastoffset in range(maxlastoffset):
                        if (ecg[nowIndex + lastoffset] == 0 or (
                                ecg[nowIndex + lastoffset] >= 0 and ecg[nowIndex + lastoffset + 1] <= 0)):
                            lastIndex = nowIndex + lastoffset
                            break

                    nowWidth = (lastIndex - preIndex + 1) * 4 / 1000.0
                    if (nowWidth >= 0.114 and nowWidth <= 0.185):  ###寬但適當的QRS，回補回去
                        QRSArray[qrs_index] = nowWidth

            BoolArray_large = (QRSArray >= 0.114)
            BoolArray_small = (QRSArray <= 0.185)
            BoolArray = np.logical_and(BoolArray_large, BoolArray_small)
            PVC_CandidateIndexArray = np.where(BoolArray)[0]
            PVC_CandidateIndexArray = [index for index in PVC_CandidateIndexArray if
                                       (index < len(QRSArray) - 1 and (Ridx[index] > 500 and Ridx[index] < 3000))]
            if (len(PVC_CandidateIndexArray) == 0):
                continue

            FilteredRRI, BoolFilteredRRI = self._rri_filter_pvc(RRIArray)
            medianRRI = np.median(FilteredRRI)
            for qrs_index in PVC_CandidateIndexArray:
                RRI_Previous = RRIArray[qrs_index - 1]
                RRI_Next = RRIArray[qrs_index]

                meanLocRRI = (RRI_Previous + RRI_Next) / 2
                if (abs(meanLocRRI - medianRRI) >= 0.3 * medianRRI):  ###當下這拍前後RRI過短或過長
                    continue

                else:  ###-----Check Packet loss----
                    PacketLossFlag = False
                    CandidatePVC_QRS = np.array(ecg[int(Ridx[qrs_index]) - 20:int(Ridx[qrs_index]) + 20])
                    for m in range(5, len(CandidatePVC_QRS)):
                        FivePointSegment = CandidatePVC_QRS[m - 5:m]
                        if (sum(abs(np.diff(FivePointSegment))) == 0):
                            PacketLossFlag = True
                            break

                    if (PacketLossFlag):  ###漏封包，這拍不計算
                        continue

                    if ((RRI_Previous <= 0.88 * medianRRI) and (RRI_Next >= 1.12 * medianRRI) and (
                            RRI_Next / RRI_Previous) >= 1.20):
                        TotalPVCCount = TotalPVCCount + 1
                        ecg_sec30 = ecg_pre  ###前後多抓10秒最後輸出ecg 30 secs片段
                        ecg_sec30.extend(RawEcg)
                        ecg_sec30.extend(ecg_next)  ###前後多抓10秒最後輸出ecg 30 secs片段
                        ecg_sec30_debasedline = BaselineRemove_Obj.baseline_remove(ecg_sec30,processnumber=1)
                        rpeak_indexarray = np.where((Ridx >= 500) & (Ridx < 3000))
                        rpeak_output = Ridx[rpeak_indexarray] - 500
                        rpeak_output = rpeak_output.astype(int)
                        newItem = pd.DataFrame([{"user_id": userid, "Measured_date": dt.strftime("%Y%m%d"),
                                                 "Measured_time": dt.strftime("%H%M%S.%f"), "HR": avgHR, "avgPR": avgPR,
                                                 "avgQRS": avgQRS, "avgQT": avgQT, "avgQTc": avgQTc, "Label": "PVC",
                                                 "Location": Ridx[qrs_index] - 500,
                                                 "ab-QRSWidth": QRSArray[qrs_index] * 1000,
                                                 "Ecg sec10": (int(ecg[500:3000])).tolist(),
                                                 "Ecg sec30": (int(ecg_sec30_debasedline)).tolist(),
                                                 "Score": QualtiyScore, "RPeaks": rpeak_output.tolist()}])
                        PVCInformation = PVCInformation.append(newItem, ignore_index=True)

        if (PVCInformation).shape[0] > 0:
            PVCInformation = PVCInformation.sort_values(by="Score", ascending=True)
            PVCInformation = PVCInformation.drop(columns="Score")
            PVCInformation = PVCInformation.reset_index(drop=True)

        return PVCInformation
    
    
    def generate_report(self, userid, data, timearray, processnum=1):
        
        """
        input ---
            userid: uuid
            data: ecg data after baseline drafting removal
            timearray:  1D array of datetime per data point corresponding to ECG data
            processnum: 處理核心數(>1為平行處理，default value=1為一般單核處理)

        output ---
            pd.DataFrame([{"user_id": userid, 
                           "Measured_date": Measuring data in format("%Y%m%d"),
                           "Measured_time": Measuring time in format("%H%M%S.%f"), 
                           "HR": avgHR,
                           "avgPR": avgPR,
                           "avgQRS": avgQRS, 
                           "avgQT": avgQT, 
                           "avgQTc": avgQTc, 
                           "Label": "PVC",
                           "Location": 1D array for PVC location,
                           "ab-QRSWidth": 1D array for QRS width,
                           "Ecg sec10": 1D array for 10-second ECG signal(當下分析的10秒片段)
                           "Ecg sec30": 1D array for 30-second ECG signal(當下分析片段與前後10秒片段，共30秒)
                           "Score": ECG quality score, 
                           "RPeaks": 1D array for r peak location
            
        """

        lock_pvc = Lock()              
        if(processnum > 1):
            manager = multiprocessing.Manager()
            ns = manager.Namespace()
            PVCInformation = pd.DataFrame()
            ns.PVCInformation = PVCInformation
            processes = [
                Process(target=self._pvc_multiprocessing, args=(processnum, data, timearray, pindex, userid, ns, lock_pvc)) for pindex in range(processnum)]  ###使用pnumber個核平行加速
            
            for process in processes:
                process.start()
            
            for process in processes:
                process.join()

            if (ns.PVCInformation).shape[0] > 0:
                ns.PVCInformation = ns.PVCInformation.sort_values(by="Score", ascending=True)
                ns.PVCInformation = ns.PVCInformation.drop(columns="Score")
                ns.PVCInformation = ns.PVCInformation.reset_index(drop=True)
            
            self.PVCInformation=ns.PVCInformation

        else: ##單核執行
            self.PVCInformation = self._pvc_singleprocess(userid, data)



    ####================PVC事件解析================================
    def pvc_event_load(self,ecg_dict=[]): 
        
        PVCInformation=self.PVCInformation
        for i in range(len(PVCInformation)):
            RowData={}
            Measured_date=PVCInformation.at[i, "Measured_date"]
            date=Measured_date[0:4]+"/"+Measured_date[4:6]+"/"+Measured_date[6:8]       
            Measured_time=PVCInformation.at[i, "Measured_time"]
            time=Measured_time[0:2]+":"+Measured_time[2:4]+":"+Measured_time[4:6]
            RowData={"date": date,
                    "time": time, ###只取到秒數
                    "unit": "10mm/mV",
                    "HR": round(PVCInformation.at[i, "HR"]),
                    "PR": PVCInformation.at[i, "avgPR"],
                    "QRS": PVCInformation.at[i, "avgQRS"],
                    "QT": PVCInformation.at[i, "avgQT"],                         
                    "QTc": PVCInformation.at[i, "avgQTc"],
                    "Irrequlars": [],
                    "PVCs": [int(PVCInformation.at[i, "Location"])],
                    "RPeaks": PVCInformation.at[i, "RPeaks"],
                    "sec10": PVCInformation.at[i, "Ecg sec10"],
                    "sec30": PVCInformation.at[i, "Ecg sec30"]
                    }                    
         
            ecg_dict.append(RowData)       

        return ecg_dict 
                      
        
    



