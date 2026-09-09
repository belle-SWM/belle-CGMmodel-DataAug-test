import numpy as np
import sys
from ..arrhythmia import multi_preprocess as mp
from datetime import datetime, timedelta
from ..ecg import fiducial as Fiducial
from multiprocessing import Pool, RawArray
from .irregular import irregular_detection
from ..ecg.qualitycheckmodel.DatasetV2 import AbnormalDetector
import time
from ..ecg import baseline as BaselineRemove_Obj

class ArrhythmiaDetection:   
    def __init__(self,user_info,data,time_array=[]):
        self._initialize()      
        self._load_data(data,time_array)
        self.Report["user_info"] = user_info        
    
    def generate_report(self, mode="Arrhythmia", processnum=1): ##主要呼叫函式
        self.PassIdxs = np.array([i for i in range(0,len(self.EcgArrays))])   
        if(processnum>1):    
            self._analyze_arrhythmia_mp(processnum) ##Number of CPUs
        else:          
            self._analyze_arrhythmia()

        self._arrhythmia_count()
        self._output_events(mode=mode)
        self.Report["report_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.Report.update(self.Output)
        
        return self.Report
        
    def _initialize(self):
       
        self.StartDT = [] # Measure Time Info
        self.EndDT = []       
        self.EcgArrays = [] # Raw Datas
        self.TimeArrays = []     
        self.PassIdxs = []   # quality-check index result       
        self.Features = [] # Fiducial Features       
        self.ArrhythmiaCounts = {} # Arrhythmia Counts
        # Output
        self.Output = {}
        self.Output["ir_statistics"] = np.zeros(31,dtype='int32').tolist()
        self.Output["ir_statistics_perhour"] = np.zeros(24,dtype='int32').tolist()
        self.Output["ir_events"] = []
        self.Output["max_hr_physio"] = []
        self.Output["min_hr_physio"] = []
        self.Output["note"] = ""
        # Report Content
        self.Report = {}
        self.Report["type"] = "P001"
        self.Report["version"] = "v1.0"
        self.Report["report_datetime"] = ""
        self.Report["measure_start_datetime"] = ""
        self.Report["measure_end_datetime"] = ""
        self.Report["user_info"] = []

    def _load_data(self,data,time_array):  ###解析data提取出ECG 
       
        if(len(time_array)>0): ###如果輸入有time_array資料，表示data為已經做過基線拉平處理的ECG
            segment_length=2500
            blocknum=int(len(data)/segment_length)
            ecgArray=np.reshape(data[0:blocknum*segment_length],(blocknum, segment_length))
            self.EcgArrays = np.array(ecgArray, dtype=np.int32)
            self.TimeArrays=((np.reshape(time_array[0:blocknum],(blocknum, 1))).flatten()).tolist()
            self.StartDT = self.TimeArrays[0]
            self.EndDT = self.TimeArrays[-1] + timedelta(seconds=10)
            self.Report["measure_start_datetime"] = self.StartDT.strftime("%Y-%m-%d %H:%M:%S")
            self.Report["measure_end_datetime"] = self.EndDT.strftime("%Y-%m-%d %H:%M:%S")           
        else:         
            data.sort(key = lambda i:i['tt'])
            self.TimeArrays = [datetime.fromtimestamp(line["tt"]/1000) for line in data]
            self.StartDT = self.TimeArrays[0]
            self.EndDT = self.TimeArrays[-1] + timedelta(seconds=10)
            self.Report["measure_start_datetime"] = self.StartDT.strftime("%Y-%m-%d %H:%M:%S")
            self.Report["measure_end_datetime"] = self.EndDT.strftime("%Y-%m-%d %H:%M:%S")
            for i in range(len(data)):
                ecg = data[i]['rows']['ecgs']
                if len(ecg) < 2500:
                    self.EcgArrays.append(ecg)
                else:
                    self.EcgArrays.append(ecg[:2500])
                
            self.EcgArrays = np.array(self.EcgArrays) 
      

    def _analyze_arrhythmia(self):
        count = 1        
        for idx in self.PassIdxs:
            print("\r  Progress of Arrhythmia Analysis: %d/%d"
                %(count,len(self.PassIdxs)), end="")
            result = self._ecg_analysis(idx)
            
            ##if result is not None:
            if len(result) >=1:
                self.Features.append(self._ecg_analysis(idx))
               
            count += 1       

       
    def _analyze_arrhythmia_mp(self, procNum):
        X_shape = (self.EcgArrays).shape
        X = RawArray('d',X_shape[0]*X_shape[1])
        # Wrap X as an numpy array so we can easily manipulates its data.
        X_np = np.frombuffer(X).reshape(X_shape)
        # Copy data to our shared array.
        np.copyto(X_np,self.EcgArrays)

        ###print('_analyze_arrhythmia_mp---> self.PassIdxs:',self.PassIdxs)  
        with Pool(processes=procNum,initializer=mp.init_worker,initargs=(X,X_shape)) as pool:                    
            for result in pool.map(mp.ecg_analysis, self.PassIdxs):
                ##if result is not None:
                if len(result) >=1:                    
                    self.Features.append(result)                 
        
        self.Features.sort(key=lambda s:s['Index'], reverse=False)      
    

    def _arrhythmia_count(self):
        startdate = self.StartDT.date()
        enddate = self.EndDT.date()
        datelist = []
        for i in range(0,(enddate-startdate).days+1):
            datelist.append(startdate+timedelta(days=i))

        arrhythmia = [evt for evt in self.Features if evt['ResultFlag'] > 0] 
        date_array = [self.TimeArrays[arr['Index']].strftime("%Y%m%d") for arr in arrhythmia]
        date_array = np.array(date_array)
        
        for date in datelist:
            Dday = date.day
            Dstr = date.strftime("%Y%m%d")
            if(date_array.size==0):
                Count=0
            else:
                Count = sum(date_array==Dstr)
            
            self.Output["ir_statistics"][Dday-1] = Count
            self.ArrhythmiaCounts[Dstr] = Count

        date_hour_list = []
        for i in range(24):
            date_hour_list.append(str(i).zfill(2))
          
        date_hour_array = [self.TimeArrays[arr['Index']].strftime("%H") for arr in arrhythmia]
        date_hour_array = np.array(date_hour_array)
        for date_hour_str in date_hour_list:
            if(date_array.size==0):
                count_hour=0
            else:
                count_hour = sum(date_hour_array==date_hour_str)
            
            self.Output["ir_statistics_perhour"][int(date_hour_str)] = count_hour


    def _output_events(self,mode):
        if mode == "Arrhythmia":           
            ir_events = self._event_summerize("Arrhythmia",10) #### 心律不整ECG訊號的輸出比數10筆
            if len(ir_events) > 0:
                self.Output["ir_events"] = ir_events
                self.Output["note"] = "建議前往醫院進行更進一步之檢測"
        else:
            ir_events = self._event_summerize("All",[])
            self.Output["ir_events"] = ir_events

        ## Output max and min HR event
        max_hr_physio = self._event_summerize("Maximum Heart Rate",10)
        
        if(len(max_hr_physio)==0): 
            self.Output["max_hr_physio"]=[]
        else:
            self.Output["max_hr_physio"] = max_hr_physio[0]

        min_hr_physio = self._event_summerize("Minimum Heart Rate",10)
        if(len(min_hr_physio)==0): 
            self.Output["min_hr_physio"]=[]
        else:
            self.Output["min_hr_physio"] = min_hr_physio[0]
        
    
    def _event_summerize(self, reason,num):
        if reason == "All":
            events = self.Features
            num = len(events)
        if reason == "Arrhythmia":            
            events = [evt for evt in self.Features if evt['ResultFlag']>0 and evt['score']>0.6] 
            events.sort(key=lambda s:(-s['ResultFlag'],-s['score']))
        if reason == "Maximum Heart Rate":
            events = sorted(self.Features, key=lambda s:s['avgHR'], reverse=True)
        if reason == "Minimum Heart Rate":
            events = sorted(self.Features, key=lambda s:s['avgHR'], reverse=False)        
             
        output = []  
     
        for evt in events:
            dt = self.TimeArrays[evt['Index']]
            ecg = self.EcgArrays[evt['Index']]
            ridx=evt['Ridx']  ###來自irregular_detection函式輸出
            eventloc=evt['location']  ###來自irregular_detection函式輸出
            if(evt['Index']>0 and evt['Index']<len(self.EcgArrays)-1): ### for extracting 30-sec ECG signal
                ecg_sec30=self.EcgArrays[evt['Index']-1]
                ecg_sec30=np.append(ecg_sec30,ecg) 
                ecg_sec30=np.append(ecg_sec30,self.EcgArrays[evt['Index']+1])
            elif(evt['Index']>0):
                ecg_sec30=self.EcgArrays[evt['Index']-1]
                ecg_sec30=np.append(ecg_sec30,ecg) 
                ecg_sec30=np.append(ecg_sec30,np.zeros(2500)) 
            elif(evt['Index']<len(self.EcgArrays)-1):
                ecg_sec30=np.zeros(2500) 
                ecg_sec30=np.append(ecg_sec30,ecg) 
                ecg_sec30=np.append(ecg_sec30,self.EcgArrays[evt['Index']+1])

            if reason=="All":
                if evt['ResultFlag'] > 0:
                    title = "Arrhythmia"
                else:
                    title = ""
            else:
                scale = evt['scale']
                ecg = ecg * scale              
                title = "%dmm/mV, score:%.2f, flag:%d" %(10*scale,evt['score'],evt['ResultFlag'])            
                
            output.append(self._events_converter(evt, title, dt, ecg, ecg_sec30, ridx, eventloc))
       
        return output
    

    def _events_converter(self, evt, reason, dt, ecg, ecg_sec30, ridx, eventloc):
        output = {}
        output["timestamp"] = dt.timestamp() * 1000
        output["reason"] = reason
        output['hr'] = str(int(evt["avgHR"])) + " bpm"
        ecg_norm = np.interp(ecg,(ecg.min(),ecg.max()),(0,1)).reshape(2500,1)       
        feature = Fiducial.feature_gen(ecg_norm,250,evt['Ridx']) 
        quality = feature["good_quality"]
        output["pr"] = self._ms_convert(feature["avgPR"],quality)
        output["qrs"] = self._ms_convert(feature["avgQRS"],quality)
        output["qt"] = self._ms_convert(feature["avgQT"],quality)
        output["qtc"] = self._ms_convert(feature["avgQTc"],quality)
        output["ridx"] = ridx
        output["eventloc"] = eventloc
        output["ecgs"] = ecg.tolist()
        output["ecgs sec30"] = ecg_sec30.tolist() 

        return output       
        

    ####===============心律不整事件解析===========================
    def arrhythmia_event_load(self): 

        report=self.Output    
        ir_statistics=report["ir_statistics"]
        ir_statistics_perhour=report["ir_statistics_perhour"]
        ir_events=report["ir_events"]
        ecg_dict=[]

        for i in range(len(ir_events)):
            RowData={}
            reason=(ir_events[i]["reason"]).split(",")
            HR=((ir_events[i]["hr"]).split(" "))[0]
            PR=((ir_events[i]["pr"]).split(" "))[0]
            QRS=((ir_events[i]["qrs"]).split(" "))[0]
            QT=((ir_events[i]["qt"]).split(" "))[0]
            QTc=((ir_events[i]["qtc"]).split(" "))[0]
                
            RowData={"date": datetime.fromtimestamp(float(ir_events[i]["timestamp"])/1000).strftime("%Y/%m/%d"),
                    "time": datetime.fromtimestamp(float(ir_events[i]["timestamp"])/1000).strftime("%H:%M:%S"), ###只取到秒數
                    "unit": reason[0],
                    "HR": HR, 
                    "PR": PR,
                    "QRS": QRS,
                    "QT": QT,                      
                    "QTc": QTc, 
                    "Irrequlars": ir_events[i]["eventloc"],
                    "PVCs": [],
                    "RPeaks": (ir_events[i]["ridx"]).tolist(), 
                    "sec10": ir_events[i]["ecgs"],
                    "sec30": ir_events[i]["ecgs sec30"]
                    }
                    
            ecg_dict.append(RowData)         
        
        return ecg_dict,ir_statistics,ir_statistics_perhour
    

    def maxmin_hr_event_load(self): 
        """
        input --- no input
        output ---
            minHRStatistics={"date": occurence date in format("%Y/%m/%d"),
                "time": occurence time in format("%H:%M:%S"),
                "unit": "Minimum Heart Rate",
                "HR": HR, 
                "PR": PR,
                "QRS": QRS,
                "QT": QT,                      
                "QTc": QTc, 
                "Irrequlars": [],
                "PVCs": [],
                "RPeaks": 1D list of R peak indexs, 
                "sec10": 1D list of ECG signal of 10 seconds where the min hr happens
                "sec30": 1D list of ECG signal of 30 seconds where the min hr happens
                }      

            maxHRStatistics={"date": occurence date in format ("%Y/%m/%d"),
                "time": occurence time in format("%H:%M:%S"),
                "unit": "Maximum Heart Rate",
                "HR": HR, 
                "PR": PR,
                "QRS": QRS,
                "QT": QT,                      
                "QTc": QTc, 
                "Irrequlars": [],
                "PVCs": [],
                "RPeaks": 1D list of R peak indexs, 
                "sec10": 1D list of ECG signal of 10 seconds where the max hr happens
                "sec30": 1D list of ECG signal of 30 seconds where the max hr happens
                }

        """
        report=self.Output 

        min_hr_physio_event=report["min_hr_physio"]
        reason=(min_hr_physio_event["reason"]).split(",")
        HR=((min_hr_physio_event["hr"]).split(" "))[0]
        PR=((min_hr_physio_event["pr"]).split(" "))[0]
        QRS=((min_hr_physio_event["qrs"]).split(" "))[0]
        QT=((min_hr_physio_event["qt"]).split(" "))[0]
        QTc=((min_hr_physio_event["qtc"]).split(" "))[0]      
        minHRStatistics={"date": datetime.fromtimestamp(float(min_hr_physio_event["timestamp"])/1000).strftime("%Y/%m/%d"),
                "time": datetime.fromtimestamp(float(min_hr_physio_event["timestamp"])/1000).strftime("%H:%M:%S"), ###只取到秒數
                "unit": reason[0],
                "HR": HR, 
                "PR": PR,
                "QRS": QRS,
                "QT": QT,                      
                "QTc": QTc, 
                "Irrequlars": [],
                "PVCs": [],
                "RPeaks": (min_hr_physio_event["ridx"]).tolist(), 
                "sec10": min_hr_physio_event["ecgs"],
                "sec30": min_hr_physio_event["ecgs sec30"]
                }                    
        
        max_hr_physio_event=report["max_hr_physio"]  
        
        reason=(max_hr_physio_event["reason"]).split(",")
        HR=((max_hr_physio_event["hr"]).split(" "))[0]
        PR=((max_hr_physio_event["pr"]).split(" "))[0]
        QRS=((max_hr_physio_event["qrs"]).split(" "))[0]
        QT=((max_hr_physio_event["qt"]).split(" "))[0]
        QTc=((max_hr_physio_event["qtc"]).split(" "))[0]             
        maxHRStatistics={"date": datetime.fromtimestamp(float(max_hr_physio_event["timestamp"])/1000).strftime("%Y/%m/%d"),
                "time": datetime.fromtimestamp(float(max_hr_physio_event["timestamp"])/1000).strftime("%H:%M:%S"), ###只取到秒數
                "unit": reason[0],
                "HR": HR, 
                "PR": PR,
                "QRS": QRS,
                "QT": QT,                      
                "QTc": QTc, 
                "Irrequlars": [],
                "PVCs": [],
                "RPeaks": (max_hr_physio_event["ridx"]).tolist(), 
                "sec10": max_hr_physio_event["ecgs"],
                "sec30": max_hr_physio_event["ecgs sec30"]
                }   
          
        return minHRStatistics,maxHRStatistics  
    

    @staticmethod
    def _ms_convert(x, quality):
        pass
        if np.isnan(x):
            return "--"
        else:
            output = str(int(x))+" ms"
            if not quality:
                output = output + "*"
            return output

    def _ecg_analysis(self, idx):        
      
        sig = self.EcgArrays[idx]
        results = irregular_detection(sig,idx)
        
        return results
    
    