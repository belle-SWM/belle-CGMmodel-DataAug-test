import os
from datetime import datetime, timedelta
import numpy as np
from ..sleep import sleeposture_analysis as SP
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import joblib
import pandas as pd
import hrvanalysis as hrv
from time import time
from multiprocessing import Pool, RawArray
##import multi_preprocess as mp
from ..sleep import multi_preprocess as mp

def filter(Ridxs, Fs, minThr, maxThr):
    RRI = np.diff(Ridxs)*1000/Fs
    RRIs = [int(rri) for rri in RRI if rri>minThr and rri<maxThr]
    Med = np.median(RRIs)
    NNIs = [rri for rri in RRIs if rri>=0.45*Med and rri<=1.62*Med]
    return NNIs

def nni_filter(Ridxs, Fs):
    RRI = np.diff(Ridxs)*1000/Fs
    RRIs = [int(rri) for rri in RRI if rri>300 and rri<2000]
    NNIs = []
    if len(RRIs) > 0:
        for i in range(1,len(RRIs)):
            rr_spread = abs((RRIs[i-1]-RRIs[i])/RRIs[i-1])
            if rr_spread < 0.3:
                NNIs.append(RRIs[i])
    return NNIs

def get_features(NNIs, feature_list):
    feature = {}
    if feature_list['time']:
        feature.update(hrv.get_time_domain_features(NNIs))
    if feature_list['freq']:
        feature.update(hrv.get_frequency_domain_features(NNIs))
    if feature_list['poincare']:
        feature.update(hrv.get_poincare_plot_features(NNIs))
    
    feature_name = list(feature.keys())
    feature_value = [float(num) for num in feature.values()]
    return feature_name, feature_value



class Device:
    def __init__(self, data, duration, iniACC):
        self._initialize()
        self._load_data(data, duration)
        
        t1 = time()
        self._detect_posture(iniACC)
        print("         [detect posture]:%.2fs" %(time()-t1))
    
        ##t1 = time()
        #self._detect_sleep_stage()
        ##print("         [sleep stage]:%.2fs" %(time()-t1))

        ##self._set_xticks()

    def _initialize(self):
        # Time of desired duration
        self.startTT = []
        self.endTT = []

        # Time of found data
        self.firstTT = []
        self.lastTT = []
        self.xticks = []
        self.xticklabels = []

        # Datas
        self.initialTimes = []
        self.motionBlocks = [[]]
        self.ecgBlocks = [[]]
        self.allRRIs = []
        self.prev_dt = 0
        self.SampleRate = []
        self.minHRHRV = {}

        # Output from "SleepPostureV3.py"
        self.State = []
        self.Analysis = []
        self.Proportion = []

        ##self.model = joblib.load("./Sleep_Strap/sleepstage/clf_mixed.pkl") ##暫時不使用ai

    def _load_data(self,data,duration):
        self.SampleRate = 2
        data.sort(key = lambda i:i['tt'])
        prev_tt = 0
        for line in data:
            magic = line['magic']
            tt = line["tt"]/1000
            dt = datetime.fromtimestamp(tt)
            if prev_tt == 0:
                motion = line['rows']['motions']
                ecg = line['rows']['ecgs']
                self._append_data(magic,dt,ecg,motion)
            else:
                motion, ecg = [], []

                ## 檢查跟上個tt的差距
                pack_diff = tt - prev_tt - 10
                if pack_diff > 5 and magic == self.prev_dt:
                    miss_motion = int(self.SampleRate * pack_diff)
                    motion.extend(np.zeros((miss_motion,10)).tolist())

                    miss_ecg = int(250 * pack_diff)
                    ecg.extend(np.zeros(miss_ecg).tolist())
                
                ## 檢查這個tt的是否有漏
                cur_motion = line['rows']['motions']
                while len(cur_motion) < self.SampleRate * 10:
                    cur_motion.append([0,0,0,0,0,0,0,0,0,0])
                motion.extend(cur_motion)

                cur_ecg = line['rows']['ecgs']
                ecg.extend(cur_ecg)

                self._append_data(magic,dt,ecg,motion)
            
            prev_tt = tt

        self.startTT = duration[0]
        self.endTT = duration[1]

    def _append_data(self,magic,dt,ecg,motion):  
        if len(self.initialTimes) == 0:
            self.initialTimes.append(dt)
        else:
            if magic != self.prev_dt:
                self.ecgBlocks.append([])
                self.motionBlocks.append([])
                self.initialTimes.append(dt)

        # 存入最新的block
        self.prev_dt = magic
        self.ecgBlocks[-1].append(ecg)
        self.motionBlocks[-1].extend(motion)
        
    def _detect_posture(self,iniACC):
        first = True
        for i in range(len(self.motionBlocks)):
            Motion = self.motionBlocks[i]
            startDT = self.initialTimes[i]
            Fs = self.SampleRate
            state, analyze, proportion = SP.PostureDetection(Motion,startDT,[],Fs,iniACC)
            self.State.append(state)
            self.Analysis.append(analyze)
            self.Proportion.append(proportion)
            if first:
                self.firstTT = state['DT'].iat[0]
                first = False
            self.lastTT = state['DT'].iat[-1]

    def _set_xticks(self):
        tick = datetime(self.firstTT.year,self.firstTT.month,self.firstTT.day,
                        self.firstTT.hour,0 if self.firstTT.minute<30 else 30)
        self.xticks.append(tick)
        while tick < self.lastTT:
            tick += timedelta(minutes=30)
            self.xticks.append(tick)
        self.xticklabels = [tick.strftime("%H:%M") for tick in self.xticks]

    def _detect_sleep_stage(self):
        for i in range(len(self.State)):
            interval = 5
            state = self.State[i]
            Rloc = self._multi_process_Rloc(self.ecgBlocks[i])
            new_state = self._predict_sleep_stage(state,Rloc,interval)
            self.State[i] = new_state

    def _multi_process_Rloc(self,ecgBlock):
        # Check length of every line are same
        ecgNum = len(ecgBlock)
        ECG = np.zeros((ecgNum,2500))
        for i in range(ecgNum):
            ecg = ecgBlock[i]
            ecgLen = len(ecg)
            if ecgLen >= 2500:
                ECG[i,:] = ecg[:2500]
            else:
                ECG[i,:ecgLen] = ecg

        # Put data into shared memory
        X_shape = ECG.shape
        # Create array in shared memory
        X = RawArray('d',X_shape[0]*X_shape[1])
        # Wrap X as an numpy array so we can easily manipulates its data.
        X_np = np.frombuffer(X).reshape(X_shape)
        # Copy data to our shared array.
        np.copyto(X_np,ECG)
        # Multi-Process to get R peak locations
        Rloc = []
        with Pool(processes=os.cpu_count(),initializer=mp.init_worker,initargs=(X,X_shape)) as pool:
            for result in pool.imap(mp.get_Rloc, range(ecgNum)):
                Rloc.append(result)
        return Rloc

    def _predict_sleep_stage(self,state,Rloc,interval):
        state['Stage'] = float('nan')
        window = int(60/10)
        for i in range(0,len(state),interval):
            start_idx, end_idx = i, i+interval
            if end_idx > len(state):
                end_idx = len(state)

            # Check if there are any "Upright" in state, set stages to awake if so.
            posture = state.State[start_idx:end_idx]
            if 1 in posture.values:
                state.loc[start_idx:end_idx,'Stage'] = 0
                continue

            # Get nni in 5 minutes
            start, end = i*window, (i+interval)*window
            Ridxs = []
            for j in range(start,end):
                if j >= len(Rloc):
                    break
                if len(Rloc[j]) > 1:
                    Ridxs.extend(Rloc[j])
            NNIs = nni_filter(Ridxs, 250)

            # Update RRIs for poincare plot
            RRIs = filter(Ridxs,250,250,2000)
            self.allRRIs.extend(RRIs)
            
            if len(NNIs) < 120: # 30(bpm) * 5(min) * 0.8(80%)
                continue
            
            # Get HRV features
            feature_list = {'time':True,'freq':True,'poincare':False}
            feature_name, feature_value = get_features(NNIs,feature_list)

            # Prediction of sleep stage
            pred = self.model.predict([feature_value])[0]
            label = []
            if pred == 0:
                # label = 0
                label = -2
            if pred == 1:
                label = -2
            if pred == 2:
                label = -3
            if pred == 3:
                label = -1

            if i+interval <= len(state):
                state.loc[i:i+interval,'Stage'] = label
            else:
                state.loc[i:,'Stage'] = label

            # Update HRV in frequency domain based on minimum HR
            if label == -3:
                self._update_HRV_minHR(NNIs,label)

        return state

    def _update_HRV_minHR(self,NNIs,label):
        hr = 60 * 1000 / np.mean(NNIs)
        features = hrv.get_frequency_domain_features(NNIs)
        output = {
            'hr':hr,
            'lf':features['lf'],
            'hf':features['hf'],
            'lf/hf':features['lf_hf_ratio'],
            'lf%':features['lf']/(features['lf']+features['hf']),
            'stage':label
            }

        if len(self.minHRHRV) == 0:
            self.minHRHRV.update(output)

        if output['hr'] < self.minHRHRV['hr']:
            self.minHRHRV.update(output)

    def poincare_figure(self,save_path,sd_feature=True):
        fig = plt.figure(figsize=(12,12))
        plt.ioff()
        ax = fig.add_subplot(1,1,1)

        if len(self.allRRIs) > 1:
            RRI = np.array(self.allRRIs)
            minRRI, maxRRI = min(RRI), max(RRI)
            X, Y = RRI[:-1], RRI[1:] # unit: second
            ax.scatter(X, Y, c='b', s=2)
            ax.set_xlabel("RR_n (ms)", fontsize=15)
            ax.set_ylabel("RR_n+1 (ms)", fontsize=15)
            ax.set_xlim(minRRI, maxRRI)
            ax.set_ylim(minRRI, maxRRI)

            if sd_feature:
                # Calculate SD1 and SD2
                dict_sd1_sd2 = hrv.get_poincare_plot_features(RRI.tolist())
                sd1 = dict_sd1_sd2["sd1"]
                sd2 = dict_sd1_sd2["sd2"]
                mean_nni = np.mean(RRI)

                # Plot ellipse contour
                ells = Ellipse(xy=(mean_nni, mean_nni), width=2 * sd2 + 1,
                       height=2 * sd1 + 1, angle=45, linewidth=2,
                       fill=False)
                ax.add_patch(ells)

                # Plot background color of ellipse
                ells = Ellipse(xy=(mean_nni, mean_nni), width=2 * sd2,
                       height=2 * sd1, angle=45)
                ells.set_alpha(0.2)
                ells.set_facecolor("blue")
                ax.add_patch(ells)

                # Plot axis of ellipse by SD1 and SD2
                arrow_style = {'length_includes_head':True,'head_width':15,'head_length':15,'linewidth':3}
                sd1_arrow = ax.arrow(mean_nni,mean_nni,-sd1*np.sqrt(2)/2,sd1*np.sqrt(2)/2,
                            ec='r',fc="r",label="SD1",**arrow_style)
                sd2_arrow = ax.arrow(mean_nni,mean_nni,sd2*np.sqrt(2)/2,sd2*np.sqrt(2)/2,
                            ec='g',fc="g",label="SD2",**arrow_style)
                ax.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="lower right")

                # Plot auxiliary lines
                line_style = {'linestyle':'--','color':'k','linewidth':1,'alpha':0.75}
                text_style = {'ha':'left','va':'bottom','alpha':0.75}
                for per in [1,1.1,1.2,1.3]:
                    if per > 1:
                        ax.plot([0,maxRRI],[0,maxRRI*per],**line_style)
                        ax.plot([0,maxRRI],[0,maxRRI/per],**line_style)
                        per_str = "%d%%"%(per*100-100)
                        ax.text(maxRRI/per,maxRRI,per_str,**text_style)
                        ax.text(maxRRI,maxRRI/per,per_str,**text_style)
                    else:
                        ax.plot([0,maxRRI],[0,maxRRI],**line_style)
                        ax.text(maxRRI,maxRRI,'0%',**text_style)

        # Save figure
        fig.tight_layout()
        save_name = "Poincare_%s_%s.png" %(self.startTT.strftime("%Y%m%d%H%M"),
                                  self.endTT.strftime("%Y%m%d%H%M"))
        kargs = {'dpi':300,'facecolor':'white','bbox_inches':'tight'}
        fig.savefig(os.path.join(save_path,save_name),**kargs)
        plt.close(fig)
        
        
    def posture_figure(self,save_path):
        fig = plt.figure(figsize=(10,2))
        plt.ioff()
        ax = fig.add_subplot(1,1,1)
        for state in self.State:
            X,Y = state['DT'],state['State']
            ax.plot(X,Y,color='b')

        ax.set_yticks(range(1,6))
        ax.set_yticklabels(['Upright','Right','Prone','Left','Supine'])
        ax.grid(alpha=0.3)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels,rotation=45)
        fig.tight_layout()

        save_name = "Posture_%s_%s.png" %(self.startTT.strftime("%Y%m%d%H%M"),
                                  self.endTT.strftime("%Y%m%d%H%M"))
        kargs = {'dpi':300,'facecolor':'white','bbox_inches':'tight'}
        fig.savefig(os.path.join(save_path,save_name),**kargs)
        plt.close(fig)
        return self.State   ##dennis add

    def stage_figure(self,save_path):
        fig = plt.figure(figsize=(10,2))
        plt.ioff()
        ax = fig.add_subplot(1,1,1)
        for state in self.State:
            X,Y = state['DT'],state['Stage']
            ax.plot(X,Y,color='b')

        ax.set_yticks(range(0,-4,-1))
        ax.set_yticklabels(['Awake','REM','Light','Deep'])
        ax.grid(alpha=0.3)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels,rotation=45)

        save_name = "Stage_%s_%s.png" %(self.startTT.strftime("%Y%m%d%H%M"),
                                  self.endTT.strftime("%Y%m%d%H%M"))
        kargs = {'dpi':300,'facecolor':'white','bbox_inches':'tight'}
        fig.savefig(os.path.join(save_path,save_name),**kargs)
        plt.close(fig)
        return self.State   ###dennis add

    def posture_analysis(self):
        on_bed_dt = self.firstTT
        found = False
        for i in range(len(self.State)):
            state = self.State[i]
            for j in range(len(state)):
                evt = state.iloc[j,:]
                if evt['State'] > 1:
                    on_bed_dt = evt['DT']
                    found = True

                if found:
                    break
            if found:
                break

        off_bed_dt = self.lastTT
        found = False
        for i in reversed(range(len(self.State))):
            state = self.State[i]
            for j in reversed(range(len(state))):
                evt = state.iloc[j,:]
                if evt['State'] > 1 and evt['DT'] > on_bed_dt:
                    off_bed_dt = evt['DT']
                    found = True
                if found:
                    break
            if found:
                break

        posture_count = np.zeros(4)
        for i in range(len(self.Proportion)):
            prop = self.Proportion[i].iloc[:,1].tolist()
            posture_count += np.array(prop)
        if sum(posture_count) > 0:
            posture_count = 100 * posture_count / sum(posture_count)
        
        result = {
            'on_bed':on_bed_dt.strftime("%Y/%m/%d %H:%M:%S"),
            'off_bed':off_bed_dt.strftime("%Y/%m/%d %H:%M:%S"),
            'right':posture_count[0],
            'prone':posture_count[1],
            'left':posture_count[2],
            'supine':posture_count[3]
        }

        return result

    def stage_analysis(self):
        sleep_dt = self.firstTT
        found = False
        for i in range(len(self.State)):
            state = self.State[i]
            for j in range(len(state)):
                evt = state.iloc[j,:]
                if evt['Stage'] < 0:
                    sleep_dt = evt['DT']
                    found = True
                if found:
                    break
            if found:
                break
        
        wake_dt = self.lastTT
        found = False
        for i in reversed(range(len(self.State))):
            state = self.State[i]
            for j in reversed(range(len(state))):
                evt = state.iloc[j,:]
                if evt['Stage'] < 0 and evt['DT'] > sleep_dt:
                    wake_dt = evt['DT'] + timedelta(minutes=5)
                    found = True
                if found:
                    break
            if found:
                break

        REM,Light,Deep = 0,0,0
        for state in self.State:
            REM += sum(state.Stage==-1)
            Light += sum(state.Stage==-2)
            Deep += sum(state.Stage==-3)
        Total = REM+Light+Deep

        result = {
            'asleep':sleep_dt.strftime("%Y/%m/%d %H:%M:%S"), 
            'wakeup':wake_dt.strftime("%Y/%m/%d %H:%M:%S"),  
            'sleephours':(wake_dt-sleep_dt).total_seconds()/(60*60),
            'rem':100*REM/Total if Total>0 else 0,
            'light':100*Light/Total if Total>0 else 0,
            'deep':100*Deep/Total if Total>0 else 0
        }
        return result

    def output_stages(self,save_path):
        Output = pd.DataFrame({})
        for state in self.State:
            Output = pd.concat([Output,state.loc[:,['DT','Stage']]])

        save_name = "Stage_%s_%s.csv" %(self.startTT.strftime("%Y%m%d%H%M"),
                                  self.endTT.strftime("%Y%m%d%H%M"))
        Output.to_csv(os.path.join(save_path,save_name))
        
    def hrv_analysis(self):
        if len(self.minHRHRV) > 0:
            output = {
                'hr':self.minHRHRV['hr'],
                'lf':self.minHRHRV['lf'],
                'hf':self.minHRHRV['hf'],
                'lf/hf':self.minHRHRV['lf/hf'],
                'lf%':self.minHRHRV['lf%'],
                'stage':self.minHRHRV['stage']
            }
        else:
            output = {'hr':0,'lf':0,'hf':0,'lf/hf':0,'lf%':0,'stage':0}
        return output

        
