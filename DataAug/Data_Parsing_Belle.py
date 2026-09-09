import os
import shutil
import sys
import numpy as np
import time
from scipy.fftpack import fft
import pywt
import torch
import neurokit2 as nk
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import ujson as json
from datetime import datetime, timedelta
import math
import glob
import csv
from time import strftime
from zipfile import ZipFile
from multiprocessing import Pool
import statistics
import torch.nn.functional as F
from pathlib import Path
import bisect

import random
from torch.utils.data.sampler import SubsetRandomSampler
##from torch.utils.data import WeightedRandomSampler


if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from SWMlib.motion import *  ##ahrs
from SWMlib.motion import motion_analysis
from SWMlib.ecg.rpeak import rpeak_detection 
from SWMlib.ecg.quality_check import ecg_quality_check_v3
from SWMlib.common import data_load_concate    
from SWMlib.common.calculation import normalization
from SWMlib.ecg.baseline import baseline_remove 
from SWMlib.ecg.noise_remove import remove_spike,emg_detector_remover 
# import Regression_Model_Predictor
# from Regression_Model_Predictor import GlucoseValuesPredictor


def get_version(): ###取得版本號

    return '007'


class DataArrangement:

    def __init__(self):
        self.basepath = os.path.dirname(__file__)
    

    ##def unzip_file(self,uuid,export_path):  ####將受測者zip檔解壓縮
    def unzip_file(self,uuid,server_db_path,export_path):  ####將受測者zip檔解壓縮


        uuid_export_path = os.path.join(export_path, uuid)        
        if not os.path.exists(uuid_export_path): 
            os.makedirs(uuid_export_path)      
       
        ##UnzipFileNameList = data_load_concate.search_unzip_file(db_path=db_path, target_uuid=uuid, start_time=start_time, end_time=end_time, export_path=uuid_export_path)  ##只適用於health server 
        ##self.serach_zip_file_and_unzip(uuid,server_db_path,start_time,end_time,uuid_export_path)   ##只適用於health server 
        
        zipfile_list=os.listdir(server_db_path)
        for zipfilename in zipfile_list:
            file_name_split_array=zipfilename.split('_') 
            if (file_name_split_array[0]==uuid):              
                print('zipped file name:',os.path.join(server_db_path,zipfilename))
                with ZipFile(os.path.join(server_db_path,zipfilename),"r") as zip:
                    zip.extractall(export_path)  ##export srj files in the path

        

    def data_processing(self,uuid,srj_db_path,glucosedata_path,base_path,server_db_path,processnum,splitting_ratio): ##測試用
    ##def data_processing(self,uuid,srj_db_path,glucosedata_path,base_path,processnum,splitting_ratio):
        
        errorcode="0"
        message=""
        
    
        ##測試用                
        if(server_db_path!=""):  ###測試用,需要抓雲端上的zip檔案 
            print('Start unzippig file!')      
            self.unzip_file(uuid,server_db_path,srj_db_path) ##自雲端資料夾中將ECG壓縮檔解壓縮成srj檔放置到srj_db_path路徑下                                 
    

        print('Start data parsing！') 
        print(glucosedata_path)
        errorcode, message = self.data_parsing(uuid,srj_db_path,glucosedata_path,base_path,processnum) ##srj檔分析後將ECG資料放置於base_path\RawData路徑下      
        if int(errorcode)<0:
            return errorcode, message
       
               
        
        print('Start data arrangement!') 
        errorcode, message = self.data_arrangement(uuid,base_path,splitting_ratio) ##根據splitting_ratio，將Dataset高中低資料分配至trainning與testing dataset
        if int(errorcode)<0:
            return errorcode, message
        
        message="data processing is done!"
        

        return errorcode, message
    

    
    def data_parsing(self,uuid,srj_db_path,glucosedata_path,base_path,processnum=8): 

        '''
        srj檔分析後得到對應的ECG訊號直接計算出每一拍R波前後的特徵，根據血糖值放置於Dataset中的高中低
        '''
        errorcode="0"
        message=""
        
        export_rawdata_path=os.path.join(base_path,"RawData")        
        if not os.path.exists(export_rawdata_path): 
            os.makedirs(export_rawdata_path) ###創建ECG擷取後存放資料夾  

        uuid_temp_path = os.path.join(export_rawdata_path, uuid)
        if not os.path.exists(uuid_temp_path): 
            os.makedirs(uuid_temp_path) ### 創建屬於uuid的資料夾 

        ###  EMG測試用
        high_emg_datapath=os.path.join(base_path,"high_emg_data")
        if not os.path.exists(high_emg_datapath): 
            os.makedirs(high_emg_datapath) ###創建ECG擷取後存放資料夾  

        uuid_high_emg_noise_path = os.path.join(high_emg_datapath, uuid)
        if not os.path.exists(uuid_high_emg_noise_path): 
            os.makedirs(uuid_high_emg_noise_path) ### 創建屬於uuid的資料夾 
        ## EMG測試用結束
        
        features_path=os.path.join(base_path, "Features")  ##創建特徵檔案資料夾
        if not os.path.exists(features_path): os.makedirs(features_path)

        features_uuid_path=os.path.join(features_path,uuid)
        if not os.path.exists(features_uuid_path): os.makedirs(features_uuid_path)   
        
        
        dataset_path=os.path.join(base_path,"Dataset") ##創建Dataset資料夾
        if not os.path.exists(dataset_path): os.makedirs(dataset_path)
            
        dataset_uuid_path_high=os.path.join(dataset_path,uuid,'High')
        if not os.path.exists(dataset_uuid_path_high): os.makedirs(dataset_uuid_path_high)
        
        dataset_uuid_path_low=os.path.join(dataset_path,uuid,'Low')
        if not os.path.exists(dataset_uuid_path_low): os.makedirs(dataset_uuid_path_low)
        
        dataset_uuid_path_normal=os.path.join(dataset_path,uuid,'Normal')
        if not os.path.exists(dataset_uuid_path_normal): os.makedirs(dataset_uuid_path_normal)        

        glucosedata_path=os.path.join(glucosedata_path,uuid+'.csv')
        glucose_csv_file_list=glob.glob(glucosedata_path)      
        ecgdata_path=os.path.join(srj_db_path,'*.srj')
        srj_file_list=glob.glob(ecgdata_path) 

        if(len(glucose_csv_file_list)==0):           
            errorcode="-102"
            message="An error occurs in the data-parsing function of Category_Regression_Model_Builder_Predictor.py: no glucose csv files exist!"          
            return errorcode, message
           
                
        ##try:
        if(True):        
            with open(glucose_csv_file_list[0], newline='', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f, delimiter=','))                
                chunks = self._split_chunks(rows,processnum)  
                args_list = [(chunk, srj_db_path, srj_file_list, uuid, uuid_temp_path, features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal) for chunk in chunks]

                with Pool(processes=processnum) as pool:
                    pool.starmap(self._process_rows, args_list)                
               
            f.close()
 
            self.leftover_feature_to_category(features_uuid_path,dataset_uuid_path_low,dataset_uuid_path_high,dataset_uuid_path_normal)  ##將剩餘的feature搬移到Dataset資料夾
             
        ##except:
        ##    errorcode="-103"
        ##    message="An error occurs in the data-parsing function of Category_Regression_Model_Builder_Predictor.py: fail to do multiprocessing analysis!"
        ##    return errorcode, message
  
        return errorcode, message
    
    
    def _process_rows(self,chunk_rows, srj_db_path, srj_file_list, uuid, uuid_temp_path, features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal):

        for row in chunk_rows:
            recored_time = row[0]
            glucose_value = row[1]
            if glucose_value != "":

                datetime_object = datetime.strptime(recored_time[2:], '%y/%m/%d %H:%M')

                if int(glucose_value) >= 250:
                    time_change = timedelta(minutes=10)
                elif int(glucose_value) >= 200:
                    time_change = timedelta(minutes=7)
                else:
                    time_change = timedelta(minutes=5)

                new_time_before = datetime_object - time_change
                new_time_after = datetime_object + time_change
                current_time_str = datetime_object.strftime("20%y%m%d %H%M%S")
                start_time_str = new_time_before.strftime("20%y%m%d %H%M%S")
                end_time_str = new_time_after.strftime("20%y%m%d %H%M%S")

                ##ecg_data_array, motion_array, _, _ = self._dataconcate(srj_db_path, srj_file_list, start_time_str, end_time_str)
                #ecg_data_array, motion_data_array,tt = self._dataconcate(srj_db_path, srj_file_list, start_time_str, end_time_str)

                sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all = self._load_all_srj_data(srj_db_path, srj_file_list, uuid)
                ecg_data_array, motion_data_array, breath_data_array, temp_data_array = self._query_by_time_range(sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all, start_time_str, end_time_str)

                ##for i, current_ecg_data in enumerate(ecg_data_array):
                end_index=len(ecg_data_array)
                for i in range(end_index):
                    current_motion_data=motion_data_array[i]
                    if(len(current_motion_data)==0): ##如果為空，這條ECG不處理                       
                        continue
                    
                    current_ecg_data=ecg_data_array[i]                   
                    ##flag=self._fast_motion_analysis(motion_data=current_motion_data)                   
                    flag = motion_analysis(motion_data=current_motion_data)

                    if (flag==0):  ##static  
                                                                                   
                        current_ecg_data=baseline_remove(current_ecg_data)  ##新增基線拉直
                        current_ecg_data = remove_spike(current_ecg_data, spike_threshold=500.0, fs=250) 
                        ecgs_10s_vg=nk.ecg_clean(current_ecg_data, sampling_rate=250, method="vg")
                        ##rpeak_array = rpeak_detection(current_ecg_data)                       
                        rpeak_array=rpeak_detection(ecgs_10s_vg)[1:] 
                        result = ecg_quality_check_v3(current_ecg_data, rpeak_array) 

                        if result == "Normal":  
                            ecg_norm=normalization(current_ecg_data)
                            clean_signal,high_noise_rpeak_indices=emg_detector_remover(ecg_norm, rpeak_array)
                            if(len(clean_signal)>0): ##低和中雜訊
                                current_time_str=current_time_str.replace(" ","")                                           
                                filename=uuid+'_'+current_time_str+'_0_'+str(i)+'_'+glucose_value+'.txt'                                                       
                                self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=0, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_temp_path)
                                self.feature_extraction_from_single_ecg(current_ecg_data,rpeak_array,int(glucose_value),filename,features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal) ###直接計算特徵後寫入到Features資料夾
                            
                            else:  
                                high_emg_datapath=os.path.join('D:\Dennis Project\DevelopingProject\server_program_2.1.1','Model','high_emg_data')
                                uuid_high_emg_noise_path = os.path.join(high_emg_datapath, uuid)
                                if not os.path.exists(uuid_high_emg_noise_path): 
                                    os.makedirs(uuid_high_emg_noise_path) ### 創建屬於uuid的資料夾 

                                self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=0, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_high_emg_noise_path)
                        else:
                            bad_datapath=os.path.join('D:\Dennis Project\DevelopingProject\server_program_2.1.1','Model','bad_data')
                            uuid_bad_path = os.path.join(bad_datapath, uuid)
                            if not os.path.exists(uuid_bad_path): 
                                os.makedirs(uuid_bad_path) ### 創建屬於uuid的資料夾 

                            self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=0, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_bad_path)
                    else:
                        dynamic_datapath=os.path.join('D:\Dennis Project\DevelopingProject\server_program_2.1.1','Model','dynamic_data')
                        uuid_dynamic_path = os.path.join(dynamic_datapath, uuid)
                        if not os.path.exists(uuid_dynamic_path): 
                            os.makedirs(uuid_dynamic_path) ### 創建屬於uuid的資料夾 

                        self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=1, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_dynamic_path)
                    
    def _load_all_srj_data(self, srj_db_path, srj_file_list, uuid):
        """
        一次性讀取並解析所有 srj 檔案，取代舊版在每筆 CGM 記錄都重新掃描的做法。
        只需要在整個 uuid 的 data_parsing 開始時呼叫一次。

        input ---
        srj_db_path: srj檔案所在路徑
        srj_file_list: srj檔案列表 (glob 取得，順序不保證按時間)

        output ---
        sorted_times: 經過排序後的 tt 時間序列 (datetime list)，供 bisect 查詢使用
        ecg_data_all, motion_data_all, breath_data_all, temp_data_all: 與 sorted_times 一一對應的資料序列
        """

        # 每個檔案先各自讀取，並在檔案內部做遞增檢查
        per_file_records = []  # list of (first_tt, [(tt, ecg, motion, breath, temp), ...])

        for index in range(len(srj_file_list)):
            current_file_path = os.path.join(srj_db_path, srj_file_list[index])
            file_records = []
            prev_tt = None

            with open(current_file_path, "r") as srj:
                line = srj.readline()
                line_no = 0
                while line:
                    line_no += 1
                    data = json.loads(line)
                    motions = data["rows"]["motions"]
                    ecgs = data["rows"]["ecgs"]
                    breaths = data["rows"]["breaths"]
                    temps = data["rows"]["temps"]
                    tt = data["tt"]
                    tt = int(tt / 1000)
                    nowtime = datetime.fromtimestamp(tt)

                    # ---- 檢查機制：確認檔案內部時間確實遞增 ----
                    if prev_tt is not None and nowtime < prev_tt:
                        raise ValueError(
                            f"[時間非遞增] uuid={uuid}, 檔案={srj_file_list[index]}, "
                            f"第 {line_no} 行時間 {nowtime} 早於前一行時間 {prev_tt}。"
                            f"請確認此 srj 檔案是否損毀或資料來源異常。"
                        )
                    prev_tt = nowtime
                    # ---------------------------------------------

                    file_records.append((nowtime, ecgs, motions, breaths, temps))
                    line = srj.readline()

            if len(file_records) > 0:
                per_file_records.append((file_records[0][0], file_records))

        # 檔案之間不重疊，但 glob 順序不保證按時間，故依每個檔案的第一筆時間排序檔案順序
        per_file_records.sort(key=lambda x: x[0])

        sorted_times = []
        ecg_data_all = []
        motion_data_all = []
        breath_data_all = []
        temp_data_all = []

        for _, file_records in per_file_records:
            for nowtime, ecgs, motions, breaths, temps in file_records:
                sorted_times.append(nowtime)
                ecg_data_all.append(ecgs)
                motion_data_all.append(motions)
                breath_data_all.append(breaths)
                temp_data_all.append(temps)

        # ---- 全域檢查機制：確認跨檔案合併後整體仍然遞增（因檔案不重疊，理論上必然成立）----
        for i in range(1, len(sorted_times)):
            if sorted_times[i] < sorted_times[i - 1]:
                raise ValueError(
                    f"[跨檔案時間非遞增] uuid={uuid}, 索引 {i} 的時間 {sorted_times[i]} "
                    f"早於索引 {i-1} 的時間 {sorted_times[i-1]}。"
                    f"請確認 srj 檔案之間是否真的不重疊。"
                )
        # -------------------------------------------------------------------------

        return sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all


    def _query_by_time_range(self, sorted_times, ecg_data_all, motion_data_all,
                            breath_data_all, temp_data_all, start_time, end_time):
        """
        在已排序的時間序列上用二元搜尋取出落在 [start_time, end_time] 區間內的資料，
        取代舊版 _dataconcate 內的一對一掃描（第604行的if判斷）。

        input ---
        sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all: _load_all_srj_data 產出的結果
        start_time, end_time: 字串，格式同原本 _dataconcate 的輸入 (可能為 "20240304 234651" 或 "20240304")

        output ---
        ecg_data_array, motion_data_array, breath_data_array, temp_data_array: 與舊版 _dataconcate 相同格式的輸出
        """

        if len(start_time) > 8:  # 有時分秒之格式
            start_time = datetime.strptime(start_time, "%Y%m%d %H%M%S")
            end_time = datetime.strptime(end_time, "%Y%m%d %H%M%S")
        else:  # 只有年月日之格式
            start_time = datetime.strptime(start_time, "%Y%m%d")
            end_time = datetime.strptime(end_time, "%Y%m%d")

        # bisect_left: 第一個 >= start_time 的位置
        # bisect_right: 第一個 > end_time 的位置
        # 這樣切出來的範圍 [left_idx, right_idx) 就等同於原本的 start_time <= nowtime <= end_time
        left_idx = bisect.bisect_left(sorted_times, start_time)
        right_idx = bisect.bisect_right(sorted_times, end_time)

        ecg_data_array = ecg_data_all[left_idx:right_idx]
        motion_data_array = motion_data_all[left_idx:right_idx]
        breath_data_array = breath_data_all[left_idx:right_idx]
        temp_data_array = temp_data_all[left_idx:right_idx]

        return ecg_data_array, motion_data_array, breath_data_array, temp_data_array

                                                             
                          

    def _split_chunks(self, data, num_chunks):
        chunk_size = math.ceil(len(data) / num_chunks)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    
    def feature_extraction_from_single_ecg(self,ecg_data_array,rpeak_array,glucose_value,file_name,features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal): 

        '''
        擷取ECG特徵放於Features資料夾，並於分配於DataSet資料夾中的normal,high,low資料夾中
        '''

        errorcode="0"
        message=""         

        if(True):
            count=0
            mean_wave=np.zeros(150) ###特徵長度150
            if(len(rpeak_array)>0):
                ##for i in range(0,len(rpeak_array)):
                for i in range(1,len(rpeak_array)-1):  ##第一拍和最後一拍不做平均
                    r_index=int(rpeak_array[i])
                    before_index=r_index-50 ##取R波前50
                    after_index=r_index+100 ##取R波後100
                    if(before_index<0 or after_index>=len(ecg_data_array)):
                        continue
            
                    current_wave=np.array(ecg_data_array[before_index:after_index])
                    mean_wave=mean_wave+current_wave
                    count=count+1

            if count>0:
                mean_wave=mean_wave/count  ###取得平均波形
                mean_wave=mean_wave.tolist()                
                if(glucose_value>=180): ###180以上視為high
                    file_destination_path=dataset_uuid_path_high+"\\"+file_name 
                elif(glucose_value<=80): ###80以下視為low                            
                    file_destination_path=os.path.join(dataset_uuid_path_low,file_name) 
                elif(glucose_value>=85 and glucose_value<=170):  ###中間血糖設定為85~170之間                                   
                    file_destination_path=os.path.join(dataset_uuid_path_normal,file_name)  
                else: ##介於80~85，170~180之間的特徵資料先放到Feature資料夾
                    file_destination_path=os.path.join(features_uuid_path,file_name)

                  
                f = open(file_destination_path, 'w')
                for i,ecgvalue in enumerate(mean_wave):
                    f.write(str(ecgvalue))
                    f.write("\n")
            
                f.close()
               

        return errorcode, message

    def leftover_feature_to_category(self,features_uuid_path,dataset_uuid_path_low,dataset_uuid_path_high,dataset_uuid_path_normal):
        
        errorcode="0"
        message=""    
        
        ##--------move leftover feature into high,low,normal------------
       
        ##features_uuid_path=os.path.join(base_path,"Features",uuid) 
        filelist=os.listdir(features_uuid_path)  ###自Features資料夾搬移到Dataset資料夾(分成高中低)
      
            
        highvalue_count=len(os.listdir(dataset_uuid_path_high))
        lowvalue_count=len(os.listdir(dataset_uuid_path_low))
        normalvalue_count=len(os.listdir(dataset_uuid_path_normal))
        
        try:           
            if(len(filelist)>0): ##有介於80~85 170~180之間的數值
                for k in range(len(filelist)):                   
                    filestr=filelist[k]
                    filestr_array=filestr.split('_')
                    valuestr=filestr_array[-1]
                    valuestr_array=valuestr.split('.')
                    value=int(valuestr_array[0])
                    file_source_path=os.path.join(features_uuid_path,filestr)
                    if(value>80 and value<85):
                        if(lowvalue_count<normalvalue_count and value<=82): ##低血糖筆數較少，且數值小於82，歸給低血糖，否則捨棄不用         
                            file_destination_path=os.path.join(dataset_uuid_path_low,filestr)
                            shutil.move(file_source_path, file_destination_path)                 
                    elif(value>170 and value<180):
                        if(normalvalue_count<highvalue_count and value<=175): ##中血糖筆數較少，且數值小於175，歸給中血糖，否則捨棄不用               
                            file_destination_path=os.path.join(dataset_uuid_path_normal,filestr) 
                            shutil.move(file_source_path, file_destination_path)
        
        except:
            errorcode="-201"
            message="An error occurs in the feature_extraction function of Category_Regression_Model_Builder_Predictor.py: fail to seperate data into high, low, normal categories!"
        
    
        return errorcode, message

    
    def data_arrangement(self,uuid,base_path,splitting_ratio=""): 
        
        errorcode="0"
        message=""
    
        
        splitting_ratio_value=float((splitting_ratio.split("_"))[0])/100.0     
             
        file_destination_path=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid)  
        if not os.path.exists(file_destination_path): os.makedirs(file_destination_path)  ##創建給AI模型訓練使用的training,testing資料夾
        
        path_train=os.path.join(file_destination_path,'Train')
        if not os.path.exists(path_train): os.makedirs(path_train)
        
        path_normal=os.path.join(file_destination_path,'Train','Normal')
        if not os.path.exists(path_normal): os.makedirs(path_normal)

        path_high=os.path.join(file_destination_path,'Train','High')
        if not os.path.exists(path_high): os.makedirs(path_high)

        path_low=os.path.join(file_destination_path,'Train','Low')
        if not os.path.exists(path_low): os.makedirs(path_low)


        path_test=os.path.join(file_destination_path,'Test')
        if not os.path.exists(path_test): os.makedirs(path_test)
        
        path_normal=os.path.join(file_destination_path,'Test','Normal')
        if not os.path.exists(path_normal): os.makedirs(path_normal)

        path_high=os.path.join(file_destination_path,'Test','High')
        if not os.path.exists(path_high): os.makedirs(path_high)

        path_low=os.path.join(file_destination_path,'Test','Low')
        if not os.path.exists(path_low): os.makedirs(path_low)
        
        if(True):
        ##try:            
            data_path=os.path.join(base_path,"Dataset",uuid,'Normal')
            ##errorcode, message = self._movedata(uuid,base_path,data_path,file_destination_path,'Normal',splitting_ratio_value)            
            errorcode, message = self._movedata(data_path,file_destination_path,'Normal',splitting_ratio_value) 

            data_path=os.path.join(base_path,"Dataset",uuid,'High')                      
            ##errorcode, message = self._movedata(uuid,base_path,data_path,file_destination_path,'High',splitting_ratio_value)            
            errorcode, message = self._movedata(data_path,file_destination_path,'High',splitting_ratio_value)    
            
            data_path=os.path.join(base_path,"Dataset",uuid,'Low')
            ##errorcode, message = self._movedata(uuid,base_path,data_path,file_destination_path,'Low',splitting_ratio_value)
            errorcode, message = self._movedata(data_path,file_destination_path,'Low',splitting_ratio_value)

        '''
        except:
            errorcode="-300"
            message="An error occurs in the data_arrangement function of Category_Regression_Model_Builder_Predictor.py: fail to run the _movedata function!"
            return errorcode,message
        '''
        
        try:
            self._data_balance(file_destination_path,"Train") ###做data balance
            self._data_balance(file_destination_path,"Test") ###做data balance
        except:
            errorcode="-301"
            message="An error occurs in the data_arrangement function of Category_Regression_Model_Builder_Predictor.py: fail to run the _data_balance function!"
            return errorcode,message
        


        return errorcode, message    
    
    
    def _data_balance(self,file_source_path,type):      
        
        '''
        將file_source_path(GlucoseData/uuid)下normal,high,low資料夾做資料平衡處理(多餘的資料會移到RemoveData資料夾)
        type: Train 或 Test
        '''
       
        print('start data balancing!')
        data_num_array=[]
        path_normal=os.path.join(file_source_path,type,'Normal')
        normal_data_num=len(os.listdir(path_normal))
        data_num_array.append(normal_data_num)
        
        path_high=os.path.join(file_source_path,type,'High')
        high_data_num=len(os.listdir(path_high))
        data_num_array.append(high_data_num)
        
        path_low=os.path.join(file_source_path,type,'Low')
        low_data_num=len(os.listdir(path_low))
        data_num_array.append(low_data_num)
      
        data_num_sorted_array=sorted(data_num_array)
        max_data_num=data_num_sorted_array[2]

        if(data_num_sorted_array[0]==0): ##最少樣本是0          
            min_data_num=data_num_sorted_array[1]  ##取非0之最少樣本         
        else:
            min_data_num=data_num_sorted_array[0]
        
        if(min_data_num>0):
            data_num_ratio=max_data_num/min_data_num
        
            if(max_data_num>3000):  ###最大樣本超過3000筆才需要考慮資料平衡
                if(data_num_ratio>=5): ##最多樣本數大於最小樣本數超過5倍才需要考慮資料平衡
                    mode_number=5
                    if(data_num_ratio>=8 and data_num_ratio<=15):
                        mode_number=(int(data_num_ratio)-2)
                    elif(data_num_ratio>15): ##超過15倍
                        mode_number=10

                    if(max_data_num==normal_data_num): ##最多樣本的是normal
                        target_dir_name="Normal"
                    elif(max_data_num==high_data_num):##最多樣本的是high
                        target_dir_name="High" 
                    else:
                        target_dir_name="Low" 

                    balance_dir_path=os.path.join(file_source_path,type,target_dir_name)
            
                    filename_list=os.listdir(balance_dir_path)
                    for i in range(len(filename_list)):
                        if(i%mode_number !=0): ##移除到remove_data
                            currentpath=os.path.join(balance_dir_path,filename_list[i])
                            copy_to_dir=os.path.join(file_source_path,'Remove_Data',type)  
                            if not os.path.exists(copy_to_dir): 
                                os.makedirs(copy_to_dir)

                            copytopath=os.path.join(copy_to_dir,filename_list[i])
                            if(not os.path.isfile(copytopath)):  
                                if(os.path.isfile(currentpath)): 
                                    shutil.move(currentpath,copytopath)     
    
    
    def _movedata(self, data_path, newpath, type, splitting_ratio_value): 
        """
        資料找時間分割點分割成training 與 testing
        
        input:
            data_path: 資料來源路徑
            newpath: 目標路徑
            type: 資料類型 (High/Low/Normal)
            splitting_ratio_value: 分割比例 (訓練集比例)
        """
        
        errorcode = "0"
        message = ""
        
        # 根據類型設定血糖值範圍
        if type == "High":
            lowest_value = 180      # 高血糖最低值
            highest_value = 500     # 高血糖最高值
        elif type == "Low":
            lowest_value = 30       # 低血糖最低值
            highest_value = 80      # 低血糖最高值
        else:  # Normal
            lowest_value = 85       # 正常血糖最低值
            highest_value = 175     # 正常血糖最高值
        
        ###rawdata_path = data_path
        
        ## 檢查原始資料路徑是否存在
        """
        if not os.path.exists(rawdata_path):
            message = f'RawData path does not exist: {rawdata_path}'
            return errorcode, message
        """

        # 取得並排序所有檔案列表
        rawdata_file_list = sorted(os.listdir(data_path)) ##rawdata_path))
               
        # ====== 步驟1: 收集所有檔案的日期資訊 ======
        dates_all = []
        
        for filename in rawdata_file_list:
            ##try:
                # 從檔名中解析日期 (格式: xxx_YYYYMMDD_xxx)
            parts = filename.split("_")
            date_str = parts[1][0:8]  # 取得前8個字元作為日期
            file_date = datetime.strptime(date_str, "%Y%m%d")
            dates_all.append(file_date)
            ##except Exception as e:
                # 如果檔名格式錯誤，跳過此檔案
            ##    pass

        ## 檢查是否有收集到日期資料
        if not dates_all:            
            message="An error occurs in the _movedata function of Category_Regression_Model_Builder_Predictor.py: no data is collected!"
            return errorcode, message

        ## 取得所有不重複的日期並排序
        dates_unique = sorted(set(dates_all))
        total_special_days = len(dates_unique)

        ## 至少需要2個不同日期才能進行分割
        if total_special_days < 2:           
            message = "An error occurs in the _movedata function of Category_Regression_Model_Builder_Predictor.py: Insufficient data for splitting (need data from at least 2 different dates)."
            return errorcode, message              
        

        ## ====== 步驟2: 計算分割時間點 ======
        ## 根據分割比例計算訓練集應包含的天數
        days_in_r_percent = int(total_special_days * splitting_ratio_value)
        if days_in_r_percent < 1:
            days_in_r_percent = 1  # 至少要有1天的訓練資料
        

        ## 設定分割日期 (此日期及之前的為訓練集，之後的為測試集)
        split_datetime = dates_unique[days_in_r_percent - 1]
              
        # ====== 步驟3: 尋找血糖值的最大值和最小值 ======
        file_list = os.listdir(data_path)
        
        min_glucose_value = 10000   # 初始化最小值 (用大數值)
        max_glucose_value = -10000  # 初始化最大值 (用小數值)
        
        # 掃描所有檔案，找出在指定範圍內的最大和最小血糖值
        for i in range(len(file_list)):
            current_item = file_list[i]
            try:
                # 從檔名中解析血糖值 (檔名最後一個數字)
                current_item_str_array = current_item.split('_')
                current_last_str = current_item_str_array[-1]
                current_last_value_str = current_last_str.split('.')[0]  # 移除副檔名
                current_glucose_value = int(current_last_value_str)
                
                # 只統計在指定範圍內的血糖值
                if current_glucose_value >= lowest_value and current_glucose_value <= highest_value:
                    if current_glucose_value < min_glucose_value:
                        min_glucose_value = current_glucose_value
                    if current_glucose_value > max_glucose_value:
                        max_glucose_value = current_glucose_value
            except Exception as e:
                # 檔名格式錯誤，跳過
                pass
        
        # ====== 步驟4: 收集所有最小值和最大值的檔案 ======
        min_files = []  # 儲存最小血糖值的檔案
        max_files = []  # 儲存最大血糖值的檔案
        
        # 確認有找到有效的最大最小值
        if min_glucose_value != 10000 and max_glucose_value != -10000:
            for i in range(len(file_list)):
                current_item = file_list[i]
                try:
                    # 再次解析血糖值
                    current_item_str_array = current_item.split('_')
                    current_last_str = current_item_str_array[-1]
                    current_last_value_str = current_last_str.split('.')[0]
                    current_glucose_value = int(current_last_value_str)
                    
                    # 將最小值和最大值的檔案分別收集起來
                    if current_glucose_value == min_glucose_value:
                        min_files.append(current_item)
                    elif current_glucose_value == max_glucose_value:
                        max_files.append(current_item)
                            
                except Exception as e:
                    pass
        
        # ====== 步驟5: 按比例分割最小值檔案 ======
        if len(min_files) > 0:
            
            # 計算分割索引位置
            split_index_min = int(len(min_files) * splitting_ratio_value)
            if split_index_min == 0 and len(min_files) > 0:
                split_index_min = 1  # 至少要有1個檔案在訓練集
            
            # 分配檔案到訓練集或測試集
            for idx, filename in enumerate(min_files):
                try:
                    current_path = os.path.join(data_path, filename)
                    
                    if idx < split_index_min:
                        # 前面的檔案分配到訓練集
                        copyto_path = os.path.join(newpath, 'Train', type)
                        os.makedirs(copyto_path, exist_ok=True)
                        
                        if os.path.isfile(current_path):
                            shutil.move(current_path, copyto_path)
                    else:
                        # 後面的檔案分配到測試集
                        copyto_path = os.path.join(newpath, 'Test', type)
                        os.makedirs(copyto_path, exist_ok=True)
                        
                        if os.path.isfile(current_path):
                            shutil.move(current_path, copyto_path)
                            
                except Exception as e:
                    pass
        
        # ====== 步驟6: 按比例分割最大值檔案 ======
        if len(max_files) > 0:
            
            # 計算分割索引位置
            split_index_max = int(len(max_files) * splitting_ratio_value)
            if split_index_max == 0 and len(max_files) > 0:
                split_index_max = 1  # 至少要有1個檔案在訓練集
            
            # 分配檔案到訓練集或測試集
            for idx, filename in enumerate(max_files):
                try:
                    current_path = os.path.join(data_path, filename)
                    
                    if idx < split_index_max:
                        # 前面的檔案分配到訓練集
                        copyto_path = os.path.join(newpath, 'Train', type)
                        os.makedirs(copyto_path, exist_ok=True)
                        
                        if os.path.isfile(current_path):
                            shutil.move(current_path, copyto_path)
                    else:
                        # 後面的檔案分配到測試集
                        copyto_path = os.path.join(newpath, 'Test', type)
                        os.makedirs(copyto_path, exist_ok=True)
                        
                        if os.path.isfile(current_path):
                            shutil.move(current_path, copyto_path)
                            
                except Exception as e:
                    pass
        
        # ====== 步驟7: 根據時間分配剩餘的檔案 ======
        
        # 處理每個剩餘的檔案
        file_list = os.listdir(data_path)  ##重讀一次資料夾看看剩下的檔案
        for filename in file_list: 
            try:
                # 從檔名中解析日期時間
                filename_splited_array = filename.split("_")
                current_file_timestr = filename_splited_array[1]
                current_datetime = datetime.strptime(current_file_timestr[0:-6], "%Y%m%d")
                
                current_path = os.path.join(data_path, filename)
                
                # 根據日期決定分配到訓練集還是測試集
                if current_datetime > split_datetime: 
                    # 日期晚於分割點 -> 測試集
                    copyto_path = os.path.join(newpath, 'Test', type)
                    os.makedirs(copyto_path, exist_ok=True)
                    
                    if os.path.isfile(current_path):
                        shutil.move(current_path, copyto_path)
                else:    
                    # 日期等於或早於分割點 -> 訓練集
                    copyto_path = os.path.join(newpath, 'Train', type)
                    os.makedirs(copyto_path, exist_ok=True)
                    
                    if os.path.isfile(current_path):
                        shutil.move(current_path, copyto_path)
                                
            except Exception as e:
                # 處理失敗的檔案跳過
                pass

        return errorcode, message
    
    def _dataconcate(self,srj_db_path,srj_file_list,start_time,end_time): ##資料串接

        """
        input ---
        db_path: srj檔案所在路徑
        srj_file_list: srj檔案列表
        start_time: 設定串接資料的開始日期(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
        end_time: 設定串接資料的結束日期(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
            
        output ---
        ecg_data_array: 所有srj檔中每個tt片段ECG資料串接後的序列
        motion_data_array:  所有srj檔中每個tt片段motion資料串接後的序列
        breath_data_array:  所有srj檔中每個tt片段breath資料串接後的序列
        temp_data_array: 所有srj檔中每個tt片段temp資料串接後的序列
        """

        ecg_data_array=[]
        motion_data_array=[]
        tt_array=[]
        ##breath_data_array=[]
        ##temp_data_array=[]

        if(len(start_time)>8): ##有時分秒之格式
            start_time = datetime.strptime(start_time,"%Y%m%d %H%M%S")
            end_time = datetime.strptime(end_time,"%Y%m%d %H%M%S")
        else:   ##只有年月日之格式
            start_time = datetime.strptime(start_time,"%Y%m%d")
            end_time=datetime.strptime(end_time,"%Y%m%d") 
      
        for index in range(len(srj_file_list)):            
            current_file_path=os.path.join(srj_db_path,srj_file_list[index])   
            with open(current_file_path,"r") as srj:
                line = srj.readline()
                while line:                    
                    data = json.loads(line)
                    motions = data["rows"]["motions"]
                    ecgs=data["rows"]["ecgs"]
                    ##breaths=data["rows"]["breaths"]
                    ##temps=data["rows"]["temps"]
                    tt_ori=data["tt"]
                    tt=int(tt_ori/1000)
                    nowtime=datetime.fromtimestamp(tt)
                    
                    if(nowtime>=start_time and nowtime<=end_time): ###針對當天的時段Concate Data                                                              
                        tt_array.append(tt_ori)
                        motion_data_array.append(motions)                    
                        ecg_data_array.append(ecgs) 
                        ##breath_data_array.append(breaths)                         
                        ##temp_data_array.append(temps)         
                
                    line = srj.readline()          

        return ecg_data_array,motion_data_array,tt_array   ##,breath_data_array,temp_data_array


    def _write_ecg_data(self,uuid,ecg_data_array,time_str,flag,index,glucose_value,uuid_temp_path):         
        
        time_str=time_str.replace(" ","")
        path = os.path.join(uuid_temp_path,uuid+'_'+time_str+'_'+str(flag)+'_'+str(index)+'_'+glucose_value+'.txt')
        f = open(path, 'w')
            
        for i,ecgvalue in enumerate(ecg_data_array): 
            f.write(str(ecgvalue))
            f.write("\n")
        
        f.close()

        

def Clean_Data(uuid,basepath,splitting_ratio=""):    
    
    errorcode="0"
    message=""

    try:
        raw_data_path=os.path.join(basepath,"RawData",uuid)
        if os.path.isdir(raw_data_path):
            shutil.rmtree(raw_data_path)
      
        dataset_data_path=os.path.join(basepath,"Dataset",uuid)
        if os.path.isdir(dataset_data_path):
            shutil.rmtree(dataset_data_path)        

        features_data_path=os.path.join(basepath,"Features",uuid)
        if os.path.isdir(features_data_path):
            shutil.rmtree(features_data_path)       
    
        motion_data_path=os.path.join(basepath,"2500_motion",uuid)
        if os.path.isdir(motion_data_path):
            shutil.rmtree(motion_data_path)   

        regression_dataset_path=os.path.join(basepath,"Dataset_Regression",uuid)
        if os.path.isdir(regression_dataset_path):
            shutil.rmtree(regression_dataset_path)     
      
        glucose_data_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid)
        if os.path.isdir(glucose_data_path):
            shutil.rmtree(glucose_data_path)  

        glucose_regression_data_path=os.path.join(basepath,splitting_ratio,"GlucoseRegressionData",uuid)
        if os.path.isdir(glucose_regression_data_path):
            shutil.rmtree(glucose_regression_data_path)  

        regression_features_path=os.path.join(basepath,splitting_ratio,"Regression_Features",uuid)
        if os.path.isdir(regression_features_path):
            shutil.rmtree(regression_features_path)     
        

        model_data_path=os.path.join(basepath,splitting_ratio,"Temp_ThreeClasses_Model",uuid)
        if os.path.isdir(model_data_path):
            shutil.rmtree(model_data_path)          

        model_data_path=os.path.join(basepath,splitting_ratio,"Temp_TwoClasses_Model",uuid)
        if os.path.isdir(model_data_path):
            shutil.rmtree(model_data_path)
   
    
        errorcode="0"
        message="All data for uuid "+uuid+" has been deleted!"     
    except:
        errorcode="-800"
        message="An error occurs in the Clean_Data function of Category_Regression_Model_Builder_Predictor.py: Some data for uuid "+uuid+" has not deleted clearly!"   

    return errorcode,message   


def Clean_Model(uuid,basepath,splitting_ratio=""):    
    
    empty_state=0
    errorcode="0"
    message=""  
    
    try:       
        model_path=os.path.join(basepath,splitting_ratio,"Best_ThreeClasses_Model",uuid)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            errorcode="0"
            message="Model for uuid "+uuid+" has been deleted!"
        else:
            empty_state=1           

        model_path=os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Model",uuid)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            errorcode="0"
            message="Model for uuid "+uuid+" has been deleted!" 
        else:
            if(empty_state==1):
                errorcode="-801"
                message="An error occurs in the Clean_Model function of Category_Regression_Model_Builder_Predictor.py: No built model for uuid "+uuid+" exists, it can not be cleaned!"            
              
    except:
        errorcode="-802"
        message="An error occurs in the Clean_Model function of Category_Regression_Model_Builder_Predictor.py: Model for uuid:"+uuid+" has not deleted clearly!"


    try:       
        model_path=os.path.join(basepath,splitting_ratio,"Best_ThreeClasses_Regression_Model",uuid)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            errorcode="0"
            message="Model for uuid "+uuid+" has been deleted!"
        else:
            empty_state=1           

        model_path=os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Regression_Model",uuid)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            errorcode="0"
            message="Model for uuid "+uuid+" has been deleted!" 
        else:
            if(empty_state==1):
                errorcode="-803"
                message="An error occurs in the Clean_Model function of Category_Regression_Model_Builder_Predictor.py: No built regression model for uuid "+uuid+" exists, it can not be cleaned!"            
              
    except:
        errorcode="-804"
        message="An error occurs in the Clean_Model function of Category_Regression_Model_Builder_Predictor.py: Regression model for uuid:"+uuid+" has not deleted clearly!"


    return errorcode,message    


def DataParsing(uuid,base_path,srj_db_path,glucosedata_path,server_db_path,processnum=8,splitting_ratio="70_30"): ##測試用
##def BuildModel(uuid,base_path,srj_db_path,glucosedata_path,processnum=8,splitting_ratio="70_30"):

    errorcode="0"
    message="" 
    status=-1   
           
    ##--------building category model-------  
    DataArrangement_Obj = DataArrangement() 
    ##errorcode, message = DataArrangement_Obj.data_processing(uuid,start_time,end_time,srj_db_path,glucosedata_path,base_path,server_db_path,processnum,splitting_ratio)  ##測試用
    errorcode, message = DataArrangement_Obj.data_processing(uuid,srj_db_path,glucosedata_path,base_path,server_db_path,processnum,splitting_ratio)  
    ##errorcode, message = DataArrangement_Obj.data_processing(uuid,srj_db_path,glucosedata_path,base_path,processnum,splitting_ratio)
        
    if(int(errorcode)<0):
        return status, errorcode, message     
    
   
    return status, errorcode, message   
 

if __name__ == "__main__":

    user_information = [
                        ['2197','20250212','20250225'],       ##T1003(績效良，但可再訓練)
                        ['2200','20250212','20250226'],       ##T1005(績效優)  
                        ['2133','20250212','20250225'],       ##T1006(績效可，70%左右)  
                        ['2131','20250212','20250225'],       ##T1007(績效良，但可再訓練)  
                        ['2208','20250217','20250302'],       ##T1011(績效良，但可再訓練)         
                        ['2205','20250217','20250302'], ##5   ##T1014(績效良，但可再訓練，注意血糖值有到500的情況)
                        ['2202','20250217','20250302'],       ##T1015(績效良，但可再訓練)                     
                        ['2215','20250219','20250304'],       ##T1020(績效優)                                                                                         
                        ['2223','20250226','20250311'],       ##T1026(績效良)                      
                        ['2230','20250226','20250311'],       ##T1028(績效超級優)                                    
                        ['2235','20250312','20250326'], ##10  ##T1032(績效普通)
                        ['2246','20250312','20250320'],       ##T1034(績效良，可在訓練)
                        ['2249','20250313','20250326'],       ##T1035(績效普通，需再訓練)                                                   
                        ['2253','20250320','20250402'],       ##T1037(績效普通，要再訓練)                     
                        ['2261','20250401','20250414'],       ##T1038(績效尚可，還可再訓練)                   
                        ['2262','20250408','20250421'], ##15  ##T1039(績效普通，train and test各有2筆血糖資料,已使用special move測試)       ##待測試                                    
                        ['2257','20250408','20250421'],       ##T1040(績效普通，可再訓練)
                        ['2199','20250212','20250225'],         ##T1001(績效尚可,使用special move後有進步) 
                        ['2198','20250212','20250225'],         ##T1004(績效極度不平衡)                                  ##待測試
                        ['2196','20250212','20250226'], ##19    ##T1002(高血糖只有2筆,未能訓練)
                        ['2206','20250217','20250228'],         ##T1008(test只有兩筆，績效尚可有點不平衡可再訓練或觀察)
                        ['2201','20250217','20250302'],         ##T1009  ##可能只做到2/28(無高、低血糖資料)
                        ['2204','20250217','20250228'],         ##T1010(test high 只有一筆，績效不好可再訓練或觀察，spcial move有進步)
                        ['2203','20250217','20250302'],         ##T1012(train中的高血糖資料只有兩筆，績效普通，可再訓練或觀察)              ##可再次測試   
                        ['2207','20250217','20250302'],         ##T1013(special move後績效依然不好)      
                        ['2210','20250217','20250302'], ##25    ##T1016(test的高血糖只有1筆，績效不好，已經測試2次)              
                        ['2209','20250219','20250304'],         ##T1017(test的高血糖只有2天4筆，test and train個只有2筆, 已經執行過4次訓練) 
                        ['2218','20250219','20250304'],         ##T1018(高血糖資料很多，但績效不好過度失衡，還沒使用specail move測試過)
                        ['2219','20250219','20250304'],         ##T1021(只有中、低血糖，無高血糖資料，已訓練回歸)                                       ##要能練low and normal
                        ['2217','20250219','20250304'],         ##T1022(只有中、低血糖，無高血糖資料，已訓練回歸)                                       ##要能練low and normal
                        ['2216','20250219','20250304'], ##30    ##T1023(績效依然不好，special move測試多次) 
                        ['2214','20250219','20250304'],         ##T1024(高血糖資料train 只有2筆，test沒有資料，已使用specail move，low血糖很多)  ##要能練low and normal 
                        ['2226','20250226','20250312'],         ##T1025(高血糖只有2筆且在同一天,未能處理) 
                        ['2224','20250226','20250311'],         ##T1027(高血糖資料很少只有3筆，績效不好)                                    ##待測試
                        ['2227','20250226','20250312'],         ##T1029(高血糖資料很多，但績效差，不平衡，使用specail move測試) 
                        ['2225','20250226','20250312'], ##35    ##T1030(高血糖只有1筆、低血糖資料只有4筆)                                  ##待測試
                        ['2236','20250310','20250323'],         ##T1031(高血糖資料只有1筆)                                                 ##待測試
                        ['2248','20250312','20250325'],         ##T1033(績效不好，需再訓練，更新後績效變差，要特別注意)                        ##待測試
                        ['2251','20250319','20250401'],         ##T1036(績效極不平衡，需再訓練)                                              ##待測試                       
                        ['2279','20250426','20250509'],         ##T1041  
                        ['2276','20250426','20250509'],  ##40   ##T1042
                        ['2278','20250426','20250509'],         ##T1043
                        ['2281','20250502','20250515'],         ##T1044
                        ['2286','20250509','20250522'],         ##T1045
                        ['2285','20250509','20250522'],         ##T1046
                        ['2294','20250520','20250531'], ##45    ##T1047
                        ['2293','20250520','20250602'],         ##T1048
                        ['2295','20250522','20250604'],         ##T1049
                        ['2291','20250527','20250610'],         ##T1050
                        ['2298','20250604','20250617'],         ##T1051(19歲，小於20歲)
                        ['2300','20250604','20250617'],  ##50   ##T1052
                        ['2299','20250604','20250617'],         ##T1053  ##訓練到一半出問題
                        ['2304','20250606','20250619'],         ##T1054
                        ['2305','20250606','20250619'],         ##T1055
                        ['2306','20250606','20250619'],         ##T1056
                        ['2130','20250623','20250706'],  ##55   ##T1057  ##test沒有高血糖
                        ['2132','20250623','20250706'],         ##T1058
                        ['2322','20250623','20250706'],         ##T1059
                        ['2329','20250701','20250713'],         ##T1060
                        ['2352','20250731','20250813'],         ##T1061 
                        ['2378','20251022','20251104'], ##60    ##T1062 
                        ['2397','20251129','20251213'],         ##T1063 
                        ['2221','20250219','20250220'],         ##提早退出                        
                        ]                                
  
    mother_path=Path(__file__).resolve().parent   ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    base_path=mother_path/"Model"
    base_path=str(base_path)
    ##glucosedata_path=r'D:\Dennis Project\Glucose_CSV\BGM_CSV\filtered_BGM' ##mother_path/"GlucoseDataCSV"  ###APP收到的血糖數值，整理成CSV檔後存放路徑
    glucosedata_path=r'D:\Dennis Project\Glucose_CSV\CGM_CSV'
    glucosedata_path=str(glucosedata_path)
    server_db_path=r'G:\.shortcut-targets-by-id\1dZUAXwQHDvBJGYwQLNpizVJHhkOfnxVa\Health_Server_Script\_rawdata_download'
   
         
    for i in range(0,63): 

        print('index:',i)       
        
        ##-------step 1. 基本設定--------
        user_info=user_information[i]
        uuid=user_info[0]                  
        start_time=user_info[1]  ##第一筆血糖資料記錄日期
        end_time=user_info[2]    ##最後一筆血糖資料紀錄日期
        print('index:',i,' uuid:',uuid)

        if(uuid !='2279' and uuid!='2276' and uuid!='2286' and uuid!='2285' and uuid!='2306' and uuid!='2322' and uuid!='2329' and uuid!='2352'
           and uuid!='2215' and uuid!='2221' and uuid!='2227' and uuid!='2226' and uuid!='2131' and uuid!='2199'):
            continue
        
        srj_db_path='D:\\DataDB\\'+uuid   ###srj檔放置路徑
                
        start_time = time.time()       
        ##-------step 2. 血糖模型建立---------
        ##status, errorcode, message = BuildModel(uuid, base_path, srj_db_path, glucosedata_path, processnum=8, splitting_ratio="70_30")  ##建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
        status, errorcode, message = DataParsing(uuid,base_path,srj_db_path,glucosedata_path,server_db_path,processnum=8,splitting_ratio="70_30")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函式執行時間: {execution_time:.6f} 秒")

        print('prcoessed uuid:', uuid, ' status:',str(status),' error code:',errorcode,' message:',message)
        

      