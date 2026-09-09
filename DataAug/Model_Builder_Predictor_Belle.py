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
from torch.utils.data import DataLoader
import torch.optim as optim
###import torch.nn.functional as F
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
from pathlib import Path
import bisect

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from SWMlib.motion import *  ##ahrs
from SWMlib.motion import static_motion_analysis
from SWMlib.motion import motion_analysis
from SWMlib.ecg.rpeak import rpeak_detection
from SWMlib.ecg.quality_check import ecg_quality_check
from SWMlib.ecg.quality_check import ecg_quality_check_v3
from SWMlib.common import data_load_concate
from SWMlib.common.calculation import normalization
from SWMlib.ecg.baseline import baseline_remove
from SWMlib.ecg.noise_remove import remove_spike, emg_detector_remover

def get_version(): ###取得版本號

    return '007'

class DataArrangement:

    def __init__(self):
        self.basepath = os.path.dirname(__file__)


    def unzip_file(self,uuid,server_db_path,export_path):  ####將受測者zip檔解壓縮

        uuid_export_path = os.path.join(export_path, uuid)
        if not os.path.exists(uuid_export_path):
            os.makedirs(uuid_export_path)

        zipfile_list=os.listdir(server_db_path)
        for zipfilename in zipfile_list:
            file_name_split_array=zipfilename.split('_')
            if (file_name_split_array[0]==uuid):
                print('zipped file name:',os.path.join(server_db_path,zipfilename))
                with ZipFile(os.path.join(server_db_path,zipfilename),"r") as zip:
                    zip.extractall(export_path)  ##export srj files in the path


    def data_processing(self,uuid,srj_db_path,glucosedata_path,base_path,server_db_path,processnum,splitting_ratio): ##測試用

        errorcode="0"
        message=""

        ##測試用
        if(server_db_path!=""):  ###測試用,需要抓雲端上的zip檔案
            print('Start unzippig file!')
            self.unzip_file(uuid,server_db_path,srj_db_path) ##自雲端資料夾中將ECG壓縮檔解壓縮成srj檔放置到srj_db_path路徑下


        print('Start data parsing！')
        print(glucosedata_path)
        errorcode, message = self.data_parsing(uuid,srj_db_path,glucosedata_path,base_path,processnum) ##srj檔分析後將ECG資料放置於base_path\RawData路徑上
        if int(errorcode)<0:
            return errorcode, message


        print('Start data arrangement!')
        errorcode, message = self.data_arrangement(uuid,base_path,splitting_ratio) ##根據splitting_ratio，將Dataset資料夾中血糖值分配至trainning和testing dataset
        if int(errorcode)<0:
            return errorcode, message

        message="data processing is done!"

        return errorcode, message


    def data_parsing(self,uuid,srj_db_path,glucosedata_path,base_path,processnum=8):

        '''
        srj檔分析得到的每一段ECG訊號直接計算出每一個R波峰值附近的特徵，根據血糖值放置於Dataset中的高中低
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
            message="An error occurs in the data-parsing function of Model_Builder_Predictor_Belle.py: no glucose csv files exist!"
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
        ##    message="An error occurs in the data-parsing function of Model_Builder_Predictor_Belle.py: fail to do multiprocessing analysis!"
        ##    return errorcode, message

        return errorcode, message


    def _process_rows(self,chunk_rows, srj_db_path, srj_file_list, uuid, uuid_temp_path, features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal):

        ##整個chunk(單一process)只需要載入一次srj資料，避免每一筆血糖記錄都重新掃描檔案
        sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all = self._load_all_srj_data(srj_db_path, srj_file_list, uuid)

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

                ecg_data_array, motion_data_array, breath_data_array, temp_data_array = self._query_by_time_range(sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all, start_time_str, end_time_str)

                end_index=len(ecg_data_array)
                for i in range(end_index):
                    current_motion_data=motion_data_array[i]
                    if(len(current_motion_data)==0): ##如果為空，逃避ECG片段
                        continue

                    current_ecg_data=ecg_data_array[i]
                    flag = motion_analysis(motion_data=current_motion_data)

                    if (flag==0):  ##static

                        current_ecg_data=baseline_remove(current_ecg_data)  ##新增基線漂移校正
                        current_ecg_data = remove_spike(current_ecg_data, spike_threshold=500.0, fs=250)
                        ecgs_10s_vg=nk.ecg_clean(current_ecg_data, sampling_rate=250, method="vg")
                        rpeak_array=rpeak_detection(ecgs_10s_vg)[1:]
                        result = ecg_quality_check_v3(current_ecg_data, rpeak_array)

                        if result == "Normal":
                            ecg_norm=normalization(current_ecg_data)
                            clean_signal,high_noise_rpeak_indices=emg_detector_remover(ecg_norm, rpeak_array)
                            if(len(clean_signal)>0): ##低雜訊過關
                                current_time_str=current_time_str.replace(" ","")
                                filename=uuid+'_'+current_time_str+'_0_'+str(i)+'_'+glucose_value+'.txt'
                                self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=0, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_temp_path)
                                self.feature_extraction_from_single_ecg(current_ecg_data,rpeak_array,int(glucose_value),filename,features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal) ###直接計算特徵並寫入到Features資料夾

                            else:
                                high_emg_datapath=os.path.join(os.path.dirname(uuid_temp_path).replace("RawData","high_emg_data"))
                                uuid_high_emg_noise_path = os.path.join(high_emg_datapath, uuid)
                                if not os.path.exists(uuid_high_emg_noise_path):
                                    os.makedirs(uuid_high_emg_noise_path) ### 創建屬於uuid的資料夾

                                self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=0, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_high_emg_noise_path)
                        else:
                            bad_datapath=os.path.join(os.path.dirname(uuid_temp_path).replace("RawData","bad_data"))
                            uuid_bad_path = os.path.join(bad_datapath, uuid)
                            if not os.path.exists(uuid_bad_path):
                                os.makedirs(uuid_bad_path) ### 創建屬於uuid的資料夾

                            self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=0, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_bad_path)
                    else:
                        dynamic_datapath=os.path.join(os.path.dirname(uuid_temp_path).replace("RawData","dynamic_data"))
                        uuid_dynamic_path = os.path.join(dynamic_datapath, uuid)
                        if not os.path.exists(uuid_dynamic_path):
                            os.makedirs(uuid_dynamic_path) ### 創建屬於uuid的資料夾

                        self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data, time_str=current_time_str, flag=1, index=i, glucose_value=glucose_value, uuid_temp_path=uuid_dynamic_path)


    def _split_chunks(self, data, num_chunks):
        chunk_size = math.ceil(len(data) / num_chunks)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


    def feature_extraction_from_single_ecg(self,ecg_data_array,rpeak_array,glucose_value,file_name,features_uuid_path,dataset_uuid_path_high,dataset_uuid_path_low,dataset_uuid_path_normal):

        '''
        將ECG特徵放於Features資料夾，並且分配於DataSet資料夾中的normal,high,low資料夾中
        '''

        errorcode="0"
        message=""

        if(True):
            count=0
            mean_wave=np.zeros(150) ###特徵長度150
            if(len(rpeak_array)>0):
                for i in range(1,len(rpeak_array)-1):  ##第一個和最後一個不列入平均
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
                    file_destination_path=os.path.join(dataset_uuid_path_high,file_name)
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
            message="An error occurs in the leftover_feature_to_category function of Model_Builder_Predictor_Belle.py: fail to seperate data into high, low, normal categories!"

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
            data_path=os.path.join(base_path,"Dataset",uuid,'Normal')
            errorcode, message = self._movedata(data_path,file_destination_path,'Normal',splitting_ratio_value)

            data_path=os.path.join(base_path,"Dataset",uuid,'High')
            errorcode, message = self._movedata(data_path,file_destination_path,'High',splitting_ratio_value)

            data_path=os.path.join(base_path,"Dataset",uuid,'Low')
            errorcode, message = self._movedata(data_path,file_destination_path,'Low',splitting_ratio_value)

        try:
            self._data_balance(file_destination_path,"Train") ###做data balance
            self._data_balance(file_destination_path,"Test") ###做data balance
        except:
            errorcode="-301"
            message="An error occurs in the data_arrangement function of Model_Builder_Predictor_Belle.py: fail to run the _data_balance function!"
            return errorcode,message


        return errorcode, message


    def _data_balance(self,file_source_path,type):

        '''
        將file_source_path(GlucoseData/uuid)中normal,high,low資料夾做資料平衡(多餘的資料搬移到RemoveData資料夾)
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
        資料放置位置依日期切割成training 和 testing

        input:
            data_path: 資料來源路徑
            newpath: 目標路徑
            type: 資料類型 (High/Low/Normal)
            splitting_ratio_value: 切割比例 (訓練集比例)
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

        # 取得並排序所有檔案列表
        rawdata_file_list = sorted(os.listdir(data_path))

        # ====== 步驟1: 收集所有檔案的日期資訊 ======
        dates_all = []

        for filename in rawdata_file_list:
            # 從檔名中解析日期 (格式: xxx_YYYYMMDD_xxx)
            parts = filename.split("_")
            date_str = parts[1][0:8]  # 取得前8個字元作為日期
            file_date = datetime.strptime(date_str, "%Y%m%d")
            dates_all.append(file_date)

        ## 檢查是否有收集到日期資料
        if not dates_all:
            message="An error occurs in the _movedata function of Model_Builder_Predictor_Belle.py: no data is collected!"
            return errorcode, message

        ## 取得所有不重複的日期並排序
        dates_unique = sorted(set(dates_all))
        total_special_days = len(dates_unique)

        ## 至少需要2個不同日期才能進行切割
        if total_special_days < 2:
            message = "An error occurs in the _movedata function of Model_Builder_Predictor_Belle.py: Insufficient data for splitting (need data from at least 2 different dates)."
            return errorcode, message


        ## ====== 步驟2: 計算切割時間點 ======
        ## 根據切割比例計算訓練集所包含的天數
        days_in_r_percent = int(total_special_days * splitting_ratio_value)
        if days_in_r_percent < 1:
            days_in_r_percent = 1  # 至少要有1天的訓練資料


        ## 設定切割日期 (此日期以之前的為訓練集，之後的為測試集)
        split_datetime = dates_unique[days_in_r_percent - 1]

        # ====== 步驟3: 找出血糖值的最大值和最小值 ======
        file_list = os.listdir(data_path)

        min_glucose_value = 10000   # 初始化最小值 (用大數值)
        max_glucose_value = -10000  # 初始化最大值 (用小數值)

        # 掃描所有檔案，找出在指定範圍內的最大和最小血糖值
        for i in range(len(file_list)):
            current_item = file_list[i]
            try:
                # 從檔名中解析血糖值 (檔名最後一段數字)
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

        # ====== 步驟7: 根據日期分配剩餘的檔案 ======

        # 處理每個剩餘的檔案
        file_list = os.listdir(data_path)  ##重新讀取資料夾中剩餘的檔案
        for filename in file_list:
            try:
                # 從檔名中解析日期時間
                filename_splited_array = filename.split("_")
                current_file_timestr = filename_splited_array[1]
                current_datetime = datetime.strptime(current_file_timestr[0:-6], "%Y%m%d")

                current_path = os.path.join(data_path, filename)

                # 根據日期決定分配到訓練集或測試集
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

    def _write_ecg_data(self,uuid,ecg_data_array,time_str,flag,index,glucose_value,uuid_temp_path):         
        
        time_str=time_str.replace(" ","")
        path = os.path.join(uuid_temp_path,uuid+'_'+time_str+'_'+str(flag)+'_'+str(index)+'_'+str(glucose_value)+'.txt')
        f = open(path, 'w')
        for i,ecgvalue in enumerate(ecg_data_array):
            f.write(str(ecgvalue))
            f.write("\n")
        
        f.close()   
    
                

class ecgDataset(Dataset):
    def __init__(self, dir_path, method='combine', classes=3, augment=False):
        self.dir_path = os.path.abspath(dir_path)
        self.method = method
        self.data_len = 150
        self.augment = augment

        if(classes==2):
            normalSigs = self.read_files('Normal')
            normalLabels = np.zeros(len(normalSigs))

            abnormSigs = self.read_files('High') 
            abnormLabels = np.ones(len(abnormSigs))
        
            self.Signals = torch.cat((normalSigs, abnormSigs), 0)      
            self.Labels = np.concatenate((normalLabels, abnormLabels), axis = 0).reshape(-1, 1)
            self.Labels = torch.from_numpy(self.Labels)
        
        else:
            normalSigs = self.read_files('Normal')
            normalLabels = np.zeros(len(normalSigs)) 
               
            abnormSigs = self.read_files('High')
            abnormLabels =  np.ones(len(abnormSigs)) 
                
            lowSigs = self.read_files('Low')
            lowLabels = 2*np.ones(len(lowSigs))      
                
            self.Signals = torch.cat((normalSigs, abnormSigs,lowSigs), 0)      
            self.Labels = np.concatenate((normalLabels, abnormLabels,lowLabels), axis = 0).reshape(-1, 1)
            self.Labels = torch.from_numpy(self.Labels)


    def __getitem__(self, index):
        signal = self.Signals[index]
        label = self.Labels[index]

        if self.augment:
            signal = self.augment_signal(signal)

        return signal, label

    def __len__(self):
        return len(self.Labels)

    # 對已完成特徵轉換的訊號做輕量資料增強：加雜訊、振幅縮放、時間平移(只用於training data)
    def augment_signal(self, signal):
        if torch.rand(1).item() < 0.5:  ##加高斯雜訊
            noise_std = 0.02
            signal = signal + torch.randn_like(signal) * noise_std

        if torch.rand(1).item() < 0.5:  ##振幅隨機縮放0.9~1.1倍
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            signal = signal * scale

        if torch.rand(1).item() < 0.5:  ##時間軸小幅平移
            shift = int(torch.randint(-5, 6, (1,)).item())
            signal = torch.roll(signal, shifts=shift, dims=-1)

        return signal

    # Parsing files in folder
    def read_files(self, foldername):
        Sig = []
        for filename in os.listdir(os.path.join(self.dir_path, foldername)):
            file_path = os.path.join(self.dir_path, foldername, filename)
            Sig.append(self.load_data(file_path))
        Sig = torch.FloatTensor(Sig)
        if self.channel == 1:
            Sig = Sig.unsqueeze(1)
        
        return Sig

    # Read data in files & Preprocessing
    def load_data(self, filepath):
        # read data in csv file
        values = np.genfromtxt(filepath, delimiter = '')
        
        # put data into fixed length signal
        sig = np.zeros(self.data_len)
        if len(values)>=self.data_len:
            sig[:self.data_len] = values[:self.data_len]
        else:
            sig[:len(values)] = values
            
        sig.reshape(self.data_len, 1)
        
        self.channel = 1
        
        if self.method == 'raw':
            return sig.tolist()
        
        if self.method == 'time':
            sig = self.normalize1(sig)
            return sig.tolist()
        
        if self.method == 'freq':
            sig = self.normalize1(sig)
            #sig = self.FFT(sig)
            sig = self.DWT(sig)
            self.channel = len(sig)          
            return sig.tolist()
        
        if self.method == 'combine':
            sig = self.normalize1(sig)
            x1 = sig.tolist()
            #x2 = self.FFT(sig).tolist()
            x3 = self.DWT(sig).tolist()
            #Comb = [x1, x2, x3]
            x3.append(x1)
            self.channel = len(x3)
            return x3
    
    def normalize1(self, x):
        x_max = np.max(x)
        x_min = np.min(x)
        x_norm = (x - x_min) / (x_max - x_min+1)  
        return x_norm
    
    def FFT(self, x):
        y = fft(x)
        P2 = abs(y) / len(x)
        P1 = P2[range(len(x)//2)]
        return P2
    
    def DWT(self, x):
        ##sig = np.zeros(2504)
        sig = np.zeros(150)
        sig[:len(x)] = x
        coeffs = pywt.swt(sig, 'sym4', level=1, trim_approx=True)
        coeffs = np.array(coeffs)

        return coeffs


class ecgDatasetSubset(Dataset):
    """
    包裝 ecgDataset 的子集合(依索引挑選)，讓 train/valid 可以共用同一份已預先載入、
    做完特徵轉換的資料，但各自獨立控制是否啟用 data augmentation。
    """
    def __init__(self, base_dataset, indices, augment=False):
        self.base_dataset = base_dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_index = self.indices[idx]
        signal = self.base_dataset.Signals[real_index]
        label = self.base_dataset.Labels[real_index]

        if self.augment:
            signal = self.base_dataset.augment_signal(signal)

        return signal, label


class CNN(torch.nn.Module):

    def __init__(self, input_channel, data_len=150): ##2500

        self.data_len = data_len
        self.input_channel = input_channel
        super(CNN, self).__init__()  
        ## define the layers
        self.conv1 = nn.Sequential(nn.Conv1d(self.input_channel,64,kernel_size=3,padding=1),
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))                              

        self.conv2 = nn.Sequential(nn.Conv1d(64,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv3 = nn.Sequential(nn.Conv1d(64,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU(),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool1d(5))                                  
        
        self.conv4 = nn.Sequential(nn.Conv1d(64,128,kernel_size=3,padding=1),      
                                   nn.BatchNorm1d(128),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv5 = nn.Sequential(nn.Conv1d(128,128,kernel_size=3,padding=1),      
                                   nn.BatchNorm1d(128),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv6 = nn.Sequential(nn.Conv1d(128,128,kernel_size=3,padding=1),      
                                   nn.BatchNorm1d(128),
                                   ##nn.ReLU(),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool1d(5))
                                   
        
        self.conv7 = nn.Sequential(nn.Conv1d(128,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv8 = nn.Sequential(nn.Conv1d(64,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv9 = nn.Sequential(nn.Conv1d(64,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU(),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool1d(5))   
        
        
        self.conv10 = nn.Sequential(nn.Conv1d(64,32,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(32),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01)) 
        
        '''
        self.conv11 = nn.Sequential(nn.Conv1d(32,32,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(32),
                                   nn.ReLU())
            
        self.conv12 = nn.Sequential(nn.Conv1d(32,32,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(32),
                                   nn.ReLU())
                            
        
        
        self.dense = nn.Sequential(nn.Linear(32*(self.data_len//(5*5*5)), 128),  
                                   nn.ReLU(),                                  
                                   nn.Linear(128,64),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),                                  
                                   nn.Linear(64,32),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),                                   
                                   nn.Linear(32, 16),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),                                  
                                   nn.Linear(16, 3)  
                                   )
        '''

                
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3= nn.Linear(32, 3)      
       
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) 
        x = self.conv6(x) 
        x = self.conv7(x) 
        x = self.conv8(x)
        x = self.conv9(x) 
        x = self.conv10(x)         
        batch_size, timesteps, featurn_size = x.size()         
        r_in = x.view(batch_size, timesteps, -1) 
        x = torch.reshape(r_in, (batch_size,32))
        x=x.unsqueeze(0)
        out, (h_n, h_c) = self.lstm(x)        
        x = self.fc1(out[-1])        
        x = self.fc2(x)
        x = self.fc3(x) 
        
        return x
    

class TwoClasses_CNN(torch.nn.Module):

    def __init__(self, input_channel, data_len=150): 

        self.data_len = data_len
        self.input_channel = input_channel
        ##print('self.input_channel:',self.input_channel)
        super(TwoClasses_CNN, self).__init__()        

        ## define the layers
        self.conv1 = nn.Sequential(nn.Conv1d(self.input_channel,64,kernel_size=3,padding=1),
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))                                 
        
        self.conv2 = nn.Sequential(nn.Conv1d(64,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv3 = nn.Sequential(nn.Conv1d(64,64,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(64),
                                   ##nn.ReLU(),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool1d(5))                                  
        
        self.conv4 = nn.Sequential(nn.Conv1d(64,128,kernel_size=3,padding=1),      
                                   nn.BatchNorm1d(128),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01)) 
        
        self.conv5 = nn.Sequential(nn.Conv1d(128,128,kernel_size=3,padding=1),      
                                   nn.BatchNorm1d(128),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv6 = nn.Sequential(nn.Conv1d(128,128,kernel_size=3,padding=1),      
                                   nn.BatchNorm1d(128),
                                   ##nn.ReLU(),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool1d(5))
                                   
        
        self.conv7 = nn.Sequential(nn.Conv1d(128,32,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(32),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv8 = nn.Sequential(nn.Conv1d(32,32,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(32),
                                   ##nn.ReLU())
                                   nn.LeakyReLU(0.01))
        
        self.conv9 = nn.Sequential(nn.Conv1d(32,32,kernel_size=3,padding=1),       
                                   nn.BatchNorm1d(32),
                                   ##nn.ReLU(),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool1d(5)) 
      
        '''
        self.dense = nn.Sequential(nn.Linear(32*(self.data_len//(5*5*5)), 128),  ##2048
                                   nn.ReLU(),
                                   ##nn.Dropout(p=0.5),
                                   nn.BatchNorm1d(128),
                                   nn.Linear(128,64),   ###(2048,1024)
                                   nn.ReLU(),
                                   ##nn.Dropout(p=0.5),
                                   nn.BatchNorm1d(64),
                                   nn.Linear(64,32),  ##(1024,512)
                                   nn.ReLU(),
                                   ##nn.Dropout(p=0.5),
                                   nn.Linear(32, 16),  ##(512, 256)
                                   nn.ReLU(),
                                   nn.BatchNorm1d(16),
                                   ##nn.Dropout(p=0.5),
                                   nn.Linear(16, 8)   ##(256, 1)
                                   )  
       
        '''
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=3)
        self.fc1 = nn.Linear(128, 64)
        #self.fc2 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) 
        x = self.conv6(x) 
        x = self.conv7(x) 
        x = self.conv8(x)
        x = self.conv9(x)
        ##x = self.conv10(x)      
        ##x = x.view(-1, 32*(self.data_len//(5*5*5))) ## reshaping        
        ##x = x.view(-1, 16*(self.data_len//(5*5*5))) ## reshaping        
        ##x = self.dense(x)
        
        batch_size, timesteps, featurn_size = x.size()         
        r_in = x.view(batch_size, timesteps, -1) 
        x = torch.reshape(r_in, (batch_size,32))
        x=x.unsqueeze(0)
        out, (h_n, h_c) = self.lstm(x)
        x = self.fc1(out[-1])        
        x = self.fc2(x)
        #x = self.fc3(x) 
     
        return x
               

def initialize_weights(model):
        for m in model.modules():
            
            if isinstance(m, nn.Conv1d):  # 判断是否為Conv1d
                ##torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)                
              
            if isinstance(m, nn.Linear):               
                torch.nn.init.kaiming_uniform_(m.weight)  
           
            elif isinstance(m, nn.BatchNorm1d):              
                m.weight.data.fill_(1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

def _majority_filtering(glucose_list):
        
        majority_result=0 ##先視為正常血糖
        category_array=[0, 0, 0]
        
        for i in range(len(glucose_list)):
            current_category=glucose_list[i]
            category_array[current_category]=category_array[current_category] + 1

        maxindex_array=[i for i,val in enumerate(category_array) if (val==max(category_array))]   
        if(len(maxindex_array)==1): ###只有一個類別佔多數
            majority_result=maxindex_array[0] ##直接輸出
        else: ###有多個類別佔多數
            for k in range(len(maxindex_array)): ###檢查是否有低或高血糖
                if(maxindex_array[k]==2):   ##低血糖priority最高
                    majority_result=2
                    break
                elif(maxindex_array[k]==1): ###高血糖次之
                    majority_result=1
        
        return majority_result


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

        high_emg_data_path=os.path.join(basepath,"high_emg_data",uuid)
        if os.path.isdir(high_emg_data_path):
            shutil.rmtree(high_emg_data_path)


        glucose_data_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid)
        if os.path.isdir(glucose_data_path):
            shutil.rmtree(glucose_data_path)

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
        message="An error occurs in the Clean_Data function: Some data for uuid "+uuid+" has not deleted clearly!"

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
                message="An error occurs in the Clean_Model function: No built model for uuid "+uuid+" exists, it can not be cleaned!"

    except:
        errorcode="-802"
        message="An error occurs in the Clean_Model function: Model for uuid:"+uuid+" has not deleted clearly!"

    return errorcode,message


##def BuildModel(uuid,basepath,srj_db_path,glucosedata_path,server_db_path=""): ##測試用
def BuildModel(uuid,basepath,srj_db_path,glucosedata_path,server_db_path="",processnum=8,splitting_ratio="70_30"):

    errorcode="0"
    message=""
    status=-1


    DataArrangement_Obj = DataArrangement()
    errorcode, message = DataArrangement_Obj.data_processing(uuid, srj_db_path, glucosedata_path, basepath, server_db_path, processnum, splitting_ratio)


    if(int(errorcode)<0):
        return status, errorcode, message

    checkedpath=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Train","Low")
    filelist_low_train=os.listdir(checkedpath)

    checkedpath=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test","Low")
    filelist_low_test=os.listdir(checkedpath)

    checkedpath=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Train","High")
    filelist_high_train=os.listdir(checkedpath)

    checkedpath=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test","High")
    filelist_high_test=os.listdir(checkedpath)

    checkedpath=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Train","Normal")
    filelist_normal_train=os.listdir(checkedpath)

    checkedpath=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test","Normal")
    filelist_normal_test=os.listdir(checkedpath)


    if(len(filelist_low_train)>0 and len(filelist_low_test)>0 and len(filelist_high_train)>0 and len(filelist_high_test)>0 and len(filelist_normal_train)>0 and len(filelist_normal_test)>0):  ###有中，低和高血糖資料
        status, errorcode, message=BuildModel_ThreeClasses(uuid,basepath,splitting_ratio)
        if(int(errorcode)>=0):
            message="Model with three classes has been built!"
    elif(len(filelist_high_train)==0 or len(filelist_high_test)==0 or len(filelist_normal_train)==0 or len(filelist_normal_test)==0):
        errorcode="-402"
        message="An error occurs in the BuildModel function: No enough normal or high glucose data!"
        status=-1
    else:
      status, errorcode, message=BuildModel_TwoClasses(uuid,basepath,splitting_ratio)
      if(int(errorcode)>=0):
        message="Model with two classes has been built!"

    return status, errorcode, message
 

def BuildModel_ThreeClasses(uuid,basepath,splitting_ratio=""):

    errorcode="0"
    message=""
    status=-1

    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間
    code_version=get_version()   ######取得模型訓練時當前程式版本

    print('Start building the model with three classes!')
    ## the model have been built or not and read performance file
    model_output_folder = os.path.join(basepath, splitting_ratio, "Best_ThreeClasses_Model", uuid)
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    model_data_path=os.path.join(basepath,splitting_ratio,"Temp_ThreeClasses_Model",uuid)  ###先清空過去訓練的模型佔存資料
    if os.path.isdir(model_data_path):
        shutil.rmtree(model_data_path)

    model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_ThreeClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")
    if not os.path.exists(model_current_best_performance_txtfile): ###沒有檔案，先創造檔案
        with open(model_current_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0")
            file.write("\n")
            file.write("Specificity:0")
            file.write("\n")
            file.write("F1:0")
            file.write("\n")
            file.write("Acc:0")
            file.write("\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(0, 0, 0, 0))


    ### Create dataset object
    Method = 'combine'

    current_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,'Train')

    filelist_low=os.listdir(os.path.join(current_path,'Low'))  
    filelist_high=os.listdir(os.path.join(current_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_path,'Normal'))    

    if(len(filelist_low)==0 or len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-400"
        message="An error occurs in the BuildModel_ThreeClasses function: No training data"      
        status=-1
        return status, errorcode, message 
    
    dataset = ecgDataset(dir_path = current_path, method=Method) 
    
    Epoch_range = range(0, 800) 
    patience =50
    Batch_size = 128 
    valid_split = 0.1
    shuffle_flag = True
    random_seed = 10

    ## Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(valid_split * dataset_size)
    if shuffle_flag :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    ## 只對training data做data augmentation, validation/test data維持原樣
    train_dataset = ecgDatasetSubset(dataset, train_indices, augment=True)
    valid_dataset = ecgDatasetSubset(dataset, val_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Batch_size)
    current_test_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,'Test')

    filelist_low=os.listdir(os.path.join(current_test_path,'Low'))
    filelist_high=os.listdir(os.path.join(current_test_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_test_path,'Normal'))

    if(len(filelist_low)==0 or len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-401"
        message="An error occurs in the BuildModel_ThreeClasses function: No testing data"

        return errorcode,message

    testdata = ecgDataset(dir_path = current_test_path, method=Method)
    test_loader = DataLoader(testdata, batch_size=Batch_size)
    train_num, valid_num, test_num = len(train_dataset),len(valid_dataset),len(testdata)
    total_num = train_num + valid_num + test_num

    print("#Train:%5d(%.2f), #Validation:%5d(%.2f), #Test:%5d(%.2f)\n" 
          %(train_num,train_num/total_num,valid_num,valid_num/total_num,test_num,test_num/total_num))   
     
       
    loopindex=0    
    current_best_sensitivity=0  ###本次訓練最佳sensitivity先設定為0
    current_best_specificity=0
    current_best_f1=0
    current_best_accs=0
    current_best_TP=0
    current_best_FP=0
    current_best_FN=0
    current_best_TN=0
    
    save_path = os.path.join(basepath,splitting_ratio,"Temp_ThreeClasses_Model",uuid) ###建立uuid專屬的模型存放資料夾  
    if os.path.isdir(save_path) is True:
        shutil.rmtree(save_path)
    os.makedirs(save_path)
     
    while(loopindex<20):  ###執行20次訓練，取最好一次

        loopindex=loopindex+1

        ## set GPU Resource
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        print('GPU Count:',torch.cuda.device_count())

        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    
        if Epoch_range[0] == 0:
            Loss_list = []
            ACCs = []
    
            model = CNN(data_len=dataset.data_len, input_channel=dataset.channel)
            ##initialize_weights(model)  ###模型權重初始化
            model = model.to(device)
            print('CNN model created.')
            
            
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-3)
        ##optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=1e-3)
       
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience, cooldown=0, min_lr=0.00001)

        n_patience = 0
        min_valid_loss = 0
        epoch_number = 0        
        

        for epoch in Epoch_range:
            print("Epoch: %d " % (epoch+1))    
            epoch_number= epoch+1    
            train_loss, valid_loss = [], []
    
            ## training phase 
            model.train()    
            correct, total = 0, 0
            temp, total_num = 0, len(train_loader)
    
            for data, target in train_loader:
                temp += 1                   
                ##print('\r' + '[Progress]:[%s%s]%.2f%%; ' 
                ##    % ('█' * int(temp*20/total_num), ' ' * (20-int(temp*20/total_num)), float(temp/total_num*100)), end='')
        
                # 資料移至 GPU加速
                data = data.to(device)
                target = target.to(device)

                # 梯度歸零
                optimizer.zero_grad()

                # Foward
                output = model(data)
                    
        
                y_pred=torch.sigmoid(output)
                result=torch.argmax(y_pred, dim=1)  
                result=torch.unsqueeze(result, 1)
                correct += (result == target).sum().item() 
        
                total += target.size(0)  
        
                # Backward + Optimize       
                loss = loss_function(output,target.squeeze().long())
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
    
            train_acc = 100*correct/total
            print("Train ACC: %.2f%% , " %(train_acc), end='')
        
            ## evaluation phase     
            model.eval()
            correct, total = 0, 0
            for data, target in valid_loader:
                   
                # 資料移至 GPU加速
                data = data.to(device)
                target = target.to(device)
 
                # 預測
                output = model(data)       
                ##correct += (torch.sigmoid(output).ge(0.5) == target).sum().item()
                y_pred=torch.sigmoid(output)
                result=torch.argmax(y_pred, dim=1) 
                result=torch.unsqueeze(result, 1)
                correct += (result == target).sum()
        
                total += target.size(0)
        
                # 損失函數      
                loss = loss_function(output, target.squeeze().long())
                valid_loss.append(loss.item())
        
     
            valid_acc = 100*correct/total
            print("Valid ACC: %.2f%% , " %(valid_acc), end='')
    
            train_loss = np.nanmean(train_loss)
            valid_loss = np.nanmean(valid_loss)
            print("Traing loss: %.4f, Valid loss: %.4f\n" %(train_loss, valid_loss))
    
            scheduler.step(valid_loss)
    
            Loss_list.append([train_loss, valid_loss]) 
            ACCs.append([train_acc, valid_acc])
    
            if valid_loss > min_valid_loss:
                if epoch == 0:
                    min_valid_loss = valid_loss
                else:
                    n_patience += 1
            else:
                min_valid_loss = valid_loss
                n_patience = 0
                torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"Model_"+str(uuid)+"_"+str(epoch+1)+".pth"))
        
            if n_patience >= patience:
                ##print("The model didn't improve for %i rounds, break it!" % patience)
                break
    
        torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"Model_"+str(uuid)+"_"+str(epoch+1)+".pth"))
    
        ####-----------------Test Model----------------------       
        num_epoch = epoch_number      
        model = CNN(data_len=dataset.data_len, input_channel=dataset.channel)
        model.to(device)        
        model_path = os.path.join(basepath,splitting_ratio,"Temp_ThreeClasses_Model",uuid,"Model_"+str(uuid)+"_"+str(num_epoch)+".pth")
        model.load_state_dict(torch.load(model_path))

        test_old_data = 0
        if test_old_data:           
            testdata = ecgDataset(dir_path = current_test_path, method=Method)
            test_loader = DataLoader(testdata, batch_size=Batch_size)
    
        model.eval()
        correct, total = 0, 0
        TN, FP, FN, TP = 0, 0, 0, 0
        for data, target in test_loader:
            data = data.to(device)
    
            ## 預測
            output = model(data)
            output = output.to('cpu')
                
            predict=np.zeros(len(output))
            answer=torch.sigmoid(output)
            predict = torch.argmax(answer, dim=1) 
            target=target.squeeze()   
            total += target.size(0)
    
            predict = predict.tolist()
            target = target.tolist()
            for i in range(0, len(predict)):
                if (predict[i] == 1 and target[i] == 1) or (predict[i] == 2 and target[i] == 2):
                    TP += 1
                    correct+=1
                elif (predict[i] == 1 and target[i] == 0) or (predict[i] == 2 and target[i] == 0):
                    FP += 1
                elif (predict[i] == 0 and target[i] == 1) or (predict[i] == 0 and target[i] == 2):
                    FN += 1
                elif (predict[i] == 0 and target[i] == 0):
                    TN += 1
                    correct+=1

        if((TP+FN)>0):                
            sensitivity=(100*TP/(TP+FN))
        else:
            sensitivity=0

        if((TN+FP)>0):    
            specificity=(100*TN/(TN+FP)) ##(100*TP/(TP+FP))
        else:
            specificity=0

        if((specificity + sensitivity)>0):
            f1=2 * (specificity * sensitivity) / (specificity + sensitivity)
        else:
            f1=0
        
        accs=100*correct/total
    
        print("TP:%d, FP:%d, FN:%d, TN:%d\n" %(TP, FP, FN, TN))
        print("Sensitivity: %.2f %%\n" %sensitivity)
        print("Specificity: %.2f %%\n" %specificity)
        print("F1: %.2f %%\n" %f1)
        print("Final ACC: %.2f %%\n" %accs) 

        diff_sensitivity = sensitivity - current_best_sensitivity
        diff_specificity = specificity - current_best_specificity
        breakthrough_flag = False
        if(diff_sensitivity > 0 and diff_specificity > 0):  ##sensitivity和specificity兩者都上升
            breakthrough_flag = True
        elif((current_best_sensitivity > current_best_specificity) and diff_sensitivity <= 0 and diff_specificity >= 0 and abs(
                diff_sensitivity) <= abs(diff_specificity)):  ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
            breakthrough_flag = True
        elif((current_best_sensitivity > current_best_specificity) and diff_sensitivity >= 3 and diff_specificity >= -2):  ##sensitivity較大，但新的sensitivity上升>=3% 且新specificity只下降2%以內
            breakthrough_flag = True
        elif((current_best_specificity > current_best_sensitivity) and diff_specificity <= 0 and diff_sensitivity >= 0 and abs(
                diff_specificity) <= abs(diff_sensitivity)):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
            breakthrough_flag = True
        elif((current_best_specificity > current_best_sensitivity) and diff_specificity >= 3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
            breakthrough_flag = True

        if (breakthrough_flag):
            current_best_specificity=specificity
            current_best_f1=f1
            current_best_sensitivity=sensitivity
            current_best_accs=accs
            current_best_TP=TP
            current_best_FP=FP
            current_best_FN=FN
            current_best_TN=TN

            torch.save(model.state_dict(),os.path.join(model_output_folder,"BestModel_"+code_version+"_"+current_time_str+".pth"))  ###比目前訓練階段的模型績效好，儲存起來            
            model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_ThreeClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")            
            with open(model_current_best_performance_txtfile, "w") as file:
                file.write("Sensitivity:%.2f" %(sensitivity))
                file.write("\n")
                file.write("Specificity:%.2f" %(specificity))
                file.write("\n")
                file.write("F1:%.2f" %(f1))
                file.write("\n")
                file.write("Acc:%.2f" %(accs))
                file.write("\n")
                file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(TP, FP, FN, TN))           


    model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_ThreeClasses_Model", uuid,"Historic_Best_Performance.txt") ###歷史績效檔案
    if not os.path.exists(model_historic_best_performance_txtfile): ###如果沒有過去歷史檔案，先創造檔案
        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0")
            file.write("\n")
            file.write("Specificity:0")
            file.write("\n")
            file.write("F1:0")
            file.write("\n")
            file.write("Acc:0")
            file.write("\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(0, 0, 0, 0))             
            file.write("Model_Name:")

    performance_txtfile = open(model_historic_best_performance_txtfile)
    performance_array=[]
    for line in performance_txtfile.readlines():
        score=line.split(":")
        performance_array.append(score[-1])
         
    performance_txtfile.close()

    historic_best_sensitivity=float(performance_array[0])
    historic_best_specificity=float(performance_array[1])
    historic_best_f1=float(performance_array[2])
    ##historic_best_accs=float(performance_array[3])    

    ##比較本次訓練最佳績效是否高於過去歷史最佳績效
    ##if((current_best_specificity>historic_best_specificity-5 and current_best_sensitivity>historic_best_sensitivity-5)
    ##   and (current_best_f1>historic_best_f1 or abs(historic_best_specificity-historic_best_sensitivity)>abs(current_best_specificity-current_best_sensitivity))):
    diff_sensitivity = current_best_sensitivity - historic_best_sensitivity
    diff_specificity = current_best_specificity - historic_best_specificity
    historic_breakthrough_flag = False
    if (diff_sensitivity >= 0 and diff_specificity >= 0):  ##sensitivity和specificity兩者都上升
        historic_breakthrough_flag = True
    elif ((current_best_sensitivity > current_best_specificity) and diff_sensitivity <= 0 and diff_specificity >= 0 and abs(
            diff_sensitivity) <= abs(diff_specificity)):  ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
        historic_breakthrough_flag = True
    elif ((current_best_sensitivity > current_best_specificity) and diff_sensitivity >= 3 and diff_specificity >= -2):  ##sensitivity較大，但新的sensitivity上升>3%且新specificity下降<2%
        historic_breakthrough_flag = True
    elif ((current_best_specificity > current_best_sensitivity) and diff_specificity <= 0 and diff_sensitivity >= 0 and abs(
            diff_specificity) <= abs(diff_sensitivity)):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
        historic_breakthrough_flag = True
    elif ((current_best_specificity > current_best_sensitivity) and diff_specificity >= 3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity上升>3%，且新sensitivity上下降<2%
        historic_breakthrough_flag = True

    if (historic_breakthrough_flag):
        model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_ThreeClasses_Model", uuid,"Historic_Best_Performance.txt")
        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:%.2f" %(current_best_sensitivity))
            file.write("\n")
            file.write("Specificity:%.2f" %(current_best_specificity))
            file.write("\n")
            file.write("F1:%.2f" %(current_best_f1))
            file.write("\n")
            file.write("Acc:%.2f" %(current_best_accs))
            file.write("\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(current_best_TP, current_best_FP, current_best_FN, current_best_TN))
            modelname="Model_Name:BestModel_"+code_version+"_"+current_time_str+".pth"
            file.write(modelname)

        status=1  ###訓練成功且績效較之前版本好
    else:
        status=0  ###訓練成功但績效沒有比之前好    

    errorcode="0"
    message="Model wiht three classes has been built!"
    print(message)

    return status, errorcode, message       
  
def BuildModel_TwoClasses(uuid,basepath,splitting_ratio=""):
    
    errorcode="0"
    message=""
    status=-1

    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間    
    code_version=get_version()   ######取得模型訓練時當前程式版本

    print('Start building the model with two classes!')
       
    model_output_folder = os.path.join(basepath, splitting_ratio, "Best_TwoClasses_Model", uuid)
    if not os.path.exists(model_output_folder): 
        os.makedirs(model_output_folder)
    
    model_data_path=os.path.join(basepath,splitting_ratio,"Temp_TwoClasses_Model",uuid)  ###先清空過去訓練的模型佔存資料
    if os.path.isdir(model_data_path):
        shutil.rmtree(model_data_path)    

    model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_TwoClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")   
    if not os.path.exists(model_current_best_performance_txtfile): ###沒有檔案，先創造檔案
        with open(model_current_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0")
            file.write("\n")
            file.write("Specificity:0")
            file.write("\n")
            file.write("F1:0")
            file.write("\n")
            file.write("Acc:0")
            file.write("\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(0, 0, 0, 0)) 

    ### Create dataset object  
    Method = 'combine'
    current_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Train")
    filelist_high=os.listdir(os.path.join(current_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_path,'Normal'))    

    if(len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-500"
        message="An error occurs in the BuildModel_TwoClasses function: No training data"
        status=-1    
        return  status, errorcode, message  
    
    dataset = ecgDataset(dir_path = current_path, method=Method, classes=2) 

    Epoch_range = range(0, 800) 
    patience =50
    Batch_size = 128 
    valid_split = 0.1
    shuffle_flag = True
    random_seed = 10

    ## Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(valid_split * dataset_size)
    if shuffle_flag :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    ## 只對training data做data augmentation, validation/test data維持原樣
    train_dataset = ecgDatasetSubset(dataset, train_indices, augment=True)
    valid_dataset = ecgDatasetSubset(dataset, val_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Batch_size)
    current_test_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test")


    filelist_high=os.listdir(os.path.join(current_test_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_test_path,'Normal'))

    if(len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-501"
        message="An error occurs in the BuildModel_TwoClasses function: No testing data"
        return errorcode,message

    testdata = ecgDataset(dir_path=current_test_path, method=Method, classes=2)

    test_loader = DataLoader(testdata, batch_size=Batch_size)

    train_num, valid_num, test_num = len(train_dataset),len(valid_dataset),len(testdata)
    total_num = train_num + valid_num + test_num

    print("#Train:%5d(%.2f), #Validation:%5d(%.2f), #Test:%5d(%.2f)\n" 
          %(train_num,train_num/total_num,valid_num,valid_num/total_num,test_num,test_num/total_num))  
     
       
    loopindex=0   
    current_best_sensitivity=0  ###本次訓練最佳sensitivity先設定為0
    current_best_specificity=0
    current_best_f1=0
    current_best_accs=0 
    current_best_TP=0
    current_best_FP=0
    current_best_FN=0
    current_best_TN=0
         
    save_path = os.path.join(basepath,splitting_ratio,"Temp_TwoClasses_Model",uuid) ###建立uuid專屬的模型存放資料夾
    if os.path.isdir(save_path) is True:
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    while(loopindex<20):  ###執行20次訓練，取最好一次

        loopindex=loopindex+1
        
        ## set GPU Resource
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        print('GPU Count:',torch.cuda.device_count())

        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        if Epoch_range[0] == 0:
            Loss_list = []
            ACCs = []
    
            model = TwoClasses_CNN(data_len=dataset.data_len, input_channel=dataset.channel)        
            initialize_weights(model)  ###模型參數初始化
  
            model = model.to(device)
            print('Two-Class CNN model created.')
       

        ##loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4)).to(device)  ###把某一類樣本權重放大4倍，解決資料不平衡問題
        loss_function = torch.nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-3)
        ###optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=100, cooldown=0, min_lr=0.00001)
    
        patience =100
        n_patience = 0
        min_valid_loss = 0
        epoch_number = 0
    
        for epoch in Epoch_range:
    
            print("Epoch: %d " % (epoch+1))
    
            epoch_number= epoch+1
    
            train_loss, valid_loss = [], []

            ## training part 
            model.train()
            correct, total = 0, 0
            temp, total_num = 0, len(train_loader)
            for data, target in train_loader:
                temp += 1
                ##print('\r' + '[Progress]:[%s%s]%.2f%%; ' 
                ##    % ('█' * int(temp*20/total_num), ' ' * (20-int(temp*20/total_num)), float(temp/total_num*100)), end='')
        
                # 資料移至 GPU加速
                data = data.to(device)
                target = target.to(device)

                # 梯度歸零
                optimizer.zero_grad()

                # Foward
                output = model(data)       
                correct += (torch.sigmoid(output).ge(0.5) == target).sum().item()
                total += target.size(0)

                # Backward + Optimize
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
    
            train_acc = 100*correct/total
            print("Train ACC: %.2f%% , " %(train_acc), end='')
    
    
            ## evaluation part     
            model.eval()
            correct, total = 0, 0
            for data, target in valid_loader:
                # 資料移至 GPU加速
                data = data.to(device)
                target = target.to(device)

                # 預測
                output = model(data)
                correct += (torch.sigmoid(output).ge(0.5) == target).sum().item()
                total += target.size(0)
                
                # 損失函數
               
                loss = loss_function(output, target)
                valid_loss.append(loss.item())
        
            valid_acc = 100*correct/total
            print("Valid ACC: %.2f%% , " %(valid_acc), end='')
    
            train_loss = np.nanmean(train_loss)
            valid_loss = np.nanmean(valid_loss)
            print("Traing loss: %.4f, Valid loss: %.4f\n" %(train_loss, valid_loss))
    
            scheduler.step(valid_loss)
    
            Loss_list.append([train_loss, valid_loss]) 
            ACCs.append([train_acc, valid_acc])
    
            if valid_loss > min_valid_loss:
                if epoch == 0:
                    min_valid_loss = valid_loss
                else:
                    n_patience += 1
            else:
                min_valid_loss = valid_loss
                n_patience = 0
                torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"TwoClassesModel_"+str(uuid)+"_"+str(epoch+1)+".pth"))
        
        
            if n_patience >= patience:
                ##print("The model didn't improve for %i rounds, break it!" % patience)
                break
    
        torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"TwoClassesModel_"+str(uuid)+"_"+str(epoch+1)+".pth"))

        ### ------------------------Testing Data--------------------
        num_epoch = epoch_number
        model = TwoClasses_CNN(data_len=dataset.data_len, input_channel=dataset.channel)
        model.to(device)        
        model_path = os.path.join(basepath,splitting_ratio,"Temp_TwoClasses_Model",uuid,"TwoClassesModel_"+str(uuid)+"_"+str(num_epoch)+".pth")
        model.load_state_dict(torch.load(model_path))

        test_old_data = 0
        if test_old_data:
            testdata = ecgDataset(dir_path = os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test"), method=Method, classes=2)
            test_loader = DataLoader(testdata, batch_size=Batch_size)
    
        ##model = model.to('cpu')

        model.eval()
        correct, total = 0, 0
        TN, FP, FN, TP = 0, 0, 0, 0
        for data, target in test_loader:

            data = data.to(device)   

            ## 預測
            output = model(data)
            output = output.to('cpu')
            predict = torch.sigmoid(output).ge(0.5)
            correct += (predict == target).sum().item()
            total += target.size(0)
    
            predict = predict.tolist()
            target = target.tolist()
            for i in range(0, len(predict)):
                if predict[i][0] is True and target[i][0] == 1:
                    TP += 1
                elif predict[i][0] is True and target[i][0] == 0:
                    FP += 1
                elif predict[i][0] is False and target[i][0] == 1:
                    FN += 1
                elif predict[i][0] is False and target[i][0] == 0:
                    TN += 1
    
        if((TP+FN)>0):                
            sensitivity=(100*TP/(TP+FN))
        else:
            sensitivity=0

        if((TN+FP)>0):    
            specificity=(100*TN/(TN+FP))
        else:
            specificity=0

        if((specificity + sensitivity)>0):
            f1=2 * (specificity * sensitivity) / (specificity + sensitivity)
        else:
            f1=0
        
        accs=100*correct/total

        print("TP:%d, FP:%d, FN:%d, TN:%d\n" %(TP, FP, FN, TN))
        print("Sensitivity: %.2f %%\n" %sensitivity)
        print("Specificity: %.2f %%\n" %specificity)
        print("F1: %.2f %%\n" %f1)
        print("Final ACC: %.2f %%\n" %accs)  
        
        ##if((specificity>current_best_specificity-5 and sensitivity>current_best_sensitivity-5) and (f1>current_best_f1 or abs(specificity-sensitivity)<abs(current_best_specificity-current_best_sensitivity))):

        diff_sensitivity = sensitivity - current_best_sensitivity
        diff_specificity = specificity - current_best_specificity
        breakthrough_flag = False
        if (diff_sensitivity > 0 and diff_specificity > 0):  ##sensitivity和specificity兩者都上升
            breakthrough_flag = True
        elif ((current_best_sensitivity > current_best_specificity) and diff_sensitivity <= 0 and diff_specificity >= 0 and abs(
                diff_sensitivity) <= abs(diff_specificity)):  ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
            breakthrough_flag = True
        elif ((current_best_sensitivity > current_best_specificity) and diff_sensitivity >= 3 and diff_specificity >= -2):  ##sensitivity較大，但新的sensitivity上升>=3% 且新specificity只下降2%以內
            breakthrough_flag = True
        elif ((current_best_specificity > current_best_sensitivity) and diff_specificity <= 0 and diff_sensitivity >= 0 and abs(
                diff_specificity) <= abs(diff_sensitivity)):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
            breakthrough_flag = True
        elif ((current_best_specificity > current_best_sensitivity) and diff_specificity >= 3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
            breakthrough_flag = True

        ##if((specificity>current_best_specificity-5 and sensitivity>current_best_sensitivity-5) and (f1>current_best_f1 or abs(specificity-sensitivity)<abs(current_best_specificity-current_best_sensitivity))):
        if(breakthrough_flag):    
            current_best_specificity=specificity
            current_best_f1=f1
            current_best_sensitivity=sensitivity
            current_best_accs=accs
            current_best_TP=TP
            current_best_FP=FP
            current_best_FN=FN
            current_best_TN=TN
            torch.save(model.state_dict(),os.path.join(model_output_folder,"BestModel_"+code_version+"_"+current_time_str+".pth"))  ###比目前訓練階段的模型績效好，儲存起來            
            model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_TwoClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")            
            with open(model_current_best_performance_txtfile, "w") as file:
                file.write("Sensitivity:%.2f" %(sensitivity))
                file.write("\n")
                file.write("Specificity:%.2f" %(specificity))
                file.write("\n")
                file.write("F1:%.2f" %(f1))
                file.write("\n")
                file.write("Acc:%.2f" %(accs))
                file.write("\n")
                file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(TP, FP, FN, TN))            


    model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio, "Best_TwoClasses_Model", uuid,"Historic_Best_Performance.txt") ###歷史最佳績效
    if not os.path.exists(model_historic_best_performance_txtfile): ###如果沒有過去歷史檔案，先創造檔案
        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0")
            file.write("\n")
            file.write("Specificity:0")
            file.write("\n")
            file.write("F1:0")
            file.write("\n")
            file.write("Acc:0")
            file.write("\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(0, 0, 0, 0))             
            file.write("Model_Name:")

    performance_txtfile = open(model_historic_best_performance_txtfile)
    performance_array=[]
    for line in performance_txtfile.readlines():
        score=line.split(":")
        performance_array.append(score[-1])
         
    performance_txtfile.close()

    historic_best_sensitivity=float(performance_array[0])
    historic_best_specificity=float(performance_array[1])
    historic_best_f1=float(performance_array[2])   

    ##比較本次訓練最佳績效是否高於過去歷史最佳績效
    ##if((current_best_specificity>historic_best_specificity-5 and current_best_sensitivity>historic_best_sensitivity-5) 
    ##   and (current_best_f1>historic_best_f1 or abs(historic_best_specificity-historic_best_sensitivity)>abs(current_best_specificity-current_best_sensitivity))):

    diff_sensitivity=historic_best_sensitivity-current_best_sensitivity
    diff_specificity=historic_best_specificity-current_best_specificity 
    historic_breakthrough_flag=False
    if(current_best_sensitivity>=historic_best_sensitivity and current_best_specificity>=historic_best_specificity): ##sensitivity和specificity兩者都上升
        historic_breakthrough_flag=True
    elif((current_best_sensitivity>current_best_specificity) and diff_sensitivity<=0 and diff_specificity>=0 and abs(diff_sensitivity)<=abs(diff_specificity)): ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
        historic_breakthrough_flag=True
    elif ((current_best_sensitivity > current_best_specificity) and diff_sensitivity >=3 and diff_specificity >=-2):  ##sensitivity較大，但新的sensitivity上升>3%且新specificity下降<2%
        historic_breakthrough_flag = True
    elif((current_best_specificity>current_best_sensitivity) and diff_specificity<=0 and diff_sensitivity>=0 and abs(diff_specificity)<=abs(diff_sensitivity)): ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
        historic_breakthrough_flag=True
    elif((current_best_specificity > current_best_sensitivity) and diff_specificity >=3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity上升>3%，且新sensitivity上下降<2%
        historic_breakthrough_flag = True

    if(historic_breakthrough_flag):
        model_historic_best_performance_txtfile=os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Model", uuid,"Historic_Best_Performance.txt")

        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:%.2f" %(current_best_sensitivity))
            file.write("\n")
            file.write("Specificity:%.2f" %(current_best_specificity))
            file.write("\n")
            file.write("F1:%.2f" %(current_best_f1))
            file.write("\n")
            file.write("Acc:%.2f" %(current_best_accs))
            file.write("\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(current_best_TP, current_best_FP, current_best_FN, current_best_TN))
            modelname_str="Model_Name:BestModel_"+code_version+"_"+current_time_str+".pth"
            file.write(modelname_str)  

        status=1
    else:
        status=0

    errorcode="0"
    message="Model with two classes has been built!"
    print(message)

    return status, errorcode, message

class GlucoseThreeClassesPredictor: ###血糖預測類別

    def __init__(self,basepath,splitting_ratio=""):
        self.basepath = basepath
        self.splitting_ratio = splitting_ratio

    def glucose_predict(self,uuid,ecgdata):

        errorcode="0"
        message=""

        current_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data')
        if not os.path.exists(current_testingdata_path): os.makedirs(current_testingdata_path)

        normal_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data','Normal') ###ECG資料都只放在Normal資料夾
        if not os.path.exists(normal_testingdata_path): os.makedirs(normal_testingdata_path)

        old_file_list=os.listdir(normal_testingdata_path)
        for i,filename in enumerate(old_file_list): ###刪除上次預測的資料
            os.remove(os.path.join(normal_testingdata_path,filename))

        high_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data','High') ###空的資料夾
        if not os.path.exists(high_testingdata_path): os.makedirs(high_testingdata_path)

        low_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data','Low') ###空的資料夾
        if not os.path.exists(low_testingdata_path): os.makedirs(low_testingdata_path)

        segment_num=int(len(ecgdata)/2500)
        ##bad_quality_count=0 
        good_quality_count=0
        if(segment_num>=6): ##至少一分鐘的資料長度
            for i in range(0,segment_num):
                current_ecg_data=ecgdata[i*2500:(i+1)*2500]  ##切成10秒一個
                rpeak_array=rpeak_detection(current_ecg_data)
                result=ecg_quality_check(current_ecg_data, rpeak_array[1:])
                if(result == "Normal"): ###正常訊號才會做特徵擷取
                    self.feature_extraction(current_ecg_data,current_testingdata_path,i)
                    good_quality_count=good_quality_count+1
                ##else:  
                ##    bad_quality_count=bad_quality_count+1
                                        

            ##if(bad_quality_count<segment_num/2): ##超過1/2的訊號品質是好的才進行辨識
            if(good_quality_count>=3): ##超過3 segments的訊號品質是好的才進行辨識
                glucose_list=self.predict(uuid,self.basepath,self.splitting_ratio)
                predicted_category=_majority_filtering(glucose_list)
                errorcode="0"
                message="Finish predicting the glucose category"

            else:
                predicted_category=-1
                errorcode="-600"
                message="An error occurs in the predict function of the GlucoseThreeClassesPredictor class: Too many segments of ECG signals are of poor quality!"
                  
        else:           
            predicted_category=-1
            errorcode="-601" 
            message="An error occurs in the predict function of the GlucoseThreeClassesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!"
                                       
    
        return errorcode, message, predicted_category    
     

    def feature_extraction(self,ecg_data_array,test_feature_file_path,index):        
            
        rpeak_array=rpeak_detection(ecg_data_array,measuring_mode='strap',method_type='vg',mode='original') 
        count=0
        feature_wave=np.zeros(150) ###特徵長度150
        for i in range(1,len(rpeak_array)):
            r_index=rpeak_array[i]
            before_index=r_index-50 ##取R波前50
            after_index=r_index+100 ##取R波後100
            if(before_index<0 or after_index>=len(ecg_data_array)):
                continue
            
            current_wave=np.array(ecg_data_array[before_index:after_index])
            feature_wave=feature_wave+current_wave
            count=count+1

        feature_wave=feature_wave/count  ###取得平均波形  
        feature_wave=feature_wave.astype(int)

        path = os.path.join(test_feature_file_path,"Normal","ecg_feature_"+str(index)+".txt") ###都只放在Normal資料夾
        f = open(path, 'w')
        for i,ecgvalue in enumerate(feature_wave):
            f.write(str(ecgvalue))
            f.write("\n")
        
        f.close() 


    def predict(self,uuid,basepath,splitting_ratio=""):
        Method='combine'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        '''
        print(device)
        print('GPU Count:',torch.cuda.device_count())

        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        '''

        current_test_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,'Current_Test_Data')

        filelist=os.listdir(current_test_path)

        if(len(filelist)==0):
            return -2   ### no testing data


        testdata = ecgDataset(dir_path = current_test_path, method=Method)
        Batch_size=len(testdata)

        test_loader = DataLoader(testdata, batch_size=Batch_size)

        model = CNN(data_len=testdata.data_len, input_channel=testdata.channel)
        model.to(device)
        best_model_performance_path = os.path.join(basepath,splitting_ratio,"Best_ThreeClasses_Model",uuid,"Historic_Best_Performance.txt")
        performance_txtfile = open(best_model_performance_path)
        count=0
        best_model_path=""
        for line in performance_txtfile.readlines():
            count=count+1
            if(count==6):
                string_array=line.split(":")
                best_model_path=string_array[1]
                print("best_model:",best_model_path)

        performance_txtfile.close()

        model_path = os.path.join(basepath,splitting_ratio,"Best_ThreeClasses_Model",uuid,best_model_path)
        model.load_state_dict(torch.load(model_path))

        test_old_data = 0
        if test_old_data:            
            testdata = ecgDataset(dir_path = current_test_path, method=Method)
            test_loader = DataLoader(testdata, batch_size=Batch_size)
    
        model.eval()
        correct, total = 0, 0
        TN, FP, FN, TP = 0, 0, 0, 0
        for data, target in test_loader:
            data = data.to(device)
    
            ## 預測
            output = model(data)
            output = output.to('cpu')           
    
            predict=np.zeros(len(output))
            answer=torch.sigmoid(output)
            predict = torch.argmax(answer, dim=1)  
            predict_list = predict.tolist()
                       
           
        return predict_list
        


class GlucoseTwoClassesPredictor: ###血糖預測類別

    def __init__(self,basepath,splitting_ratio=""):
        self.basepath = basepath
        self.splitting_ratio = splitting_ratio


    def glucose_predict(self,uuid,ecgdata):

        errorcode="0"
        message=""
        glucose_list=[]

        current_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data')
        if not os.path.exists(current_testingdata_path): os.makedirs(current_testingdata_path)

        normal_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data','Normal') ###ECG資料都只放在Normal資料夾
        if not os.path.exists(normal_testingdata_path): os.makedirs(normal_testingdata_path)

        old_file_list=os.listdir(normal_testingdata_path)
        for i,filename in enumerate(old_file_list): ##刪除上次預測的資料
            os.remove(os.path.join(normal_testingdata_path,filename))

        high_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data','High') ###空的資料夾
        if not os.path.exists(high_testingdata_path): os.makedirs(high_testingdata_path)

        low_testingdata_path=os.path.join(self.basepath,self.splitting_ratio,"GlucoseData",uuid,'Current_Test_Data','Low') ###空的資料夾
        if not os.path.exists(low_testingdata_path): os.makedirs(low_testingdata_path)

        segment_num=int(len(ecgdata)/2500)
        ##bad_quality_count=0 
        good_quality_count=0
        if(segment_num>=6):  ##至少一分鐘的資料長度
            for i in range(0,segment_num):
                current_ecg_data=ecgdata[i*2500:(i+1)*2500]  ###切成10秒一個
                rpeak_array=rpeak_detection(current_ecg_data)
                result=ecg_quality_check(current_ecg_data, rpeak_array[1:])
                if(result == "Normal"): 
                    self.feature_extraction(current_ecg_data,current_testingdata_path,i)
                    good_quality_count=good_quality_count+1
                ##else:  
                ##    bad_quality_count=bad_quality_count+1
                                        

            ##if(bad_quality_count<segment_num/2): ##超過1/2的訊號品質是好的才進行辨識
            if(good_quality_count>=3): ##超過3 segments的訊號品質是好的才進行辨識
                glucose_list=self.predict(uuid,self.basepath,self.splitting_ratio)
                predicted_category=_majority_filtering(glucose_list)
                errorcode="0"
                message="Finish predicting the glucose category"

            else:
                predicted_category=-1
                errorcode="-700"
                message="An error occurs in the predict function of the GlucoseTwoClassesPredictor class: Too many segments of ECG signals are of poor quality!"
                  
        else:   
            predicted_category=-1        
            errorcode="-701" 
            message="An error occurs in the predict function of the GlucoseTwoClassesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!"
                                       
    
        return errorcode, message, predicted_category      
  
   

    def feature_extraction(self,ecg_data_array,test_feature_file_path,index):        
            
        rpeak_array=rpeak_detection(ecg_data_array,measuring_mode='strap',method_type='vg',mode='original') 
        count=0
        feature_wave=np.zeros(150) ###特徵長度150
        for i in range(1,len(rpeak_array)):
            r_index=rpeak_array[i]
            before_index=r_index-50 ##取R波前50
            after_index=r_index+100 ##取R波後100
            if(before_index<0 or after_index>=len(ecg_data_array)):
                continue
            
            current_wave=np.array(ecg_data_array[before_index:after_index])
            feature_wave=feature_wave+current_wave
            count=count+1

        feature_wave=feature_wave/count  ###取得平均波形  
        feature_wave=feature_wave.astype(int)

        path = os.path.join(test_feature_file_path,"Normal","ecg_feature_"+str(index)+".txt") ###都只放在Normal資料夾
        f = open(path, 'w')
        for i,ecgvalue in enumerate(feature_wave):
            f.write(str(ecgvalue))
            f.write("\n")
        
        f.close() 


    def predict(self,uuid,basepath,splitting_ratio=""):

        Method='combine'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''
        print(device)
        print('GPU Count:',torch.cuda.device_count())

        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        '''

        current_test_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,'Current_Test_Data')

        filelist=os.listdir(current_test_path)
        if(len(filelist)==0):
            return -2   ### no testing data


        testdata = ecgDataset(dir_path = current_test_path, method=Method,classes=2)
        Batch_size=len(testdata)
        test_loader = DataLoader(testdata, batch_size=Batch_size)

        model = TwoClasses_CNN(data_len=testdata.data_len, input_channel=testdata.channel)
        model.to(device)
        best_model_performance_path = os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Model",uuid,"Historic_Best_Performance.txt")
        performance_txtfile = open(best_model_performance_path)
        count=0
        best_model_path=""
        for line in performance_txtfile.readlines():
            count=count+1
            if(count==6):
                string_array=line.split(":")
                best_model_path=string_array[1]
                print("best_model:",best_model_path)

        performance_txtfile.close()

        model_path = os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Model",uuid,best_model_path)
        model.load_state_dict(torch.load(model_path))

        test_old_data = 0
        if test_old_data:            
            testdata = ecgDataset(dir_path = current_test_path, method=Method,classes=2)
            test_loader = DataLoader(testdata, batch_size=Batch_size)
    
        model.eval()
        
        predict_list=[]

        for data, target in test_loader:
            data = data.to(device)  
            ## 預測
            output = model(data)
            output = output.to('cpu')
            predict = torch.sigmoid(output).ge(0.5)            
         
        for i in range(len(predict)):
            if(predict[i]==True):
                predict_list.append(1)
            else:
                predict_list.append(0)    

        return predict_list


class GlucosePredictor:

    def __init__(self,uuid,basepath,splitting_ratio=""):
        self.uuid=uuid
        self.basepath=basepath
        self.splitting_ratio=splitting_ratio
        self.errorcode="0"
        self.set_model()

    def set_model(self):
        historic_best_sensitivity=0
        historic_best_specificity=0
        model_historic_best_performance_txtfile = os.path.join(self.basepath,self.splitting_ratio,"Best_ThreeClasses_Model",self.uuid,"Historic_Best_Performance.txt")
        if(os.path.exists(model_historic_best_performance_txtfile)):
            performance_txtfile = open(model_historic_best_performance_txtfile)
            performance_array=[]
            for line in performance_txtfile.readlines():
                score=line.split(":")
                performance_array.append(score[-1])

            performance_txtfile.close()

            historic_best_sensitivity=float(performance_array[0])
            historic_best_specificity=float(performance_array[1])

        if (historic_best_sensitivity>0 and historic_best_specificity>0): ###如果三類別歷史績效sensitivity,specificity皆不為0，表示三類別最佳模型存在
            self.GlucosePredictor_Obj=GlucoseThreeClassesPredictor(self.basepath,self.splitting_ratio)
        else:  ##檢查二類別模型是否存在
            model_historic_best_performance_txtfile = os.path.join(self.basepath,self.splitting_ratio,"Best_TwoClasses_Model",self.uuid,"Historic_Best_Performance.txt")
            if(os.path.exists(model_historic_best_performance_txtfile)):
                performance_txtfile = open(model_historic_best_performance_txtfile)
                performance_array=[]
                for line in performance_txtfile.readlines():
                    score=line.split(":")
                    performance_array.append(score[-1])

                performance_txtfile.close()
                historic_best_sensitivity=float(performance_array[0])
                historic_best_specificity=float(performance_array[1])

            if (historic_best_sensitivity>0 and historic_best_specificity>0): ###如果二類別模型存在
                self.GlucosePredictor_Obj=GlucoseTwoClassesPredictor(self.basepath,self.splitting_ratio)
            else: ###連二類模型都不存在
                self.errorcode="-1"
                self.message="Model has not been built!"
            

    def predict(self,ecgdata):               
        if(self.errorcode=="-1"):
            errorcode="-602"
            message=self.message
            glucose_category=-1  ###無預測
        else:
            errorcode, message, glucose_category=(self.GlucosePredictor_Obj).glucose_predict(self.uuid,ecgdata)
        
        return errorcode, message, glucose_category
    

if __name__ == "__main__":

    run_start_time = time.perf_counter()

    ##uuid 對應的CSV檔必須已存在於 glucosedata_path 下(例如 2208.csv)
    ##ECG srj檔：本機 DataDB/<uuid> 已經有資料的話，server_db_path 留空字串即可跳過下載；
    ##           本機沒有的uuid，把 server_db_path 指到雲端zip所在的實際路徑，會自動下載解壓縮到 DataDB/<uuid>
    user_information = ['2197','2199','2204','2206','2208','2210','2215','2216','2223','2249']

    mother_path=Path(__file__).resolve().parent
    basepath=str(mother_path/"Model")  ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    glucosedata_path=str(mother_path/"GlucoseDataCSV") ###APP收到的血糖數值，整理成CSV檔後存放路徑(已存在)
    server_db_path=r'G:\.shortcut-targets-by-id\1dZUAXwQHDvBJGYwQLNpizVJHhkOfnxVa\Health_Server_Script\_rawdata_download'   ###雲端zip檔所在路徑，請改成實際路徑(要能存取雲端磁碟的電腦才能執行這段)

    splitting_ratio="70_30"
    processnum=8

    ##-------step 1. 依 GlucoseDataCSV 現有的uuid，逐一下載雲端ECG(如需要)並做資料處理，尚未包含模型訓練-------
    DataArrangement_Obj = DataArrangement()
    for uuid in user_information:
        print('uuid:',uuid)
        srj_db_path=str(mother_path/"DataDB"/uuid)   ###srj檔放置路徑(自雲端下載解壓縮後，或已存在的本機srj檔都放這裡)

        errorcode, message = DataArrangement_Obj.data_processing(uuid, srj_db_path, glucosedata_path, basepath, server_db_path, processnum, splitting_ratio)
        print('uuid:',uuid,' errorcode:',errorcode,' message:',message)

    ##-------step 2. 資料備齊後，才需要用 BuildModel 做模型訓練(每個uuid最多20輪x800 epochs，非常耗時，請確認資料處理都成功後再打開這段)-------
    ##for uuid in user_information:
    ##    srj_db_path=str(mother_path/"DataDB"/uuid)
    ##    status, errorcode, message = BuildModel(uuid, basepath, srj_db_path, glucosedata_path, server_db_path, processnum, splitting_ratio)  ###建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
    ##    print('uuid:',uuid,' status:',str(status),' error code:',errorcode,' message:',message)

    ##-------step 3. 輸入ECG進行血糖類別分類(模型訓練完成後才能用)-------
    ##uuid = user_information[0]
    ##testdatapath=os.path.join(basepath,"TestData",uuid) ##測試用程式
    ##if not os.path.exists(testdatapath): os.makedirs(testdatapath)  ##測試用程式
    ##
    ##filelist=os.listdir(testdatapath) ##測試用程式
    ##ecgdata=[] ##測試用程式
    ##for i in range(0,len(filelist)): ##測試用程式
    ##    filename=filelist[i]
    ##    print('filename:',filename)
    ##    f = open(os.path.join(testdatapath,filename), "r")
    ##    for x in f:
    ##        ecgdata.append(int(x.rstrip('\n')))
    ##
    ##GlucosePredictor_Obj=GlucosePredictor(uuid,basepath,splitting_ratio)
    ##errorcode, message, glucose_category=GlucosePredictor_Obj.predict(ecgdata)
    ##print('glucose_category:',glucose_category)
    ##print('errorcode:',str(errorcode),' message:',message)

    run_end_time = time.perf_counter()
    print("Total running time: %.2f seconds" %(run_end_time - run_start_time))
