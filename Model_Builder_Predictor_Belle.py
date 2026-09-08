import os
import shutil
import sys
import numpy as np
import time
from scipy.fftpack import fft
import pywt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
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
import bisect

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from SWMlib.motion import *  ##ahrs
from SWMlib.motion import static_motion_analysis
from SWMlib.ecg.rpeak import rpeak_detection 
from SWMlib.ecg.quality_check import ecg_quality_check 
from SWMlib.common import data_load_concate  

def get_version(): ###取得版本號

    return '006'

class DataArrangement:

    def __init__(self):
        self.basepath = os.path.dirname(__file__)      

    def unzip_file(self,uuid,start_time,end_time,db_path,export_path):  ####將受測者zip檔解壓縮

        uuid_export_path = os.path.join(export_path, uuid)        
        if not os.path.exists(uuid_export_path): 
            os.makedirs(uuid_export_path)      
       
        UnzipFileNameList = data_load_concate.search_unzip_file(db_path=db_path, target_uuid=uuid, start_time=start_time, end_time=end_time, export_path=uuid_export_path)   

    
    ##def data_processing(self,uuid,start_time,end_time,srj_db_path,glucosedata_path,basepath,server_db_path): ##測試用
    def data_processing(self,uuid,srj_db_path,glucosedata_path,basepath):
        
        errorcode="0"
        message=""
        
        ##測試用  
        
        if(server_db_path!=""):  ###測試用,需要抓雲端上的zip檔案 
            print('Start unzippig file!')      
            self.unzip_file(uuid,start_time,end_time,server_db_path,srj_db_path) ##自雲端資料夾中將ECG壓縮檔解壓縮成srj檔放置到srj_db_path路徑下                  
        
        
        print('Start data parsing！') 
        errorcode, message = self.data_parsing(uuid,srj_db_path,glucosedata_path,basepath) ##srj檔分析後將ECG資料放置於export_txtfile_path路徑下      
        if int(errorcode)<0:
            return errorcode, message


        # print('Start feature extracting!') 
        # errorcode, message = self.feature_extraction(uuid,basepath) ##擷取ECG特徵          
        # if int(errorcode)<0:
        #     return errorcode, message
      
       

        # print('Start data arrangement!') 
        # errorcode, message = self.data_arrangement(uuid,glucosedata_path,basepath) 
        # if int(errorcode)<0:
        #     return errorcode, message
        
        # message="data processing is done!"
        
        return errorcode, message
    
    def data_parsing(self,uuid,srj_db_path,glucosedata_path,export_txtfile_path): 

        errorcode="0"
        message=""
        
        export_rawdata_path=os.path.join(export_txtfile_path,"RawData")        
        if not os.path.exists(export_rawdata_path): 
            os.makedirs(export_rawdata_path) ###創建ECG擷取後存放資料夾  

        uuid_temp_path = os.path.join(export_rawdata_path, uuid)
        if not os.path.exists(uuid_temp_path): 
            os.makedirs(uuid_temp_path) ### 創建屬於uuid的資料夾 
        
        glucosedata_path=os.path.join(glucosedata_path,uuid+'.csv')
        #print('glucosedata_path:',glucosedata_path)
        glucose_csv_file_list=glob.glob(glucosedata_path) 

        ecgdata_path=os.path.join(srj_db_path, uuid, uuid +'*.srj')
        #print('ecgdata_path:',ecgdata_path)
        srj_file_list=glob.glob(ecgdata_path) 
        #print('srj_file_list:',srj_file_list)

        if(len(glucose_csv_file_list)==0):           
            errorcode="-102"
            message="An error occurs in the data-parsing function: no glucose csv files exist!"          
            return errorcode, message

        # ---- 新增：迴圈外只呼叫一次 ----
        sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all = self._load_all_srj_data(srj_db_path, srj_file_list, uuid)
        # --------------------------------
                
        try:
            with open(glucose_csv_file_list[0], newline='', encoding='utf-8') as f:
                rows = csv.reader(f, delimiter=',')
                for row in rows:               
                    recored_time=row[0]
                    glucose_value=row[1]                   

                    if(glucose_value!=""):                    
                        
                        datetime_object = datetime.strptime(recored_time[2:],'%y/%m/%d %H:%M')
                    
                        if(int(glucose_value)>=250): ##血糖值高到250以上
                            time_change = timedelta(minutes=10) ##取前後10分鐘
                        elif(int(glucose_value)>=200):
                            time_change = timedelta(minutes=7) ##取前後7分鐘
                        else:                        
                            time_change = timedelta(minutes=5) ##取前後5分鐘

                        new_time_before = datetime_object - time_change 
                        new_time_after= datetime_object + time_change 
                        current_time_str= datetime_object.strftime("20%y%m%d %H%M%S")
                        start_time_str = new_time_before.strftime("20%y%m%d %H%M%S")
                        end_time_str = new_time_after.strftime("20%y%m%d %H%M%S")
                        
                        #ecg_data_array,motion_data_array,breath_data_array,temp_data_array = self._dataconcate(srj_db_path,srj_file_list,start_time_str,end_time_str,uuid)
                        ecg_data_array, motion_data_array, breath_data_array, temp_data_array = self._query_by_time_range(sorted_times, ecg_data_all, motion_data_all, breath_data_all, temp_data_all, start_time_str, end_time_str)
                        #print(ecg_data_array)

                        for i in range(len(ecg_data_array)):
                            #current_motion_data=motion_data_array[i]
                            current_ecg_data = ecg_data_array[i]

                            #print('current_time_str:',current_time_str,' glucose_value:',glucose_value,' ecg_data_len:',len(current_ecg_data))

                            #if(len(current_ecg_data)==2500 and len(current_motion_data)>0): ###確定ecg資料和motion資料都有
                            rpeak_array = rpeak_detection(current_ecg_data)
                            result = ecg_quality_check(current_ecg_data, rpeak_array[1:])
                            if (result == "Normal"):
                                # static_flag,std=static_motion_analysis(motion_data=current_motion_data,sample_rate=2)
                                # flagsum=sum(static_flag)
                                # if(flagsum==0): ##static
                                #     self._write_ecg_data(uuid=uuid,ecg_data_array=current_ecg_data,time_str=current_time_str,flag=0,index=i,glucose_value=glucose_value,uuid_temp_path=uuid_temp_path)
                                # else:  ##dynamic
                                self._write_ecg_data(uuid=uuid, ecg_data_array=current_ecg_data,
                                                     time_str=current_time_str, flag=0, index=i,
                                                     glucose_value=glucose_value, uuid_temp_path=uuid_temp_path)
                                ##else:
                            ##    errorcode=-1
                            ##    message='error occurs in the data-parsing function: ecg data is too short!'

                        '''
                        for i in range(len(motion_data_array)):
                            current_motion_data=motion_data_array[i]
                            current_ecg_data=ecg_data_array[i]
                        
                            if(len(current_ecg_data)==2500 and len(current_motion_data)>0): ###確定ecg資料和motion資料都有                        
                                rpeak_array=rpeak_detection(current_ecg_data)
                                result=ecg_quality_check(current_ecg_data, rpeak_array[1:])
                                if(result=="Normal"):                               
                                    static_flag,std=static_motion_analysis(motion_data=current_motion_data,sample_rate=2)
                                    flagsum=sum(static_flag)
                                    if(flagsum==0): ##static                                                            
                                        self._write_ecg_data(uuid=uuid,ecg_data_array=current_ecg_data,time_str=current_time_str,flag=0,index=i,glucose_value=glucose_value,uuid_temp_path=uuid_temp_path)                         
                                    else:  ##dynamic        
                                        self._write_ecg_data(uuid=uuid,ecg_data_array=current_ecg_data,time_str=current_time_str,flag=1,index=i,glucose_value=glucose_value,uuid_temp_path=uuid_temp_path)                            
                            else:                           
                                errorcode="-101" 
                                message='An error occurs in the data-parsing function: ecg data is too short or no motion data!'                                                   
                        '''
            f.close()

        except:
            errorcode="-100"
            message="An error occurs in the data-parsing function: fail to read glucose csv files"
            return errorcode, message
  
        return errorcode, message
    

    def feature_extraction(self,uuid,export_txtfile_path): 

        errorcode="0"
        message=""    
        
        basepath=os.path.join(export_txtfile_path,"RawData",uuid)  
        file_name_list=os.listdir(basepath)

        os.chdir(basepath)
        for index,file_name in enumerate(file_name_list):
            ecg_data_array = []
            with open(file_name) as file:
                for line in file: 
                    line = line.strip() 
                    ecg_data_array.append(int(line)) 
            
            rpeak_array=rpeak_detection(ecg_data_array,measuring_mode='strap',method_type='vg',mode='original') 
            ##rpeak_array=rpeak_detection(ecg_data_array,measuring_mode='strap',method_type='bandpass',mode='original')
           
            count=0
            mean_wave=np.zeros(150) ###特徵長度150
            if(len(rpeak_array)>0):
                for i in range(1,len(rpeak_array)):
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
            else:
                errorcode="-200"
                message="An error occurs in the feature_extraction function: the number of R wave peak is zero!"  

            mean_wave=mean_wave.astype(int)
           
            features_path=os.path.join(export_txtfile_path,"Features")  ##創建特徵檔案資料夾
            if not os.path.exists(features_path): os.makedirs(features_path)

            output_path=os.path.join(features_path,uuid)
            if not os.path.exists(output_path): os.makedirs(output_path)      

            path = os.path.join(output_path,file_name)            
            f = open(path, 'w')
            mean_wave=mean_wave.tolist()
            for i,ecgvalue in enumerate(mean_wave):
                f.write(str(ecgvalue))
                f.write("\n")
        
            f.close()

        ##--------seperate data into high,low,normal------------
        basepath=os.path.join(export_txtfile_path,'Features',uuid)
        filelist=os.listdir(basepath)  ###自Features資料夾搬移到Dataset資料夾(分成高中低)

        dataset_path=os.path.join(export_txtfile_path,"Dataset") ##創建Dataset資料夾
        if not os.path.exists(dataset_path): os.makedirs(dataset_path)
            
        path_high=os.path.join(dataset_path,uuid,'High')
        if not os.path.exists(path_high): os.makedirs(path_high)
        
        path_low=os.path.join(dataset_path,uuid,'Low')
        if not os.path.exists(path_low): os.makedirs(path_low)
        
        path_normal=os.path.join(dataset_path,uuid,'Normal')
        if not os.path.exists(path_normal): os.makedirs(path_normal)        
        
        other_value_index_array=[]
        highvalue_count=0
        lowvalue_count=0
        normalvalue_count=0
        try:
            for i in range(len(filelist)):
                filestr=filelist[i]
                filestr_array=filestr.split('_')
                valuestr=filestr_array[-1]
                valuestr_array=valuestr.split('.')
                value=int(valuestr_array[0])
                oripath=os.path.join(basepath,filestr)

                if(value>=180): ###180以上視為high                       
                    newpath=os.path.join(export_txtfile_path,"Dataset",uuid,'High',filestr)               
                    shutil.move(oripath, newpath)
                    highvalue_count=highvalue_count+1
                elif(value<=80): ###80以下視為low
                    newpath=os.path.join(export_txtfile_path,"Dataset",uuid,'Low',filestr)                
                    shutil.move(oripath, newpath) 
                    lowvalue_count=lowvalue_count+1
                elif(value>=85 and value<=170):  ###中間血糖設定為85~170之間
                    newpath=os.path.join(export_txtfile_path,"Dataset",uuid,'Normal',filestr)                
                    shutil.move(oripath, newpath)
                    normalvalue_count=normalvalue_count+1 
                else:
                    other_value_index_array.append(i)     

            if(len(other_value_index_array)>0): ##有介於80~85 170~180之間的數值
                for k in range(len(other_value_index_array)):
                    index=other_value_index_array[k]
                    filestr=filelist[index]
                    filestr_array=filestr.split('_')
                    valuestr=filestr_array[-1]
                    valuestr_array=valuestr.split('.')
                    value=int(valuestr_array[0])
                    oripath=os.path.join(basepath,filestr)
                    if(value>80 and value<85):
                        if(lowvalue_count<normalvalue_count and value<=82): ##低血糖筆數較少，且數值小於82，歸給低血糖，否則捨棄不用
                            newpath=os.path.join(export_txtfile_path,"Dataset",uuid,'Low',filestr)                
                            shutil.move(oripath, newpath)                 
                    elif(value>170 and value<180):
                        if(normalvalue_count<highvalue_count and value<=175): ##中血糖筆數較少，且數值小於175，歸給中血糖，否則捨棄不用
                            newpath=os.path.join(export_txtfile_path,"Dataset",uuid,'Normal',filestr)                
                            shutil.move(oripath, newpath)
        except:
            errorcode="-201"
            message="An error occurs in the feature_extraction function: fail to seperate data into high, low, normal categories!"
            return errorcode, message

        return errorcode, message
    

    def data_arrangement(self,uuid,glucosedata_path,export_txtfile_path):

        errorcode="0"
        message=""
    
        newpath=os.path.join(export_txtfile_path,"GlucoseData_"+uuid)        
        if not os.path.exists(newpath): os.makedirs(newpath)  ###創建給AI模型訓練使用的training,testing資料夾
        
        path_train=os.path.join(newpath,'Train')
        if not os.path.exists(path_train): os.makedirs(path_train)
        
        path_normal=os.path.join(newpath,'Train','Normal')
        if not os.path.exists(path_normal): os.makedirs(path_normal)

        path_high=os.path.join(newpath,'Train','High')
        if not os.path.exists(path_high): os.makedirs(path_high)

        path_low=os.path.join(newpath,'Train','Low')
        if not os.path.exists(path_low): os.makedirs(path_low)


        path_test=os.path.join(newpath,'Test')
        if not os.path.exists(path_test): os.makedirs(path_test)
        
        path_normal=os.path.join(newpath,'Test','Normal')
        if not os.path.exists(path_normal): os.makedirs(path_normal)

        path_high=os.path.join(newpath,'Test','High')
        if not os.path.exists(path_high): os.makedirs(path_high)

        path_low=os.path.join(newpath,'Test','Low')
        if not os.path.exists(path_low): os.makedirs(path_low)
        
        ##try: 
        
        if(True):    
            errorcode, message = self._movedata(uuid,glucosedata_path,export_txtfile_path,newpath,'Normal')
            errorcode, message = self._movedata(uuid,glucosedata_path,export_txtfile_path,newpath,'High')
            errorcode, message = self._movedata(uuid,glucosedata_path,export_txtfile_path,newpath,'Low')

            self._data_balance(newpath,"Train") ###做data balance
            self._data_balance(newpath,"Test") ###做data balance
            
        ##except:
        ##    errorcode="-300"
        ##    message="An error occurs in the data_arrangement function: fail to move data to normal, high and low categories!"
        ##    return errorcode,message

        return errorcode, message

    def _data_balance(self,newpath,type):      
        
        """
        newpath: Glucose_uuid directory
        type: Train or Test
        """
       
        ##print('start data balancing!')
        data_num_array=[]
        path_normal=os.path.join(newpath,type,'Normal')
        normal_data_num=len(os.listdir(path_normal))
        data_num_array.append(normal_data_num)
        
        path_high=os.path.join(newpath,type,'High')
        high_data_num=len(os.listdir(path_high))
        data_num_array.append(high_data_num)
        
        path_low=os.path.join(newpath,type,'Low')
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

                    balance_dir_path=os.path.join(newpath,type,target_dir_name)
            
                    filename_list=os.listdir(balance_dir_path)
                    for i in range(len(filename_list)):
                        if(i%mode_number !=0): ##移除到remove_data
                            currentpath=os.path.join(balance_dir_path,filename_list[i])
                            copy_to_dir=os.path.join(newpath,'Remove_Data',type)  ## for exmaple: .\GlucoseData_48\\Remove_Data\\Train
                            if not os.path.exists(copy_to_dir): 
                                os.makedirs(copy_to_dir)

                            copytopath=os.path.join(copy_to_dir,filename_list[i])
                            if(not os.path.isfile(copytopath)):  
                                if(os.path.isfile(currentpath)): 
                                    shutil.move(currentpath,copytopath)     

    def _movedata(self,uuid,glucosedata_path,export_txtfile_path,newpath,type):
        
        errorcode="0"
        message=""
              
        bin_num=0
        if(type=="High"): ##180~400,每差10視為一個區間
            bin_num=23
            lowest_value=180
            highest_value=400
        elif(type=="Low"):  ##30~80,每差10視為一個區間
            bin_num=6
            lowest_value=30
            highest_value=80
        else:  ##85~175,每差10視為一個區間
            bin_num=10
            lowest_value=85 
            highest_value=175
        
        
        data_path=os.path.join(export_txtfile_path,"Dataset",uuid,type)
        file_list=os.listdir(data_path) ###自dataset搬移資料
        
        min_glucose_value=10000
        max_glucose_value=-10000

        for i in range(len(file_list)):
            current_item=file_list[i]           
            current_item_str_array=current_item.split('_')
            current_last_str=current_item_str_array[-1]
            current_last_value_str=current_last_str.split('.')[0]
            current_glucose_value=int(current_last_value_str)
            if(current_glucose_value>=lowest_value and current_glucose_value<=highest_value): ###先確定落在目前處理的區間中
                if(current_glucose_value<min_glucose_value): ###找最小血糖值
                    min_glucose_value=current_glucose_value
            
                if(current_glucose_value>max_glucose_value): ###找最大血糖值
                    max_glucose_value=current_glucose_value            


        for i in range(len(file_list)):
            current_item=file_list[i]           
            current_item_str_array=current_item.split('_')
            current_last_str=current_item_str_array[-1]
            current_last_value_str=current_last_str.split('.')[0] 
            current_glucose_value=int(current_last_value_str)
            if(current_glucose_value==min_glucose_value or current_glucose_value==max_glucose_value): ###把最大最小值放在Training data                
                current_path=os.path.join(data_path,current_item)
                copyto_path=os.path.join(newpath,'Train',type)  
                if(not os.path.isfile(copyto_path)):  
                    if (os.path.isfile(current_path)):            
                        shutil.move(current_path,copyto_path)                

            
        ##----以上先把最低最高血糖值放到training set中---------------
        ##----剩下的資料依照血糖數值分到不同區間，譬如高血糖180~400之間每10切成一等分，共23等分，第一等分為180~189，第二等分為190~199，依此類推        
        glucosedata_path=os.path.join(glucosedata_path,uuid+'.csv')       
        glucose_csv_file_list=glob.glob(glucosedata_path)      
        value_date_histogram_pre = pd.DataFrame(columns=['time_strings'])
        value_date_histogram = pd.DataFrame(columns=['time_strings'])       

        for i in range(bin_num):
            value_date_histogram_pre.at[i, 'time_strings']=[] ## 初始化data frame
            value_date_histogram.at[i, 'time_strings']=[] ## 初始化data frame    
        
        try:
            data_path=os.path.join(export_txtfile_path,"Dataset",uuid,type)           
            file_list=os.listdir(data_path)            
            time_str_pre=""
            for index, filename in enumerate(file_list): ##統計時間質方圖
                filename_splited_array=filename.split("_")
                current_file_timestr=filename_splited_array[1]
                if(current_file_timestr!=time_str_pre): ##上一筆時間字串與目前分析的這筆是不同時間字串
                    time_str_pre=current_file_timestr
                    current_last_str=filename_splited_array[-1]
                    current_last_value_str=current_last_str.split('.')[0]
                    current_glucose_value=int(current_last_value_str)
                    if(current_glucose_value>=lowest_value and current_glucose_value<=highest_value): ##數值落在目前處理的區間才處理                        
                        current_bin_index=math.floor((current_glucose_value-lowest_value)/10) ###此血糖值落在哪一個bin中          
                        value_date_histogram.at[current_bin_index, 'time_strings'].append(current_file_timestr)  ##將時間字串放入此區間(可能重複)
                   
            
            ##for i in range(bin_num):  
            ##    print('bin_',str(i),': ', value_date_histogram.at[i, 'time_strings'])            

                
            for index, filename in enumerate(file_list): ###比對時間字串是否位於前80%
                filename_splited_array=filename.split("_")
                current_file_timestr=filename_splited_array[1]
                current_last_str=filename_splited_array[-1]
                current_last_value_str=current_last_str.split('.')[0]
                current_glucose_value=int(current_last_value_str)
                if(current_glucose_value>=lowest_value and current_glucose_value<=highest_value): ##數值落在目前處理的區間才處理      
                    current_bin_index=math.floor((current_glucose_value-lowest_value)/10)                   
                    current_timestr_array=value_date_histogram.at[current_bin_index,'time_strings']
                    current_trainingset_timestr_array=[] 
                    if(len(current_timestr_array)>=2): ###血糖區間範圍內有超過筆2時間才需要切train and test set
                        trainingset_num=math.floor(len(current_timestr_array)*0.8)                      
                        current_trainingset_timestr_array=current_timestr_array[0:trainingset_num]

                    else: ###不超過2筆時間，就直接給training set
                        current_trainingset_timestr_array=current_timestr_array  
                
                    if(current_file_timestr not in current_trainingset_timestr_array): ## 不在trainingset的time string陣列中，就分到testing set                                          
                        current_path=os.path.join(data_path,filename)
                        copyto_path=os.path.join(newpath,'Test',type)                       
                        if(not os.path.isfile(copyto_path)):
                            if (os.path.isfile(current_path)):
                                shutil.move(current_path,copyto_path)   

                    else:    ## 在trainingset的time string陣列中，分到training set                       
                        current_path=os.path.join(data_path,filename)
                        copyto_path=os.path.join(newpath,'Train',type)
                        if(not os.path.isfile(copyto_path)):
                            if (os.path.isfile(current_path)):
                                shutil.move(current_path,copyto_path)           
            
        
        except:
            errorcode="-101"
            message="An error occurs in the _movedata function: fail to read glucose csv files"
        
        return errorcode, message   

       
    def _dataconcate(self,srj_db_path,srj_file_list,start_time,end_time,uuid): ####資料串接

        """
        input ---
        db_path: srj檔案所在路徑
        srj_file_list: srj檔案列表
        start_time: 設定串接資料的開始日期(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
        end_time: 設定串接資料的結束日期(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
            
        output ---
        original_timearray: 所有srj檔中每個tt片段的開始時間序列
        ecg_data_array: 所有srj檔中每個tt片段ECG資料串接後的序列
        motion_data_array:  所有srj檔中每個tt片段motion資料串接後的序列
        breath_data_array:  所有srj檔中每個tt片段breath資料串接後的序列
        temp_data_array: 所有srj檔中每個tt片段temp資料串接後的序列
        """
   
        original_timearray=[]
        ecg_data_array=[]
        motion_data_array=[]
        breath_data_array=[]
        temp_data_array=[]
   

        if(len(start_time)>8): ####有時分秒之格式       
            start_time = datetime.strptime(start_time,"%Y%m%d %H%M%S")
            end_time = datetime.strptime(end_time,"%Y%m%d %H%M%S")
        else:   ###只有年月日之格式     
            start_time = datetime.strptime(start_time,"%Y%m%d")
            end_time=datetime.strptime(end_time,"%Y%m%d") 
      
        for index in range(len(srj_file_list)):
            ##current_file_path=os.path.join(db_path,uuid,srj_file_list[index])   
            current_file_path=os.path.join(srj_db_path,srj_file_list[index])   
            with open(current_file_path,"r") as srj:
                line = srj.readline()
                while line:                    
                    data = json.loads(line)
                    motions = data["rows"]["motions"]
                    ecgs=data["rows"]["ecgs"]
                    breaths=data["rows"]["breaths"]
                    temps=data["rows"]["temps"]
                    tt=data["tt"]
                    tt=int(tt/1000)
                    nowtime=datetime.fromtimestamp(tt)
                    
                    if(nowtime>=start_time and nowtime<=end_time): ###針對當天的時段Concate Data                                                              
                        motion_data_array.append(motions)                    
                        ecg_data_array.append(ecgs) 
                        breath_data_array.append(breaths)                         
                        temp_data_array.append(temps)         
                
                    line = srj.readline()          

        return ecg_data_array,motion_data_array,breath_data_array,temp_data_array        


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
    def __init__(self, dir_path, method='combine', classes=3):
        self.dir_path = os.path.abspath(dir_path)
        self.method = method
        self.data_len = 150 
        
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
        
        return signal, label

    def __len__(self):
        return len(self.Labels)

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


def Clean_Data(uuid,basepath):    
    
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
    
      
        glucose_data_path=os.path.join(basepath,"GlucoseData_"+uuid)
        if os.path.isdir(glucose_data_path):
            shutil.rmtree(glucose_data_path)        
    
        model_data_path=os.path.join(basepath,"Temp_ThreeClasses_Model",uuid)
        if os.path.isdir(model_data_path):
            shutil.rmtree(model_data_path)          

        model_data_path=os.path.join(basepath,"Temp_TwoClasses_Model",uuid)
        if os.path.isdir(model_data_path):
            shutil.rmtree(model_data_path)
    
        errorcode="0"
        message="All data for uuid "+uuid+" has been deleted!"     
    except:
        errorcode="-800"
        message="An error occurs in the Clean_Data function: Some data for uuid "+uuid+" has not deleted clearly!"   

    return errorcode,message   


def Clean_Model(uuid,basepath):    
    
    empty_state=0
    errorcode="0"
    message=""  
    
    try:       
        model_path=os.path.join(basepath,"Best_ThreeClasses_Model",uuid)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            errorcode="0"
            message="Model for uuid "+uuid+" has been deleted!"
        else:
            empty_state=1           

        model_path=os.path.join(basepath,"Best_TwoClasses_Model",uuid)
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
def BuildModel(uuid,basepath,srj_db_path,glucosedata_path):

    errorcode="0"
    message="" 
    status=-1
    
    
    DataArrangement_Obj = DataArrangement() 
    ##errorcode, message = DataArrangement_Obj.data_processing(uuid,start_time,end_time,srj_db_path,glucosedata_path,basepath,server_db_path) ##測試用
    errorcode, message = DataArrangement_Obj.data_processing(uuid, srj_db_path, glucosedata_path, basepath)
    

    if(int(errorcode)<0):
        return status, errorcode, message     
       
    # checkedpath=os.path.join(basepath,"GlucoseData_"+uuid,"Train","Low")
    # filelist_low_train=os.listdir(checkedpath)

    # checkedpath=os.path.join(basepath,"GlucoseData_"+uuid,"Test","Low")
    # filelist_low_test=os.listdir(checkedpath)

    # checkedpath=os.path.join(basepath,"GlucoseData_"+uuid,"Train","High")
    # filelist_high_train=os.listdir(checkedpath)

    # checkedpath=os.path.join(basepath,"GlucoseData_"+uuid,"Test","High")
    # filelist_high_test=os.listdir(checkedpath)

    # checkedpath=os.path.join(basepath,"GlucoseData_"+uuid,"Train","Normal")
    # filelist_normal_train=os.listdir(checkedpath)

    # checkedpath=os.path.join(basepath,"GlucoseData_"+uuid,"Test","Normal")
    # filelist_normal_test=os.listdir(checkedpath)


    # if(len(filelist_low_train)>0 and len(filelist_low_test)>0 and len(filelist_high_train)>0 and len(filelist_high_test)>0 and len(filelist_normal_train)>0 and len(filelist_normal_test)>0):  ###有中，低和高血糖資料
    #     status, errorcode, message=BuildModel_ThreeClasses(uuid,basepath)
    #     if(int(errorcode)>=0):
    #         message="Model with three classes has been built!"
    # elif(len(filelist_high_train)==0 or len(filelist_high_test)==0 or len(filelist_normal_train)==0 or len(filelist_normal_test)==0):
    #     errorcode="-402"
    #     message="An error occurs in the BuildModel function: No enough normal or high glucose data!"
    #     status=-1
    # else:
    #   status, errorcode, message=BuildModel_TwoClasses(uuid,basepath)
    #   if(int(errorcode)>=0):
    #     message="Model with two classes has been built!"

    return status, errorcode, message   
 

def BuildModel_ThreeClasses(uuid,basepath):
   
    errorcode="0"
    message=""
    status=-1

    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間
    code_version=get_version()   ######取得模型訓練時當前程式版本

    print('Start building the model with three classes!')
    ## the model have been built or not and read performance file
    model_output_folder = os.path.join(basepath, "Best_ThreeClasses_Model", uuid)
    if not os.path.exists(model_output_folder): 
        os.makedirs(model_output_folder)
    
    model_data_path=os.path.join(basepath,"Temp_ThreeClasses_Model",uuid)  ###先清空過去訓練的模型佔存資料
    if os.path.isdir(model_data_path):
        shutil.rmtree(model_data_path)    

    model_current_best_performance_txtfile=os.path.join(basepath, "Best_ThreeClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")   
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

    current_path=os.path.join(basepath,'GlucoseData_'+uuid,'Train')

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

    ## Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=Batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=Batch_size, sampler=valid_sampler)
    current_test_path=os.path.join(basepath,'GlucoseData_'+uuid,'Test')

    filelist_low=os.listdir(os.path.join(current_test_path,'Low'))
    filelist_high=os.listdir(os.path.join(current_test_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_test_path,'Normal'))

    if(len(filelist_low)==0 or len(filelist_high)==0 or len(filelist_normal)==0):        
        errorcode="-401"
        message="An error occurs in the BuildModel_ThreeClasses function: No testing data"
        
        return errorcode,message  
    
    testdata = ecgDataset(dir_path = current_test_path, method=Method)
    test_loader = DataLoader(testdata, batch_size=Batch_size)
    train_num, valid_num, test_num = len(train_sampler),len(valid_sampler),len(testdata)
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
    
    save_path = os.path.join(basepath,"Temp_ThreeClasses_Model",uuid) ###建立uuid專屬的模型存放資料夾  
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
       
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=1,patience=patience, cooldown=0, min_lr=0.00001)   

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
        model_path = os.path.join(basepath,"Temp_ThreeClasses_Model",uuid,"Model_"+str(uuid)+"_"+str(num_epoch)+".pth")
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
            model_current_best_performance_txtfile=os.path.join(basepath, "Best_ThreeClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")            
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


    model_historic_best_performance_txtfile=os.path.join(basepath, "Best_ThreeClasses_Model", uuid,"Historic_Best_Performance.txt") ###歷史績效檔案
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
        model_historic_best_performance_txtfile=os.path.join(basepath, "Best_ThreeClasses_Model", uuid,"Historic_Best_Performance.txt")
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
  
def BuildModel_TwoClasses(uuid,basepath):
    
    errorcode="0"
    message=""
    status=-1

    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間    
    code_version=get_version()   ######取得模型訓練時當前程式版本

    print('Start building the model with two classes!')
       
    model_output_folder = os.path.join(basepath, "Best_TwoClasses_Model", uuid)
    if not os.path.exists(model_output_folder): 
        os.makedirs(model_output_folder)
    
    model_data_path=os.path.join(basepath,"Temp_TwoClasses_Model",uuid)  ###先清空過去訓練的模型佔存資料
    if os.path.isdir(model_data_path):
        shutil.rmtree(model_data_path)    

    model_current_best_performance_txtfile=os.path.join(basepath, "Best_TwoClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")   
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
    current_path=os.path.join(basepath,'GlucoseData_'+uuid,'Train')
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

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=Batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=Batch_size, sampler=valid_sampler)
    current_test_path=os.path.join(basepath,'GlucoseData_'+uuid,'Test')

   
    filelist_high=os.listdir(os.path.join(current_test_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_test_path,'Normal'))

    if(len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-501"
        message="An error occurs in the BuildModel_TwoClasses function: No testing data"
        return errorcode,message 
    
    testdata = ecgDataset(dir_path=current_test_path, method=Method, classes=2)

    test_loader = DataLoader(testdata, batch_size=Batch_size)

    train_num, valid_num, test_num = len(train_sampler),len(valid_sampler),len(testdata)
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
         
    save_path = os.path.join(basepath,"Temp_TwoClasses_Model",uuid) ###建立uuid專屬的模型存放資料夾
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=1, patience=100, cooldown=0, min_lr=0.00001)
    
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
        model_path = os.path.join(basepath,"Temp_TwoClasses_Model",uuid,"TwoClassesModel_"+str(uuid)+"_"+str(num_epoch)+".pth")
        model.load_state_dict(torch.load(model_path))

        test_old_data = 0
        if test_old_data:
            testdata = ecgDataset(dir_path = './GlucoseData_'+uuid+'/Test', method=Method, classes=2)
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
            model_current_best_performance_txtfile=os.path.join(basepath, "Best_TwoClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")            
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


    model_historic_best_performance_txtfile=os.path.join(basepath, "Best_TwoClasses_Model", uuid,"Historic_Best_Performance.txt") ###歷史最佳績效
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
        model_historic_best_performance_txtfile=os.path.join(basepath,"Best_TwoClasses_Model", uuid,"Historic_Best_Performance.txt")

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

    def __init__(self,basepath):
        self.basepath = basepath   

    def glucose_predict(self,uuid,ecgdata): 
        
        errorcode="0"
        message=""

        current_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data')
        if not os.path.exists(current_testingdata_path): os.makedirs(current_testingdata_path)

        normal_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data','Normal') ###ECG資料都只放在Normal資料夾
        if not os.path.exists(normal_testingdata_path): os.makedirs(normal_testingdata_path)
        
        old_file_list=os.listdir(normal_testingdata_path)        
        for i,filename in enumerate(old_file_list): ###刪除上次預測的資料
            os.remove(os.path.join(normal_testingdata_path,filename))

        high_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data','High') ###空的資料夾
        if not os.path.exists(high_testingdata_path): os.makedirs(high_testingdata_path)

        low_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data','Low') ###空的資料夾
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
                glucose_list=self.predict(uuid,self.basepath)
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


    def predict(self,uuid,basepath):        
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

        current_test_path=os.path.join(basepath,'GlucoseData_'+uuid,'Current_Test_Data')
        
        filelist=os.listdir(current_test_path)
     
        if(len(filelist)==0):
            return -2   ### no testing data
    
        
        testdata = ecgDataset(dir_path = current_test_path, method=Method)
        Batch_size=len(testdata)
        
        test_loader = DataLoader(testdata, batch_size=Batch_size)

        model = CNN(data_len=testdata.data_len, input_channel=testdata.channel)
        model.to(device)        
        best_model_performance_path = os.path.join(basepath,"Best_ThreeClasses_Model",uuid,"Historic_Best_Performance.txt")
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

        model_path = os.path.join(basepath,"Best_ThreeClasses_Model",uuid,best_model_path)
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

    def __init__(self,basepath):
        self.basepath = basepath        
    
   
    def glucose_predict(self,uuid,ecgdata): 
        
        errorcode="0"
        message=""
        glucose_list=[]

        current_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data')
        if not os.path.exists(current_testingdata_path): os.makedirs(current_testingdata_path)

        normal_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data','Normal') ###ECG資料都只放在Normal資料夾
        if not os.path.exists(normal_testingdata_path): os.makedirs(normal_testingdata_path)
        
        old_file_list=os.listdir(normal_testingdata_path)        
        for i,filename in enumerate(old_file_list): ##刪除上次預測的資料
            os.remove(os.path.join(normal_testingdata_path,filename))

        high_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data','High') ###空的資料夾
        if not os.path.exists(high_testingdata_path): os.makedirs(high_testingdata_path)

        low_testingdata_path=os.path.join(self.basepath,'GlucoseData_'+uuid,'Current_Test_Data','Low') ###空的資料夾
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
                glucose_list=self.predict(uuid,self.basepath)
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


    def predict(self,uuid,basepath):     

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

        current_test_path=os.path.join(basepath,'GlucoseData_'+uuid,'Current_Test_Data')
        
        filelist=os.listdir(current_test_path)
        if(len(filelist)==0):
            return -2   ### no testing data
    
        
        testdata = ecgDataset(dir_path = current_test_path, method=Method,classes=2)
        Batch_size=len(testdata)
        test_loader = DataLoader(testdata, batch_size=Batch_size)

        model = TwoClasses_CNN(data_len=testdata.data_len, input_channel=testdata.channel)
        model.to(device)        
        best_model_performance_path = os.path.join(basepath,"Best_TwoClasses_Model",uuid,"Historic_Best_Performance.txt")
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

        model_path = os.path.join(basepath,"Best_TwoClasses_Model",uuid,best_model_path)        
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
    
    def __init__(self,uuid,basepath):
        self.uuid=uuid
        self.basepath=basepath  
        self.errorcode="0"     
        self.set_model()  

    def set_model(self):      
        historic_best_sensitivity=0
        historic_best_specificity=0
        model_historic_best_performance_txtfile = os.path.join(self.basepath,"Best_ThreeClasses_Model",self.uuid,"Historic_Best_Performance.txt")
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
            self.GlucosePredictor_Obj=GlucoseThreeClassesPredictor(self.basepath)
        else:  ##檢查二類別模型是否存在     
            model_historic_best_performance_txtfile = os.path.join(self.basepath,"Best_TwoClasses_Model",self.uuid,"Historic_Best_Performance.txt")
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
                self.GlucosePredictor_Obj=GlucoseTwoClassesPredictor(self.basepath)
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

    user_information = [['67', '20240717', '20240730'],
                        ['1619', '20230909', '20230923'],
                        ['1631', '20230914', '20230929'],                        
                        ['1662', '20230922', '20231006'],
                        ['1672', '20230925', '20231009'],
                        ['1677', '20230926', '20231029'],  ##5
                        ['1692', '20231013', '20231110'],  ##criswu(應該超過80歲)
                        ['1693', '20231016', '20231027'],
                        ['1726', '20231020', '20231103'],
                        ['1727', '20231020', '20231101'],
                        ['1732', '20231028', '20231110'],  ##10(測試集高血糖只有一筆)
                        ['1737', '20231102', '20231116'],
                        ['1741', '20231103', '20231117'],
                        ['1748', '20231105', '20231118'],
                        ['1828', '20240112', '20240126'],
                        ['1426', '20230604', '20230628'],  ##15
                        ['1417', '20230602', '20230611'],  ##目前CSV檔格式有問題
                        ['1669', '20230924', '20231007'],
                        ['1798', '20231212', '20231226'],  ##20
                        ['1742', '20231103', '20231117'],
                        ['599', '20221129', '20221223'],
                        ['67', '20240716', '20240724'],
                        ['88', '20240716', '20240724'],
                        ['92', '20240607', '20240813'],
                        ['9', '20240611', '20240614'],
                        ['88', '20240716', '20240730'],
                        ['700', '20240716', '20240808']]

    basepath=r'C:\Users\belle\Documents\algorithm_blood_glucose-main\Model'  ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    glucosedata_path=r'C:\Users\belle\Documents\algorithm_blood_glucose-main\GlucoseDataCSV' ###APP收到的血糖數值，整理成CSV檔後存放路徑
    server_db_path=r'G:\.shortcut-targets-by-id\1mVd2VN2iGQWp7B1dN1zE55tszPd4fL9j\Health_Server'   ###ECG資料所在雲端路徑(Dennis測試用)

    for i in range(8,9):  ##range(0,len(user_information)):
        user_info=user_information[i]
        ##-------step 1. 基本設定--------
        uuid=user_info[0]  ##'1662' ##'1732' ##   ##'1828' ##'1741'  ##'1742' ##'1727' ##'1631' ##'1748'
        start_time=user_info[1] ##'20230922' ##'20231028' ##'20240112' ##'20231103' ##'20231020' ##'20230914' ##'20231105'         ##第一筆血糖資料記錄日期
        end_time=user_info[2] ##'20231006' ##'20231110'  ##'20240126' ##'20231117' ##'20231102' ##'20230929' ##'20231118'         ##最後一筆血糖資料紀錄日期
        print('index:',i,' uuid:',uuid)
        srj_db_path='C:\\Users\\belle\\Documents\\algorithm_blood_glucose-main\\DataDB\\'   ###srj檔放置路徑

        ##-------step 2. 血糖模型建立---------
        #status, errorcode, message = BuildModel(uuid,basepath,srj_db_path,glucosedata_path,server_db_path)  ###建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
        status, errorcode, message = BuildModel(uuid, basepath, srj_db_path, glucosedata_path)  ###建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
        print('status:',str(status),' error code:',errorcode,' message:',message)

    
        ##--------step 3. 輸入ECG進行血糖類別分類---------   
        testdatapath=os.path.join(basepath,"TestData",uuid) ##測試用程式
        if not os.path.exists(testdatapath): os.makedirs(testdatapath)  ##測試用程式 

        filelist=os.listdir(testdatapath) ##測試用程式
        ecgdata=[] ##測試用程式
        for i in range(0,len(filelist)): ##測試用程式
            filename=filelist[i]
            print('filename:',filename)
            f = open(os.path.join(testdatapath,filename), "r")
            for x in f:
                ecgdata.append(int(x.rstrip('\n')))       
    
        GlucosePredictor_Obj=GlucosePredictor(uuid,basepath) 
        errorcode, message, glucose_category=GlucosePredictor_Obj.predict(ecgdata)
        print('glucose_category:',glucose_category)
        print('errorcode:',str(errorcode),' message:',message)     
     
    run_end_time = time.perf_counter()
    print("Total running time: %.2f seconds" %(run_end_time - run_start_time))
