import os
import sys
from datetime import datetime, timedelta
import numpy as np
from zipfile import ZipFile
import pandas as pd
import ujson as json
       
    
def search_unzip_file(db_path,target_uuid,start_time,end_time,export_path):  ##檔案解壓縮      
    
    """
    input ---
        db_path: 本地端電腦資料庫路徑，例如  DBPath = "C:\\Users\\SWM-Jared\\Desktop\\DataDB"
        target_uuid: 要找尋的uuid檔案
        start_time: 開始時間(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
        end_time: 結束時間(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
        export_path: 解壓縮檔案放置路徑       
    
    output ---
        unzip_file_name_list: 回傳符合時間起迄下target_uuid所屬的srj檔案列表(已經解壓縮)
    """

    if(len(start_time)>8): ####有時分秒之格式 
        startTimestamp = datetime.strptime(start_time,"%Y%m%d %H%M%S").timestamp()
        endTimestamp = datetime.strptime(end_time,"%Y%m%d %H%M%S").timestamp()        
        startDT = datetime.strptime(start_time.split(" ")[0],"%Y%m%d") - timedelta(days=1)
        endDT = datetime.strptime(end_time.split(" ")[0],"%Y%m%d")
    else: ###只有年月日之格式
        startTimestamp = datetime.strptime(start_time,"%Y%m%d").timestamp()
        endTimestamp = ( datetime.strptime(end_time,"%Y%m%d")  + timedelta(days=1) ).timestamp()
        startDT = datetime.strptime(start_time,"%Y%m%d") - timedelta(days=1)
        endDT = datetime.strptime(end_time,"%Y%m%d")
   
    unzip_file_name_list=[]  
    file_timestamps = []
       

    ## 遍蒞所有資料夾
    for D in os.listdir(db_path):
        
        if(len(D)<8): ##uuid的資料夾不需要parsing資料
            continue
                
        Folder = os.path.join(db_path,D)
        # 符合日期的資料夾
        if os.path.isdir(Folder):
            date = datetime.strptime(D,"%Y%m%d")
            if date >= startDT and date <= endDT:
                # 遍歷該日期資料夾的檔案
                for files in os.listdir(Folder):
                    fileparts = files.split("_")
                    uuid = fileparts[0]
                        
                    # 檢查檔案是否已解壓縮過
                    filename = files.split(".")[0]
                    ##if filename in existFiles:
                    ##    continue
                    if "(" in filename or ")" in filename: ###有些檔案是重復版本，不用再次解壓縮
                        continue
                    # 檢查符合UUID且為zip檔
                    if uuid == target_uuid and files.endswith(".zip"):
                        filePath = os.path.join(Folder,files)                            
                        file_timestamps.append(int(fileparts[1])/1000)                            
                         
                        # 解壓縮至指定路徑
                        with ZipFile(filePath,"r") as zip:
                            zip.extractall(export_path)
                            unzip_file_name_list.append(filename[0:len(filename)]+".srj")  
                              
       
        
    start_index = -1
    end_index = -1        
    unzip_file_name_list = sorted(unzip_file_name_list)
    file_timestamps = sorted(file_timestamps)
        
    if len(file_timestamps) > 0:
        
        if startTimestamp <= file_timestamps[0] and endTimestamp >= file_timestamps[0]:
            start_index = 0
                
        for file_i, file_timestamp in enumerate(file_timestamps):
                
            if startTimestamp >= file_timestamp and start_index == -1:
                    
                start_index = file_i
                
            if start_index != -1 and end_index == -1 and endTimestamp <= file_timestamp:
                    
                end_index = file_i
            
        if start_index != -1 and end_index == -1:
            end_index = len(file_timestamps)
        
    if start_index == -1:
        return []
    else :
        return unzip_file_name_list[start_index:end_index]        


def dataconcate(db_path,srj_file_list,start_time,end_time,uuid): ####資料串接

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
        current_file_path=os.path.join(db_path,uuid,srj_file_list[index])   
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
                    original_timearray.append(nowtime)                
                    for m in motions:
                        motion_data_array.append(m)
                    
                    for n in ecgs:
                        ecg_data_array.append(n) 

                    for b in breaths:
                        breath_data_array.append(b) 

                    for t in temps:
                        temp_data_array.append(t)         
                
                line = srj.readline()          

    return original_timearray,ecg_data_array,motion_data_array,breath_data_array,temp_data_array        



