# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:39:56 2022

@author: SWM-Jared
"""

import os
import json
import numpy as np
from scipy import signal   ##, interpolate
from scipy.interpolate import interp1d
from ..ecg.upsidedown.upsidedown_correction import load_cnn, _checking_ecg_upsidedown_v2


def _delete_impulse(ecgs,diff_threshold):
    
    ecgs_diff = np.diff(ecgs)
    diff_len = len(ecgs_diff) - 1
    
    j = 0
    
    for i, (ecg, ecg_diff) in enumerate(zip(ecgs[:-1], ecgs_diff)):
        
        if j == 0 :
        
            if ecg_diff < -diff_threshold:
                
                j = 1
                
                try:
                
                    while ecgs_diff[i+j] == 0:
                        j += 1
                        
                except:
                    j -= 1
                
                if ecgs_diff[i+j] > diff_threshold:
                    
                    for k in range(i+1,i+j+1): ecgs[k]=(ecgs[i]+ecgs[i+j+1])/2
        
            elif ecg_diff > diff_threshold:
                
                j = 1
                
                try:
                
                    while ecgs_diff[i+j] == 0:
                        j += 1
                        
                except:
                    j -= 1
                
                if ecgs_diff[i+j] < -diff_threshold:
                    
                    for k in range(i+1,i+j+1): ecgs[k]=(ecgs[i]+ecgs[i+j+1])/2
        else :
            
            j = j - 1
           
    return ecgs


def _remove_abnormal_data_segment(srj_lines, verbose):
    
    srj_tt_list = []
    new_srj_lines = []
    
    srj_tt_check = 0
    srj_ecg_number = 2500
    
    lack_ecg_count = 0
    repeat_count = 0
    lack_2hz_count = 0
    
    for srj_line in srj_lines:
        
        srj_json = json.loads(srj_line)
        
        srj_tt = srj_json["tt"]
        
        srj_temp_num = len(srj_json["rows"]["temps"])
        srj_motion_num = len(srj_json["rows"]["motions"])
        
        if srj_ecg_number != int(srj_json["ecgno"]):
            
            lack_ecg_count += 1
            
        elif srj_tt_check == srj_tt:
            
            repeat_count += 1
        
        elif ( srj_temp_num == 0 or srj_motion_num == 0 ) and lack_2hz_count == 0:
            
            lack_2hz_count += 1
        
        else :
            
            srj_tt_list.append(srj_tt)
            new_srj_lines.append(srj_line)
        
        srj_tt_check = srj_tt
    
    if verbose:
        print("1. Delete Insufficient & Duplicate Data Segment ({} -> {}).".format(len(srj_lines),len(new_srj_lines)))
        print("   --- ECG Insufficient: {}".format(lack_ecg_count))
        print("   --- 2HZ Insufficient: {}".format(lack_2hz_count))
        print("   --- Duplicate: {}".format(repeat_count))
    
    return new_srj_lines, srj_tt_list


def _divide_time_slices(srj_tt_list, version, verbose):
    
    if version == 'new':
    
        srj_tt_list = np.linspace(srj_tt_list[0], srj_tt_list[-1]+10028, num=len(srj_tt_list)+1, endpoint=True)
    
    elif version == 'old':
        
        srj_tt_list.append(srj_tt_list[-1]+10028)
    
    srj_td_list = np.diff(srj_tt_list)
    
    srj_tt_list = srj_tt_list[:-1]
    
    if verbose:
        
        if version == 'new':
            
            print("2. Divide All Time Slices of Data Segments Equally ({} ms).".format(round(np.nanmean(srj_td_list))))
        
        elif version == 'old':
            
            print("2. Divide All Time Slices of Data Segments ({} ms).".format(round(np.nanmean(srj_td_list))))
    
    return srj_tt_list, srj_td_list


def _assign_all_time_to_signal(srj_tt_list, srj_td_list, new_srj_lines, verbose):
    
    srj_ecg_times = []
    srj_breath_times = []
    srj_temp_times = []
    srj_motion_times = []
    
    srj_ecg_signals = []
    srj_breath_signals = []
    srj_temp_signals = []
    srj_motion_signals = []
    
    srj_last_breath = 0
    srj_last_temp = 0
    srj_last_motion = 0
    
    for srj_tt, srj_td, srj_line in zip(srj_tt_list, srj_td_list, new_srj_lines):
        
        srj_json = json.loads(srj_line)
        
        srj_ecgs = srj_json["rows"]["ecgs"]
        srj_breaths = srj_json["rows"]["breaths"]
        srj_temps = srj_json["rows"]["temps"]
        srj_motions = srj_json["rows"]["motions"]
        
        srj_ecg_times.extend(np.linspace(srj_tt, srj_tt+srj_td, num=2500, endpoint=False))
        srj_ecg_signals.extend(srj_ecgs)
        
        if len(srj_breaths) > 0:
            srj_breath_times.extend(np.linspace(srj_tt, srj_tt+srj_td, num=len(srj_breaths), endpoint=False))
            srj_breath_signals.extend(srj_breaths)
        
        elif len(srj_breaths) == 0:
            srj_breath_times.append(srj_tt)
            srj_breath_signals.append(srj_last_breath)
        
        if len(srj_temps) > 0:
            srj_temp_times.extend(np.linspace(srj_tt, srj_tt+srj_td, num=len(srj_temps), endpoint=False))
            srj_temp_signals.extend(srj_temps)
        
        elif len(srj_temps) == 0:
            srj_temp_times.append(srj_tt)
            srj_temp_signals.append(srj_last_temp)
        
        if len(srj_motions) > 0:
            srj_motion_times.extend(np.linspace(srj_tt, srj_tt+srj_td, num=len(srj_motions), endpoint=False))
            srj_motion_signals.extend(srj_motions)
        
        elif len(srj_motions) == 0:
            srj_motion_times.append(srj_tt)
            srj_motion_signals.append(srj_last_motion)
        
        srj_last_breath = srj_breath_signals[-1]
        
        srj_last_temp = srj_temp_signals[-1]
            
        srj_last_motion = srj_motion_signals[-1]
    
    srj_ecg_times.append(srj_tt+srj_td)
    srj_breath_times.append(srj_tt+srj_td)
    srj_temp_times.append(srj_tt+srj_td)
    srj_motion_times.append(srj_tt+srj_td)
    
    srj_ecg_signals.append(srj_ecgs[-1])
    srj_breath_signals.append(srj_breaths[-1])
    srj_temp_signals.append(srj_temps[-1])
    srj_motion_signals.append(srj_motions[-1])
    
    srj_ecg_signals = _delete_impulse(srj_ecg_signals)
    
    if verbose:
        print("3. Assign Time Points to All Signals.")
        print("   --- ECG: {} - {}".format(len(srj_ecg_times), len(srj_ecg_signals)))
        print("   --- BREATH: {} - {}".format(len(srj_breath_times), len(srj_breath_signals)))
        print("   --- TEMPERATURE: {} - {}".format(len(srj_temp_times), len(srj_temp_signals)))
        print("   --- MOTION: {} - {}".format(len(srj_motion_times), len(srj_motion_signals)))
    
    return srj_ecg_times, srj_ecg_signals, srj_breath_times, srj_breath_signals, srj_temp_times, srj_temp_signals, srj_motion_times, srj_motion_signals


def _resample_signal(srj_ecg_times, srj_ecg_signals, srj_breath_times, srj_breath_signals, srj_temp_times, srj_temp_signals, srj_motion_times, srj_motion_signals, verbose):
    
    srj_time_s = int(srj_ecg_times[0])
    srj_time_e = int(srj_ecg_times[-1])
    
    new_srj_segment_num = int( srj_time_e - srj_time_s ) // 10000
        
    srj_250hz_times = range( srj_time_s, srj_time_s + 10000*new_srj_segment_num, 4)
    srj_50hz_times = range( srj_time_s, srj_time_s + 10000*new_srj_segment_num, 20)
    srj_2hz_times = range( srj_time_s, srj_time_s + 10000*new_srj_segment_num, 500)
    
    f_ecg = interp1d(srj_ecg_times, srj_ecg_signals)
    new_srj_ecg_signals = f_ecg( srj_250hz_times )
    
    f_breath = interp1d(srj_breath_times, srj_breath_signals)
    new_srj_breath_signals = f_breath( srj_50hz_times )
    
    srj_temp_signals = np.array(srj_temp_signals).T
    
    new_srj_temp_signals = []
    
    for srj_temp_signal in srj_temp_signals:
        
        f_temp = interp1d(srj_temp_times, srj_temp_signal)
        
        new_srj_temp_signals.append( f_temp( srj_2hz_times ) )
    
    new_srj_temp_signals = np.array(new_srj_temp_signals).T
    
    srj_motion_signals = np.array(srj_motion_signals).T
    
    new_srj_motion_signals = []
    
    for srj_motion_signal in srj_motion_signals:
        
        f_motion = interp1d(srj_motion_times, srj_motion_signal)
        
        new_srj_motion_signals.append( f_motion( srj_2hz_times ) )
    
    new_srj_motion_signals = np.array(new_srj_motion_signals).T
    
    new_srj_time_signals = np.array(srj_2hz_times).reshape(-1, 20)[:,0]
    new_srj_ecg_signals = np.array(new_srj_ecg_signals).reshape(-1, 2500).astype('int')
    new_srj_breath_signals = np.array(new_srj_breath_signals).reshape(-1, 500).astype('int')
    new_srj_temp_signals = np.array(new_srj_temp_signals).reshape(-1, 20, 2)
    new_srj_motion_signals = np.array(new_srj_motion_signals).reshape(-1, 20, 10).astype('float')
    
    if verbose:
        print("4. Resample all signals to 250 hz, 50 hz & 2 hz.")
        print("   --- TIME: {}".format(new_srj_time_signals.shape))
        print("   --- ECG: {}".format(new_srj_ecg_signals.shape))
        print("   --- BREATH: {}".format(new_srj_breath_signals.shape))
        print("   --- TEMPERATURE: {}".format(new_srj_temp_signals.shape))
        print("   --- MOTION: {}".format(new_srj_motion_signals.shape))
    
    return new_srj_time_signals, new_srj_ecg_signals, new_srj_breath_signals, new_srj_temp_signals, new_srj_motion_signals


def _create_data_segment(new_srj_lines, new_srj_time_signals, new_srj_ecg_signals, new_srj_breath_signals, new_srj_temp_signals, new_srj_motion_signals, verbose):
    
    srj_line_sample = new_srj_lines[0]
    srj_json_sample = json.loads(srj_line_sample)
    srj_json_sample["ecgno"] = 2500
    
    final_srj_lines = []
    
    for time, ecg, breath, temp, motion in zip( new_srj_time_signals, new_srj_ecg_signals, new_srj_breath_signals, new_srj_temp_signals, new_srj_motion_signals ):
        
        srj_json_sample["tt"] = time.tolist()
        srj_json_sample["rows"]["ecgs"] = ecg.tolist()
        srj_json_sample["rows"]["breaths"] = breath.tolist()
        srj_json_sample["rows"]["temps"] = temp.tolist()
        srj_json_sample["rows"]["motions"] = motion.tolist()
    
        final_srj_lines.append(json.dumps(srj_json_sample))
    
    if verbose:
        print("5. Create New Data Segment ({}).".format(len(final_srj_lines)))
    
    return final_srj_lines


def _create_srj_file(export_file, final_srj_lines, verbose):
    
    export_string = ""
    
    for srj_line in final_srj_lines:
        
        export_string += str(srj_line).replace("\'","\"").replace(" ","") + '\n'
    
    with open(export_file, 'w') as f:
        f.write(export_string)
        f.close()
    
    if verbose:
        print("6. Create SRJ File and Export.")
        print("   --- {}".format(export_file))


def _normalize_signal(srj_signal, last_signal, standard_number, data_type):
    
    signal_number = len(srj_signal)
    
    if signal_number > standard_number:
        
        if data_type != '':
            signal_index = np.linspace(0, signal_number-1, num=standard_number, endpoint=True).astype(data_type)
        else :
            signal_index = np.linspace(0, signal_number-1, num=standard_number, endpoint=True)
        
        srj_signal = [ srj_signal[int(i)] for i in signal_index]
    
    elif signal_number < standard_number :
        
        srj_signal.extend([last_signal for i in range(standard_number-signal_number)])
    
    return srj_signal


def _filter_signal(srj_tt_list, srj_td_list, new_srj_lines, verbose):
    
    new_srj_time_signals = []
    new_srj_ecg_signals = []
    new_srj_breath_signals = []
    new_srj_temp_signals = []
    new_srj_motion_signals = []
    
    last_srj_json = json.loads(new_srj_lines[0])
    
    if len(last_srj_json["rows"]["breaths"]) == 0:
        last_srj_breath = 0
    else :
        last_srj_breath = last_srj_json["rows"]["breaths"][0]
    
    if len(last_srj_json["rows"]["temps"]) == 0:
        last_srj_temp = [0, 0]
    else :
        last_srj_temp = last_srj_json["rows"]["temps"][0]
    
    if len(last_srj_json["rows"]["motions"]) == 0:
        last_srj_motion = [0,0,0,0,0,0,0,0,0,0]
    else :
        last_srj_motion = last_srj_json["rows"]["motions"][0]
    
    for srj_line in new_srj_lines:
        
        srj_json = json.loads(srj_line)
        
        srj_tt = srj_json["tt"]
        srj_ecg = srj_json["rows"]["ecgs"]
        srj_breath = srj_json["rows"]["breaths"]
        srj_temp = srj_json["rows"]["temps"]
        srj_motion = srj_json["rows"]["motions"]
        
        new_srj_time_signals.append(srj_tt)
        new_srj_ecg_signals.extend(srj_ecg)
        
        srj_breath = _normalize_signal(srj_breath, last_srj_breath, 500, 'int')
        srj_temp = _normalize_signal(srj_temp, last_srj_temp, 20, '')
        srj_motion = _normalize_signal(srj_motion, last_srj_motion, 20, 'float')
        
        new_srj_breath_signals.extend(srj_breath)
        new_srj_temp_signals.extend(srj_temp)
        new_srj_motion_signals.extend(srj_motion)
    
        last_srj_breath = srj_breath[-1]
        last_srj_temp = srj_temp[-1]
        last_srj_motion = srj_motion[-1]
    
    new_srj_ecg_signals = _delete_impulse(new_srj_ecg_signals)
    
    new_srj_time_signals = np.array(new_srj_time_signals)
    new_srj_ecg_signals = np.array(new_srj_ecg_signals).reshape(-1, 2500).astype('int')
    new_srj_breath_signals = np.array(new_srj_breath_signals).reshape(-1, 500).astype('int')
    new_srj_temp_signals = np.array(new_srj_temp_signals).reshape(-1, 20, 2)
    new_srj_motion_signals = np.array(new_srj_motion_signals).reshape(-1, 20, 10).astype('float')
    
    return new_srj_time_signals, new_srj_ecg_signals, new_srj_breath_signals, new_srj_temp_signals, new_srj_motion_signals


def file_process(import_file_path, export_file_path, export_type='file', upsidedown_flag=True,ecg_rate=250,breath_rate=50,temp_rate=2,motion_rate=2): ### 主程式
    
    """
    input ---
        import_file_path: the path of the file to be handled (example: C:/Users/User/Desktop/DataDB/48/48_1709537445733_546C0EDE3E34.srj)
        export_file_path: the path of the file to be exported if export='file' (example:  C:/Users/User/Desktop/DataDB\48\48_1709537445733_546C0EDE3E34.srj)
        export_type: input string "file" to export file, otherwise, export variable srj_tts, srj_ecgs, srj_breaths, srj_temps, srj_motions 
        upsidedown_flag: if True, correct the upsidedown ecg segment, otherwise, keep it originally
        ecg_rate= sampling rate of ECG signal, default vale = 250
        motion_rate: sampling rate of motion signal, default vale = 2
        breath_rate: sampling rate of breath signal, default vale = 50
        temp_rate: sampling rate of temprature signal, default vale = 2
            
    output ---    
        if export_type='file', export srj file in export_file_path 
        otherwise, export variables srj_tts, srj_ecgs, srj_breaths, srj_temps, srj_motions

    """

    srj_tts = []
    srj_ecgs = []
    srj_breaths = []
    srj_temps = []
    srj_motions = []
    srj_lines = []
    
    p_srj_tt = -1
    p_srj_breath = 0
    p_srj_motion = np.zeros(motion_rate*10).tolist()   ## 10  motion_rate*10 
    p_srj_temp = np.zeros(temp_rate*10).tolist()  ## 2 
    
    with open(import_file_path, 'r') as f:
        
        for srj_line in f:
        
            if len(srj_line) > 1:
                
                srj_json = json.loads(srj_line)
                srj_tt = srj_json["tt"]
                srj_ecgno = srj_json["ecgno"]
                
                if srj_tt != p_srj_tt and srj_ecgno == ecg_rate*10:  ###2500:
                    
                    srj_lines.append(srj_line)
                    
                    srj_tts.append(srj_json["tt"])
                    srj_ecgs.append(srj_json["rows"]["ecgs"])
                    
                    srj_motion = srj_json["rows"]["motions"]
                    srj_breath = srj_json["rows"]["breaths"]
                    srj_temp = srj_json["rows"]["temps"]
                    
                    srj_breath_len = len(srj_breath)
                    
                    if srj_breath_len == 0:
                        srj_breath = [ p_srj_breath for i in range(breath_rate*10) ]  ##500
                        
                    elif srj_breath_len < breath_rate*10:  ##500
                        srj_breath.extend([srj_breath[-1] for i in range(breath_rate*10-srj_breath_len)])  ##500
                    
                    elif srj_breath_len > breath_rate*10:  ###500
                        srj_breath = signal.resample(srj_breath, breath_rate*10).tolist()  ###500
                        
                    srj_breaths.append(srj_breath)
                    
                    srj_temp_len = len(srj_temp)
                    
                    if srj_temp_len == 0:
                        srj_temp = [ p_srj_temp for i in range(temp_rate*10) ]  ###20
                        
                    elif srj_temp_len < temp_rate*10:  ##20
                        srj_temp.extend([srj_temp[-1] for i in range(temp_rate*10-srj_temp_len)]) ###20
                    
                    elif srj_temp_len > temp_rate*10: ###20
                        srj_temp = signal.resample(srj_temp, temp_rate*10).tolist()  ###10
                        
                    srj_temps.append(srj_temp)
                    
                    srj_motion_len = len(srj_motion)
                    
                    if srj_motion_len == 0:
                        srj_motion = [ p_srj_motion for i in range(motion_rate*10) ]   ###20
                        
                    elif srj_motion_len < motion_rate*10:  ###20
                        srj_motion.extend([srj_motion[-1] for i in range(motion_rate*10-srj_motion_len)])   ###20
                    
                    elif srj_motion_len > motion_rate*10: ###20
                        srj_motion = signal.resample(srj_motion, motion_rate*10).tolist()   ###20
                        
                    srj_motions.append(srj_motion)
                    
                p_srj_breath = srj_breath[-1]
                p_srj_motion = srj_motion[-1]
                p_srj_temp = srj_temp[-1]
                p_srj_tt = int(srj_tt)
                
    f.close()
    
    ###srj_tts = np.linspace(srj_tts[0], srj_tts[-1], len(srj_tts), endpoint=True).astype('int64').tolist() ###測試先遮掉
    
    ### Load CNN model
    ##model = load_cnn()

    # checking upside-down result list
      
    if(upsidedown_flag):
    
        new_srj_ecgs = []
    
        ###Predict srj_ecgs  
        usd_flag = _checking_ecg_upsidedown_v2(srj_ecg=srj_ecgs, evaluate_quality=True)

        if usd_flag == 1:
        
            for ecgs in srj_ecgs:
            
                ecgs = np.array(ecgs) * -1
                new_srj_ecgs.append(ecgs.tolist())
    
        else :
        
            for ecgs in srj_ecgs:
                new_srj_ecgs.append(ecgs)
      
        srj_ecgs = new_srj_ecgs
    

    if export_type == 'file':

        srj_tts = np.linspace(srj_tts[0], srj_tts[-1], len(srj_tts), endpoint=True).astype('int64').tolist()
        new_srj_lines = []
        
        for srj_line, srj_tt, srj_ecg, srj_breath, srj_temp, srj_motion in zip(srj_lines, srj_tts, srj_ecgs, srj_breaths, srj_temps, srj_motions):
            
            srj_json = json.loads(srj_line)
            srj_json["tt"] = srj_tt
            srj_json["rows"]["ecgs"] = srj_ecg
            srj_json["rows"]["breaths"] = srj_breath
            srj_json["rows"]["temps"] = srj_temp
            srj_json["rows"]["motions"] = srj_motion
            
            srj_line = json.dumps(srj_json)
            
            new_srj_lines.append(srj_line)
    
        export_string = ""
            
        for srj_line in new_srj_lines:
            
            export_string += str(srj_line).replace("\'","\"").replace(" ","") + '\n'
        
        with open(export_file_path, 'w') as f:
            f.write(export_string)
            f.close()
        
        return True
    
    elif export_type == 'data':
        
        return srj_tts, srj_ecgs, srj_breaths, srj_temps, srj_motions


if __name__ == '__main__':
    
    """
    verbose = True or False
    version = 'new' or 'old'
    resample = True or False
    export = 'file', 'data', or 'signal'
    """
    
    import_path = 'DATA/'
    export_path = 'EXPORT/'
    
    srj_file = '139_1668433273112_546C0EDE589D.srj'
    
    import_file = import_path + srj_file
    export_file = export_path + srj_file
    
    ##result = document_process_(import_path, export_path, verbose=True, version='new', resample=False, export='file')
    result = file_process(import_file, export_file, verbose=True, version='new', resample=False, export='file')
    