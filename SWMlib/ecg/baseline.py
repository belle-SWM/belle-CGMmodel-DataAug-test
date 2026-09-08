import numpy as np
from scipy.ndimage import median_filter
import pandas as pd
import math
from multiprocessing import Process,Lock
import multiprocessing
from ..ecg import noise_remove

def _baseline_remove_sp(ecg):
    ecg = np.array(ecg, dtype='int32')
    # Remove Baseline
    baseline = median_filter(ecg, size=int(0.2*250), mode='nearest')
    baseline = median_filter(baseline, size=int(0.6*250), mode='nearest')
    ecg_filt = ecg - baseline

    # Remove Pulse
    Diff = np.diff(ecg_filt)
    pulseIdx = np.argwhere(abs(Diff) > 1000).flatten()
    if len(pulseIdx) > 0:
        for idx in pulseIdx:
            ecg_filt[idx+1] = ecg_filt[idx]

    return ecg_filt

###-----multiprocessing method------
def _baseline_removal_mp(ConcateEcgArray, processnumber, pindex, ns, lock):
    datalength = len(ConcateEcgArray)
    block_length = math.ceil(datalength / processnumber)
    startindex = pindex * block_length - 250
    startflag = 0
    if (startindex < 0):
        startindex = 0
        startflag = 1

    endindex = (pindex + 1) * block_length + 250  ##多放寬1秒，避免銜接處artifact
    endflag = 0
    if (endindex > datalength):
        endindex = datalength
        endflag = 1

    lock.acquire()
    nowbaseline = _baseline_remove_sp(ConcateEcgArray[startindex:endindex])
    lock.release()

    if (startflag == 0 and endflag == 0):  ###中間段落
        newitem = pd.DataFrame([{"pindex": pindex, "baseline": nowbaseline[250:-250]}])
        ns.baseline_ecg_structure = pd.concat([newitem,(ns.baseline_ecg_structure)]).reset_index(drop=True)  ####ECG基線拉直


    elif (startflag == 1):  ###第一個段落
        newitem = pd.DataFrame([{"pindex": pindex, "baseline": nowbaseline[0:-250]}])
        ns.baseline_ecg_structure = pd.concat([newitem,(ns.baseline_ecg_structure)]).reset_index(drop=True)  ####ECG基線拉直


    elif (endflag == 1):  ###最後一個段落
        newitem = pd.DataFrame([{"pindex": pindex, "baseline": nowbaseline[250:len(nowbaseline)]}])
        ns.baseline_ecg_structure = pd.concat([newitem,(ns.baseline_ecg_structure)]).reset_index(drop=True)  ####ECG基線拉直



def baseline_remove(concated_ecg_array, processnum=1): ###主程式
    
    """
    input ---
        concated_ecg_array: ECG raw data array
        processnumber: the number of cpu used to run this function, if not given, the default value is 1
    output ---
        debasedline_ecg: 1D numpy array which contain ECG signal after baseline draft removal
    """    
    
    if (processnum>1):
        lock = Lock()
        manager = multiprocessing.Manager()
        ns = manager.Namespace()
        baseline_ecg_structure = pd.DataFrame()
        ns.baseline_ecg_structure = baseline_ecg_structure
        processes = [
            Process(target=_baseline_removal_mp, args=(concated_ecg_array, processnum, pindex, ns, lock)) for pindex in range(processnum)]
        ## start all processes
        for process in processes:
            process.start()

        # wait for all processes to complete
        for process in processes:
            process.join()

        ns.baseline_ecg_structure = (ns.baseline_ecg_structure).sort_values(by="pindex", ascending=True)
        ns.baseline_ecg_structure = ns.baseline_ecg_structure.drop(columns="pindex")
        ns.baseline_ecg_structure = ns.baseline_ecg_structure.reset_index(drop=True)

        debasedline_ecg = np.array([])
        for i in range(processnum):
            temparray = np.array((ns.baseline_ecg_structure[i:i + 1].values.tolist())).flatten()
            debasedline_ecg = np.concatenate((debasedline_ecg, temparray))  ##因為長度不同，需要分別處理

        debasedline_ecg = debasedline_ecg.astype(int)

    else:  ##"singleprocess"
        debasedline_ecg = _baseline_remove_sp(concated_ecg_array)  ####ECG基線拉直

    debasedline_ecg = noise_remove.remove_impulse(debasedline_ecg)

    return debasedline_ecg  
