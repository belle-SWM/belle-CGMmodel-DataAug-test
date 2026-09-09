from ..ecg.rpeak import rpeak_detection
from ..ecg.quality_check import *
from ..ecg.noise_remove import *


def _rri_filter(rri, method='mean', rri_outlier=(272, 2100), rri_thh=(0.55, 0.9)):    

    ## remove extreme outlier
    new_rri = [x for x in rri if rri_outlier[1] >= x >= rri_outlier[0]]
    reference_value = 0
    if method == 'mean':
        reference_value = np.mean(new_rri)
    elif method == 'median':
        reference_value = np.median(new_rri)

    lower = reference_value - rri_thh[0] * reference_value
    upper = reference_value + rri_thh[1] * reference_value  
    output = rri.copy()

    for i in range(len(rri)):
        if lower < rri[i] < upper and rri_outlier[0] <= rri[i] <= rri_outlier[1]:
            output[i] = rri[i]
        else:
            output[i] = -1 * rri[i]  # minus the filtered rri

    output2 = [y for y in new_rri if lower < y < upper]  # Delete the filtered RRI

    return output, output2

def irregular_detection(sig,idx=-1): ##主程式
    
    """
    input ---
        sig: ECG訊號
        idx: 在ECG資料結構中第幾個index, 可不輸入此參數直接只提供ECG訊號
    
    output ---
        result_dict: 定義的資料結構 result_dict 
                    = { 
                        'Index':輸入的idx,
                        'scale':訊號放大倍數,
                        'ResultFlag':心律不整嚴重程度(0為正常無心律不整，1為輕微，2為嚴重),
                        'STD':0,
                        'minHR':最小心率,
                        'maxHR':最大心率,
                        'avgHR':平均心率,
                        'Ridx':R波 index,
                        'score':訊號品質檢查後分數,
                        'location':心律不整heart beat位置
                      }  
    """

    ## Parameters
    thr1 = 0.2    ##偵測輕微心律不整的情況
    thr2 = 0.35   ##偵測嚴重心律不整的情況
    thr3 = 0.13   ##偵測R波多打的情況

    ## rescale signal    
    scale = 1 
    max_sig=max(sig)  
    if 300 >= max_sig > 150:
        scale = 2
    elif 150 >= max_sig > 100:
        scale = 4
    elif 100 >= max_sig > 50:
        scale = 6
    elif 50 >= max_sig:        
        scale = 8
    else:
        scale = 1

    ecg = sig * scale

    ## detect R peaks
    rpeaks = rpeak_detection(ecg) ##使用vg方法

    result_flag = 0
    location1 = []
    location2 = []
    location = []
    ridx = []
    score = 0
    result_dict = {}    
    
    if len(rpeaks) > 1:
        
        ridx = rpeaks[1:] 
        ridx=np.array(ridx,dtype=np.int32)  
        if len(ridx) >= 2:  
            
            ## calculate confidence score
            score0 = pattern_clustering(ecg, ridx, th=0.9)
            score1 = area_ratio(ecg, ridx)        
            score = score0 * score1
            ##bat_score_impulse = int(np.nanmean(wavelet_detect_noise(ecg, ridx, fs=250)) * 100)      
            if(score>=60): ##and bat_score_impulse>=60):                
               
                RRIArray_ori = np.diff(ridx)*1000/250 ## calculate RRI
                RRIArray_ori.astype('int32')               
                RRIArray_MinusSign,RRIArray = _rri_filter(RRIArray_ori)  
                       
                if len(RRIArray) >= 2:
                    nowMaxRRI = max(RRIArray)
                    nowMinRRI = min(RRIArray)
                    medianRRI = np.median(RRIArray)                   
                    meanRRI = np.mean(RRIArray)

                    ## detect irreqular heart beats                
                    count1, count2 = 0, 0
                    FlagArray=np.zeros([1,len(RRIArray_ori)],dtype=int)
                    for k in range(1,len(RRIArray_ori)):
                        sum_RRI = RRIArray_ori[k]+RRIArray_ori[k-1]
                        if (RRIArray_ori[k]>=meanRRI*1.6 or RRIArray_ori[k-1]>=meanRRI*1.6): ##較長拍漏封包的情況
                            [score1,index1] = packet_loss_check(ecg,[ridx[k-1],ridx[k]])
                            [score2,index2] = packet_loss_check(ecg,[ridx[k],ridx[k+1]])
                            if(score1==0 or score2==0):
                                FlagArray[0,k]=-1
                                FlagArray[0,k-1]=-1
                               
                    
                        if ((abs(sum_RRI - medianRRI) < medianRRI*thr3) or (RRIArray_ori[k]<272 or RRIArray_ori[k-1]<272)): ##R波多打的情況                            
                            FlagArray[0,k]=-1
                            FlagArray[0,k-1]=-1
                           
                
                    ConsideredCount=0   ###統計累計可分析Beat數，太少的也會濾除     
                    for k in range(1,len(RRIArray_MinusSign)):
                        if(FlagArray[0,k]!=-1 and FlagArray[0,k-1]!=-1 and RRIArray_MinusSign[k]>0 and RRIArray_MinusSign[k-1]>0):
                            ConsideredCount=ConsideredCount+1
                            delta_RRI = (RRIArray_MinusSign[k]-RRIArray_MinusSign[k-1])/min(RRIArray_MinusSign[k],RRIArray_MinusSign[k-1])                            
                            noiseflag=0                            
                            if abs(delta_RRI) >= thr1:                                
                                noiseflagArray = check_high_freq_noise(ecg,np.array([ridx[k-1],ridx[k],ridx[k+1]])) ###RRI對應到R peak index需要+1
                                if(noiseflagArray[0]==1 or noiseflagArray[1]==1 or noiseflagArray[2]==1):
                                    noiseflag=1
                                    break          
                               
                                if(noiseflag==0):
                                    count1 += 1   
                                    location1.append(int(ridx[k-1])) 

                            if abs(delta_RRI) >= thr2:  
                                if(noiseflag==0):
                                    count2 += 1
                                    location2.append(int(ridx[k-1]))
                  
                    if(ConsideredCount<=2):
                        result_flag = -1 
                    else:
                        if count1 >= 2:
                            result_flag = 1   
                            location=location1             
                        if count2 >= 2:
                            result_flag = 2  
                            location=location2
                        
                       
                                                    
                    result_dict = {'Index':idx,
                        'scale':scale,
                        'ResultFlag':result_flag,
                        'STD':0,
                        'minHR':60000/nowMaxRRI,
                        'maxHR':60000/nowMinRRI,
                        'avgHR':60000/meanRRI,
                        'Ridx':ridx,
                        'score':score,
                        'location':location
                        }
                else:
                    result_flag=-1
            else:
                result_flag=-1
        else:
            result_flag=-1
    else:
        result_flag=-1


    return result_dict      
    
              
