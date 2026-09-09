from ..motion import ahrs
from ..motion.ahrs.filters.madgwick import Madgwick
from ..motion.ahrs.common.orientation import acc2q, q2R
import numpy as np

def static_motion_analysis(motion_data,sample_rate):
        """
        Determine static or dynamic of motion data.
        
        Calculate the stantard deviation of the linear acceleration in global coordinate within 2 seconds.
        Set a threshold of stantard deviation to determine static or dynamic.
        
        inpuyt ---
            motion_data: 2D list of raw motion data
            sample_rate: int or float(Sample rate of motion data)

        output ---
              1-D list of flags 
              0: dynamic state
              1: static state
        """

        # 將Motion raw data 轉成 numpy array
        sample_num = len(motion_data)
        motion_data = np.array(motion_data)

        # 將Motion raw data 轉成對應的數值(dps, g, ...)
        F = np.concatenate((np.ones(3)/114.28, np.ones(3)*0.000061, np.ones(3)*1.5, 1), axis = None)
        Values = motion_data * F

        # 擷取ACC and GYR 
        ACC = Values[:,3:6]
        GYR = Values[:,0:3]
        
        # 計算 global & body 座標系的轉換矩陣: C
        orientation = ahrs.filters.Madgwick(acc=ACC*9.80665, gyr=GYR*np.pi/180, frequency=sample_rate)
        C = ahrs.common.orientation.q2R(orientation.Q)
        
        # 將 ACC 轉至 global 坐標系, 扣除 gravity 後得到空間中實際的加速度
        linACC = np.zeros(ACC.shape)
        for i in range(0, sample_num):
            ACCglob = np.dot(ACC[i], C[i].T)
            linACC[i] = ACCglob - [0,0,1] # notice: C in matlab is the transpose of C in here
        linACC = linACC * 9.81 # g to m/s^2
        
        # 計算實際加速度 2 秒內的標準差
        STD = np.zeros(linACC.shape)
        w_size = 2 * sample_rate
        for i in range(0, sample_num):
            if i < w_size:
                window = linACC[:i+1]
            else:
                window = linACC[i-w_size+1:i+1]

            if window.shape[0] > 1:
                STD[i] = np.std(window, axis=0)
                
        # 設定標準差的閥值，以定義動態或靜態
        movingTH = 0.5
        static_label = np.ones(sample_num,dtype=int)
        for i in range(w_size, sample_num):
            s = STD[i]

            # 三軸中任一軸的標準差大於閥值就視為動態
            flag = np.sum(s > movingTH)
            if flag == 0:
                static_label[i] = 0 ##1 # Static
            else:
                static_label[i] = 1 ##0 # Dynamic

    
        return static_label,STD  
