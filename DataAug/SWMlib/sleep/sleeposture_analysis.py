
import datetime
import numpy as np
import pandas as pd

from ..motion import ahrs
from ..motion.ahrs.filters.complementary import ComplementaryQ
from ..motion.ahrs.common.orientation import acc2q
from scipy import stats


def DeterminePosture(ACCref):
    SampleNum = len(ACCref)
    Label = []
    Label_list = ['Upright','Right','Prone','Left','Supine']
    Angle = np.zeros((SampleNum, 4))
    
    for i in range(0, SampleNum):
        Ax, Ay, Az = ACCref[i]

        pitch = np.arctan2(Az, Ay) * 180 / np.pi
        roll = np.arctan2(Ax, Ay) * 180 / np.pi
        yaw = np.arctan2(Ax, Az) * 180 / np.pi
        Angle[i] = [Ay, pitch, roll, yaw]
        
        if Ay > 0.7:
            Label.append(1) #Upright
        else:
            if abs(yaw) <= 30:
                Label.append(5) #Supine
            elif abs(yaw) >= 135:
                Label.append(3) #Prone
            else:
                if yaw < 0:
                    Label.append(4) #Left
                else:
                    Label.append(2) #Right
      
    return Angle, np.array(Label), Label_list

def RotationCorrection(acc, ref):
    acc = acc / np.linalg.norm(acc)
    N = np.cross(ref, acc)
    N = N / np.linalg.norm(N)
    theta = np.arccos(np.dot(ref, acc.T))

    comb = [[1,1], [-1,1], [1,-1], [-1,-1]]
    R = np.zeros((len(comb),3,3))
    Error = np.ones((len(comb),1))

    for i in range(0, len(comb)):
        sign_n = comb[i][0]
        Nsign = sign_n * N
        Nx = Nsign[0]
        Ny = Nsign[1]
        Nz = Nsign[2]

        sign_theta = comb[i][1]
        Tsign = sign_theta * theta

        R[i] = np.identity(3) * np.cos(Tsign) \
        + np.sin(Tsign) * np.array([[0,-Nz,Ny],[Nz,0,-Nx],[-Ny,Nx,0]]) \
        + (1-np.cos(Tsign)) * np.outer(Nsign.T, Nsign)

        dif = np.dot(R[i],acc.T) - ref
        Error[i] = np.linalg.norm(dif)

    Imin = np.argmin(Error)
    RM = R[Imin]
    return RM

def TurnCapability(Angle, moveLabel, SampleRate):
    SampleNum = len(Angle)
    Ay = Angle[:,0]
    Yaw = Angle[:,3]
    Yaw[Ay>0.7] = np.nan

    # 偵測每次動作的起訖
    dif_moveLabel = np.zeros(len(moveLabel))
    dif_moveLabel[1:] = np.diff(moveLabel)
    dif_moveLabel[Ay>0.7] = 0

    startFlag, endFlag = 0, 0
    startIdx, endIdx = 0, 0
    Blocks = []
    for i in range(0, len(dif_moveLabel)):
        value = dif_moveLabel[i]
        if value == 0:
            continue
        # 動作開始
        if value == 1:
            startFlag = 1
            startIdx = i
        # 動作結束
        if startFlag and value == -1:
            endFlag = 1
            endIdx = i
        # 紀錄起訖
        if endFlag:
            startFlag = 0
            endFlag = 0
            Blocks.append([startIdx, endIdx])

    failTurnIdx = np.zeros(SampleNum)
    for i in range(len(Blocks)):
        block = range(Blocks[i][0]-5*SampleRate, Blocks[i][1]+5*SampleRate)
        if block.start < 0 or block.stop >= SampleNum:
            continue

        Y = Yaw[block]
        if sum(np.isnan(Y)) > 0:
            continue
        
        # 前後的角度變化
        D1 = abs(Y[0] - Y[-1])
        # 最大的角度變化
        D2 = abs(max(Y) - min(Y))
        if D2 > 300: #避免Yaw的正負180切換
            continue
        
        # 前後角度變化遠小於過程中最大的角度變化-->翻身失敗
        if 0.5*D2 > D1 and D2 > 60:
            failTurnIdx[int(np.median(block))] = 1
            
    return failTurnIdx

def SummerizePosture(Label,moveLabel,failTurnIdx,offline,startdt,SampleRate):
    SampleNum = len(Label)
    if SampleNum <= 0:
        return

    # 輸出時間間隔
    timedelta = 1 * 60 * SampleRate
    td = datetime.timedelta(minutes=1)

    # Resample offline (250Hz->SampleRate)
    #idx = np.arange(0,len(offline),250/SampleRate,dtype=int)
    new_offline = np.zeros(len(Label))
    #new_offline[0:len(idx)] = offline[idx]

    TimeStamps = []
    Dates, Hours, Mins = [], [], []
    Label_mins = []
    currentdt = startdt
    turnoverFlag = []
    movingFlag = []
    failTurnFlag = []
    offlineFlag = []
    for i in range(0, SampleNum, timedelta):
        TimeStamps.append(currentdt)

        if i + timedelta < SampleNum:
            idx = range(i,i+timedelta)
        else:
            idx = range(i,SampleNum)

        # 統計該分鐘最多的姿態
        w_min = Label[idx]
        M = stats.mode(w_min)
        label_min = int(M[0])

        # 檢查該分鐘是否有翻身&移動
        w_stat = moveLabel[idx]
        turn_flag = 0
        move_flag = 0
        if len(Label_mins) >= 1:
            if label_min > 1 and Label_mins[-1] > 1:
                if Label_mins[-1] != label_min:
                    turn_flag = 1
                if sum(w_stat) > 0:
                    move_flag = 1

        # 檢查該分鐘是否有翻身失敗
        fail_min = failTurnIdx[idx]
        fail_flag = 0
        if sum(fail_min) > 0:
            fail_flag = 1

        # 檢查該分鐘是否有離線
        off_min = new_offline[idx]
        off_flag = 0
        if sum(off_min) > 0:
            off_flag = 1
            label_min = np.nan
            turn_flag = np.nan
            move_flag = np.nan
            fail_flag = np.nan

        Label_mins.append(label_min)
        movingFlag.append(move_flag)
        turnoverFlag.append(turn_flag)
        failTurnFlag.append(fail_flag)
        offlineFlag.append(off_flag)

        # 分離: 日期/時/分
        Dates.append(currentdt.strftime('%Y/%m/%d'))
        Hours.append(currentdt.hour)
        Mins.append(currentdt.minute)

        currentdt += td
    
    # 輸出表格
    State = pd.DataFrame({
        'DT':TimeStamps,
        "Date":Dates,
        "H":Hours,
        "M":Mins,
        "State":Label_mins,
        "Roll Over":turnoverFlag,
        "Move":movingFlag,
        "Fail Turn":failTurnFlag,
        "Offline":offlineFlag
        })

    ## 統計每種姿態所佔的時間
    Labelcount = np.zeros(6)
    count = np.bincount(Label_mins)
    Labelcount[:len(count)] = count
    bins = np.arange(0,len(Labelcount))

    Proportion = pd.DataFrame({
        "Posture":['Right','Prone','Left','Supine'],
        "Minutes":Labelcount[2:]
    })

    ## 輸出每小時翻身與移動次數
    date_list, hour_list = [], []
    for dt in TimeStamps:
        if dt.hour not in hour_list:
            date_list.append(dt.strftime('%Y/%m/%d'))
            hour_list.append(dt.hour)
    hourNum = len(hour_list)

    Hours = np.array(Hours, dtype=int)
    moveCount = np.zeros(hourNum)
    turnCount = np.zeros(hourNum)
    failCount = np.zeros(hourNum)
    movingFlag = np.array(movingFlag)
    turnoverFlag = np.array(turnoverFlag)
    failTurnFlag = np.array(failTurnFlag)
    for i in range(0,hourNum):
        idx = np.argwhere(Hours==hour_list[i]).flatten()
        moveWindow = movingFlag[idx]
        moveCount[i] = np.nansum(moveWindow)

        turnWindow = turnoverFlag[idx]
        turnCount[i] = np.nansum(turnWindow)

        failWindow = failTurnFlag[idx]
        failCount[i] = np.nansum(failWindow)

    Analyze = pd.DataFrame({
        "Date":date_list,
        "H":hour_list,
        "Roll Over Times":turnCount,
        "Moving Times":moveCount,
        "Failed Turn Over Times":failCount  
    })

    return State, Analyze, Proportion
    
def PostureDetection(Motions,initialTime,offline,SampleRate,iniACC):

    ##### Preprocessing #####
    SampleNum = len(Motions)

    # 將Motion raw data 轉成對應的數值(dps, g, ...)
    F = np.concatenate((np.ones(3)/114.28,np.ones(3)*0.000061,np.ones(3)*1.5, 1),axis=None)
    Values = Motions * F

    # 擷取ACC and GYR 
    GYR = Values[:,0:3] * np.pi/180
    ACC = Values[:,3:6] * 9.80665
    
    ##### Determine Static or Dynamic ? #####
    # 計算 global & body 座標系的轉換矩陣: C
    # orientation = ComplementaryQ(acc=ACC,
    #                              gyr=GYR,
    #                              frequency=SampleRate,
    #                              gain=0.06)
    # C = ahrs.common.orientation.q2R(orientation.Q) 
    
    FUSE = ComplementaryQ(frequency=SampleRate,gain=0.06)
    Q = np.zeros((SampleNum,4))
    Q[0] = acc2q(ACC[0])
    for i in range(1,SampleNum):
        if Motions[i] == [0,0,0,0,0,0,0,0,0,0]:
            Q[i] = Q[i-1]
        else:
            Q[i] = FUSE.updateIMU(Q[i-1],gyr=GYR[i],acc=ACC[i])
    C = ahrs.common.orientation.q2R(Q)

    
    ##### Determine Body Posture #####
    ## 計算 body 坐標系至 reference 坐標系的旋轉矩陣 RM
    # 將 global 的 gravity[0 0 1] 旋轉至 body
    ACCbody = np.zeros(ACC.shape)
    for i in range(0, SampleNum):
        ACCbody[i] = np.dot([0,0,1], C[i])
    
    ref = [0,1,0]
    RM = RotationCorrection(iniACC, ref)
    ACCref = np.dot(ACCbody, RM.T)   
    Angle, Label, Label_list = DeterminePosture(ACCref)
    ##### Output Results #####
    moveLabel = np.zeros(SampleNum)
    failTurnIdx = np.zeros(SampleNum)
    State, Analyze, Proportion = SummerizePosture(Label,moveLabel,failTurnIdx,
                                                  offline,initialTime,SampleRate)

    return State, Analyze, Proportion
    