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

import random
from torch.utils.data.sampler import SubsetRandomSampler


if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))


from common import (
##    ECGDataset,
    FocalLoss,
    BalancedBatchSampler,
    initialize_weights,
    train_valid_split_indices,
    save_training_history,
)

  
from SWMlib.common.calculation import normalization


def get_version(): ###取得版本號

    return '007'


###──── Data Augmentation 開關 ──────────────────────────────────────────
###這次要單獨測試「頻域平滑幅度擾動」的效果，所以先把原本三種時域增強關掉。
###用開關而不是註解掉：之後要評估綜合方式時只要改這幾個值，不必再動 augment_signal。
AUG_GAUSSIAN_NOISE   = False   ##加高斯雜訊
AUG_AMPLITUDE_SCALE  = False   ##振幅隨機縮放 0.9~1.1 倍
AUG_TIME_SHIFT       = False   ##時間軸剛性平移(邊緣補值)
AUG_FREQ_MAGNITUDE   = True    ##頻域平滑幅度擾動(只動幅度包絡，相位不動)

###頻域擾動的參數
FREQ_AUG_AMPLITUDE   = 0.4     ##增益包絡的擾動幅度(±40%)
FREQ_AUG_CONTROL_PTS = 5       ##控制點數，越少包絡越平滑、越不易產生 ringing


class ECGDataset(Dataset):
    def __init__(self, dir_path, method='raw', classes=3, augment=False):
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

    # 對已完成特徵轉換的訊號做輕量資料增強(只用於training data)
    # 每一種增強由檔案開頭的 AUG_* 開關控制，方便單獨測試或之後評估組合
    def augment_signal(self, signal):
        if AUG_GAUSSIAN_NOISE and torch.rand(1).item() < 0.5:  ##加高斯雜訊
            noise_std = 0.02
            signal = signal + torch.randn_like(signal) * noise_std

        if AUG_AMPLITUDE_SCALE and torch.rand(1).item() < 0.5:  ##振幅隨機縮放0.9~1.1倍
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            signal = signal * scale

        if AUG_TIME_SHIFT and torch.rand(1).item() < 0.5:  ##時間軸小幅平移
            ###不用 torch.roll：roll 是循環的，被推出一端的取樣點會從另一端繞回來，
            ###在訊號中間接出一個真實 ECG 不會出現的跳階。改成邊緣補值(replicate)：
            ###平移後空出來的那幾點用最靠近的邊緣值填，另一端多出來的直接截掉。
            ###這裡 signal 是 (1, 150)，F.pad 的 replicate 模式需要 3D 輸入，所以直接用 cat。
            shift = int(torch.randint(-5, 6, (1,)).item())
            if shift != 0:
                pad = abs(shift)
                if shift > 0:   ###往右移，左端用第一點補、截掉尾端
                    signal = torch.cat([signal[..., :1].expand(*signal.shape[:-1], pad),
                                        signal[..., :-pad]], dim=-1)
                else:           ###往左移，右端用最後一點補、截掉開頭
                    signal = torch.cat([signal[..., pad:],
                                        signal[..., -1:].expand(*signal.shape[:-1], pad)], dim=-1)

        if AUG_FREQ_MAGNITUDE and torch.rand(1).item() < 0.5:  ##頻域平滑幅度擾動
            ###只擾動頻譜的「幅度包絡」，相位完全不動。相位攜帶時間資訊(T波位置、R-T間期)，
            ###而那正是判別血糖的依據(High 的 T 波比 Normal 提前 8~32ms，10/10 個 uuid 一致)，
            ###動了相位等於把標籤本身抹掉，跟 time warping 是同一類錯誤。
            ###包絡用少數控制點內插成平滑曲線：per-bin 亂數等於在時域跟隨機 kernel 卷積，
            ###會產生 ringing 把波峰位置抹糊，間接破壞時序(實測效率只有平滑包絡的 1/15)。
            ###注意這裡不能把頻譜命名為 F，F 在本檔開頭已經是 torch.nn.functional。
            sig_len = signal.shape[-1]
            spec = torch.fft.rfft(signal, dim=-1)
            ctrl = 1.0 + (torch.rand(FREQ_AUG_CONTROL_PTS, device=signal.device) - 0.5) * 2 * FREQ_AUG_AMPLITUDE
            gain = F.interpolate(ctrl.view(1, 1, -1), size=spec.shape[-1],
                                 mode='linear', align_corners=True).view(-1).clone()
            gain[0] = 1.0   ###DC 不動，避免整體基線飄移
            signal = torch.fft.irfft(spec * gain, n=sig_len, dim=-1)

        return signal

    # Parsing files in folder
    def read_files(self, foldername):
        Sig = []
        for filename in os.listdir(os.path.join(self.dir_path, foldername)):
            file_path = os.path.join(self.dir_path, foldername, filename)
            Sig.append(self.load_data(file_path))
        
        Sig = torch.FloatTensor(Sig)

        if Sig.ndimension() == 2:
            Sig = Sig.unsqueeze(1)

        ##if self.channel == 1:
        ##    Sig = Sig.unsqueeze(1)
        
        return Sig
    
    
    # Read data in files & Preprocessing
    def load_data(self, filepath):
        
        values = np.genfromtxt(filepath, delimiter = '')
        
        # put data into fixed length signal
        sig = np.zeros(self.data_len)
        if len(values)>=self.data_len:
            sig[:self.data_len] = values[:self.data_len]
        else:
            print("values",values)
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


class ECGDatasetSubset(Dataset):
    """
    包裝 ECGDataset 的子集合(依索引挑選)，讓 train/valid 可以共用同一份已預先載入、
    做完特徵轉換的資料，但各自獨立控制是否啟用 data augmentation
    (train 才需要 augment=True，valid/test 維持 augment=False，避免驗證/測試資料被污染)。
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


class CNN_Transformer(torch.nn.Module):

    def __init__(self, input_channel, data_len=150): 
        super(CNN_Transformer, self).__init__()  
        self.data_len = data_len
        self.input_channel = input_channel
        
        # ----------------- 第一階段 CNN (150 -> 50 點) -----------------
        self.conv1 = nn.Sequential(nn.Conv1d(self.input_channel, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64))                                    
        self.relu  = nn.LeakyReLU(inplace=True) 

        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(64))
        
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(64))  
                                           
        self.maxpool = nn.MaxPool1d(3) # 💡 修改：調整為 3
        
        # ----------------- 第二階段 CNN (50 -> 16 點) -----------------
        # 修正：第一階段 concat 後通道變 128
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1),      
                                   nn.BatchNorm1d(128))
        
        self.conv5 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1),      
                                   nn.BatchNorm1d(128))
        
        self.conv6 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1),      
                                   nn.BatchNorm1d(128))  
                                   
        # ----------------- 第三階段 CNN (16 -> 5 點) -----------------
        # 修正：第二階段 concat 後通道變 256
        self.conv7 = nn.Sequential(nn.Conv1d(256, 32, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(32))
        
        self.conv8 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(32))
        
        self.conv9 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(32)) 
         
        # ----------------- Transformer 架構層 -----------------
        # 修正：conv9 輸出 32，經第三階段 concat 後變成 64 通道
        self.d_model = 64 
        # 修正：150 點經 3 次 MaxPool(3) 後，剩餘長度為 5 點
        self.reduced_seq_len = 5  
        
        # 舊版本 Transformer 預設形狀是 [Seq_Len, Batch, Feature] -> [5, 1, 64]
        self.pos_embedding = nn.Parameter(torch.randn(self.reduced_seq_len, 1, self.d_model))
        
        # 拿掉 batch_first=True 以相容舊版本 PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4,              # 4頭注意力機制 (64 可被 4 整除)
            dim_feedforward=128,   
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # ----------------- 全連接分類層 -----------------
        # 修正：進入全連接層的總特徵數更新為 64 通道 * 5 個時間點 = 320
        self.fc_input_dim = self.d_model * self.reduced_seq_len
        
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3) # 三分類輸出

    def forward(self, x):
        # 輸入 x 形狀: [Batch, Channel, 150]
        
        # 第一階段 CNN + 正負特徵提取
        x = self.conv1(x)
        x = self.relu(x) 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        rx = self.relu(x)
        mrx = self.maxpool(rx)
        y = self.relu(-x)
        mry = self.maxpool(y)    
        # 輸出形狀: [Batch, 128, 50]
        x = torch.cat([mrx, mry], dim=1) 
        
        # 第二階段 CNN
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x) 
        x = self.relu(x)
        x = self.conv6(x) 
        rx6 = self.relu(x)
        mrx6 = self.maxpool(rx6)
        y6 = self.relu(-x)
        mry6 = self.maxpool(y6)   
        # 輸出形狀: [Batch, 256, 16]
        x = torch.cat([mrx6, mry6], dim=1) 
        
        # 第三階段 CNN
        x = self.conv7(x) 
        x = self.conv8(x)
        x = self.conv9(x) 
        rx9 = self.relu(x)
        mrx9 = self.maxpool(rx9)
        y9 = self.relu(-x)
        mry9 = self.maxpool(y9)    
        # 最終 CNN 輸出形狀: [Batch, 64, 5]
        x = torch.cat([mrx9, mry9], dim=1) 

        # --- 舊版 Transformer 維度調整：[Batch, 64, 5] 轉換為 [5, Batch, 64] ---
        x = x.permute(2, 0, 1)
        
        # 2. 注入位置編碼
        x = x + self.pos_embedding
        
        # 3. 送入 Transformer 編碼器 (輸出形狀維持為 [5, Batch, 64])
        x = self.transformer_encoder(x)  
        
        # --- 轉回全連接層要用的形狀：[5, Batch, 64] 轉回 [Batch, 5, 64] ---
        x = x.permute(1, 0, 2)
        
        # 展平特徵 (Flatten) -> [Batch, 320]
        x = x.reshape(x.size(0), -1)    
        
        # 全連接分類層
        x = self.fc1(x)        
        x = self.fc2(x)
        x = self.fc3(x) 
        
        return x
            

    

class TwoClasses_Transformer(torch.nn.Module):

    def __init__(self, input_channel, data_len=150): 
        super(TwoClasses_Transformer, self).__init__()  
        self.data_len = data_len
        self.input_channel = input_channel      

        # ----------------- 第一階段 CNN (150 -> 50 點) -----------------
        self.conv1 = nn.Sequential(nn.Conv1d(self.input_channel, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64))
        
        self.relu  = nn.LeakyReLU(inplace=True)                                                 
        
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(64))
        
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(64))
        # 修改：改為 MaxPool1d(3)
        self.maxpool = nn.MaxPool1d(3)                                                 
        
        # ----------------- 第二階段 CNN (50 -> 16 點) -----------------
        # 修正：第一階段 concat 後通道變 128
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1),      
                                   nn.BatchNorm1d(128)) 
        
        self.conv5 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1),      
                                   nn.BatchNorm1d(128))
        
        self.conv6 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1),      
                                   nn.BatchNorm1d(128))
                                   
        # ----------------- 第三階段 CNN (16 -> 5 點) -----------------
        # 修正：第二階段 concat 後通道變 256
        self.conv7 = nn.Sequential(nn.Conv1d(256, 32, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(32))
        
        self.conv8 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(32))
        
        self.conv9 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding=1),       
                                   nn.BatchNorm1d(32)) 
        
        # ----------------- Transformer 架構層 -----------------
        # 修正：conv9 輸出 32 通道，經第三階段 concat 後變成 64 通道
        self.d_model = 64 
        # 修正：150 點經 3 次 MaxPool(3) 後，剩餘長度為 5 點
        self.reduced_seq_len = 5  
        
        # 舊版本 Transformer 預設形狀是 [Seq_Len, Batch, Feature] -> [5, 1, 64]
        self.pos_embedding = nn.Parameter(torch.randn(self.reduced_seq_len, 1, self.d_model))
        
        # 拿掉 batch_first=True 以相容舊版本 PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4,              
            dim_feedforward=128,   
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # ----------------- 全連接層 -----------------
        # 修正：進入全連接層的總特徵數更新為 64 通道 * 5 個時間點 = 320
        self.fc1 = nn.Linear(self.d_model * self.reduced_seq_len, 64)
        self.fc2 = nn.Linear(64, 1)        

    def forward(self, x):
        # 輸入 x 的形狀: [Batch, Channel, 150]
        
        # 第一階段 CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        rx = self.relu(x)
        mrx = self.maxpool(rx)
        y = self.relu(-x)
        mry = self.maxpool(y)    
        # 輸出形狀: [Batch, 128, 50]
        x = torch.cat([mrx, mry], dim=1)            
        
        # 第二階段 CNN
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x) 
        rx6 = self.relu(x)
        mrx6 = self.maxpool(rx6)
        y6 = self.relu(-x)
        mry6 = self.maxpool(y6)    
        # 輸出形狀: [Batch, 256, 16]
        x = torch.cat([mrx6, mry6], dim=1)

        # 第三階段 CNN
        x = self.conv7(x) 
        x = self.conv8(x)
        x = self.conv9(x)        
        rx9 = self.relu(x)
        mrx9 = self.maxpool(rx9)
        y9 = self.relu(-x)
        mry9 = self.maxpool(y9)    
        # 最終 CNN 輸出形狀: [Batch, 64, 5]
        x = torch.cat([mrx9, mry9], dim=1)
        
        # --- 舊版 Transformer 維度調整：[Batch, 64, 5] 轉換為 [5, Batch, 64] ---
        x = x.permute(2, 0, 1)
        
        # 2. 注入位置編碼
        x = x + self.pos_embedding
        
        # 3. 送入 Transformer 計算 (輸出形狀依然是 [5, Batch, 64])
        x = self.transformer_encoder(x)  
        
        # --- 轉回全連接層要用的形狀：[5, Batch, 64] 轉回 [Batch, 5, 64] ---
        x = x.permute(1, 0, 2)
        
        # 展平特徵向量 (Flatten) -> [Batch, 320]
        x = x.reshape(x.size(0), -1)    
        
        # 全連接層分類
        x = self.fc1(x)        
        x = self.fc2(x)
            
        return x    
                  
'''
def initialize_weights(model):
               
        
        for m in model.modules():
            
            if isinstance(m, nn.Conv1d):  # 判断是否為Conv1d
                ##torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)                
              
            if isinstance(m, nn.Linear):               
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None: ## 加上偏差初始化
                    torch.nn.init.zeros_(m.bias.data)  
           
            elif isinstance(m, nn.BatchNorm1d):              
                m.weight.data.fill_(1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
'''        

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


'''
class FocalLoss(nn.Module):
    """
    自動支援：
    1. Binary classification: logits shape [N] or [N,1]
    2. Multi-class classification: logits shape [N,C], C >= 2

    參數：
    - alpha:
        * binary: 可給 float，例如 0.25
        * multi-class: 可給 tensor/list，例如 [1.0, 2.0, 2.0]
    - gamma: focal loss 的 focusing parameter
    - reduction: "mean" / "sum" / "none"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:
            - binary: [N] or [N,1]
            - multi-class: [N,C]

        targets:
            - binary: [N] or [N,1], values in {0,1}
            - multi-class: [N], values in {0,1,...,C-1}
        """
        # -------------------------
        # Case 1: Binary classification
        # logits.shape = [N] or [N,1]
        # -------------------------
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
            logits = logits.view(-1)
            targets = targets.float().view(-1)

            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            pt = torch.where(targets == 1, probs, 1 - probs)

            if self.alpha is None:
                alpha_t = 1.0
            else:
                # binary alpha: 建議給 float，例如 0.25
                alpha = float(self.alpha)
                alpha_t = torch.where(
                    targets == 1,
                    torch.full_like(targets, alpha),
                    torch.full_like(targets, 1 - alpha)
                )

            fl = alpha_t * ((1 - pt) ** self.gamma) * bce

        # -------------------------
        # Case 2: Multi-class classification
        # logits.shape = [N,C], C>=2
        # -------------------------
        elif logits.ndim == 2 and logits.size(1) >= 2:
            targets = targets.view(-1).long()

            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            fl = ((1 - pt) ** self.gamma) * ce

            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    alpha = torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device)
                elif isinstance(self.alpha, torch.Tensor):
                    alpha = self.alpha.to(logits.device, dtype=logits.dtype)
                else:
                    raise TypeError("For multi-class, alpha must be list, tuple, or torch.Tensor.")

                fl = alpha[targets] * fl

        else:
            raise ValueError(
                f"Unsupported logits shape: {logits.shape}. "
                "Expected [N], [N,1], or [N,C]."
            )

        # -------------------------
        # Reduction
        # -------------------------
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        elif self.reduction == "none":
            return fl
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class BalancedBatchSampler:
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        self.n_samples_per_class = batch_size // self.n_classes
        self.batch_size = self.n_samples_per_class * self.n_classes
        
        self.class_indices = {
            cls: np.where(self.labels == cls)[0].tolist()
            for cls in self.classes
        }
        
        self.n_batches = len(labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for cls, indices in self.class_indices.items():
                sampled = np.random.choice(indices, self.n_samples_per_class, replace=True)
                batch.extend(sampled)
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
            
def train_valid_split_indices(labels_array, valid_ratio=0.1, seed=10):
    """
    依照類別分層切分 train / valid，避免 valid 幾乎沒有 minority class
    labels_array: 1D numpy array
    """
    rng = np.random.RandomState(seed)

    train_idx_all = []
    val_idx_all = []

    unique_classes = np.unique(labels_array)
    for cls in unique_classes:
        cls_idx = np.where(labels_array == cls)[0]
        rng.shuffle(cls_idx)

        n_val = int(len(cls_idx) * valid_ratio)
        # 至少保留 1 筆到 validation（若該類別本來有資料）
        if len(cls_idx) > 1:
            n_val = max(1, n_val)
        else:
            n_val = 0

        val_idx_all.extend(cls_idx[:n_val].tolist())
        train_idx_all.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx_all)
    rng.shuffle(val_idx_all)

    return train_idx_all, val_idx_all
'''

def BuildModel(uuid,base_path,srj_db_path,glucosedata_path,server_db_path,processnum=8,splitting_ratio="70_30",model_subdir="",augment=False): ##測試用

    errorcode="0"
    message=""
    status=-1


    checkedpath=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid,"Train","Low")
    filelist_low_train=os.listdir(checkedpath)

    checkedpath=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid,"Test","Low")
    filelist_low_test=os.listdir(checkedpath)

    checkedpath=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid,"Train","High")
    filelist_high_train=os.listdir(checkedpath)

    checkedpath=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid,"Test","High")
    filelist_high_test=os.listdir(checkedpath)

    checkedpath=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid,"Train","Normal")
    filelist_normal_train=os.listdir(checkedpath)

    checkedpath=os.path.join(base_path,splitting_ratio,"GlucoseData",uuid,"Test","Normal")
    filelist_normal_test=os.listdir(checkedpath)


    if(len(filelist_low_train)>0 and len(filelist_low_test)>0 and len(filelist_high_train)>0 and len(filelist_high_test)>0 and len(filelist_normal_train)>0 and len(filelist_normal_test)>0):  ###有中，低和高血糖資料
        status, errorcode, message=BuildModel_ThreeClasses(uuid,base_path,splitting_ratio,model_subdir,augment)
        if(int(errorcode)>=0):
            message="Category model with three classes has been built!"

    elif(len(filelist_high_train)==0 or len(filelist_high_test)==0 or len(filelist_normal_train)==0 or len(filelist_normal_test)==0):
        errorcode="-402"
        message="An error occurs in the BuildModel function of M5.py: No enough normal or high glucose data!"
        status=-1

    else:
      status, errorcode, message=BuildModel_TwoClasses(uuid,base_path,splitting_ratio,model_subdir,augment)
      if(int(errorcode)>=0):
        message="Category model with two classes has been built!"


    print('message:',message)

    return status, errorcode, message



def BuildModel_ThreeClasses(uuid,basepath,splitting_ratio="",model_subdir="",augment=False):

    errorcode="0"
    message=""
    status=-1

    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間
    code_version=get_version()   ######取得模型訓練時當前程式版本

    print('Start building the model with three classes!')
    model_output_folder = os.path.join(basepath, splitting_ratio,model_subdir,"Best_ThreeClasses_Model", uuid)
    if not os.path.exists(model_output_folder): 
        os.makedirs(model_output_folder)
    
    model_data_path=os.path.join(basepath, splitting_ratio,model_subdir,"Temp_ThreeClasses_Model",uuid)  ###先清空過去訓練的模型佔存資料
    if os.path.isdir(model_data_path):
        shutil.rmtree(model_data_path)    

    model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir, "Best_ThreeClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")   
    if not os.path.exists(model_current_best_performance_txtfile): ###沒有檔案，先創造檔案
        with open(model_current_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0\n" )
            file.write("Specificity:0\n" )
            file.write("-----------------------")
            file.write("High_Sensitivity:0\n" )
            file.write("Low_Sensitivity:0\n")
            file.write("Normal_Sensitivity:0\n")
            file.write("High_Specificity:0\n")
            file.write("Low_Specificity:0\n")
            file.write("Normal_Specificity:0\n")
            file.write("-----------------------\n")
            file.write("F1:0\n")
            file.write("Acc:0\n")
            file.write("TP_Normal:%d, FP_Normal:%d, FN_Normal:%d, TN_Normal:%d\n" %(0, 0, 0, 0))
            file.write("TP_High:%d, FP_High:%d, FN_High:%d, TN_High:%d\n" %(0, 0, 0, 0))
            file.write("TP_Low:%d, FP_Low:%d, FN_Low:%d, TN_Low:%d\n" %(0, 0, 0, 0))  
        
    ### Create dataset object  
    Method = 'time' ##'combine'
    current_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,'Train')
    filelist_low=os.listdir(os.path.join(current_path,'Low'))  
    filelist_high=os.listdir(os.path.join(current_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_path,'Normal'))    

    if(len(filelist_low)==0 or len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-400"
        message="An error occurs in the BuildModel_ThreeClasses function of M5.py: No training data"      
        status=-1
        return status, errorcode, message 
    
    dataset = ECGDataset(dir_path = current_path, method = Method)
    
    Epoch_range = range(0, 300)
    patience =30
    scheduler_patience = 15  ###要小於early-stopping的patience，否則LR還沒衰減就先被early stop停掉
    Batch_size = 32
    valid_split = 0.1
    shuffle_flag = True
    random_seed = 10

    ##-----Balanced Batch------
    ## --- 1.設定資料根目錄------
    glucosedata_path = os.path.join(basepath, splitting_ratio, "GlucoseData", uuid, 'Train')
    data_dir = Path(glucosedata_path)
    classes = ['Normal', 'High']  ##, 'Low']


    ## --- 2. 遍歷資料夾取得路徑與標籤 ---
    all_file_paths = []
    target_list = []
    for idx, cls_name in enumerate(classes):
        cls_folder = data_dir / cls_name
        files = list(cls_folder.glob('*.txt'))
        print(f"類別 {cls_name} (Index {idx}) 共有 {len(files)} 個檔案")
        all_file_paths.extend(files)
        target_list.extend([idx] * len(files))

    target_list = torch.tensor(target_list)

    ## --- 3. 進行分層 Train / Valid Split ---
    labels_np_all = dataset.Labels.numpy().flatten().astype(int)   
    train_indices, val_indices = train_valid_split_indices(labels_array=labels_np_all, valid_ratio=valid_split, seed=random_seed)

    # --- 4. 建立 BalancedBatchSampler ---
    train_labels = labels_np_all[train_indices]
    train_sampler = BalancedBatchSampler(labels=train_labels, batch_size=Batch_size)

    # --- 5. 建立 Dataset 與 DataLoader ---
    # 訓練集：使用 batch_sampler；只對training data做data augmentation
    train_subset = ECGDatasetSubset(dataset, train_indices, augment=augment)
    train_loader = DataLoader(train_subset, batch_sampler=train_sampler)

    # 驗證集：保持原始分布，不做augmentation
    val_subset = ECGDatasetSubset(dataset, val_indices, augment=False)
    valid_loader = DataLoader(val_subset, batch_size=Batch_size, shuffle=False)

    total_train_num =  len(train_loader)
    total_valid_num = len(valid_loader)

    print("total_train_num",total_train_num)
    print("total_valid_num",total_valid_num)
    print("data augmentation:",augment)
    if(total_train_num<=1 or total_valid_num<=1):
        errorcode="-500"
        message="An error occurs in the BuildModel_ThreeClasses function of M5.py: No training data"
        status=-1
        return  status, errorcode, message


    current_test_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test")
    testdata = ECGDataset(dir_path = current_test_path, method = Method)        
    test_loader = DataLoader(testdata, batch_size=Batch_size)
          
    loopindex=0    
    current_best_sensitivity=0  ###本次訓練最佳sensitivity先設定為0
    current_best_specificity=0
    current_best_sensitivity_high=0  
    current_best_sensitivity_low=0
    current_best_sensitivity_normal=0  
    current_best_specificity_high=0
    current_best_specificity_low=0
    current_best_specificity_normal=0
    current_best_f1=0
    current_best_accs=0
    current_best_TP_normal=0
    current_best_FP_normal=0
    current_best_FN_normal=0
    current_best_TN_normal=0
    current_best_TP_high=0
    current_best_FP_high=0
    current_best_FN_high=0
    current_best_TN_high=0
    current_best_TP_low=0
    current_best_FP_low=0
    current_best_FN_low=0
    current_best_TN_low=0

    save_path = os.path.join(basepath,splitting_ratio,model_subdir, "Temp_ThreeClasses_Model",uuid) ###建立uuid專屬的模型存放資料夾  
    if os.path.isdir(save_path) is True:
        shutil.rmtree(save_path)
    os.makedirs(save_path)
     
    ''' 
    seed=42  ##為了減少訓練時的隨機性因素導致每次訓練績效差異很大
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #禁用 CUDNN 內部優化，以確保確定性（可能會輕微降低速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    '''

    while(loopindex<5):  ###執行5次訓練，取最好一次

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


        ###Loss_list/ACCs 移到 if 外面：續訓(Epoch_range[0] != 0)時原本不會被初始化，迴圈裡的 append 會 NameError
        Loss_list = []
        ACCs = []

        if Epoch_range[0] == 0:

            model=CNN_Transformer(data_len=dataset.data_len, input_channel=dataset.channel)
            initialize_weights(model)  ###模型權重初始化
            model = model.to(device)
            print('CNN transformer model created.')
            
            
        ##loss_function = torch.nn.CrossEntropyLoss().to(device)
        loss_function = FocalLoss(gamma=2.0).to(device)  ###三分類改用FocalLoss；如需針對類別加權可傳入 alpha=[w_Normal, w_High, w_Low]
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-3)
        ##optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=1e-3)
       
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=1,patience=scheduler_patience, cooldown=0, min_lr=0.00001)

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

                if data.size(0) <= 1: continue  ###避免batch_size=1時squeeze()把target壓成0維，造成 "target batch_size (0)" 錯誤

                # 梯度歸零
                optimizer.zero_grad()
                
                # Foward
                output = model(data)
              
        
                y_pred=torch.sigmoid(output)
                result=torch.argmax(y_pred, dim=1)  
                result=torch.unsqueeze(result, 1)
                target=target.view(-1,1)             
                correct += (result == target).sum().item() 
                
                total += target.size(0)  
                
                # Backward + Optimize     

                loss = loss_function(output, target.view(-1).long())  ###改用 view(-1) 取代 squeeze()，batch_size=1時仍保留維度；FocalLoss內部也會再做一次view(-1)
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

                if data.size(0) <= 1: continue  ###同樣避免batch_size=1造成squeeze()把target壓成0維

                # 預測
                output = model(data)       
                ##correct += (torch.sigmoid(output).ge(0.5) == target).sum().item()
                y_pred=torch.sigmoid(output)
                result=torch.argmax(y_pred, dim=1) 
                result=torch.unsqueeze(result, 1)
                target=target.view(-1,1)
                correct += (result == target).sum()
        
                total += target.size(0)
                
                # 損失函數      
                loss = loss_function(output, target.view(-1).long())
                valid_loss.append(loss.item())
        
                ##predict_list = result.squeeze().tolist() 
                ##target_list = target.squeeze().tolist()
                
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
                n_patience =0
                torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"Model_"+str(uuid)+"_"+str(epoch+1)+".pth"))

            if n_patience >= patience:
                break
            
        torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"Model_"+str(uuid)+"_"+str(epoch+1)+".pth"))
        
        ###把這次訓練每個 epoch 的 loss / accuracy 落地成 CSV + 曲線圖，跟 .pth 放在同一個資料夾
        save_training_history(save_path, uuid, Loss_list, ACCs,
                              tag='ThreeClasses', start_epoch=Epoch_range[0] + 1)

        ####-----------------Test Model----------------------       
        num_epoch = epoch_number      
        ##model = CNN(data_len=dataset.data_len, input_channel=1)
        
        model=CNN_Transformer(data_len=dataset.data_len, input_channel=dataset.channel)
        model.to(device)        
        model_path = os.path.join(basepath,splitting_ratio,model_subdir,"Temp_ThreeClasses_Model",uuid,"Model_"+str(uuid)+"_"+str(num_epoch)+".pth")
        
        if device.type == 'cuda':
            model.load_state_dict(torch.load(model_path))
        else:
            map_location=torch.device('cpu')
            model.load_state_dict(torch.load(model_path,map_location))     
        
        ##model.load_state_dict(torch.load(model_path))

        test_old_data = 0
        if test_old_data:           
            testdata = ECGDataset(dir_path = current_test_path, method = Method)
            test_loader = DataLoader(testdata, batch_size=Batch_size)
    
        model.eval()
        correct, total = 0, 0
        
        ##-----新方法使用 3x3 混淆矩陣儲存空間 cm[target][predict]-----
       
        class_indices = [0, 1, 2]
        cm = {t: {p: 0 for p in class_indices} for t in class_indices}

        for data, target in test_loader:
            data = data.to(device)

            ## 預測
            output = model(data)
            output = output.to('cpu')
                
            answer = torch.sigmoid(output)
            predict = torch.argmax(answer, dim=1) 
            target = target.squeeze()   
            total += target.size(0)

            # 轉為 list 方便迭代
            predict_list = predict.tolist()
            target_list = target.tolist()

            for p, t in zip(predict_list, target_list):
                # 紀錄混淆矩陣：實際為 t，預測為 p
                if t in class_indices and p in class_indices:
                    cm[t][p] += 1
                
                # 計算總正確數
                if p == t:
                    correct += 1

        # 計算總準確率
        accs = 100 * correct / total
        class_names = {0: 'Normal', 1: 'High', 2: 'Low'}

        current_sens = {}
        current_specs = {}

        # 取得總樣本數 (用於計算 TN)
        total_samples = sum(sum(preds.values()) for preds in cm.values())

        for j, name in class_names.items():
            # 1. 從 cm 提取 TP, FP, FN, TN
            tp = cm[j][j]
            
            # FN: 實際為 j 但預測為其他 (Row sum of j - TP)
            fn = sum(cm[j].values()) - tp
            
            # FP: 預測為 j 但實際為其他 (Column sum of j - TP)
            fp = sum(cm[actual][j] for actual in class_names.keys()) - tp
            
            # TN: 剩下的所有樣本 (Total - TP - FP - FN)
            tn = total_samples - (tp + fp + fn)

            # 2. 計算指標 (加上極小值防止除以零)
            s_en = (100 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            s_pe = (100 * tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            current_sens[j] = s_en + 0.000001
            current_specs[j] = s_pe + 0.000001
            
            # 3. 列印各類別詳細資訊
            print(f'TP_{name}:{tp}, FP_{name}:{fp}, FN_{name}:{fn}, TN_{name}:{tn}')
            # 這裡順便印出該類別預測成各類的詳細細節 (對應你的 Excel 需求)
            print(f'Details for Actual {name}: -> Pred Normal: {cm[j][0]}, -> Pred High: {cm[j][1]}, -> Pred Low: {cm[j][2]}')
            print(f"Sensitivity_{name}: {s_en:.2f} %")
            print(f"Specificity_{name}: {s_pe:.2f} %\n")

               
       
        sensitivity= statistics.geometric_mean(current_sens.values())
        specificity= statistics.geometric_mean(current_specs.values())
        if((specificity + sensitivity)>0):
            f1=2 * (specificity * sensitivity) / (specificity + sensitivity)
        else:
            f1=0.0

        print(f"Total Sensitivity (G-mean): {sensitivity:.2f} %\n")
        print(f"Total Specificity (G-mean): {specificity:.2f} %\n")
        print("F1: %.2f %%\n" %f1)
        print("Final ACC: %.2f %%\n" %accs) 
        
        diff_sensitivity = sensitivity - current_best_sensitivity
        diff_specificity = specificity - current_best_specificity
        diff_current_sen_spe=abs(current_best_sensitivity-current_best_specificity)  ##Aegen新增
        diff_sen_spe=abs(sensitivity-specificity)                                    ##Aegen新增
        breakthrough_flag = False

        if(diff_sensitivity >= 0 and diff_specificity >= 0):  ##sensitivity和specificity兩者都上升
            breakthrough_flag = True
        elif((current_best_sensitivity > current_best_specificity) and diff_sensitivity <= 0 and diff_specificity >= 0 and abs(diff_sensitivity) <= abs(diff_specificity)):  ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
            breakthrough_flag = True
        elif((current_best_sensitivity > current_best_specificity) and diff_sensitivity >= 3 and diff_specificity >= -2):  ##sensitivity較大，但新的sensitivity上升>=3% 且新specificity只下降2%以內
            breakthrough_flag = True
        elif((current_best_specificity > current_best_sensitivity) and diff_specificity <= 0 and diff_sensitivity >= 0 and abs(diff_specificity) <= abs(diff_sensitivity)):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
            breakthrough_flag = True
        elif((current_best_specificity > current_best_sensitivity) and diff_specificity >= 3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity上升>=3%，且新sensitivity只下降2%以內
            breakthrough_flag = True
        elif(diff_sensitivity>0 or diff_specificity >0) and (diff_sen_spe < diff_current_sen_spe): ##sensitivity或specificity上升，新的sensitivity與specificity差異小於舊的sensitivity與specificity差異時更新
            breakthrough_flag = True


        if (breakthrough_flag):
            current_best_specificity=specificity
            current_best_f1=f1
            current_best_sensitivity=sensitivity
            current_best_accs=accs
            current_best_sensitivity_high=current_sens[1]
            current_best_sensitivity_low=current_sens[2]
            current_best_sensitivity_normal=current_sens[0]
            current_best_specificity_high = current_specs[1]
            current_best_specificity_low = current_specs[2]
            current_best_specificity_normal = current_specs[0]

            '''
            current_best_TP_normal=stats[0]["TP"]
            current_best_FP_normal=stats[0]["FP"]
            current_best_FN_normal=stats[0]["FN"]
            current_best_TN_normal=stats[0]["TN"]
            current_best_TP_high=stats[1]["TP"]
            current_best_FP_high=stats[1]["FP"]
            current_best_FN_high=stats[1]["FN"]
            current_best_TN_high=stats[1]["TN"]
            current_best_TP_low=stats[2]["TP"]
            current_best_FP_low=stats[2]["FP"]
            current_best_FN_low=stats[2]["FN"]
            current_best_TN_low=stats[2]["TN"]
            '''

           
            total_samples = sum(sum(preds.values()) for preds in cm.values())
            # 針對各類別計算 TP, FP, FN, TN 並賦值給 current_best 變數
            # --- Normal (Index 0) ---
            tp0 = cm[0][0]
            fn0 = sum(cm[0].values()) - tp0
            fp0 = sum(cm[act][0] for act in [0, 1, 2]) - tp0
            tn0 = total_samples - (tp0 + fp0 + fn0)

            current_best_TP_normal = tp0
            current_best_FP_normal = fp0
            current_best_FN_normal = fn0
            current_best_TN_normal = tn0

            # --- High (Index 1) ---
            tp1 = cm[1][1]
            fn1 = sum(cm[1].values()) - tp1
            fp1 = sum(cm[act][1] for act in [0, 1, 2]) - tp1
            tn1 = total_samples - (tp1 + fp1 + fn1)

            current_best_TP_high = tp1
            current_best_FP_high = fp1
            current_best_FN_high = fn1
            current_best_TN_high = tn1

            # --- Low (Index 2) ---
            tp2 = cm[2][2]
            fn2 = sum(cm[2].values()) - tp2
            fp2 = sum(cm[act][2] for act in [0, 1, 2]) - tp2
            tn2 = total_samples - (tp2 + fp2 + fn2)

            current_best_TP_low = tp2
            current_best_FP_low = fp2  
            current_best_FN_low = fn2  
            current_best_TN_low = tn2   


            torch.save(model.state_dict(),os.path.join(model_output_folder,"BestModel_"+code_version+"_"+current_time_str+".pth"))  ###比目前訓練階段的模型績效好，儲存起來            
            model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir,"Best_ThreeClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")            
            with open(model_current_best_performance_txtfile, "w") as file:
                file.write("Sensitivity:%.2f\n" %(sensitivity))
                ##file.write("\n")
                file.write("Specificity:%.2f\n" %(specificity))
                ##file.write("\n")
                file.write("-----------------------\n")
                ##file.write("\n")
                file.write("High_Sensitivity:%.2f\n" %(current_sens[1]))
                ##file.write("\n")
                file.write("Low_Sensitivity:%.2f\n" %(current_sens[2]))
                ##file.write("\n")
                file.write("Normal_Sensitivity:%.2f\n" %(current_sens[0]))
                ##file.write("\n")
                file.write("High_Sepcificity:%.2f\n" %(current_specs[1]))
                ##file.write("\n")
                file.write("Low_Sepcificity:%.2f\n" %(current_specs[2]))
                ##file.write("\n")
                file.write("Normal_Sepcificity:%.2f\n" %(current_specs[0]))
                ##file.write("\n")
                file.write("-----------------------\n")
                ##file.write("\n")
                file.write("F1:%.2f\n" %(f1))
                ##file.write("\n")
                file.write("Acc:%.2f\n" %(accs))
                ##file.write("\n")
                file.write("TP_Normal:%d, FP_Normal:%d, FN_Normal:%d, TN_Normal:%d\n" % (current_best_TP_normal, current_best_FP_normal, current_best_FN_normal, current_best_TN_normal))
                file.write("TP_High:%d, FP_High:%d, FN_High:%d, TN_High:%d\n" % (current_best_TP_high, current_best_FP_high, current_best_FN_high, current_best_TN_high))
                file.write("TP_Low:%d, FP_Low:%d, FN_Low:%d, TN_Low:%d\n" % (current_best_TP_low, current_best_FP_low, current_best_FN_low, current_best_TN_low))

                '''
                file.write("TP_Normal:%d, FP_Normal:%d, FN_Normal:%d, TN_Normal:%d\n" %(stats[0]["TP"], stats[0]["FP"], stats[0]["FN"], stats[0]["TN"]))
                file.write("TP_High:%d, FP_High:%d, FN_High:%d, TN_High:%d\n" %(stats[1]["TP"], stats[1]["FP"], stats[1]["FN"], stats[1]["TN"]))
                file.write("TP_Low:%d, FP_Low:%d, FN_Low:%d, TN_Low:%d\n" %(stats[2]["TP"], stats[2]["FP"], stats[2]["FN"], stats[2]["TN"]))        
                '''
    model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio,model_subdir, "Best_ThreeClasses_Model", uuid,"Historic_Best_Performance.txt") ###歷史績效檔案
    if not os.path.exists(model_historic_best_performance_txtfile): ###如果沒有過去歷史檔案，先創造檔案
        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0\n" )
            file.write("Specificity:0\n" )
            file.write("-----------------------")
            file.write("\n")
            file.write("High_Sensitivity:0\n" )
            file.write("Low_Sensitivity:0\n")
            file.write("Normal_Sensitivity:0\n")
            file.write("High_Specificity:0\n")
            file.write("Low_Specificity:0\n")
            file.write("Normal_Specificity:0\n")
            file.write("-----------------------\n")
            file.write("F1:0\n")
            file.write("Acc:0\n")
            file.write("TP_Normal:%d, FP_Normal:%d, FN_Normal:%d, TN_Normal:%d\n" %(0, 0, 0, 0))
            file.write("TP_High:%d, FP_High:%d, FN_High:%d, TN_High:%d\n" %(0, 0, 0, 0))
            file.write("TP_Low:%d, FP_Low:%d, FN_Low:%d, TN_Low:%d\n" %(0, 0, 0, 0)) 
            file.write("Model_Name:") 
            
            
    performance_txtfile = open(model_historic_best_performance_txtfile)
    performance_array=[]
    for line in performance_txtfile.readlines():
        score=line.split(":")
        performance_array.append(score[-1])
         
    performance_txtfile.close()

    historic_best_sensitivity=float(performance_array[0])
    historic_best_specificity=float(performance_array[1])    

    ##比較本次訓練最佳績效是否高於過去歷史最佳績效
    diff_sensitivity = current_best_sensitivity - historic_best_sensitivity
    diff_specificity = current_best_specificity - historic_best_specificity
    diff_historic_best_sen_spe=abs(historic_best_sensitivity-historic_best_specificity)  ##Aegen 新增
    diff_current_best_sen_spe=abs(current_best_sensitivity-current_best_specificity)     ##Aegen 新增

    historic_breakthrough_flag = False
    if (diff_sensitivity >= 0 and diff_specificity >= 0):  ##sensitivity和specificity兩者都上升
        historic_breakthrough_flag = True
    elif ((historic_best_sensitivity > historic_best_specificity) and diff_sensitivity <= 0 and diff_specificity >= 0 and abs(
            diff_sensitivity) <= abs(diff_specificity)):  ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
        historic_breakthrough_flag = True
    elif ((historic_best_sensitivity > historic_best_specificity) and diff_sensitivity >= 3 and diff_specificity >= -2):  ##sensitivity較大，但新的sensitivity上升>3%且新specificity下降<2%
        historic_breakthrough_flag = True
    elif ((historic_best_specificity > historic_best_sensitivity) and diff_specificity <= 0 and diff_sensitivity >= 0 and abs(
            diff_specificity) <= abs(diff_sensitivity)):  ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
        historic_breakthrough_flag = True
    elif ((historic_best_specificity > historic_best_sensitivity) and diff_specificity >= 3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity上升>3%，且新sensitivity上下降<2%
        historic_breakthrough_flag = True
    elif(diff_sensitivity > 0 or diff_specificity > 0) and (diff_current_best_sen_spe < diff_historic_best_sen_spe):#sensitivity或specificity上升，新的sensitivity與specificity差異小於舊的sensitivity與specificity差異時更新
        historic_breakthrough_flag = True


    if (historic_breakthrough_flag):
        model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir, "Best_ThreeClasses_Model", uuid,"Historic_Best_Performance.txt")
        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:%.2f\n" %(current_best_sensitivity))
            file.write("Specificity:%.2f\n" %(current_best_specificity))
            file.write("-----------------------\n")           
            file.write("High_Sensitivity:%.2f\n" %(current_best_sensitivity_high))           
            file.write("Low_Sensitivity:%.2f\n" %(current_best_sensitivity_low))            
            file.write("Normal_Sensitivity:%.2f\n" %(current_best_sensitivity_normal))           
            file.write("High_Specificity:%.2f\n" %(current_best_specificity_high))            
            file.write("Low_Specificity:%.2f\n" %(current_best_specificity_low))
            file.write("Normal_Specificity:%.2f\n" %(current_best_specificity_normal))
            file.write("-----------------------\n")            
            file.write("F1:%.2f\n" %(current_best_f1))           
            file.write("Acc:%.2f\n" %(current_best_accs))          
            file.write("TP_Normal:%d, FP_Normal:%d, FN_Normal:%d, TN_Normal:%d\n" %(current_best_TP_normal, current_best_FP_normal, current_best_FN_normal, current_best_TN_normal))
            file.write("TP_High:%d, FP_High:%d, FN_High:%d, TN_High:%d\n" %(current_best_TP_high, current_best_FP_high, current_best_FN_high, current_best_TN_high))
            file.write("TP_Low:%d, FP_Low:%d, FN_Low:%d, TN_Low:%d\n" %(current_best_TP_low, current_best_FP_low, current_best_FN_low, current_best_TN_low)) 
            modelname="Model_Name:BestModel_"+code_version+"_"+current_time_str+".pth"
            file.write(modelname)          

        status=1  ###訓練成功且績效較之前版本好
    else:
        status=0  ###訓練成功但績效沒有比之前好    

    ##-------寫出confusion matrix到excel檔案----
    labels = ['Low', 'Normal', 'High']

    # 2. 依照順序從 cm 字典中提取數據 (cm[實際][預測])
    # 注意：您的 index 映射為 0:Normal, 1:High, 2:Low
    matrix_data = [
        [cm[2][2], cm[2][0], cm[2][1]], # 實際 Low -> 預測 Low, Normal, High
        [cm[0][2], cm[0][0], cm[0][1]], # 實際 Normal -> 預測 Low, Normal, High
        [cm[1][2], cm[1][0], cm[1][1]]  # 實際 High -> 預測 Low, Normal, High
    ]

    # 3. 建立 DataFrame
    df = pd.DataFrame(matrix_data, index=labels, columns=labels)

    # 4. 計算橫列與直欄的總和 
    df['Total_Reference'] = df.sum(axis=1)
    df.loc['Total_Predicted'] = df.sum(axis=0)

    # 5. 儲存為 Excel 檔案
    ###改存到 basepath/perf_Matrix/<model_subdir>/ 底下。原本直接寫在 basepath 根目錄，
    ###路徑不含 splitting_ratio 與 model_subdir，導致 NoAug/WithAug 跑完會互相覆蓋同名檔案
    ###(實際已經發生過：NoAug 與 WithAug_1 的混淆矩陣都被 WithAug_2 蓋掉了)。
    ###model_subdir 為空字串時 os.path.join 會自動略過，退回 perf_Matrix/ 根目錄。
    output_excel = os.path.join(basepath,"perf_Matrix",model_subdir,uuid+"_Performance_Matrix.xlsx")
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df.to_excel(output_excel, index_label="Reference \ Predicted")

    print(f"混淆矩陣已成功導出至: {output_excel}")
    ##----------------------

    errorcode="0"
    message="Model wiht three classes has been built!"
    print(message)

    return status, errorcode, message          

def BuildModel_TwoClasses(uuid,basepath,splitting_ratio="",model_subdir="",augment=False):

    errorcode="0"
    message=""
    status=-1

    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間    
    code_version=get_version()   ######取得模型訓練時當前程式版本

    print('Start building the model with two classes!')       
    model_output_folder = os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model", uuid)
    if not os.path.exists(model_output_folder): 
        os.makedirs(model_output_folder)
    
    model_data_path=os.path.join(basepath, splitting_ratio, model_subdir, "Temp_TwoClasses_Model", uuid)  ###先清空過去訓練的模型佔存資料
    if os.path.isdir(model_data_path):
        shutil.rmtree(model_data_path)    


    if not os.path.exists(os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model",uuid)):
        os.makedirs(os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model",uuid))

    model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model", uuid, "Performance_"+code_version+"_"+current_time_str+".txt")   
    if not os.path.exists(model_current_best_performance_txtfile): ##沒有檔案，先創造檔案
        with open(model_current_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0\n")
            file.write("Specificity:0\n")
            file.write("-----------------------\n")
            file.write("High_Sensitivity:0\n")
            file.write("Low_Sensitivity:0\n")
            file.write("Normal_Sensitivity:0\n")
            file.write("-----------------------\n")
            file.write("F1:0\n")
            file.write("Acc:0\n")
            file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(0, 0, 0, 0)) 
        
    ## Create dataset object  
    Method = 'time' ##'raw' ##'combine'
    current_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Train")
    
    '''
    filelist_high=os.listdir(os.path.join(current_path,'High'))
    filelist_normal=os.listdir(os.path.join(current_path,'Normal'))    
    
    
    if(len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-500"
        message="An error occurs in the BuildModel_TwoClasses function of M5.py: No training data"
        status=-1    
        return  status, errorcode, message  
    '''

    dataset = ECGDataset(dir_path = current_path, method=Method, classes=2)

    Epoch_range = range(0, 300)
    patience =30
    scheduler_patience = 15  ###要小於early-stopping的patience，否則LR還沒衰減就先被early stop停掉
    Batch_size = 32 ##256 ###128
    valid_split = 0.1
    random_seed = 10
           
    ##-----Balanced Batch------
    ## --- 1.設定資料根目錄------
    glucosedata_path = os.path.join(basepath, splitting_ratio, "GlucoseData", uuid, 'Train')
    data_dir = Path(glucosedata_path)
    classes = ['Normal', 'High']  ##, 'Low']

    ## --- 2. 遍歷資料夾取得路徑與標籤 ---
    all_file_paths = []
    target_list = []
    for idx, cls_name in enumerate(classes):
        cls_folder = data_dir / cls_name
        files = list(cls_folder.glob('*.txt'))
        print(f"類別 {cls_name} (Index {idx}) 共有 {len(files)} 個檔案")
        all_file_paths.extend(files)
        target_list.extend([idx] * len(files))

    target_list = torch.tensor(target_list)

    ## --- 3. 進行分層 Train / Valid Split ---
    labels_np_all = dataset.Labels.numpy().flatten().astype(int)   
    train_indices, val_indices = train_valid_split_indices(labels_array=labels_np_all, valid_ratio=valid_split, seed=random_seed)

    # --- 4. 建立 BalancedBatchSampler ---
    train_labels = labels_np_all[train_indices]
    train_sampler = BalancedBatchSampler(labels=train_labels, batch_size=Batch_size)

    # --- 5. 建立 Dataset 與 DataLoader ---
    # 訓練集：使用 batch_sampler；只對training data做data augmentation
    train_subset = ECGDatasetSubset(dataset, train_indices, augment=augment)
    train_loader = DataLoader(train_subset, batch_sampler=train_sampler)

    # 驗證集：保持原始分布，不做augmentation
    val_subset = ECGDatasetSubset(dataset, val_indices, augment=False)
    valid_loader = DataLoader(val_subset, batch_size=Batch_size, shuffle=False)

    total_train_num =  len(train_loader)
    total_valid_num = len(valid_loader)

    print("total_train_num",total_train_num)
    print("total_valid_num",total_valid_num)
    print("data augmentation:",augment)
    if(total_train_num<=1 or total_valid_num<=1):
        errorcode="-500"
        message="An error occurs in the BuildModel_TwoClasses function of M5.py: No training data"
        status=-1
        return  status, errorcode, message



    current_test_path=os.path.join(basepath,splitting_ratio,"GlucoseData",uuid,"Test")
    
    '''
    if(type_array[0]=='High' or type_array[1]=='High'):
        filelist_high=os.listdir(os.path.join(current_test_path,'High'))
    if(type_array[0]=='Normal'):
        filelist_normal=os.listdir(os.path.join(current_test_path,'Normal'))
    if(type_array[0]=='Low' or type_array[1]=='Low'):
        filelist_low=os.listdir(os.path.join(current_test_path,'Low'))    

    
    if(len(filelist_high)==0 or len(filelist_normal)==0):
        errorcode="-501"
        message="An error occurs in the BuildModel_TwoClasses function of M5.py: No testing data"
        return errorcode,message 
    '''

    testdata = ECGDataset(dir_path=current_test_path, method=Method, classes=2)
    test_loader = DataLoader(testdata, batch_size=Batch_size) 
           
    loopindex=0   
    current_best_sensitivity=0  ###本次訓練最佳sensitivity先設定為0
    current_best_specificity=0
    current_best_sensitivity_high=0 
    current_best_sensitivity_normal=0  
    current_best_f1=0
    current_best_accs=0 
    current_best_TP=0
    current_best_FP=0
    current_best_FN=0
    current_best_TN=0
         
    save_path = os.path.join(basepath, splitting_ratio, model_subdir, "Temp_TwoClasses_Model", uuid) ###建立uuid專屬的模型存放資料夾
    if os.path.isdir(save_path) is True:
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    '''
    seed=42  ##為了減少訓練時的隨機性因素導致每次訓練績效差異很大
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #禁用 CUDNN 內部優化，以確保確定性（可能會輕微降低速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    '''


    while(loopindex<5):  ###執行5次訓練，取最好一次

        print('uuid:',uuid)
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

        ###Loss_list/ACCs 移到 if 外面：續訓(Epoch_range[0] != 0)時原本不會被初始化，迴圈裡的 append 會 NameError
        Loss_list = []
        ACCs = []

        if Epoch_range[0] == 0:
    
            ##model = TwoClasses_CNN(data_len=dataset.data_len, input_channel=dataset.channel)        
            model = TwoClasses_Transformer(data_len=dataset.data_len, input_channel=dataset.channel) 
            initialize_weights(model)  ###模型參數初始化
  
            model = model.to(device)
            print('Two-Class CNN model created.')
       

        ##loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4)).to(device)  ###把某一類樣本權重放大4倍，解決資料不平衡問題
        ##loss_function = torch.nn.BCEWithLogitsLoss().to(device)
        loss_function = FocalLoss(gamma=2.0).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-3)
        ###optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=1, patience=scheduler_patience, cooldown=0, min_lr=0.00001)

        n_patience = 0
        min_valid_loss = 0
         
        
        for epoch in Epoch_range:
    
            print("Epoch: %d " % (epoch+1))
            
            ## 初始化累加器（保持在 GPU）
            train_loss_total = torch.tensor(0.0, device=device)
            valid_loss_total = torch.tensor(0.0, device=device)

            ## --- Training Part ---
            model.train()
            correct, total = torch.tensor(0, device=device), 0

            for data, target in train_loader:
                # 使用 non_blocking 加速
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                if data.size(0) <= 1: continue 

                optimizer.zero_grad(set_to_none=True) # 比 zero_grad() 稍微快一點
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

                # 在 GPU 上直接計算累加，不使用 .item()
                with torch.no_grad():
                    train_loss_total += loss.detach()
                    # 優化：output > 0 等同於 sigmoid(output) > 0.5
                    correct += (output >= 0).eq(target.byte()).sum()
                    total += target.size(0)

            train_acc = (100.0 * correct / total).item() # 整個 Epoch 只調用一次 .item()
            train_loss_avg = (train_loss_total / len(train_loader)).item()
            print(f"Train ACC: {train_acc:.2f}%, ", end='')


            ## --- Evaluation Part ---
            model.eval()
            correct_v, total_v = torch.tensor(0, device=device), 0

            with torch.no_grad(): # 驗證時絕對要加，省去梯度計算空間與時間
                for data, target in valid_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    
                    if data.size(0) <= 1: continue

                    output = model(data)
                    loss = loss_function(output, target)
                    
                    valid_loss_total += loss
                    correct_v += (output >= 0).eq(target.byte()).sum()
                    total_v += target.size(0)

            valid_acc = (100.0 * correct_v / total_v).item()
            valid_loss_avg = (valid_loss_total / len(valid_loader)).item()
            print(f"Valid ACC: {valid_acc:.2f}%, Traing loss: {train_loss_avg:.4f}, Valid loss: {valid_loss_avg:.4f}")

            ## --- 後續邏輯 ---
            scheduler.step(valid_loss_avg)
            Loss_list.append([train_loss_avg, valid_loss_avg])
            ACCs.append([train_acc, valid_acc])
            
            if valid_loss_avg > min_valid_loss:
                if epoch == 0: ##epoch第一次，模型不論好壞都先存起來
                    min_valid_loss = valid_loss_avg
                    torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"TwoClassesModel_"+str(uuid)+"_Best.pth"))
                else:
                    n_patience += 1
            else:
                min_valid_loss = valid_loss_avg
                n_patience = 0              
                torch.save(model.state_dict(),os.path.join(os.path.abspath(save_path),"TwoClassesModel_"+str(uuid)+"_Best.pth"))
                       
        
            if n_patience >= patience:
                ##print("The model didn't improve for %i rounds, break it!" % patience)
                break
             
           
        ###把這次訓練每個 epoch 的 loss / accuracy 落地成 CSV + 曲線圖，跟 .pth 放在同一個資料夾
        save_training_history(save_path, uuid, Loss_list, ACCs,
                              tag='TwoClasses', start_epoch=Epoch_range[0] + 1)

        ## ------------------------Testing Data--------------------      
        ##model = TwoClasses_CNN(data_len=dataset.data_len, input_channel=dataset.channel)
        
        model = TwoClasses_Transformer(data_len=dataset.data_len, input_channel=dataset.channel) 
        model.to(device)        
   
        model_path = os.path.join(basepath,splitting_ratio,model_subdir,"Temp_TwoClasses_Model",uuid,"TwoClassesModel_"+str(uuid)+"_Best.pth")

        if device.type == 'cuda':
            model.load_state_dict(torch.load(model_path))
        else:
            map_location=torch.device('cpu')
            model.load_state_dict(torch.load(model_path,map_location))     
        
      
        test_old_data = 0
        if test_old_data:
            testdata = ECGDataset(dir_path = './'+splitting_ratio+'/GlucoseData/'+uuid+'/Test', method=Method, classes=2)
            test_loader = DataLoader(testdata, batch_size=Batch_size) 
           
       
        model.eval()
        correct, total = 0, 0
        TN, FP, FN, TP = 0, 0, 0, 0
        TP_Normal, TP_High = 0, 0
        FN_Normal, FN_High = 0, 0
        for data, target in test_loader:
            data = data.to(device) 

            ## 預測
            output = model(data)
            output = output.to('cpu')
            predict = torch.sigmoid(output).ge(0.5)
            correct += (predict == target).sum().item()
            
            current_batch_size=0 
            if target.dim() == 0:
                current_batch_size = 1
            else:
                current_batch_size = target.size(0)

            total += current_batch_size  
           
            
            predict = predict.tolist()
            target = target.tolist()
            for i in range(0, len(predict)):
                if predict[i][0] is True and target[i][0] == 1:
                    TP += 1
                    TP_High+=1
                elif predict[i][0] is True and target[i][0] == 0:
                    FP += 1
                    FN_Normal+=1
                elif predict[i][0] is False and target[i][0] == 1:
                    FN += 1
                    FN_High+=1
                elif predict[i][0] is False and target[i][0] == 0:
                    TN += 1
                    TP_Normal+=1
    
        if((TP+FN)>0):                
            sensitivity=(100*TP/(TP+FN))
        else:
            sensitivity=0.0

        if((TN+FP)>0):    
            specificity=(100*TN/(TN+FP))
        else:
            specificity=0.0

        if((specificity + sensitivity)>0):
            f1=2 * (specificity * sensitivity) / (specificity + sensitivity)
        else:
            f1=0.0
        
        accs=100*correct/total

        if((TP_Normal+FN_Normal)>0):                
            sensitivity_normal=(100*TP_Normal/(TP_Normal+FN_Normal))
        else:
            sensitivity_normal=0.0

        if((TP_High+FN_High)>0):                
            sensitivity_high=(100*TP_High/(TP_High+FN_High))
        else:
            sensitivity_high=0.0    

        print("TP:%d, FP:%d, FN:%d, TN:%d\n" %(TP, FP, FN, TN))
        print("Sensitivity: %.2f %%\n" %sensitivity)
        print("Specificity: %.2f %%\n" %specificity)
        print("F1: %.2f %%\n" %f1)
        print("Final ACC: %.2f %%\n" %accs) 
        print("Sensitivity_High: %.2f %%\n" %sensitivity_high)
        print("Sensitivity_Normal: %.2f %%\n" %sensitivity_normal)

        
        diff_sensitivity = sensitivity - current_best_sensitivity
        diff_specificity = specificity - current_best_specificity
        diff_current_sen_spe=abs(current_best_sensitivity-current_best_specificity)  
        diff_sen_spe=abs(sensitivity-specificity)                                   
        breakthrough_flag = False
        
        if (diff_sensitivity >= 0 and diff_specificity >= 0):  ##sensitivity和specificity兩者都上升
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
        elif(diff_sensitivity>0 or diff_specificity>0) and (diff_sen_spe < diff_current_sen_spe): ##sensitivity或specificity上升，新的sensitivity與specificity差異小於舊的sensitivity與specificity差異時更新
            breakthrough_flag = True   

        if(breakthrough_flag):    
            current_best_specificity=specificity
            current_best_f1=f1
            current_best_sensitivity=sensitivity
            current_best_sensitivity_high=sensitivity_high
            current_best_sensitivity_normal=sensitivity_normal
            current_best_accs=accs
            current_best_TP=TP
            current_best_FP=FP
            current_best_FN=FN
            current_best_TN=TN
            torch.save(model.state_dict(),os.path.join(model_output_folder, "BestModel_"+code_version+"_"+current_time_str+".pth"))  ###比目前訓練階段的模型績效好，儲存起來            
            model_current_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model", uuid,"Performance_"+code_version+"_"+current_time_str+".txt")            
            with open(model_current_best_performance_txtfile, "w") as file:
                file.write("Sensitivity:%.2f\n" %(sensitivity))
                file.write("Specificity:%.2f\n" %(specificity))
                file.write("-----------------------\n")
                file.write("High_Sensitivity:%.2f\n" %(sensitivity_high))
                file.write("Normal_Sensitivity:%.2f\n" %(sensitivity_normal))
                file.write("-----------------------\n")
                file.write("F1:%.2f\n" %(f1))
                file.write("Acc:%.2f\n" %(accs))
                file.write("TP:%d, FP:%d, FN:%d, TN:%d\n" %(TP, FP, FN, TN)) 
              

    model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model", uuid,"Historic_Best_Performance.txt") ###歷史最佳績效
    if not os.path.exists(model_historic_best_performance_txtfile): ###如果沒有過去歷史檔案，先創造檔案
        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:0\n")
            file.write("Specificity:0\n")
            file.write("-----------------------\n")
            file.write("High_Sensitivity:0\n")
            file.write("Low_Sensitivity:0\n")
            file.write("Normal_Sensitivity:0\n")
            file.write("-----------------------\n")
            file.write("F1:0\n")
            file.write("Acc:0\n")
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
  

    ##比較本次訓練最佳績效是否高於過去歷史最佳績效  
    diff_sensitivity=current_best_sensitivity-historic_best_sensitivity
    diff_specificity=current_best_specificity-historic_best_specificity
    diff_historic_best_sen_spe=abs(historic_best_sensitivity-historic_best_specificity)  ##Aegen 新增
    diff_current_best_sen_spe=abs(current_best_sensitivity-current_best_specificity)     ##Aegen 新增
    historic_breakthrough_flag=False
    
    if(current_best_sensitivity>=historic_best_sensitivity and current_best_specificity>=historic_best_specificity): ##sensitivity和specificity兩者都上升
        historic_breakthrough_flag=True
    elif((historic_best_sensitivity>historic_best_specificity) and diff_sensitivity<=0 and diff_specificity>=0 and abs(diff_sensitivity)<=abs(diff_specificity)): ##sensitivity較大，但新的sensitivity下降，且下降差值<新specificity上升差值
        historic_breakthrough_flag=True
    elif ((historic_best_sensitivity > historic_best_specificity) and diff_sensitivity >=3 and diff_specificity >=-2):  ##sensitivity較大，但新的sensitivity上升>3%且新specificity下降<2%
        historic_breakthrough_flag = True
    elif((historic_best_specificity> historic_best_sensitivity) and diff_specificity<=0 and diff_sensitivity>=0 and abs(diff_specificity)<=abs(diff_sensitivity)): ##specificity較大，但新的specificity下降，且下降差值<新sensitivity上升差值
        historic_breakthrough_flag=True
    elif((historic_best_specificity > historic_best_sensitivity) and diff_specificity >=3 and diff_sensitivity >= -2):  ##specificity較大，但新的specificity上升>3%，且新sensitivity上下降<2%
        historic_breakthrough_flag = True
    elif(diff_sensitivity > 0 or diff_specificity > 0) and (diff_current_best_sen_spe < diff_historic_best_sen_spe):#sensitivity或specificity上升，新的sensitivity與specificity差異小於舊的sensitivity與specificity差異時更新
        historic_breakthrough_flag = True


    if(historic_breakthrough_flag):
        model_historic_best_performance_txtfile=os.path.join(basepath, splitting_ratio, model_subdir, "Best_TwoClasses_Model", uuid, "Historic_Best_Performance.txt")

        with open(model_historic_best_performance_txtfile, "w") as file:
            file.write("Sensitivity:%.2f" %(current_best_sensitivity))
            file.write("\n")
            file.write("Specificity:%.2f" %(current_best_specificity))
            file.write("\n")
            file.write("-----------------------")
            file.write("\n")
            file.write("High_Sensitivity:%.2f" %(current_best_sensitivity_high))
            file.write("\n")
            file.write("Normal_Sensitivity:%.2f" %(current_best_sensitivity_normal))
            file.write("\n")
            file.write("-----------------------")
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

    ##-----存初二分類的confusion matrix----
    matrix_data = [
    [TN, FP],  # 實際為 0 (Normal) 的分佈
    [FN, TP]   # 實際為 1 (Abnormal) 的分佈
    ]

    labels = ['Normal', 'Abnormal']

    # 2. 建立 DataFrame
    df = pd.DataFrame(matrix_data, index=labels, columns=labels)

    # 3. 計算橫列總和 (Total_Reference) -> 寫在第四個 Column (以二分類來說是第三欄)
    df['Total_Reference'] = df.sum(axis=1)

    # 4. 計算直欄總和 (Total_Predicted) -> 寫在下方的 Row
    df.loc['Total_Predicted'] = df.sum(axis=0)

    # 5. 儲存為 Excel 檔案
    ###改存到 basepath/perf_Matrix/<model_subdir>/ 底下。原本直接寫在 basepath 根目錄，
    ###路徑不含 splitting_ratio 與 model_subdir，導致 NoAug/WithAug 跑完會互相覆蓋同名檔案
    ###(實際已經發生過：NoAug 與 WithAug_1 的混淆矩陣都被 WithAug_2 蓋掉了)。
    ###model_subdir 為空字串時 os.path.join 會自動略過，退回 perf_Matrix/ 根目錄。
    output_excel = os.path.join(basepath,"perf_Matrix",model_subdir,uuid+"_Performance_Matrix.xlsx")
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df.to_excel(output_excel, index_label="Reference \ Predicted")

    print(f"二分類混淆矩陣已成功導出至: {output_excel}")

    errorcode="0"
    message="Model with two classes has been built!"
    print(message)

    return status, errorcode, message



if __name__ == "__main__":

    user_information = [['2197','20250212','20250225'],       ##T1003(績效良，但可再訓練)
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
                        ['2221','20250219','20250220'],         ##提早退出

                        
                        ['2279','20250426','20250509'], ##40    ##T1041  
                        ['2276','20250426','20250509'],         ##T1042
                        ['2278','20250426','20250509'],         ##T1043
                        ['2281','20250502','20250515'],         ##T1044
                        ['2286','20250509','20250522'],         ##T1045
                        ['2285','20250509','20250522'], ##45    ##T1046
                        ['2294','20250520','20250531'],         ##T1047
                        ['2293','20250520','20250602'],         ##T1048
                        ['2295','20250522','20250604'],         ##T1049
                        ['2291','20250527','20250610'],         ##T1050
                        ['2298','20250604','20250617'],  ##50   ##T1051(19歲，小於20歲)
                        ['2300','20250604','20250617'],         ##T1052
                        ['2299','20250604','20250617'],         ##T1053  ##訓練到一半出問題
                        ['2304','20250606','20250619'],         ##T1054
                        ['2305','20250606','20250619'],         ##T1055
                        ['2306','20250606','20250619'],  ##55   ##T1056
                        ['2130','20250623','20250706'],         ##T1057  ##test沒有高血糖
                        ['2132','20250623','20250706'],         ##T1058
                        ['2322','20250623','20250706'],  ##58   ##T1059
                        ['2329','20250701','20250713'],         ##T1060
                        ['2352','20250731','20250813'],         ##T1061 
                        ['2378','20251022','20251104'],         ##T1062 
                        ['2397','20251129','20251213'],         ##T1063 


                        ##['91', '20240731', '20250319'],
                        ##['91', '20240812', '20250319'],
                        ['91', '20241108', '20250719'],
                        ['2283','20260214','20260227'],
                        ['2423','20260214','20260227'],
                        ['798','20260227','20260312'],
                        ['746','20250619','20250626']]         ##David資料測試                        
  
    mother_path=Path(__file__).resolve().parent   ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    base_path=mother_path/"Model"
    base_path=str(base_path)
    ##glucosedata_path=r'D:\Dennis Project\Glucose_CSV\BGM_CSV\filtered_BGM' ##mother_path/"GlucoseDataCSV"  ###APP收到的血糖數值，整理成CSV檔後存放路徑
    glucosedata_path=r'D:\Dennis Project\Glucose_CSV\CGM_CSV'
    ##glucosedata_path=r'D:\Dennis Project\Glucose_CSV\BGM_CSV\filtered_BGM'
    glucosedata_path=str(glucosedata_path)
    server_db_path='G:\\.shortcut-targets-by-id\\1Mc_sTYrGzDau1JPki2AQDpV4FeX5tKu3\\SWM_DataCenter\\Health_Server_Script\\_rawdata_download'
   
         
    target_uuids=['2197','2199','2204','2206','2208','2210','2215','2216','2223','2249']  ###這次要訓練的uuid(Model/70_30/GlucoseData/<uuid>底下已有Train/Test資料)

    model_subdir="WithAug_2"   ###無augmentation版本，輸出到Model/70_30/NoAug底下；要跑有augmentation版本請改成"WithAug"並把augment改成True，否則會覆蓋掉這次的結果
    augment=True

    for i in range(0,len(user_information)):

        user_info=user_information[i]

        ##-------step 1. 基本設定--------
        uuid=user_info[0]

        if uuid not in target_uuids:
            continue

        start_time=user_info[1]  ##第一筆血糖資料記錄日期
        end_time=user_info[2]    ##最後一筆血糖資料紀錄日期
        print('index:',i,' uuid:',uuid)
        srj_db_path=str(mother_path/"DataDB"/uuid)   ###srj檔放置路徑


        start_time = time.time()
        ##-------step 2. 血糖模型建立---------
        ##status, errorcode, message = BuildModel(uuid, base_path, srj_db_path, glucosedata_path, processnum=8, splitting_ratio="70_30")  ##建立個人化模型(自動根據是否有低血糖資料，決定訓練中高血糖模型或是高中低血糖模型)
        status, errorcode, message = BuildModel(uuid,base_path,srj_db_path,glucosedata_path,server_db_path,processnum=8,splitting_ratio="70_30",model_subdir=model_subdir,augment=augment)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函式執行時間: {execution_time:.6f} 秒")

        print('prcoessed uuid:', uuid, ' status:',str(status),' error code:',errorcode,' message:',message)
        

        '''
        ##--------step 3. 輸入ECG進行血糖類別分類--------- 
        testdatapath=os.path.join(base_path,"TestData",uuid) ##測試用程式
        if not os.path.exists(testdatapath): os.makedirs(testdatapath)  ##測試用程式 

        filelist=os.listdir(testdatapath) ##測試用程式
        ecgdata=[] 
        for i in range(0,len(filelist)): ##測試用程式
            filename=filelist[i]
            print('filename:',filename)
            f = open(os.path.join(testdatapath,filename), "r")
            for x in f:
                ecgdata.append(int(x.rstrip('\n')))       
    
        print('length ecg:',len(ecgdata))
        GlucosePredictor_Obj=GlucosePredictor(uuid,base_path)
        ##ecgdata=ecgdata[10*60*250:20*60*250]
    
        errorcode, message, glucose_category, glucose_value=GlucosePredictor_Obj.predict(ecgdata)
        print('glucose_category:',glucose_category,' glucose_value:',glucose_value)
        print('errorcode:',str(errorcode),' message:',message)     
        '''