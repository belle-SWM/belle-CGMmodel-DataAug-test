from .Model import CNN
import os
import numpy as np
import pywt
import torch

class ECG_Dataset:
    """Create dataset for abnormal ecg detection.

    Attributes:
        data: Raw data in dataset
        correct_flag: A boolean flag of correct input or not.
        model_path: Path of PyTorch model.
        device: Indicator to show is there any gpu in device.
    """

    def __init__(self, data):
        self.data = np.array(data)
        self.correct_flag = self.check_shape()
        self.model_path = os.path.join(os.path.dirname(__file__), 'abnormal_ECG_model.pth')
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def check_shape(self):
        # 輸入為一維向量
        if len(self.data.shape) == 1:
            # 增加維度至二維向量
            self.data = np.expand_dims(self.data, axis=0)
        
        # 輸入為二維向量
        if len(self.data.shape) == 2:
            # data_len: 輸入訊號個數, sig_len: 訊號長度
            self.data_len, self.sig_len = self.data.shape
            if self.sig_len != 2500:
                print('Input shape should be [~, 2500]')
                return False
        else:
            print('Input dimension should be 2 or lower.')
            return False
        
        return True    
    
    @staticmethod
    def normalize1(x):
        '''
        x_max = np.max(x)
        x_min = np.min(x)
        x_norm = (x - x_min) / (x_max - x_min)
        '''
        x_norm = np.interp(x,(x.min(),x.max()),(0,1))
        return x_norm

    @staticmethod
    def DWT(x):
        # pywt.swt require the length to be 2504
        sig = np.zeros(2504)
        sig[:len(x)] = x
        # stationary wavelet transform
        coeffs = pywt.swt(sig, 'sym4', level=3, trim_approx=True)
        coeffs = np.array(coeffs)[:,:2500]
        return coeffs

    def feature(self):
        # 確認shape是否正確
        if not self.correct_flag:
            return None
        
        # 對每段sig擷取特徵  7 seconds for 1000 data
        F = []
        for sig in self.data:
            # 強度標準化至0-1
            sig = self.normalize1(sig)

            # 小標轉換
            sig = self.DWT(sig).tolist()
            F.append(sig)
            
        # 將特徵轉成3維tensor輸出
        # shape: [data_len, 4, sig_len]
        F = torch.tensor(F)
        
        # 如果輸入為一向量，輸出時增加維度
        if self.data_len == 1:
            F.unsqueeze(0)
        
        return F 
    
    def create_model(self):
        model = CNN(data_len=2500, input_channel=4)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        
        return model

    def release_GPU(self):
        torch.cuda.empty_cache()
        
    def detect_abnormal(self, use_gpu=0):
        
        """Detect abnormal ECG in dataset.

        Load existing PyTorch model and do prediction.

        Args:
            use_gpu: use GPU or not. Default: 0

        Returns:
            A list of abnormal indicators.
        """

        device = self.device
        if self.device != 'cpu' and use_gpu == 0:
            device = 'cpu'
        
        # 對輸入取feature後，移至device
        intput = self.feature().to(device)
        # 載入model後，移至device，開啟evaluation mode
        model = self.create_model().to(device).eval()
        # 預測結果
        output = model(intput).to('cpu')
        predict = torch.sigmoid(output).ge(0.5).tolist()
        # 整理結果為list輸出
        predict_flat = [item for sublist in predict for item in sublist]
        
        return predict_flat
