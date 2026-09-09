from .Model import CNN
import os
import numpy as np
import pywt
import torch

class AbnormalDetector:
    def __init__(self, use_gpu=1):
        self._initialize()
        self.device = ("cuda:0" if use_gpu==1 else "cpu")
        self._create_model()

    def detect_abnormal(self, Data):
        self.data = np.array(Data)
        self.correct_flag = self._check_shape()

        # 對輸入取feature後，移至device
        intput = self._feature().to(self.device)
        # 預測結果
        with torch.no_grad():
            output = self.model(intput).to('cpu')
        predict = torch.sigmoid(output).ge(0.5).tolist()
        # 整理結果為list輸出
        predict_flat = [item for sublist in predict for item in sublist]
        
        return predict_flat

    def _initialize(self):
        self.device = []
        self.model = []
        self.data = []
        self.data_len = []
        self.correct_flag = []

    def _create_model(self):
        model_path = os.path.join(os.path.dirname(__file__),'abnormal_ECG_model.pth')
        Model = CNN(data_len=2500, input_channel=4)
        Model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # 載入model後，移至device，開啟evaluation mode
        self.model = Model.to(self.device).eval()

    def _check_shape(self):
        # 輸入為一維向量
        if len(self.data.shape) == 1:
            # 增加維度至二維向量
            self.data = np.expand_dims(self.data, axis=0)
        
        # 輸入為二維向量
        if len(self.data.shape) == 2:
            # data_len: 輸入訊號個數, sig_len: 訊號長度
            self.data_len, sig_len = self.data.shape
            if sig_len != 2500:
                print('Input shape should be [~, 2500]')
                return False
        else:
            print('Input dimension should be 2 or lower.')
            return False

        return True    
    
    @staticmethod
    def _normalize1(x):
        x_norm = np.interp(x,(x.min(),x.max()),(0,1))
        return x_norm

    @staticmethod
    def _DWT(x):
        # pywt.swt require the length to be 2504
        sig = np.zeros(2504)
        sig[:len(x)] = x
        # stationary wavelet transform
        coeffs = pywt.swt(sig, 'sym4', level=3, trim_approx=True)
        coeffs = np.array(coeffs)[:,:2500]
        return coeffs

    def _feature(self):
        # 確認shape是否正確
        if not self.correct_flag:
            return None
        
        # 對每段sig擷取特徵  7 seconds for 1000 data
        F = []
        for sig in self.data:
            # 強度標準化至0-1
            sig = self._normalize1(sig)

            # 小標轉換
            sig = self._DWT(sig).tolist()
            F.append(sig)
            
        # 將特徵轉成3維tensor輸出
        # shape: [data_len, 4, sig_len]
        F = torch.tensor(F)
        
        # 如果輸入為一向量，輸出時增加維度
        if self.data_len == 1:
            F.unsqueeze(0)
        
        return F 

    
    