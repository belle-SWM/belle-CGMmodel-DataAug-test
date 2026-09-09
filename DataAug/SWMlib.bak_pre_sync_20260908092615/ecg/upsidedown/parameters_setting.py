"""

Setting the parameters for the ECG Upside-Down detection project.

Project: Upside-down ECG signal classification
Create time: Feb. 22, 2023
Author: Benjamin

"""

from .cnn_model_upsidedown import *
import argparse


parser = argparse.ArgumentParser(description='Parameters')

# 待檢查顛倒的ECG資料夾路徑設定
parser.add_argument('--EcgFolder_for_test',
                    default='./Folder_for_checking/',  # 確認
                    help="ECG data folder for checking weather signal is upside-down.")

# 載入模型設定
parser.add_argument('--ModelFrame', default=model_v1_1, help="Which model frame?")
parser.add_argument('--Load_ModelDate', default='20230217_1',  # 確認
                    help="Which date model to be loaded.")
parser.add_argument('--Load_ModelName', default='latest_model',   # 確認
                    help="Model name.")

# CNN模型設定
parser.add_argument('--CNN_norm_mean', default=0.459,
                    help='ECG normalization mean for CNN model. (model 20230217_1 = 0.459)')
parser.add_argument('--CNN_norm_std', default=0.286,
                    help='ECG normalization std for CNN model. (model 20230217_1 = 0.286)')

# 其餘設定
parser.add_argument('--Using_GPU', default=False, help='Use GPU or CPU to run CNN model?')

# 建立
### args = parser.parse_args() ### server 不可執行
args, unknown = parser.parse_known_args() ### sever 可執行