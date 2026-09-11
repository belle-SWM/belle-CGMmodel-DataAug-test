# ============================================================
# 共用函式 - M1~M5.py、Fusion.py、Regression_Model_Predictor_meta_OOF.py 都會 import 這個檔案
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.fftpack import fft
import pywt

# 版本號
def get_version():
    return '2.1.3'

# ─── GlucoseData / Category_Features CSV 輔助函式 ──────────────────────
# 資料格式：
#   Model/<splitting_ratio>/GlucoseData/<uuid>/<uuid>_ecg_waveforms.csv
#       欄位：source_file, uuid, timestamp, type, index, glucose, split(train/valid/test),
#             BG_Level(Normal/High/Low), group_id, split_method, status, ecg_0 ~ ecg_149
#   Model/<splitting_ratio>/Category_Features/<uuid>/<uuid>_category_features.csv
#       欄位：與上面相同的 meta 欄位 + 一批 ECG 回歸特徵欄位（例如 rr_interval, hr, ...）
# 兩者透過 source_file 欄位一一對應，各自代表同一批樣本的「訊號」與「回歸特徵」視角。
#
# 這裡刻意不定義任何回歸特徵清單（例如 USED_FEATURE_DIC）——要用哪些回歸特徵
# 是模型層（M3.py）的決定，由呼叫端透過 ECGDataset(..., reg_feature_names=[...]) 注入，
# 避免 common.py 反過來 import M3.py 造成循環引用。

_META_COLS = {
    'source_file', 'uuid', 'timestamp', 'type', 'index', 'glucose',
    'split', 'BG_Level', 'group_id', 'split_method', 'status',
}


# 回傳指定 uuid 的 ecg_waveforms.csv 完整路徑
def get_ecg_csv_path(basepath, splitting_ratio, uuid):
    return os.path.join(basepath, splitting_ratio, "GlucoseData", str(uuid), f"{uuid}_ecg_waveforms.csv")


# 回傳指定 uuid 的 category_features.csv 完整路徑
def get_category_features_csv_path(basepath, splitting_ratio, uuid):
    return os.path.join(basepath, splitting_ratio, "Category_Features", str(uuid), f"{uuid}_category_features.csv")


# 回傳指定 uuid 的 regression_features.csv 完整路徑
def get_regression_features_csv_path(basepath, splitting_ratio, uuid):
    return os.path.join(basepath, splitting_ratio, "Regression_Features", str(uuid), f"{uuid}_regression_features.csv")


# 嘗試把 dir_path 解析成新格式 CSV 的路徑（ecg_waveforms.csv 或 category_features.csv）：
# - dir_path 本身就是 .csv 檔案 -> 直接回傳
# - dir_path 是資料夾，底下有 *_ecg_waveforms.csv 或 *_category_features.csv -> 回傳該檔案
# - 都不是（例如舊格式 Normal/High/Low 資料夾）-> 回傳 None
def resolve_ecg_csv_path(dir_path):
    if not isinstance(dir_path, str):
        return None
    if dir_path.lower().endswith('.csv') and os.path.isfile(dir_path):
        return dir_path
    if os.path.isdir(dir_path):
        for pattern in ('*_ecg_waveforms.csv', '*_category_features.csv'):
            matches = sorted(glob.glob(os.path.join(dir_path, pattern)))
            if matches:
                return matches[0]
    return None


# 讀取 ecg_waveforms.csv，回傳指定 split（train/valid/test，None=全部）下
# 每個 BG_Level 的筆數，例如 {'Normal': 100, 'High': 20}
def csv_split_class_counts(csv_path, split=None):
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path, usecols=['split', 'BG_Level'])
    if split is not None:
        if isinstance(split, (list, tuple, set)):
            df = df[df['split'].isin(split)]
        else:
            df = df[df['split'] == split]
    return df['BG_Level'].value_counts().to_dict()


# PyTorch Dataset：從新格式 ecg_waveforms.csv / category_features.csv 讀取 ECG 訊號、
# 血糖值、分類標籤與回歸特徵，供 M1~M5.py 訓練/評估使用。
class ECGDataset(Dataset):
    # dir_path:
    #     * GlucoseData/<uuid>/<uuid>_ecg_waveforms.csv（含 ecg_0~ecg_149 訊號欄位）
    #     * Category_Features/<uuid>/<uuid>_category_features.csv（不含訊號，含回歸特徵欄位）
    #     可傳 .csv 檔案路徑，或包含該 csv 檔的資料夾。
    # split:
    #     'train' / 'valid' / 'test'，或多個 split 組成的 list/tuple（例如 ['train','valid']）。
    #     None 代表不篩選 split（全部資料）。
    # reg_feature_names:
    #     要從 CSV 抓哪些欄位當回歸特徵（例如 M3.py 的 REGRESSION_FEATURE_NAMES）。
    #     - 未指定 -> 自動抓 CSV 中除了 meta 欄位、ecg_* 欄位以外的所有數值欄位。
    #     - 有指定 -> 只抓指定的欄位（CSV 缺欄位就補 0，並印警告）。
    # df：可選，直接傳入已經在記憶體裡處理好的 DataFrame（例如血糖點數縮減後的
    #     子集），跳過讀檔；此時 dir_path 可以不給。df 內容格式需跟 ecg_waveforms.csv
    #     一致（含 ecg_0~ecg_149、BG_Level、split 等欄位）。
    def __init__(self, dir_path=None, method='time', classes=3, split=None, reg_feature_names=None, df=None):
        self.method = method
        self.data_len = 150
        self.channel = 1
        self.split = split

        if df is not None:
            self.csv_path = None
            self.dir_path = None
            self._load_from_df(df.copy(), classes=classes, split=split, reg_feature_names=reg_feature_names)
            return

        csv_path = resolve_ecg_csv_path(dir_path)
        self.csv_path = csv_path
        if csv_path is None:
            raise FileNotFoundError(
                f"ECGDataset: 在 {dir_path!r} 找不到 *_ecg_waveforms.csv 或 *_category_features.csv")

        self.dir_path = csv_path
        self._load_from_csv(csv_path, classes=classes, split=split, reg_feature_names=reg_feature_names)

    # 從新格式 CSV（ecg_waveforms.csv 或 category_features.csv）讀取訊號、血糖值、標籤、回歸特徵
    def _load_from_csv(self, csv_path, classes, split, reg_feature_names=None):
        df = pd.read_csv(csv_path)
        self._load_from_df(df, classes=classes, split=split, reg_feature_names=reg_feature_names)

    # 共用主體：df 已經是完整的（或已篩過 split 的）DataFrame，處理訊號/標籤/回歸特徵
    def _load_from_df(self, df, classes, split, reg_feature_names=None):

        if split is not None:
            if isinstance(split, (list, tuple, set)):
                df = df[df['split'].isin(split)]
            else:
                df = df[df['split'] == split]

        class_names = ['Normal', 'High'] if classes == 2 else ['Normal', 'High', 'Low']
        label_map = {name: idx for idx, name in enumerate(class_names)}

        df = df[df['BG_Level'].isin(class_names)].reset_index(drop=True)

        ecg_cols = [f'ecg_{i}' for i in range(self.data_len)]
        has_signal = all(c in df.columns for c in ecg_cols)

        if has_signal:
            raw_signals = df[ecg_cols].to_numpy(dtype=np.float64) if len(df) > 0 else np.zeros((0, self.data_len))
            processed = [self._transform_signal(self._pad_to_len(row)) for row in raw_signals]
            if processed:
                Sig = torch.FloatTensor(processed)
                if self.channel == 1 and Sig.dim() == 2:
                    Sig = Sig.unsqueeze(1)
            else:
                Sig = torch.zeros((0, self.channel, self.data_len))
        else:
            # 例如 category_features.csv：沒有訊號欄位，固定回傳全 0 訊號
            self.channel = 1
            Sig = torch.zeros((len(df), 1, self.data_len))

        self.Signals = Sig
        labels_np = df['BG_Level'].map(label_map).to_numpy(dtype=np.float64).reshape(-1, 1)
        self.Labels = torch.from_numpy(labels_np)
        self.GlucoseValues = df['glucose'].tolist() if 'glucose' in df.columns else [-1] * len(df)
        if 'source_file' in df.columns:
            self.Filenames = df['source_file'].astype(str).tolist()
        else:
            self.Filenames = [str(i) for i in range(len(df))]

        # ─ 回歸特徵：直接從同一份 CSV 取值（新格式不再逐檔讀取）──────────────────
        if reg_feature_names is not None:
            names = list(reg_feature_names)
            missing = [c for c in names if c not in df.columns]
            if missing:
                print(f"[ECGDataset] 警告：CSV 缺少下列回歸特徵欄位，將以 0 補齊：{missing}")
        else:
            candidate_cols = [c for c in df.columns if c not in _META_COLS and c not in ecg_cols]
            names = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

        self.RegFeatureNames = names
        if names and len(df) > 0:
            reg_df = pd.DataFrame({name: (df[name] if name in df.columns else 0.0) for name in names})
            reg_np = reg_df.to_numpy(dtype=np.float64)
        else:
            reg_np = np.zeros((len(df), len(names)))
        self.RegFeatures = torch.nan_to_num(
            torch.tensor(reg_np, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # 計算回歸特徵各欄位的平均值與標準差（標準差下限 1e-8，避免除以 0），供 apply_reg_zscore 使用
    def get_reg_feature_stats(self):
        mean = self.RegFeatures.mean(dim=0)
        std = self.RegFeatures.std(dim=0)
        std = torch.clamp(std, min=1e-8)
        return mean, std

    # 用指定的 mean/std 對 RegFeatures 做 z-score 標準化
    def apply_reg_zscore(self, mean, std):
        self.RegFeatures = (self.RegFeatures - mean) / std

    # 依索引回傳單筆樣本 (訊號, 回歸特徵, 標籤, 檔名)
    def __getitem__(self, index):
        signal = self.Signals[index]
        reg_feat = self.RegFeatures[index]
        label = self.Labels[index]
        filename = self.Filenames[index]
        return signal, reg_feat, label, filename

    # 取得指定索引的血糖值
    def get_glucose(self, index):
        return self.GlucoseValues[index]

    # 取得指定索引的檔名
    def get_filename(self, index):
        return self.Filenames[index]

    # 回傳資料集筆數
    def __len__(self):
        return len(self.Labels)

    # 將訊號補齊/截斷成固定長度 data_len（長度不足補 0，超過則截斷）
    def _pad_to_len(self, values):
        sig = np.zeros(self.data_len)
        n = min(len(values), self.data_len)
        sig[:n] = values[:n]
        return sig

    # 依 self.method 對訊號做前處理：raw=原始值、time=正規化、freq=正規化後做 DWT、
    # combine=正規化後把 DWT 係數與原始正規化訊號合併成多通道
    def _transform_signal(self, sig):
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

        if self.method == 'combine':
            sig = self.normalize1(sig)
            x1 = sig.tolist()
            #x2 = self.FFT(sig).tolist()
            x3 = self.DWT(sig).tolist()
            #Comb = [x1, x2, x3]
            x3.append(x1)
            self.channel = len(x3)
            return x3

    # 對訊號做 min-max 正規化（分母多加 1，避免 x_max == x_min 時除以 0）
    def normalize1(self, x):
        x_max = np.max(x)
        x_min = np.min(x)
        x_norm = (x - x_min) / (x_max - x_min+1)
        return x_norm


    # 對訊號做 FFT 轉換，回傳正規化後的頻譜幅值 P2
    def FFT(self, x):
        y = fft(x)
        P2 = abs(y) / len(x)
        P1 = P2[range(len(x)//2)]
        return P2

    # 對訊號做離散平穩小波轉換（SWT，sym4，level=1），回傳係數陣列
    def DWT(self, x):
        ##sig = np.zeros(2504)
        sig = np.zeros(150)
        sig[:len(x)] = x
        coeffs = pywt.swt(sig, 'sym4', level=1, trim_approx=True)
        coeffs = np.array(coeffs)

        return coeffs


# 對模型各層做權重初始化：Conv1d 用 kaiming_normal_，Linear 用 kaiming_uniform_，
# 兩者的 bias 皆初始化為 0；BatchNorm1d 的權重初始化為 1、bias 初始化為 0
def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, nn.Conv1d):  # 判斷是否為Conv1d
            ##torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None: # 加上偏差初始化
                torch.nn.init.zeros_(m.bias.data)

        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)


# Focal Loss，自動支援：
# 1. Binary classification: logits shape [N] or [N,1]
# 2. Multi-class classification: logits shape [N,C], C >= 2
#
# 參數：
# - alpha:
#     * binary: 可給 float，例如 0.25
#     * multi-class: 可給 tensor/list，例如 [1.0, 2.0, 2.0]
# - gamma: focal loss 的 focusing parameter
# - reduction: "mean" / "sum" / "none"
class FocalLoss(nn.Module):
    # 儲存 alpha / gamma / reduction / label_smoothing 等超參數
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    # logits:
    #     - binary: [N] or [N,1]
    #     - multi-class: [N,C]
    #
    # targets:
    #     - binary: [N] or [N,1], values in {0,1}
    #     - multi-class: [N], values in {0,1,...,C-1}
    def forward(self, logits, targets):
        # -------------------------
        # Case 1：二元分類（Binary classification）
        # logits.shape = [N] or [N,1]
        # -------------------------
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
            logits = logits.view(-1)
            targets = targets.float().view(-1)
            
            if self.label_smoothing > 0:
                # 將 1 變為 1 - smoothing/2 (例如 0.95)，0 變為 smoothing/2 (例如 0.05)
                targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            pt = torch.exp(-bce)

            if self.alpha is None:
                alpha_t = 1.0
            else:
                # binary alpha: 建議給 float，例如 0.25
                alpha = float(self.alpha)
                alpha_t = torch.where(
                    targets > 0.5,
                    torch.full_like(targets, alpha),
                    torch.full_like(targets, 1 - alpha)
                )

            fl = alpha_t * ((1 - pt) ** self.gamma) * bce

        # -------------------------
        # Case 2：多類別分類（Multi-class classification）
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
        # 依 reduction 設定彙整結果
        # -------------------------
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        elif self.reduction == "none":
            return fl
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

# 各類別樣本數平均的 batch sampler：每個 batch 從每個類別各抽取相同數量的樣本
# （不足時可重複抽樣 replace=True），再打亂順序，用於類別不平衡的訓練情境。
# 修正版避開三個舊寫法的地雷（大資料量下偶發 access violation 的來源）：
# (1) 每個 batch 都把大 list 轉 numpy → 改成 __init__ 一次轉好；
# (2) yield numpy int64 純量 → 改成純 Python int；
# (3) 舊版全域 np.random → 改用較穩的 default_rng。
class BalancedBatchSampler:
    # 依 batch_size 與類別數算出每類別要抽的樣本數，並把各類別索引「一次」轉成
    # numpy 陣列存起來（不要每個 batch 再從 list 轉）；濾掉沒有樣本的類別，
    # 並保底至少 1 個 batch，避免資料量小於一個 batch 時靜默跑出 0 個 batch。
    def __init__(self, labels, batch_size):
        self.labels = np.asarray(labels)
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        self.n_samples_per_class = max(1, batch_size // self.n_classes)
        self.batch_size = self.n_samples_per_class * self.n_classes
        self.class_indices = [np.where(self.labels == c)[0] for c in self.classes]
        self.class_indices = [a for a in self.class_indices if len(a) > 0]
        self.n_batches = max(1, len(self.labels) // self.batch_size)

    # 逐一產生 n_batches 個 batch：用整數位置抽樣（replace=True）直接對已存好的
    # numpy 陣列取值，shuffle 後轉成純 Python int 再 yield（最保險，避免 numpy 純量跨行程傳遞出包
    def __iter__(self):
        rng = np.random.default_rng()
        n = self.n_samples_per_class
        for _ in range(self.n_batches):
            parts = [a[rng.integers(0, len(a), size=n)] for a in self.class_indices]
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield batch.tolist()

    # 回傳總 batch 數
    def __len__(self):
        return self.n_batches


# 依照類別分層切分 train / valid，避免 valid 幾乎沒有 minority class
# labels_array: 1D numpy array
def train_valid_split_indices(labels_array, valid_ratio=0.1, seed=10):
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


# ─── 閾值移動（Threshold Moving）共用工具 ──────────────────────────────────
# 在 Valid split 上，掃描候選 threshold，挑一個讓 G-mean（各類別 Sensitivity /
# Specificity 的幾何平均）最大化的 threshold，取代預設的 0.5 / argmax，用於類別
# 不平衡（例如血糖 High/Low 樣本遠少於 Normal）時的後處理決策邊界調整。
# M1~M5.py 訓練流程共用；邏輯與 Fusion.py 的 ProbSumTh 系列一致。

# 依機率矩陣與 threshold 規則決定預測類別：
# - threshold=None -> 直接 argmax（沒有閾值移動時的原始行為）
# - n_classes==2   -> probs[:,1] >= threshold 判定為類別 1，否則類別 0
# - n_classes>=3   -> 機率 >= threshold 的類別視為「命中」；多個同時命中時
#                     Low(2) > High(1) > Normal(0) 優先；都沒命中則退回 argmax
# probs: (N, n_classes) 的機率矩陣（numpy array 或可轉換為 numpy 的物件）
def apply_threshold_prediction(probs, n_classes, threshold=None):
    probs = np.asarray(probs)
    n = probs.shape[0]
    if threshold is None:
        return np.argmax(probs, axis=1).astype(int)

    y_pred = np.zeros(n, dtype=int)
    for i in range(n):
        ratios = probs[i]
        if n_classes == 2:
            y_pred[i] = 1 if float(ratios[1]) >= threshold else 0
        else:
            above = [c for c in range(n_classes) if ratios[c] >= threshold]
            if len(above) == 1:
                y_pred[i] = above[0]
            elif len(above) > 1:
                pred = 0
                for idx in above:
                    if idx == 2:
                        pred = 2
                        break
                    elif idx == 1:
                        pred = 1
                y_pred[i] = pred
            else:
                y_pred[i] = int(np.argmax(ratios))
    return y_pred


# 計算各類別 Sensitivity / Specificity 的幾何平均後，再取整體 G-mean
# （= sqrt(OverallSens * OverallSpec)），跟 Fusion.py compute_metrics() 的 Gmean 定義一致。
def compute_gmean(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    epsilon = 1e-6

    sens_list, spec_list = [], []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))

        sens = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = (100.0 * tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sens_list.append(sens + epsilon)
        spec_list.append(spec + epsilon)

    overall_sens = float(np.prod(sens_list) ** (1.0 / n_classes))
    overall_spec = float(np.prod(spec_list) ** (1.0 / n_classes))
    return (overall_sens * overall_spec) ** 0.5


# 在候選 threshold 中，挑一個讓 apply_threshold_prediction 後的 G-mean 最大的。
# candidates 預設為 [None(=argmax)] + 0.10, 0.15, ..., 0.90。
# 回傳 (best_threshold, best_gmean)。
def find_best_gmean_threshold(probs, y_true, n_classes, candidates=None):
    if candidates is None:
        candidates = [None] + [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]

    best_th, best_gmean = None, -1.0
    for th in candidates:
        preds = apply_threshold_prediction(probs, n_classes, th)
        gmean = compute_gmean(y_true, preds, n_classes)
        if gmean > best_gmean:
            best_gmean = gmean
            best_th = th
    return best_th, best_gmean


# 從訓練/測試 ECGDataset 與測試集逐筆預測結果，組出 Generate_Classification_Analysis_Report 需要的資料結構。
# class_names: {int_label: 類別名稱字串}，例如 {0:'Normal',1:'High',2:'Low'}。
# 回傳: (train_glucose_values, train_glucose_by_class, test_glucose_values, test_glucose_by_class,
#        prediction_results, correct_glucose, wrong_glucose, wrong_predictions)
def build_analysis_report_inputs(traindata, testdata, filenames, predict_list, target_list, class_names):
    train_glucose_values = list(traindata.GlucoseValues)
    train_glucose_by_class = {name: [] for name in class_names.values()}
    for g, lbl in zip(traindata.GlucoseValues, traindata.Labels.view(-1).tolist()):
        train_glucose_by_class[class_names[int(lbl)]].append(g)

    filename_to_glucose = dict(zip(testdata.Filenames, testdata.GlucoseValues))

    prediction_results = []
    test_glucose_by_class = {name: [] for name in class_names.values()}
    for idx, (fname, pred, targ) in enumerate(zip(filenames, predict_list, target_list)):
        true_cls = class_names[int(targ)]
        pred_cls = class_names[int(pred)]
        glucose = filename_to_glucose.get(fname, -1)
        test_glucose_by_class[true_cls].append(glucose)
        prediction_results.append({
            '編號': idx, '檔名': fname,
            '真實類別': true_cls, '預測類別': pred_cls,
            '血糖值': glucose,
            '結果': '正確' if pred_cls == true_cls else '錯誤',
        })

    test_glucose_values = [p['血糖值'] for p in prediction_results]
    correct_glucose = [p['血糖值'] for p in prediction_results if p['結果'] == '正確']
    wrong_glucose = [p['血糖值'] for p in prediction_results if p['結果'] == '錯誤']
    wrong_predictions = [
        {k: p[k] for k in ('編號', '檔名', '真實類別', '預測類別', '血糖值')}
        for p in prediction_results if p['結果'] == '錯誤'
    ]
    return (train_glucose_values, train_glucose_by_class, test_glucose_values, test_glucose_by_class,
            prediction_results, correct_glucose, wrong_glucose, wrong_predictions)



# ─── 訓練歷程記錄 ──────────────────────────────────────────────────
# 訓練迴圈裡累積的 Loss_list / ACCs 原本只存在記憶體，跑完就沒了，
# 只能從 console 的 print 回頭翻。這裡把它們落地成 CSV + 曲線圖，
# 跟 .pth checkpoint 放在同一個 save_path 底下。
#
#   <uuid>_<tag>_training_history.csv   欄位：epoch, train_loss, valid_loss, train_acc, valid_acc
#   <uuid>_<tag>_training_curve.png     上圖 loss、下圖 accuracy
#
# start_epoch 用來支援續訓(Epoch_range[0] != 0)：傳 Epoch_range[0]+1 進來，
# epoch 編號才會接續，CSV 也會保留前一次的紀錄而不是整個蓋掉。
def _history_value(v):
    ###train_acc/valid_acc 在不同迴圈裡可能是 float、numpy 純量或還沒 .item() 的 torch tensor，這裡統一轉成 float
    try:
        if hasattr(v, 'item'):
            return float(v.item())
        return float(v)
    except Exception:
        return float('nan')


def save_training_history(save_path, uuid, loss_list, accs, tag='', start_epoch=1):
    if not loss_list:
        return None, None

    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)
    prefix = f"{uuid}_{tag}" if tag else str(uuid)
    csv_path = os.path.join(save_path, prefix + "_training_history.csv")
    png_path = os.path.join(save_path, prefix + "_training_curve.png")

    rows = []
    for i, pair in enumerate(loss_list):
        acc_pair = accs[i] if i < len(accs) else (float('nan'), float('nan'))
        rows.append({
            'epoch': start_epoch + i,
            'train_loss': _history_value(pair[0]),
            'valid_loss': _history_value(pair[1]),
            'train_acc': _history_value(acc_pair[0]),
            'valid_acc': _history_value(acc_pair[1]),
        })
    df = pd.DataFrame(rows)

    ###續訓時把舊紀錄接回來，重跑到的 epoch 以這次的結果為準
    if start_epoch > 1 and os.path.exists(csv_path):
        try:
            old = pd.read_csv(csv_path)
            old = old[old['epoch'] < start_epoch]
            df = pd.concat([old, df], ignore_index=True)
        except Exception as e:
            print("讀取舊的 training history 失敗，這次直接覆寫：%s" % e)

    df.to_csv(csv_path, index=False)
    print("Training history saved: %s" % csv_path)

    ###畫圖失敗(例如沒有 matplotlib、字型問題)不該讓整個訓練掛掉，所以包 try
    try:
        import matplotlib
        matplotlib.use('Agg')  ###伺服器/容器沒有顯示器，固定用非互動式 backend
        import matplotlib.pyplot as plt

        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        ax_loss.plot(df['epoch'], df['train_loss'], marker='o', markersize=3, label='Train loss')
        ax_loss.plot(df['epoch'], df['valid_loss'], marker='o', markersize=3, label='Valid loss')
        ###標出 valid loss 最低的那個 epoch，也就是 early stopping 真正選中的模型
        if df['valid_loss'].notna().any():
            best = df.loc[df['valid_loss'].idxmin()]
            ax_loss.axvline(best['epoch'], color='gray', linestyle='--', linewidth=1)
            ###最佳 epoch 靠近右邊界時，標註改放到虛線左側，避免被圖框切掉
            on_right = best['epoch'] > (df['epoch'].min() + df['epoch'].max()) / 2
            ax_loss.annotate("best epoch %d\nvalid loss %.4f" % (int(best['epoch']), best['valid_loss']),
                             xy=(best['epoch'], best['valid_loss']),
                             xytext=(-5 if on_right else 5, 12), textcoords='offset points',
                             ha='right' if on_right else 'left', fontsize=8, color='gray')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title("%s  training history" % prefix)
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        ax_acc.plot(df['epoch'], df['train_acc'], marker='o', markersize=3, label='Train ACC')
        ax_acc.plot(df['epoch'], df['valid_acc'], marker='o', markersize=3, label='Valid ACC')
        ax_acc.set_xlabel('Epoch')
        ax_acc.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  ###epoch 是整數，不要出現 1.5 這種刻度
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print("Training curve saved: %s" % png_path)
    except Exception as e:
        print("畫 training curve 失敗(CSV 已存檔，不影響訓練)：%s" % e)
        png_path = None

    return csv_path, png_path
