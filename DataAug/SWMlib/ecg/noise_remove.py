import numpy as np
##from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def remove_impulse(ecgs):
    ecgs_diff = np.diff(ecgs)
    j = 0
    for i, (ecg, ecg_diff) in enumerate(zip(ecgs[:-1], ecgs_diff)):
        if j == 0:
            if ecg_diff < -500:
                j = 1
                try:
                    while ecgs_diff[i + j] == 0:
                        j += 1
                except:
                    j -= 1

                if ecgs_diff[i + j] > 500:
                    for k in range(i + 1, i + j + 1): ecgs[k] = (ecgs[i] + ecgs[i + j + 1]) / 2

            elif ecg_diff > 700:
                j = 1
                try:
                    while ecgs_diff[i + j] == 0:
                        j += 1

                except:
                    j -= 1

                if ecgs_diff[i + j] < -700:
                    for k in range(i + 1, i + j + 1): ecgs[k] = (ecgs[i] + ecgs[i + j + 1]) / 2
        else:
            j = j - 1

    return ecgs




def remove_spike(ecg, spike_threshold=500.0, fs=250):
    
    """
    去突波處理
    
    參數：
        ecg: 輸入 ECG 訊號
        spike_threshold: 突波偵測閾值（預設500.0）
        fs: 取樣率（Hz）
    
    回傳：
        去除突波後的訊號
    """
       
    # 差分找突波
    diff_signal = np.diff(ecg)
    abs_diff = np.abs(diff_signal)
    spike_indices = np.where(abs_diff > spike_threshold)[0]
    
    spike_count = len(spike_indices)    
    
    # 修補突波
    if spike_count > 0:
        ecg_despike = ecg  ##.copy()
        for spike_idx in spike_indices:
            if spike_idx + 1 < len(ecg_despike):
                ecg_despike[spike_idx + 1] = ecg_despike[spike_idx]
    else:
        ecg_despike = ecg  ##.copy()
    
    return ecg_despike




def _estimate_noise_level(segment):
    """
    估計 ECG 片段的雜訊程度
    
    使用兩個指標來量化雜訊等級：
    1. 二階導數平均絕對值 (d2)：反映信號變化的劇烈程度
    2. 零交叉率 (ZCR)：反映信號振盪的頻繁程度
    
    參數:
        segment: ECG 信號片段 (numpy array)
        
    回傳:
        noise_score: 雜訊分數 (0.0~1.0)，數值越高表示雜訊越嚴重
    """
    N = len(segment)
    sig = np.asarray(segment, dtype=float)

    # 片段太短無法可靠評估
    if N < 5:
        return 0.0

    # === 指標 1: 二階導數 (Second Derivative) ===
    # 計算二階導數：反映信號曲率變化
    # EMG 雜訊會導致信號快速、不規則變化，使二階導數較大
    d2 = np.diff(sig, n=2)
    d2_mean = np.mean(np.abs(d2))
    
    # 正規化：將二階導數映射到 0~1 範圍
    # 分母 0.02 是經驗值，根據實際 ECG 數據調整
    d2_normalization_divisor = 0.02
    d2_score = np.clip(d2_mean / d2_normalization_divisor, 0, 1)

    # === 指標 2: 零交叉率 (Zero Crossing Rate) ===
    # 計算信號穿越均值的次數
    # EMG 雜訊會使信號頻繁振盪，導致零交叉率增加
    mean_val = np.mean(sig)
    zero_crossings = np.sum(np.diff(np.sign(sig - mean_val)) != 0)
    zcr = zero_crossings / N
    
    # 正規化：將 ZCR 映射到 0~1 範圍
    # 分母 0.15 是經驗閾值
    zcr_score = np.clip(zcr / 0.15, 0, 1)

    # === 加權組合兩個指標 ===
    w_d2, w_zcr = 0.60, 0.40
    noise_score = (w_d2 * d2_score + w_zcr * zcr_score)
    
    return float(noise_score)


def _evaluate_rr_intervals_noise_early_stop(ecg_normalized,r_peaks,low_threshold=0.25,high_threshold=0.60):

    """
    評估每個 RR 間距的雜訊等級，並進行早停優化
    
    策略：
    - 將相鄰兩個 RR 間距配對評估（提高評估穩定性）
    - 當高雜訊 RR 間距超過總數的 1/3 時提前停止（early stop）
    - 根據雜訊程度將信號分類為 low / middle / high
    
    參數:
        ecg_normalized: 標準化後的 ECG 信號
        r_peaks: Rpeak 位置索引 (numpy array)
        low_threshold: 低雜訊閾值（預設 0.25）
        high_threshold: 高雜訊閾值（預設 0.60）
        
    回傳:
        final_classification: 整體分類 ("low" / "middle" / "high")
        RR_scores: 每個 RR 間距的詳細評估結果 (list of dict)
    """
    r_peaks = np.asarray(r_peaks, dtype=int)

    # 計算總 RR 間距數量和早停閾值
    total_RR = len(r_peaks) - 1
    high_stop_threshold = total_RR * (1.0 / 3.0)  # 1/3

    # 初始化計數器
    low_count = 0      # 低雜訊 RR 間距數量
    high_count = 0     # 高雜訊 RR 間距數量
    middle_count = 0   # 中等雜訊 RR 間距數量

    RR_scores = []           # 儲存每個 RR 間距的評估結果
    early_stopped = False    # 早停標記
    final_classification = "middle"  # 預設分類

    # === 逐對評估 RR 間距 ===
    i = 0
    while i < total_RR:
        current_rr_idx = i
        next_rr_idx = i + 1

        # 提取當前 RR 間距的信號片段
        start1 = r_peaks[current_rr_idx]
        end1 = r_peaks[current_rr_idx + 1]
        # 避開 Rpeak 附近 20 個樣本點（避免 QRS 波群干擾）
        seg1 = ecg_normalized[start1 + 20 : end1 - 20]

        # 檢查是否有下一個 RR 間距可以配對
        has_pair = (next_rr_idx < total_RR)

        if has_pair:
            # 提取下一個 RR 間距的信號片段
            start2 = r_peaks[next_rr_idx]
            end2 = r_peaks[next_rr_idx + 1]
            seg2 = ecg_normalized[start2 + 20 : end2 - 20]
            
            # 將兩個片段對齊並相加（配對評估策略）
            min_len = min(len(seg1), len(seg2))
            combined_segment = seg1[:min_len] + seg2[:min_len]
        else:
            # 最後一個 RR 間距無法配對，單獨評估
            combined_segment = seg1

        # 計算組合片段的雜訊分數
        combined_noise_score = _estimate_noise_level(combined_segment)

        # === 根據雜訊分數分類 ===
        if combined_noise_score < low_threshold:
            # 低雜訊
            classification = "low"
            increment = 2 if has_pair else 1  # 配對時計數 +2
            low_count += increment
        elif combined_noise_score >= high_threshold:
            # 高雜訊
            classification = "high"
            increment = 2 if has_pair else 1
            high_count += increment
        else:
            # 中等雜訊
            classification = "middle"
            increment = 2 if has_pair else 1
            middle_count += increment

        # 記錄當前 RR 間距的評估結果
        RR_scores.append({
            "RR_interval_index": current_rr_idx + 1,           # RR 間距編號（從 1 開始）
            "start_r_peak_idx": int(current_rr_idx),           # 起始 Rpeak 索引
            "start_idx": int(start1),                          # 起始樣本點索引
            "end_idx": int(end1),                              # 結束樣本點索引
            "RR_interval_classification": classification,       # 分類結果
        })

        # 記錄配對的下一個 RR 間距（如果存在）
        if has_pair:
            RR_scores.append({
                "RR_interval_index": next_rr_idx + 1,
                "start_r_peak_idx": int(next_rr_idx),
                "start_idx": int(start2),
                "end_idx": int(end2),
                "RR_interval_classification": classification,
            })

        # === 早停檢查 ===
        # 如果高雜訊 RR 間距超過 1/3，提前判定為高雜訊信號
        if high_count > high_stop_threshold:
            final_classification = "high"
            early_stopped = True
            break

        # 跳過下一個（因為已經配對評估過）
        i += 2

    # === 最終分類判定 ===
    if not early_stopped:
        if low_count == total_RR:
            # 所有 RR 間距都是低雜訊
            final_classification = "low"
        else:
            # 混合情況：至少有一些中等或高雜訊
            final_classification = "middle"

    return final_classification, RR_scores


def emg_detector_remover(ecg_norm, r_peaks):
    """
    EMG 雜訊檢測與移除的主函式
    
    處理流程：
    1. 評估整體信號雜訊等級（low / middle / high）
    2. 根據雜訊等級採取不同策略：
       - high: 信號品質太差，無法處理 → 回傳 ([], [])
       - low: 信號乾淨，無需處理 → 回傳原始信號
       - middle: 部分區段有雜訊 → 局部 Savitzky-Golay 平滑處理
    
    參數:
        ecg_norm: 標準化後的 ECG 信號 (numpy array)
        r_peaks: Rpeak 位置索引 (numpy array)
        
    回傳:
        clean_signal: 去雜訊後的信號 (numpy array，中與低雜訊) 或 [](高雜訊)
        high_noise_rpeak_indices:                                                       
        若為低雜訊，high_noise_rpeak_indices為[]
        若為中雜訊，high_noise_rpeak_indices有數值
        若為高雜訊，clean_signal, high_noise_rpeak_indices都為[]

    """
    # === 參數設定 ===
    FS = 250  # 採樣率 (Hz)
    
    # 雜訊閾值
    LOW_NOISE_THRESHOLD = 0.30   # 低於此值視為低雜訊
    HIGH_NOISE_THRESHOLD = 0.50  # 高於此值視為高雜訊

    # Savitzky-Golay 濾波器參數
    SG_WINDOW_BASE = 11    # 基礎窗口大小（樣本點數）
    SG_POLY = 3            # 多項式階數
    SG_PROTECT_SEC = 0.04  # Rpeak 保護區間（秒）
    SG_TAPER_SEC = 0.05    # 漸變區間（秒）

    # === 步驟 1: 評估雜訊等級 ===
    RR_classification, RR_scores = _evaluate_rr_intervals_noise_early_stop(ecg_norm,r_peaks,low_threshold=LOW_NOISE_THRESHOLD,high_threshold=HIGH_NOISE_THRESHOLD)
    clean_signal=[]
    high_noise_rpeak_indices=[]
    # === 步驟 2: 根據分類採取對應策略 ===
    
    # 高雜訊 → 不處理
    if RR_classification == "high":
        return clean_signal, high_noise_rpeak_indices

    # 低雜訊 → 回傳原始信號
    if RR_classification == "low":
        return ecg_norm, high_noise_rpeak_indices

    # 中雜訊 → Savitzky-Golay 平滑
    clean_signal = ecg_norm

    # 將時間轉換為樣本點數
    p_samples = int(SG_PROTECT_SEC * FS)  # 保護區樣本數
    t_samples = int(SG_TAPER_SEC * FS)    # 漸變區樣本數

    # 逐個處理標記為 "middle" 的 RR 間距
    for rr_info in RR_scores:
        # 跳過非中等雜訊的區段
        if rr_info["RR_interval_classification"] != "middle":
            continue

        # 提取 RR 間距片段
        start_idx = int(rr_info["start_idx"])
        end_idx = int(rr_info["end_idx"])
        segment = clean_signal[start_idx:end_idx]
        N = len(segment)
        
        # 片段太短無法濾波
        if N < SG_POLY + 2:
            continue

        # 調整窗口大小（必須為奇數且 >= 多項式階數 + 2）
        window = SG_WINDOW_BASE
        if window >= N:
            window = (N // 2) * 2 - 1  # 調整為小於 N 的最大奇數
        if window < SG_POLY + 2:
            continue

        # === Savitzky-Golay 平滑濾波 ===
        # 保留信號趨勢的同時去除高頻雜訊
        smoothed = savgol_filter(segment, window_length=window, polyorder=SG_POLY)

        # 創建混合遮罩（Mask）
        # 目的：在 Rpeak 附近保留原始信號，中間區域使用平滑信號
        mask = np.zeros(N)

        # 左側（RR 間距起始）保護區與漸變區
        left_protect_end = min(N, p_samples)  # 保護區結束位置
        left_taper_end = min(N, left_protect_end + t_samples)  # 漸變區結束位置

        # 右側（RR 間距結束）保護區與漸變區
        right_protect_start = max(0, N - p_samples)  # 保護區開始位置
        right_taper_start = max(0, right_protect_start - t_samples)  # 漸變區開始位置

        # 設置保護區：mask = 1.0（完全使用原始信號）
        mask[:left_protect_end] = 1.0
        mask[right_protect_start:] = 1.0

        # 設置左側漸變區：mask 從 1.0 線性遞減到 0.0
        if left_taper_end > left_protect_end:
            mask[left_protect_end:left_taper_end] = np.linspace(
                1.0, 0.0, left_taper_end - left_protect_end
            )

        # 設置右側漸變區：mask 從 0.0 線性遞增到 1.0
        if right_protect_start > right_taper_start:
            mask[right_taper_start:right_protect_start] = np.linspace(
                0.0, 1.0, right_protect_start - right_taper_start
            )

        # === 混合原始信號與平滑信號 ===
        # 公式: output = mask * original + (1 - mask) * smoothed
        # - 保護區（mask=1）：使用原始信號
        # - 中間區（mask=0）：使用平滑信號
        # - 漸變區（0<mask<1）：平滑過渡
        clean_signal[start_idx:end_idx] = mask * segment + (1.0 - mask) * smoothed

    # === 收集高雜訊區段的 Rpeak 索引 ===
    high_noise_rpeak_indices = np.array(
        [
            item["start_r_peak_idx"]
            for item in RR_scores
            if item["RR_interval_classification"] == "high"
        ],
        dtype=int,
    )

    return clean_signal, high_noise_rpeak_indices


'''
def remove_baseline_drift(ecg, fs=250, first_window_sec=0.2, second_window_sec=0.6):
    
    """
    去基線飄移 - 使用 Savitzky-Golay 濾波器
    
    參數：
        ecg: 輸入 ECG 訊號
        fs: 取樣率（Hz）
        first_window_sec: 第一次濾波視窗大小（秒）
        second_window_sec: 第二次濾波視窗大小（秒）
    
    回傳：
        去除基線飄移後的訊號
    """

    
    ##print(f"    開始去基線飄移...")
    
    # 全程維持浮點，不要轉 int32
    x = np.asarray(ecg, dtype=np.float32)
    n = len(x)
    
    # 取得秒為單位的視窗設定，換成樣本點；確保 odd 且 < n
    def _odd_under_len(win_sec):
        w = int(round(max(5, win_sec * fs)))  # 至少 5 點
        if w % 2 == 0:
            w += 1
        if w >= n:
            # 視窗太大就縮到 n-1（需為 odd）
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
            if w < 5:
                w = 5 if n > 5 else (n - 1 if (n - 1) % 2 == 1 else max(3, n - 2))
        return w
    
    win1 = _odd_under_len(first_window_sec)   # 小視窗0.2（秒）
    win2 = _odd_under_len(second_window_sec)  # 大視窗0.6（秒）
    
    ##print(f"    第一次 SG 濾波，視窗={win1} 點")
    baseline1 = savgol_filter(x, window_length=win1, polyorder=2, mode='interp')
    
    ##print(f"    第二次 SG 濾波，視窗={win2} 點")
    baseline2 = savgol_filter(baseline1, window_length=win2, polyorder=2, mode='interp')
    
    # 用浮點相減
    ecg_clean = x - baseline2
    
    ##print(f"    去基線完成：原始範圍[{x.min():.3f}, {x.max():.3f}] → 清理後[{ecg_clean.min():.3f}, {ecg_clean.max():.3f}]")
    
    return ecg_clean


def estimate_noise_level(ecg, fs=250):
    
    """
    估計訊號的雜訊程度
    
    input: 
    ecg: 正規化[0,1]之間的10秒ECG訊號
    fs: ECG取樣率

    output:
    noise_level_flag: 0(正常無EMG雜訊),1(中等EMG雜訊),2(嚴重EMG雜訊)
    """

    LOW_NOISE_THRESHOLD = 0.05 # 低雜訊的上限值
    HIGH_NOISE_THRESHOLD = 0.11 # 高雜訊的下限值

    # (1) 一階導數 STD
    derivative = np.gradient(ecg)
    derivative_std = np.std(derivative)
    derivative_score = np.clip(derivative_std / 50, 0, 1)

    # (2) 一階差分 STD
    diff_signal = np.diff(ecg)
    diff_std = np.std(diff_signal)
    diff_score = np.clip(diff_std / 50, 0, 1)

    # (3) 零交叉率（快速振盪代表雜訊）
    zero_crossings = np.sum(np.diff(np.sign(ecg - np.mean(ecg))) != 0)
    zcr = zero_crossings / len(ecg)
    zcr_score = np.clip(zcr / 0.15, 0, 1)

    # 綜合雜訊評分
    noise_score = 0.4 * derivative_score + 0.4 * diff_score + 0.2 * zcr_score
    
    if noise_score<LOW_NOISE_THRESHOLD:
        noise_level_flag=0
    elif(noise_score>=HIGH_NOISE_THRESHOLD):
        noise_level_flag=2
    else:
        noise_level_flag=1

    return noise_level_flag    


def remove_emg_noise(ecg, r_peaks, fs=250): 
    
    """
    ECG 肌電雜訊去除
    
    參數：
        ecg: 原始 ECG 訊號 (numpy array)
        fs: 取樣率 (Hz)
        
    
    流程：
        1. EMD 分解 → 多個 IMF
        2. 估計前 P 個 IMF 為噪訊
        3. 偵測 R peak 位置
        4. 建立 Tukey 保護窗（保護 QRS）
        5. 混合 IMF 重建乾淨訊號
    """
    
   
    noise_level_flag=estimate_noise_level(ecg)

    if (noise_level_flag !=1): ##沒有EMG或很嚴重EMG雜訊，不修正
        return noise_level_flag,ecg
    
    else: ##中等EMG雜訊，要清理

        alpha=0.006 ##噪訊判斷閾值,實驗後的參數,在這個數值以下才視為雜訊需要去除
        a_table=[0.05, 0.1, 0.1, 0.18] ##各 IMF 保留係數,實驗後的參數 
        qrs_width=0.25 ## qrs_width: QRS 波寬度 (秒),實驗後的參數
        
        x = np.asarray(ecg, dtype=float)
        N, dc = len(x), np.mean(x)
        x0 = x - dc
    
        imfs = EMD()(x0)          
        ##imfs = EMD(max_imf=5)(x0)
        P = _estimate_noise_order(imfs, alpha)  ##估計噪訊 IMF 數量 P
        
        qrs_w = int(qrs_width * fs)  # QRS 寬度（樣本數）
        
    
        #建立 Tukey 保護窗  
        psi_list = []
        for i in range(1, P+1):
            # 每個 IMF 的窗長度與錐形比例不同
            L = qrs_w + int(0.4*i*qrs_w)
            r = min(0.95, 0.45*i)
            tw = _tukey_window(L, r) 
            
            # 在每個 R peak 位置放置保護窗
            psi = np.zeros(N)
            half = L // 2
            for rp in r_peaks:
                s, e = max(0, rp-half), min(N, rp-half+L)
                seg_len = e - s
                # 完整窗或邊界縮短窗
                if seg_len == L:
                    psi[s:e] = np.maximum(psi[s:e], tw)
                else:
                    tw_resized = np.interp(np.linspace(0, L-1, seg_len), np.arange(L), tw)
                    psi[s:e] = np.maximum(psi[s:e], tw_resized)
            psi_list.append(psi)
        
        # 混合 IMF 重建訊號  
        out = np.zeros_like(x0)
        
        # 前 P 個 IMF（噪訊）：QRS 處保留，非 QRS 處衰減
        for i in range(P):
            ai = a_table[i] if i < len(a_table) else a_table[-1]
            out += psi_list[i]*imfs[i] + ai*(1-psi_list[i])*imfs[i]            
    
        out += imfs[P:].sum(axis=0)       
        # 加回 DC 偏移
        filtered_ecg=out + dc

        return noise_level_flag,filtered_ecg


def _estimate_noise_order(imfs, alpha=0.006): 
    
    """
    估計噪訊 IMF 數量 P (Abdollahpoor方法)
    
    原理：累加 IMF1~IMFM，若 |mean(累加值)| > alpha，則 P = M-1
    表示前 M-1 個 IMF 屬於噪訊成分
    """
    csum = np.zeros(imfs.shape[1])
    for M in range(1, imfs.shape[0] + 1):
        csum += imfs[M-1]
        mean_abs = abs(np.mean(csum))
               
        if mean_abs > alpha:
            return max(0, M-1)
    return 1

def _tukey_window(L, r):
    
    """
    產生 Tukey window
    
    參數：
        L: 窗長度
        r: 錐形比例 (0~1)，0=矩形窗，1=Hann窗
    
    用途：保護 QRS 波段，邊緣平滑過渡，中間保持原值
    """
    L, r = int(L), np.clip(r, 0, 1)
    if L <= 1 or r == 0:
        return np.ones(L)
    
    n, w = np.arange(L), np.ones(L)
    edge = int(r * (L-1) / 2)  # 計算錐形邊緣長度
    if edge > 0:
        # 前緣：餘弦過渡從 0→1
        w[:edge] = 0.5 * (1 + np.cos(np.pi * (2*n[:edge]/(r*(L-1)) - 1)))
        # 後緣：餘弦過渡從 1→0
        w[-edge:] = 0.5 * (1 + np.cos(np.pi * (2*(n[-edge:]-(L-1))/(r*(L-1)) + 1)))
    return w
'''

        
            
    