# 2026/09/08 開發環境設定紀錄

## 今天做了什麼
沒有修改任何模型/程式邏輯，只是把專案的執行環境建立起來：

1. 建立虛擬環境 `venv/`（使用 Python 3.9，路徑：`C:\Users\belle\AppData\Local\Programs\Python\Python39\python.exe`）
2. 掃描 [Model_Builder_Predictor_Belle.py](Model_Builder_Predictor_Belle.py)、[Data_Parsing_Belle.py](Data_Parsing_Belle.py)、[SWMlib/](SWMlib/) 用到的所有 import，整理出 [requirements.txt](requirements.txt) 並安裝
3. 實際 `import SWMlib` 跑過一次，抓出 3 個套件版本不相容的問題，並把版本鎖在 requirements.txt：

   | 套件 | 裝到的版本會壞 | 改鎖版本 | 原因 |
   |---|---|---|---|
   | neurokit2 | 0.2.12 | **0.2.9** | 0.2.12 的 `read_xdf.py` 用了 `float \| None` 這種新版 type hint 語法，Python 3.9 不支援，import 就會炸 |
   | nolds | 0.6.3 | **0.5.2** | 0.6.3 在 `nolds/datasets.py` 用 `importlib.resources.files(__name__)`，在 Python 3.9 上對非 package 的 module 會丟 `TypeError: 'nolds.datasets' is not a package` |
   | scipy | 1.13.1 | **1.10.1** | `SWMlib/motion/ahrs/filters/flae.py` 有 `from scipy import sqrt`，這個 numpy 別名在 scipy 1.12+ 已經被移除，1.10.1 還保留（雖然會有 deprecation warning） |

   目前 [requirements.txt](requirements.txt) 是用 `pip freeze` 產生的完整鎖定版本，之後如果要重建環境，直接用它安裝即可。

4. 建立 git 追蹤（只追蹤程式碼，不追蹤 data/model/csv）：

   ```powershell
   cd C:\Users\belle\Documents\algorithm_blood_glucose-main
   git init
   git config user.name "Belle"
   git config user.email "belle@singularwings.com"
   ```

   （只設定在這個 repo，不是全域設定，所以不會影響其他專案的 git 身份）

   接著建立 [.gitignore](.gitignore)，排除以下內容，讓 `git status` 只會看到真正的程式碼變動：

   ```gitignore
   # 虛擬環境
   venv/

   # Python 快取
   __pycache__/
   *.pyc

   # 資料與模型輸出（體積大、常變動，不進版控）
   DataDB/
   DataDB_fromNEW/
   GlucoseDataCSV/
   Model/
   Model_fromNEW/

   # 資料檔案格式
   *.csv
   *.json
   *.zip
   *.xlsx

   # 訓練好的模型權重檔（體積大，不進版控）
   *.pth

   # IDE
   .vscode/
   .idea/
   desktop.ini
   ```

   最後做第一次 commit：

   ```powershell
   git add -A
   git commit -m "Initial commit: track source code, exclude data/model/venv artifacts"
   ```

   目前只有這一個本地 commit，**沒有設定任何遠端 remote，也沒有 push 到任何地方**。

## 之後每次要跑程式前

**一定要先啟動虛擬環境**，不然會用到系統的 Python，套件版本不對就會出現上面那些 import 錯誤。

在 PowerShell：
```powershell
cd C:\Users\belle\Documents\algorithm_blood_glucose-main
.\venv\Scripts\Activate.ps1
```

啟動後看到提示字元前面出現 `(venv)` 就表示成功了。

要離開虛擬環境時：
```powershell
deactivate
```

## 如果要在別台電腦/重灌後重建環境

```powershell
cd C:\Users\belle\Documents\algorithm_blood_glucose-main
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 平常改完程式後怎麼用 git

```powershell
git status          # 先看改了哪些檔案（venv/data/csv 不會出現，因為被 .gitignore 排除了）
git diff            # 看實際改了什麼內容
git add <檔案>       # 只加要追蹤的檔案，避免誤加大檔案
git commit -m "說明這次改了什麼"
```

目前沒有設定 remote，所以這些 commit 都只存在本機，不會不小心上傳出去。

## 小提醒
- 一定要用 Python 3.9，不要用 3.12，目前鎖定的 neurokit2/nolds 版本在 3.12 上沒驗證過。
- 之後若要升級 neurokit2、nolds、scipy 版本，記得先 `import SWMlib` 測試一次確認不會炸，再更新 requirements.txt。
- 新增資料夾若也是放 data/model 產出，記得補進 [.gitignore](.gitignore)，避免不小心被 `git add -A` 加進去。

# 2026/09/08 新增 Data Augmentation（ECG 訊號訓練資料增強）

## 這次做了什麼
針對 [Model_Builder_Predictor_Belle.py](Model_Builder_Predictor_Belle.py) 的模型訓練流程加入 data augmentation，改善 Low/High 血糖類別樣本較少的問題（過去只有靠 `_data_balance` 做 under-sampling，沒有做過任何資料增強）。

1. **`ecgDataset`** class 新增 `augment=False` 參數，並加上 `augment_signal()` 方法：對已完成特徵轉換（normalize + DWT）的訊號做三種隨機增強，各自 50% 機率觸發：
   - 疊加高斯雜訊（std=0.02）
   - 振幅隨機縮放 0.9~1.1 倍
   - 時間軸左右平移最多 ±5 個取樣點

2. 新增 **`ecgDatasetSubset`** 包裝類別（緊接在 `ecgDataset` 之後），依索引挑選子集合，讓 train/valid 可以共用同一份已載入的資料，但各自獨立控制是否啟用 augmentation，避免驗證/測試資料被污染。

3. **`BuildModel_ThreeClasses`** 和 **`BuildModel_TwoClasses`** 的 train/valid 切分方式，從 `SubsetRandomSampler` 改成：
   ```python
   train_dataset = ecgDatasetSubset(dataset, train_indices, augment=True)   # 訓練集：啟用增強
   valid_dataset = ecgDatasetSubset(dataset, val_indices, augment=False)    # 驗證集：不增強
   ```
   測試集（`testdata`/`test_loader`）完全沒變動，維持原本 `ecgDataset(...)` 預設 `augment=False`。

4. 移除不再使用的 `from torch.utils.data.sampler import SubsetRandomSampler` import。

## 為什麼這樣改
- Augmentation 只能套用在訓練集，不能碰驗證/測試集，否則績效評估會失真。
- 原本 train/valid 共用同一個 `ecgDataset` 實例（只用 sampler 切索引），沒辦法對同一實例的不同索引分別開關 augmentation，所以才另外包一層 `ecgDatasetSubset`。
- 增強是套用在「DWT + normalize 之後」的特徵向量上，不是原始 ECG 波形。如果之後想改成在原始波形上做增強（例如更貼近生理訊號的 time warp），要改到 `load_data()`，且需要讓 augment=True/False 兩份資料各自重新讀檔算 DWT，寫法會更複雜。

## 如果要調整增強強度
都在 `ecgDataset.augment_signal()` 這個方法裡面，改雜訊標準差、縮放範圍、平移距離的數字即可，一處修改就好。

## 驗證
`python -m py_compile Model_Builder_Predictor_Belle.py` 通過，只做語法檢查，尚未實際跑過完整訓練驗證績效數字（後續驗證見下面「實際跑訓練做比較」章節）。

# 2026/09/08 實際跑訓練做比較：Data Augmentation 有無的效果

## 跑之前先修掉兩個擋路的既有 bug（跟 augmentation 無關）
1. **`ReduceLROnPlateau(..., verbose=1, ...)` 在目前環境會崩潰**：目前 venv 裝的是 PyTorch 2.8，`verbose` 參數已經被移除，`BuildModel_ThreeClasses`/`BuildModel_TwoClasses` 原本一啟動建立 scheduler 就會丟 `TypeError`，跟這次的 augmentation 修改無關，是既有程式與新版 PyTorch 不相容。已把兩處（[L1522](Model_Builder_Predictor_Belle.py#L1522) 三分類、[L1942](Model_Builder_Predictor_Belle.py#L1942) 二分類）的 `verbose=1` 拿掉。
2. **`data_processing()`／`BuildModel()` 內部原本呼叫訓練流程的部分被註解掉**：`feature_extraction`、`data_arrangement`、以及 `BuildModel_ThreeClasses`/`BuildModel_TwoClasses` 的實際呼叫都被註解，代表跑 `BuildModel()` 只會做完 `data_parsing` 就結束，不會真的訓練模型（看起來是之前測試 `_load_all_srj_data` 效能優化時暫時關閉的）。已重新打開這些呼叫。
3. 過程中曾經意外發現 `Model_Builder_Predictor_Belle.py` 裡 `ecgDatasetSubset.__len__` 那行被存成亂碼字元（IDE 開著檔案時發生），已修正並重新用 `py_compile` 確認整支檔案語法正常。之後如果編輯這支檔案，存檔後最好順手 `py_compile` 一次確認沒有異常字元。

## 雲端下載那段程式碼有恢復
`data_processing()` 裡呼叫 `unzip_file` 抓雲端 zip 檔的那段（給有需要從雲端下載資料的同事使用）已經恢復，但簽名多補了三個「有預設值、不影響既有呼叫方式」的參數，避免恢復後又立刻炸掉：
```python
def data_processing(self,uuid,srj_db_path,glucosedata_path,basepath,start_time=None,end_time=None,server_db_path=""):
```
原因：原本這段程式碼引用的 `server_db_path`、`start_time`、`end_time` 根本不是這個函式簽名的參數（只存在於上面被註解掉的另一版簽名），只要函式被呼叫、走到 `if(server_db_path!="")` 這行就會 `NameError` 崩潰。現在改成有預設值的參數：不傳的話（現有本機資料流程）行為不變，`server_db_path=""` 時不會觸發雲端下載；同事需要雲端下載時，改呼叫時多傳 `start_time`、`end_time`、`server_db_path` 三個參數即可觸發。

## 比較測試設定
- 對象：uuid `1726`（本機唯一有完整 RawData 的受測者），血糖分類走**二分類**（Normal vs High），因為這個人沒有低血糖(Low)資料。
- Train：High 337 筆、Normal 3093 筆；Test：High 139 筆、Normal 797 筆。
- 因為完整協定（20輪 x 最多800 epochs）在CPU上實際跑完可能要 3.5~19 小時，先跑**縮小規模版本**看趨勢：3輪 x 最多150 epochs，其餘設定（batch size、optimizer、loss、scheduler patience=100）都跟正式程式一致。
- 只跑了縮小規模版本，**完整協定這次沒有跑**（使用者決定先不跑完整版）。

## 結果（3輪中依 sensitivity/specificity 綜合最佳的一次）

| 指標 | 無 Augmentation | 有 Augmentation | 差異 |
|---|---|---|---|
| Sensitivity | 87.77% | 94.24% | +6.47pp |
| Specificity | 99.75% | 99.87% | +0.12pp |
| F1(調和平均) | 93.38% | 96.98% | +3.60pp |
| Accuracy | 97.97% | 99.04% | +1.07pp |
| TP/FP/FN/TN | 122/2/17/795 | 131/1/8/796 | FN 17→8 |
| 耗時 | 1452秒(≈24分) | 1620秒(≈27分，慢約12%) | |

三輪個別 sensitivity 也是有 augmentation 版本普遍較高（94.24 / 92.09 / 86.33 對比無augmentation的 87.77 / 87.77 / 86.33）。

**解讀**：High 在訓練集是少數類別（337 筆 vs Normal 3093 筆），augmentation 主要幫助模型更抓得住這個少數類別——FN（把 High 誤判成 Normal）從17筆降到8筆，Sensitivity 因此明顯提升，Specificity/FP幾乎沒有代價。方向符合預期。

**限制，不是最終結論**：
- 這是縮小規模（3輪、上限150 epoch）的趨勢測試，不是原始設計的完整協定（20輪、上限800 epoch），完整協定的數字可能不同。
- 只測了 uuid 1726 這一個人、二分類(Normal/High)的情境，沒有涵蓋三分類(Low/Normal/High)、也沒有跨受測者驗證。
- 目前 augmentation 是「線上即時擾動」（每個 epoch 對同一批原始檔案做隨機加雜訊/縮放/平移），並不會真的把 Train 資料夾裡 High/Normal 的檔案數量增加或改變，純粹增加訓練時看到的樣本多樣性。

## Data Augmentation 對「資料量／類別平衡」的實際影響量化
這點需要特別澄清：目前的實作方式**不會改變 Train 資料夾裡的檔案數量**（High 337、Normal 3093 這兩個數字，不管 augment 開關與否都不變），augmentation 只是讓模型每個 epoch 看到的 337 筆 High 都是「同樣 337 筆、但每次隨機加了一點雜訊/縮放/平移的版本」，增加的是**多樣性**，不是**筆數**，跟 `_data_balance` 那種直接刪減多數類別筆數的做法是不同機制、目前也沒有互相取代。

如果想要用 augmentation 做到「真的把少數類別筆數拉高、逼近多數類別」，需要另外加上**過採樣（oversampling）**機制（例如 `WeightedRandomSampler`，讓 High 這 337 筆每個 epoch被多抽幾次，每次抽到都套用不同的隨機增強，效果上就像是產生了不同的合成樣本）。以 uuid 1726 現有的 Train 筆數（Normal 3093 : High 337，目前約 9.18 : 1）換算幾種目標比例，需要的過採樣倍率：

| 目標比例(Normal:High) | High 需要的「有效」筆數/epoch | 過採樣倍率 |
|---|---|---|
| 9.18 : 1（現況，不做任何處理） | 337 | 1.0x |
| 5 : 1 | 619 | 1.84x |
| 3 : 1 | 1031 | 3.06x |
| 2 : 1 | 1547 | 4.59x |
| 1 : 1（完全平衡） | 3093 | 9.18x |

**建議**：不建議直接衝到 1:1，因為 High 實際上就只有 337 筆「不重複」的原始樣本，硬過採樣到跟 Normal 一樣多，等於同一筆資料的增強版本會被重複看很多次，模型容易記住這337筆的特徵而非真的學到高血糖的通用特徵，過擬合風險高。比較穩健的做法通常抓 **3~5倍**（對應 Normal:High ≈ 3:1~5:1）之間，如果之後要做這個，需要另外實作 sampler，目前程式還沒有這部分。

# 2026/09/08 合併雲端下載＋加速資料處理管線到 Model_Builder_Predictor_Belle.py

## 這次做了什麼
目的：加速 [Model_Builder_Predictor_Belle.py](Model_Builder_Predictor_Belle.py) 的資料處理部分，把 [Data_Parsing_Belle.py](Data_Parsing_Belle.py) 裡「下載雲端 zip → 解壓 → srj 解析 → 特徵擷取 → 分 Train/Test」這一整套較快的管線，整個併進 `Model_Builder_Predictor_Belle.py` 的 `DataArrangement` class，取代原本較慢的單進程版本。

1. **`DataArrangement` class 整個換成新管線**：
   - `unzip_file`：直接掃 `server_db_path` 底下檔名前綴等於 uuid 的 zip 檔解壓，取代舊版呼叫 `data_load_concate.search_unzip_file`。
   - `data_parsing`：改用 `multiprocessing.Pool`，把血糖 CSV 的每一列切成 `processnum` 份 chunk 平行處理（`_process_rows`），每個 chunk 內對 ECG 訊號做 `baseline_remove` → `remove_spike` → `nk.ecg_clean` → `motion_analysis`（分 static/dynamic）→ `ecg_quality_check_v3` → `emg_detector_remover`，品質好的訊號才進入 `feature_extraction_from_single_ecg` 算特徵，**直接**分類寫入 Dataset 的 High/Low/Normal 資料夾，取代舊版「先把原始 ECG 全部寫檔 → 事後再讀一次做特徵擷取」的兩階段做法，省了一次完整的檔案 I/O。
   - 順手修掉一個效能 bug：Data_Parsing_Belle.py 原本的 `_process_rows` 在**每一筆**血糖記錄都重新呼叫 `_load_all_srj_data` 重讀全部 srj 檔；改成每個 process（每個 chunk）只讀一次。
   - `data_arrangement`／`_movedata`：依日期切 Train/Test（取代舊版依血糖值直方圖切的方式），並支援 `splitting_ratio`（例如 `"70_30"`）決定訓練集比例。

2. **資料夾結構改成 `splitting_ratio` 子資料夾**（例如 `Model/70_30/GlucoseData/<uuid>/Train|Test/High|Low|Normal`），取代舊版扁平的 `Model/GlucoseData_<uuid>/...`。原因：這正是既有的 [M5_DataAug.py](M5_DataAug.py) 訓練程式期待吃到的輸入格式，對齊之後資料處理完可以直接餵給 M5 訓練，不用再搬檔案。連動修改：
   - `BuildModel`、`BuildModel_ThreeClasses`、`BuildModel_TwoClasses`：都多了 `splitting_ratio` 參數，內部所有 `GlucoseData_`/`Best_*Model`/`Temp_*Model` 路徑都改成 `basepath/splitting_ratio/...`。
   - `Clean_Data`、`Clean_Model`：一樣補上 `splitting_ratio` 參數。
   - `GlucoseThreeClassesPredictor`、`GlucoseTwoClassesPredictor`、`GlucosePredictor`：建構子多了 `splitting_ratio`，預測時讀模型的路徑也一起改。

3. **`__main__` 區塊**改成用 `Path(__file__)` 相對路徑（`Model/`、`GlucoseDataCSV/`、`DataDB/<uuid>`），不再寫死 Windows 專屬路徑，並示範新的呼叫方式：
   ```python
   status, errorcode, message = BuildModel(
       uuid, basepath, srj_db_path, glucosedata_path,
       server_db_path,   # 本機已有 srj 檔就填 ""，跳過雲端下載
       processnum,       # 平行處理程序數，是加速的關鍵旋鈕，依CPU核心數調整
       splitting_ratio,  # 例如 "70_30"
   )
   ```

## 環境修復（合併過程中發現、跟這次程式改動本身無關）
1. **`SWMlib` 是舊版，缺新管線需要的函式**：`motion_analysis`、`ecg_quality_check_v3`、`normalization`、`remove_spike`、`emg_detector_remover` 在原本的 [SWMlib/](SWMlib/) 裡都不存在。已從 `/share/Ivy/bgm_cls/SWMlib`（同事 Ivy 的新版）整份複製過來覆蓋，舊版備份在 `SWMlib.bak_pre_sync_20260908092615/`，之後確認沒問題可以刪除。
2. **缺頂層 `common.py`**：[M5_DataAug.py](M5_DataAug.py) 有 `from common import (...)`，但這個 repo 原本沒有這個檔案。已從 `/share/Ivy/bgm_cls/common.py` 複製過來。
3. **殘留問題，尚未解決**：複製過來的 `common.py` 版本比 `M5_DataAug.py` 預期的舊，裡面缺少 `train_valid_split_indices` 這個函式，所以 `import M5_DataAug` 目前還是會炸（`ImportError: cannot import name 'train_valid_split_indices'`）。這個跟這次「合併資料處理管線」的任務無關，之後要跑 M5 訓練前需要再跟 Ivy 對版本，或請她補這個函式到 common.py。

## 驗證
- `python -m py_compile Model_Builder_Predictor_Belle.py`：通過。
- `import Model_Builder_Predictor_Belle`：SWMlib 更新後可以正常 import（改動前會因為缺函式直接炸）。
- 之後有實際拿 uuid `2208` 跑過一次完整的 `DataArrangement.data_processing(...)`（見下一節），驗證了資料處理管線本身可以正常跑完、產出正確的 Train/Test 資料夾結構。

> **⚠️ 以下 2208、2197、2204、2199 這 4 組「實測」數字都是修正時區 bug 之前跑的，已知是錯的，不要拿來用。正確數字請看後面「重大bug：時區錯位」那個章節的修正後結果。**

## 實測：uuid 2208 資料處理（本機已有 srj，未含雲端下載）
資料規模：`DataDB/2208` 306 個 srj 檔、共 1.5GB；`GlucoseDataCSV/2208.csv` 1267 筆血糖記錄。`processnum=8`、`splitting_ratio="70_30"`，`server_db_path=""`（本機已有 ECG，跳過雲端下載）。

- 耗時：485 秒（≈8 分鐘），`errorcode="0"`。
- 產出 `Model/70_30/GlucoseData/2208/`：
  | | High | Low | Normal |
  |---|---|---|---|
  | Train | 195 | 431 | 706 |
  | Test | 76 | 222 | 227 |
- `Model/RawData/2208` 累積 8501 筆通過品質檢查的 ECG 片段（`Model/Dataset/2208/*` 在 `data_arrangement` 之後會被清空，因為檔案都被搬到上面的 Train/Test 資料夾了，這是正常現象）。
- 三個類別（High/Low/Normal）在 Train/Test 都有資料，代表這個 uuid 可以直接用 `BuildModel_ThreeClasses` 訓練三分類模型。
- 觀察到的效能瓶頸：8 個 process 各自獨立載入完整的 306 個 srj 檔進記憶體（`_load_all_srj_data` 設計上是每個 process 各載入一次），2208 這個案例每個 process 吃到約 9GB RAM。之後如果遇到 srj 檔案量更大的 uuid，記憶體可能會是比 CPU 更早撞到的瓶頸，需要視情況調低 `processnum`。

## 實測：uuid 2197 資料處理（本機已有 srj，未含雲端下載）
資料規模：`DataDB/2197` 14 個 srj 檔、共 2.2GB；`GlucoseDataCSV/2197.csv`。設定同上（`processnum=8`、`splitting_ratio="70_30"`、`server_db_path=""`）。

- 耗時：557 秒（≈9.3 分鐘），`errorcode="0"`。
- 產出 `Model/70_30/GlucoseData/2197/`：
  | | High | Low | Normal |
  |---|---|---|---|
  | Train | 364 | 0 | 1676 |
  | Test | 166 | 0 | 510 |
- `Model/RawData/2197` 累積 22440 筆通過品質檢查的 ECG 片段。
- 這個 uuid **沒有 Low 資料**，`BuildModel()` 裡的判斷邏輯會自動偵測到並改走 `BuildModel_TwoClasses`（Normal vs High 二分類），不需要額外處理。

## `__main__` 改成批次下載＋處理迴圈
把原本單一 uuid 的示範，改成依 `GlucoseDataCSV` 現有的全部 10 個 uuid（`2197,2199,2204,2206,2208,2210,2215,2216,2223,2249`）跑 `DataArrangement.data_processing(...)`（只做下載+資料處理，**不含模型訓練**）。`BuildModel`（含模型訓練，每個 uuid 最多 20 輪 x 800 epochs，非常耗時）和預測 demo 都先註解掉，避免不小心觸發長時間訓練，需要的話手動打開。

`server_db_path` 目前填的是 Data_Parsing_Belle.py 原本用的雲端路徑，只有在能存取那個 Google Drive 網路磁碟的電腦（例如 Belle 的工作機）才能真的觸發下載；這個容器環境沒有掛載雲端磁碟，所以雲端下載本身沒辦法在這裡驗證，只驗證了「本機已有 srj 檔」這條路徑（就是上面 2208 的實測）。

# 2026/09/09 重大bug：時區錯位造成ECG-血糖配對錯誤（已修正）

## 問題
這台機器（容器）系統時區是 **UTC**，但 srj 的 `tt` 時間戳與血糖 CSV 記錄的時間，實際上都是**台北時間（UTC+8）**。`_load_all_srj_data`／舊版 `_dataconcate` 都用 `datetime.fromtimestamp(tt)` 把 epoch 轉成日期時間——這個函式是用「執行當下系統的時區」解讀時間，在系統時區本身就是 Asia/Taipei 的機器（例如 Belle 平常用的 Windows 電腦，多半預設就是台北時區）上跑沒問題，但在這台 UTC 容器上跑，換算出來的時間會**系統性慢 8 小時**，導致每一筆 ECG 片段配對到的血糖記錄時間點是錯的。

這不是新引入的 bug，是原本程式（新舊管線都一樣）就存在的「假設執行環境時區＝資料記錄時區」的隱性依賴，只是在這個 UTC 容器上才會出問題。

## 怎麼發現的
處理 uuid `2204` 時，發現 5 筆高血糖記錄（例如 `2025-02-18 08:46`）在這台機器上完全查不到任何 ECG 片段，一開始以為是那個時段裝置沒收訊。往前查證：
- 用系統當地時間（UTC）查 `08:41-08:51` → 0 筆（剛好落在 08:30:48~15:58:35 的真實資料空窗期）
- 往前推 8 小時查 `00:41-00:51`（UTC）→ 60 筆（滿的）
- 用 `TZ=Asia/Taipei` 重新執行同一段查詢邏輯，直接查 `08:41-08:51` → 60 筆（跟上面 -8小時查到的結果一致）

另外，srj 每日檔案的邊界都剛好落在 UTC 16:00（例如 `2204_20250218.srj` 涵蓋 `2025-02-17 16:00:06` ~ `2025-02-18 15:59:57`），這正好對應「台北午夜」，也佐證了資料本來就是按台北日曆日切檔的。

## 影響範圍
不只是查不到資料這種明顯狀況——**因為 ECG 大部分時段是連續量測的，時間差 8 小時通常還是能查到「一樣多筆數」的片段，只是查到的其實是另一個時段、跟這筆血糖記錄無關的 ECG**，也就是說在這台容器上處理過的資料，ECG 特徵配對到的血糖標籤幾乎全部是錯的，不是只有查不到資料的那幾筆而已。

在此之前用舊時區跑過的 `2208`、`2197`、`2204`、`2199` 這 4 組資料（見上面幾個「實測」章節）都受影響，已標記為不可用。

## 修正方式
在 [Model_Builder_Predictor_Belle.py](Model_Builder_Predictor_Belle.py) 開頭（import 區塊之後）加上：
```python
os.environ['TZ'] = 'Asia/Taipei'
if hasattr(time, 'tzset'):
    time.tzset()
```
把時區直接寫死在程式裡，不管之後在哪一台機器/容器上執行都不會再受系統預設時區影響。`time.tzset()` 只有 Unix 系統才有，Windows 沒有這個函式，但 Windows 上 `datetime.fromtimestamp()` 本來就不會吃 `TZ` 環境變數，所以在 Windows（Belle平常的電腦）上這行只是沒作用、不會報錯，前提是 Windows 系統時區本身就是 Asia/Taipei（目前為止都是這樣運作的）。

`multiprocessing.Pool` 在 Linux 預設用 `fork`，子行程會繼承父行程呼叫過 `tzset()` 之後的狀態，所以這個修正對 `data_parsing` 內部平行處理的 worker 也一樣有效，不用額外處理。

## 全部 4 個 uuid 用修正後的版本重新跑過，結果如下

先用 `Clean_Data(uuid, basepath, "70_30")` 清掉舊（錯誤時區）產出的 RawData/Dataset/Features/high_emg_data/GlucoseData，並手動清掉 `bad_data`/`dynamic_data` 底下的殘留，才重新執行 `DataArrangement.data_processing(...)`（設定同前：`processnum=8`、`splitting_ratio="70_30"`、`server_db_path=""`）。

| uuid | 耗時(秒) | Train High/Low/Normal | Test High/Low/Normal | RawData片段數 |
|---|---|---|---|---|
| 2208 | 699 | 50 / 514 / 753 | 210 / 750 / 204 | 9614 |
| 2197 | 733 | 737 / 0 / 1549 | 261 / 0 / 590 | 22594 |
| 2204 | 784 | 83 / 0 / 1669 | 32 / 0 / 379 | 20704 |
| 2199 | 1041 | 1140 / 0 / 2278 | 167 / 0 / 1403 | 38401 |

跟修正前的數字比對，變化都很明顯（例如 2208 修正前 Train High/Low/Normal 是 195/431/706，修正後變成 50/514/753；2204 修正前 High 完全是 0，修正後 Train/Test 各有 83/32 筆），證實這是真的會實質影響訓練資料內容的問題，不是可忽略的小誤差。

2204、2197、2199 這三個 uuid 的 Low 都還是 0，這點跟時區無關：實際查證過 2204 的低血糖記錄時間點（2025-03-01之後）本來就超出這個人 srj 資料收集的範圍（最晚到2025-02-28），是資料本身的完整度問題，不是程式錯誤。

## 給之後的提醒
如果之後要在別的機器/容器上跑這支程式，或是同事的環境時區未知，建議先確認：
```python
import time
print(time.tzname)  # 應該要顯示台北對應的時區資訊，不是 UTC 或別的時區
```
如果不放心，也可以比照這次的除錯方式，挑一筆已知血糖記錄時間、直接查那個時間窗口的 ECG 片段數，跟往前後8小時的查詢結果比對，確認沒有錯位。

# 2026/09/09 幫 M5_DataAug.py 補上 Data Augmentation 開關，並修好兩個既有 bug

## 為什麼要動 M5_DataAug.py
之前的 data augmentation（見上面「新增 Data Augmentation」章節）只加在 [Model_Builder_Predictor_Belle.py](Model_Builder_Predictor_Belle.py) 裡，[M5_DataAug.py](M5_DataAug.py) 完全沒有 augmentation 相關程式碼。這次要用 M5 訓練「無 augmentation」跟「有 augmentation」兩個版本來比較績效，所以先把同一套 augmentation 機制搬過來，並讓它可以用參數開關，而不是寫死開啟。

## 修掉的既有 bug（跟這次加 augmentation 無關，但擋路）
1. **`import M5_DataAug` 會直接炸**：[common.py](common.py) 缺少 `train_valid_split_indices` 這個函式（`M5_DataAug.py` 有 import 它），這是先前合併資料處理管線時就發現、記錄在上面「殘留問題，尚未解決」章節、但一直沒補的坑。已經從其他同事（例如 `jane/Result/v2.1.2/common.py`）的版本把同一份函式實作補進 `common.py`。
2. **`BuildModel()` 整個函式被註解掉，但 `__main__` 還在呼叫它**：`M5_DataAug.py` 最上面那段測試用的 `BuildModel(uuid, base_path, srj_db_path, glucosedata_path, server_db_path, processnum=8, splitting_ratio="70_30")` 整段被註解，`__main__` 直接呼叫會 `NameError`。已經照原本註解掉的邏輯還原成正常函式（依 uuid 底下 Train/Test 資料夾是否同時有 High/Low/Normal 三類資料，自動決定呼叫 `BuildModel_ThreeClasses` 還是 `BuildModel_TwoClasses`），並補上下面第 3 點的 `model_subdir`／`augment` 參數讓它可以往下傳。

驗證：`python -m py_compile M5_DataAug.py` 通過，`import M5_DataAug` 也確認可以正常 import。

## 這次加的 Data Augmentation 開關（做法跟 Model_Builder_Predictor_Belle.py 的版本一致）
1. **`ECGDataset`** 新增 `augment=False` 參數，並加上 `augment_signal()` 方法：對已完成特徵轉換（normalize + DWT）的訊號做三種隨機增強，各自 50% 機率觸發：
   - 疊加高斯雜訊（std=0.02）
   - 振幅隨機縮放 0.9~1.1 倍
   - 時間軸左右平移最多 ±5 個取樣點

2. 新增 **`ECGDatasetSubset`** 包裝類別（緊接在 `ECGDataset` 之後），依索引挑選子集合，讓 train/valid 可以共用同一份已載入的資料，但各自獨立控制是否啟用 augmentation，避免驗證/測試資料被污染。

3. **`BuildModel_ThreeClasses`**、**`BuildModel_TwoClasses`**、以及還原後的 **`BuildModel`** 三個函式都新增 `model_subdir=""`、`augment=False` 兩個參數（`BuildModel` 會把這兩個參數原封不動往下傳給 `BuildModel_ThreeClasses`/`BuildModel_TwoClasses`）：
   ```python
   train_subset = ECGDatasetSubset(dataset, train_indices, augment=augment)  # 訓練集：依參數決定是否增強
   valid_subset = ECGDatasetSubset(dataset, val_indices, augment=False)      # 驗證集：一律不增強
   ```
   測試集（`testdata`/`test_loader`）完全沒變動，維持原本 `ECGDataset(...)` 預設 `augment=False`。

4. 移除不再使用的 `from torch.utils.data import Subset` import（train/valid 改用 `ECGDatasetSubset`，原本的 `Subset` 已經沒有地方用到）。

## 如何用來做「有無 augmentation」比較
呼叫兩次，`model_subdir` 一定要給不同值，否則第二次會把第一次的 `Best_*Model`/`Temp_*Model` 輸出覆蓋掉：
```python
BuildModel_TwoClasses(uuid, basepath, splitting_ratio, model_subdir="NoAug",   augment=False)
BuildModel_TwoClasses(uuid, basepath, splitting_ratio, model_subdir="WithAug", augment=True)
```
三分類同理呼叫 `BuildModel_ThreeClasses`，或直接呼叫還原後的 `BuildModel(...)`（會自動判斷二分類/三分類）並多傳 `model_subdir`、`augment`。

## 尚未驗證
這次只確認了程式可以正常 import／`py_compile`，還沒有實際跑過訓練比較兩者績效數字（跟前面「有無 augmentation」在 Model_Builder_Predictor_Belle.py 上跑的縮小規模測試不是同一次驗證，`M5_DataAug.py` 這邊用的是不同的 Transformer 架構、分層切分＋BalancedBatchSampler，需要另外實際跑一次才能拿到數字）。等資料傳輸完成、`Model/70_30/GlucoseData/<uuid>` 底下有完整 Train/Test 資料後再實測。


# 2026/09/09 準備 M5_DataAug.py 跑 NoAug 全量訓練，並把訓練規模降級

## 背景
上一節「尚未驗證」的前置條件已經滿足：10 個 uuid 的 `Model/70_30/GlucoseData/<uuid>` 都已經有完整 Train/Test 資料（`2215`、`2216`、`2223`、`2249` 是這天補跑 `Model_Builder_Predictor_Belle.py` 的 `data_processing` 產生的，2223+2249 共 1390 秒、2215+2216 共 1077 秒）。所以這次開始實際用 M5 跑「無 augmentation」那一邊。

各 uuid 資料現況與 `BuildModel()` 會自動選到的分支：

| uuid | Train High/Normal/Low | Test High/Normal/Low | 分支 |
|---|---|---|---|
| 2197 | 765 / 2948 / 0 | 289 / 1120 / 0 | 二分類 |
| 2199 | 1140 / 2278 / 0 | 167 / 1403 / 0 | 二分類 |
| 2204 | 83 / 1669 / 0 | 32 / 379 / 0 | 二分類 |
| 2206 | 57 / 1048 / 0 | 79 / 2278 / 0 | 二分類 |
| 2249 | 2563 / 2817 / 0 | 1162 / 1622 / 0 | 二分類 |
| 2208 | 50 / 753 / 514 | 210 / 204 / 750 | 三分類 |
| 2210 | 246 / 2082 / 418 | 18 / 681 / 542 | 三分類 |
| 2215 | 228 / 1169 / 1483 | 276 / 468 / 1127 | 三分類 |
| 2216 | 270 / 1054 / 541 | 197 / 1920 / 1339 | 三分類 |
| 2223 | 844 / 1403 / 166 | 271 / 910 / 88 | 三分類 |

## `__main__` 的設定改動
原本的 `__main__` 是 `for i in range(0,17)` 搭配寫死的 `if(uuid !='2131'): continue`，只會跑 uuid 2131（而 2131 根本沒有資料，等於跑不動）。改成：

```python
target_uuids=['2197','2199','2204','2206','2208','2210','2215','2216','2223','2249']

model_subdir="NoAug"   ###要跑有augmentation版本請改成"WithAug"並把augment改成True
augment=False

for i in range(0,len(user_information)):
    uuid=user_info[0]
    if uuid not in target_uuids:
        continue
    ...
    status, errorcode, message = BuildModel(...,model_subdir=model_subdir,augment=augment)
```

- 迴圈改成掃完整個 `user_information`（原本只掃前 17 筆，但 `2210`、`2216`、`2199` 等 uuid 在第 17 筆之後，掃不到），再用 `target_uuids` 過濾。實際命中順序是 `2197, 2208, 2215, 2223, 2249, 2199, 2206, 2204, 2210, 2216`（照 `user_information` 原順序）。
- `srj_db_path` 從 `C:\Users\User\Desktop\DataDB\<uuid>` 改成本機的 `DataDB/<uuid>`。這個參數 `BuildModel` 實際沒有用到（只有資料處理階段才需要 srj），改掉純粹是避免留著 Windows 路徑誤導。
- 輸出會進 `Model/70_30/NoAug/Best_TwoClasses_Model/<uuid>/` 或 `Best_ThreeClasses_Model/<uuid>/`（依分支）。之後跑 WithAug 只要改 `model_subdir`／`augment` 兩個值，不會覆蓋這次結果。

## 訓練規模降級（兩條路徑都改成同一組）
原設定跑起來太慢：2197 第一輪（二分類、patience 100）就花了約 9 分鐘，換算 20 輪約 2.5～3 小時／uuid，10 個 uuid 要 25～30 小時，而且 NoAug/WithAug 要各跑一次。所以先降級：

| 參數 | 原本（三分類 / 二分類） | 改成 | 位置 |
|---|---|---|---|
| 訓練輪數 | 20 / 20 | **5** | [L912](M5_DataAug.py#L912)、[L1562](M5_DataAug.py#L1562) |
| Epoch 上限 | 800 / 800 | **300** | [L808](M5_DataAug.py#L808)、[L1454](M5_DataAug.py#L1454) |
| Early-stop patience | 50 / **100** | **30** | [L809](M5_DataAug.py#L809)、[L1455](M5_DataAug.py#L1455) |
| Scheduler patience | 50 / 100 | **15** | [L945](M5_DataAug.py#L945)、[L1598](M5_DataAug.py#L1598) |

預估降到每個 uuid 約 20～25 分鐘、10 個 uuid 約 3.5～4 小時，NoAug＋WithAug 兩邊合計約 7～8 小時。

**選這組數字的理由**：那 20 輪是「用不同隨機初始化重跑整個訓練、取最好的一次」（best-of-N），輪數越多只是把最佳值往上推，對「有無 augmentation 誰比較好」這個比較本身幫助有限；降到 5 輪仍然取樣得到初始化的變異。Epoch 上限 300 搭配 patience 30，實務上大多會在早停就結束，上限只是防呆。**最重要的是 NoAug 和 WithAug 必須用完全同一組設定，否則兩邊數字不能比。**

## 順手修掉的兩個參數問題
1. **二分類的 `patience` 被覆蓋成 100**：`BuildModel_TwoClasses` 開頭設了 `patience=50`，但迴圈裡（原 L1546）又有一行 `patience = 100` 每輪重新賦值，所以實際生效的 early-stop patience 一直是 100，跟三分類的 50 不一致（也是 2197 第一輪要跑 9 分鐘的原因）。已移除那行覆蓋，現在兩條路徑都由函式開頭的 `patience` 統一控制。
2. **scheduler patience 拆成獨立變數 `scheduler_patience`**：三分類原本是 `ReduceLROnPlateau(..., patience=patience)`，直接複用 early-stop 的 `patience`；兩者相等時 LR 永遠不會衰減就先被 early stop 停掉，`ReduceLROnPlateau` 等於白設。現在 `scheduler_patience=15` < `patience=30`，LR 至少有一次衰減的機會。

## 這次沒有改的東西
`ReduceLROnPlateau(..., verbose=1, ...)` 保留不動。前面 2026/09/08 那節提到這個參數在 PyTorch 2.8 會丟 `TypeError`，但這台機器（H100 80GB HBM3）的環境是 **torch 2.5.1+cu121**，`verbose` 還在（只是 deprecated），不會炸。如果之後換到 2.8 以上的環境跑 M5，這兩處（[L945](M5_DataAug.py#L945)、[L1598](M5_DataAug.py#L1598)）要比照 Model_Builder_Predictor_Belle.py 拿掉 `verbose`。

## 怎麼確認訓練真的在 GPU 上
容器環境裡 `nvidia-smi` 下方的 Processes 表格會是空的（PID namespace 隔離，查詢會顯示 `[Not Found]`），**不能**因為那張表沒列出 process 就以為沒用到 GPU。可靠的確認方式：

1. 看程式自己印的 log（[L917-925](M5_DataAug.py#L917-L925)、[L1568-1576](M5_DataAug.py#L1568-L1576)）每輪開頭會印 `cuda:0`、`GPU Count: 1`、`NVIDIA H100 80GB HBM3`；掉回 CPU 會印 `cpu` 且沒有裝置名稱。
2. `ls -l /proc/$(pgrep -f M5_DataAug)/fd | grep nvidia` 看 process 有沒有開 `/dev/nvidiaN`（`nvidia-smi -q | grep -i "minor number"` 可以對應是哪張卡）。
3. `nvidia-smi` 看 GPU-Util／Memory-Usage 有沒有在動。

## 進度追蹤的建議
訓練 log 只印在終端機，事後查不到。建議啟動時留檔：
```bash
cd /share/Belle/DataAug && python M5_DataAug.py 2>&1 | tee train_noaug_$(date +%m%d_%H%M).log
```
沒留 log 的話只能從輸出檔案的時間推：`Best_*Model/<uuid>/BestModel_*.pth` 的 mtime 是最後一次「突破」的時間，`Temp_*Model/<uuid>/` 有更新代表還在訓練中。注意二分類的 Temp 只存一個固定檔名 `TwoClassesModel_<uuid>_Best.pth` 反覆覆蓋，看不出輪數／epoch；三分類則是每個 epoch 存一個 `Model_<uuid>_<epoch>.pth`，可以直接從檔名看到 epoch 進度。

## 尚未驗證
降級後的設定還沒有完整跑過。先前用舊設定（20 輪／800 epochs）跑的那次只完成 2197 的第一輪就停掉，輸出（`Model/70_30/NoAug/`）已經刪除，避免跟降級後的結果混在一起。

# 2026/09/11 補上 training loss / accuracy 的記錄與曲線圖

## 為什麼要改
前一節「進度追蹤的建議」提到訓練 log 只印在終端機、事後查不到。實際翻程式碼後發現情況比預期更徹底：

四個訓練迴圈裡都有 `Loss_list` 和 `ACCs` 兩個 list，每個 epoch 把 train/valid 的 loss 與 accuracy append 進去 —— 但**全專案沒有任何一處把它們 return、畫圖或寫檔**。grep 過所有出現位置，只有初始化和 append 兩種，函式一結束就被回收。專案裡也沒有 tensorboard / wandb / mlflow / logging / np.save / json.dump 任何一種實驗追蹤機制（全部零命中），唯一會落地的檔案是 confusion matrix 的 `*_Performance_Matrix.xlsx` 和 `.pth` 權重。

也就是說 loss 真正的用途只有控制訓練流程兩件事：餵給 `scheduler.step(valid_loss)` 讓 ReduceLROnPlateau 決定要不要降 LR，以及用 `min_valid_loss` / `n_patience` 做 early stopping 與存檔判斷。曲線長什麼樣、有沒有 overfitting、best epoch 落在哪裡，跑完就查不到了。

## 這次做了什麼

### 1. [common.py](common.py) 新增共用函式 `save_training_history()`

位置 [L582 起](common.py#L582)。每次訓練結束後輸出兩個檔案，跟 `.pth` checkpoint 放在同一個 `save_path`：

| 檔案 | 內容 |
|---|---|
| `<uuid>_<tag>_training_history.csv` | 欄位 `epoch, train_loss, valid_loss, train_acc, valid_acc` |
| `<uuid>_<tag>_training_curve.png` | 上圖 loss、下圖 accuracy，並用灰色虛線標出 valid loss 最低的 epoch（也就是 early stopping 實際選中的那個模型） |

`tag` 用來區分同一個 uuid 底下的兩條路徑，值是 `ThreeClasses` 或 `TwoClasses`。

實作上處理掉的幾個細節：

- **型別不一致**：`valid_acc` 在三分類路徑是 `100*correct/total`，`correct` 來自 `.sum()` 所以是還沒 `.item()` 的 torch tensor；二分類路徑則已經 `.item()` 成 float。統一用 `_history_value()` 轉 float，順便擋掉 numpy 純量。
- **沒有顯示器**：固定 `matplotlib.use('Agg')`，容器裡不會因為找不到 display 而炸。matplotlib 3.9.4 本來就在 [requirements.txt](requirements.txt) 裡，不用額外安裝。
- **畫圖失敗不能拖垮訓練**：繪圖整段包在 try 裡，出錯只印一行訊息，CSV 照樣寫出去。訓練跑了幾小時，不該因為畫圖失敗而什麼都沒留下。
- **續訓**：傳 `start_epoch=Epoch_range[0]+1`，epoch 編號會接續；如果 CSV 已存在，會把 `epoch < start_epoch` 的舊列保留下來再接上新的，不是整個蓋掉。

### 2. 四個訓練迴圈各加一處呼叫

都放在迴圈結束、最後一次 `torch.save` 之後、測試階段之前：

| 檔案 | 位置 | tag |
|---|---|---|
| M5_DataAug.py | [L1055](M5_DataAug.py#L1055) | ThreeClasses |
| M5_DataAug.py | [L1686](M5_DataAug.py#L1686) | TwoClasses |
| Model_Builder_Predictor_Belle.py | [L1625](Model_Builder_Predictor_Belle.py#L1625) | ThreeClasses |
| Model_Builder_Predictor_Belle.py | [L2037](Model_Builder_Predictor_Belle.py#L2037) | TwoClasses |

`Model_Builder_Predictor_Belle.py` 原本沒有 import `common`，補了一行 [L50](Model_Builder_Predictor_Belle.py#L50)（檔案開頭本來就有把自己的目錄加進 `sys.path`，所以直接 import 得到）。`M5_DataAug.py` 則是加進既有的 `from common import (...)` 區塊。

## 順手修掉的既有 bug：續訓會 NameError

四個迴圈原本都長這樣：

```python
if Epoch_range[0] == 0:
    Loss_list = []
    ACCs = []
    model = ...   ###建立新模型
```

`Loss_list` / `ACCs` 的初始化被包在「從第 0 個 epoch 開始」的守衛裡，但迴圈內的 `Loss_list.append(...)` 是無條件執行的。只要用非零的 `Epoch_range` 續訓，第一個 epoch 就會 `NameError: name 'Loss_list' is not defined`。

已把兩行初始化移到 `if` 外面（[M5 L929](M5_DataAug.py#L929)、[M5 L1579](M5_DataAug.py#L1579)、[MBP L1507](Model_Builder_Predictor_Belle.py#L1507)、[MBP L1926](Model_Builder_Predictor_Belle.py#L1926)），建立模型那段仍然留在守衛內。目前的 `__main__` 都是從 0 開始跑，所以這個 bug 還沒被踩到，但新加的記錄功能會用到續訓路徑，先修掉。

## 驗證
- `save_training_history()` 單獨測過：一般情況、續訓合併（epoch 1–3 存檔後再存 4–5，CSV 正確接成 1–5）、混入 torch tensor 的 accuracy、空 list 回傳 `(None, None)` 不炸，PNG 有實際開起來看過。
- 兩個模組都 `import` 成功，且 `save_training_history` 解析到同一個 common.py 的函式物件。
- 尚未在真實訓練中跑過（等下一次 NoAug 全量訓練驗證）。

## 順帶更新：前一節的行號連結
這次在 `M5_DataAug.py` 插入 import 與呼叫後，前一節 2026/09/09 引用的行號全部往後位移（位移量 +1 到 +9 不等，因為插入點分散）。已把那一節表格與內文裡的 `M5_DataAug.py#L...` 連結一併更新成現在的行號，內容本身沒有改動。

## 踩到的坑：換行符
`M5_DataAug.py` 是 **CRLF** 換行（其他 .py 和 .md 都是 LF）。用 Python 以文字模式讀寫會把 CRLF 吃成 LF，導致整個檔案 2000 多行全部變成 diff。已還原成 CRLF，現在 diff 只有實際改動的部分。**之後用腳本批次改這個檔案時要注意**，用 binary 模式讀寫或改完檢查 `git diff --stat` 的行數是否合理。

# 2026/09/11 時間軸平移改用邊緣補值，取代 torch.roll

## 問題
Data augmentation 的時間軸平移原本用 `torch.roll`：

```python
shift = int(torch.randint(-5, 6, (1,)).item())
signal = torch.roll(signal, shifts=shift, dims=-1)
```

`torch.roll` 是**循環**的 —— 被推出右端的取樣點會從左端繞回來，等於在訊號中間接出一個真實 ECG 不會出現的跳階。

## 先量測，再決定要不要改
沒有直接改，先掃了 8 個 uuid、7360 個 training beat 實測這個跳階到底多大（訊號經 `normalize1` 正規化後全幅 = 1）：

| 量測 | 數值 |
|---|---|
| 接縫跳階 `\|x[0]-x[149]\|` 中位數 | 0.0086 |
| 接縫跳階 p99 / 最大值 | 0.040 / 0.084 |
| 自然逐點變化 `\|x[i+1]-x[i]\|` 中位數 | 0.0043 |
| 每個 beat 自己的 QRS 最大斜率（中位數） | 0.276 |
| 接縫 ÷ 該 beat 最大自然斜率（中位數） | **3.1%** |
| 接縫大於該 beat 最大自然斜率的比例 | **0 / 7360 = 0.0%** |

**結論是這個 bug 的實際傷害很小**，原因是 beat 切割時 R 峰被鎖在 index 50（實測 argmax 位置 p5=50、中位數=50、p95=51），所以視窗兩端都落在平坦的 TP 基線區（頭 5 點平均 0.123、尾 5 點平均 0.136），接起來的落差自然就小。沒有任何一個 beat 的接縫跳階超過它自己 QRS 上升沿的大小，對 CNN 來說比較像是基線雜訊稍大，而不是一個結構性的假訊號。

## 那為什麼還是改了
1. 上面的結論**是建立在「R 峰鎖在 50」這個前提上的**。哪天 beat 切割方式改了、或 shift 從 ±5 放大，這個前提就不一定還成立，又要重新驗證一次。邊緣補值則無論如何都不會產生非生理的不連續。
2. **現在改是免費的**：WithAug 還沒跑過（目前 `augment=False`／`model_subdir="NoAug"`）。一旦 WithAug 跑完再改 augmentation，那批結果就跟之後的不能比，得整批重跑。
3. 截掉邊緣 5 點的代價實測過，是可以接受的：頭尾 5 點偏離基線只有 0.058 / 0.045，對照 QRS 峰值偏離基線 0.816，**截掉的純粹是基線，沒有波形資訊**。T 波峰在 idx 110–120，離邊界很遠。

## 改法
[M5_DataAug.py L123](M5_DataAug.py#L123)、[Model_Builder_Predictor_Belle.py L866](Model_Builder_Predictor_Belle.py#L866)（兩個檔案是同一份重複的 `augment_signal`，一起改）：

```python
shift = int(torch.randint(-5, 6, (1,)).item())
if shift != 0:
    pad = abs(shift)
    if shift > 0:   ###往右移，左端用第一點補、截掉尾端
        signal = torch.cat([signal[..., :1].expand(*signal.shape[:-1], pad),
                            signal[..., :-pad]], dim=-1)
    else:           ###往左移，右端用最後一點補、截掉開頭
        signal = torch.cat([signal[..., pad:],
                            signal[..., -1:].expand(*signal.shape[:-1], pad)], dim=-1)
```

`signal` 在這裡是 `(1, 150)`，而 `F.pad` 的 `mode='replicate'` 對 1D 需要 3D 輸入 `(N, C, W)`，所以沒有用 `F.pad`，直接用 `expand` + `cat`。

## 驗證
- 小張量逐一檢查平移結果正確（`shift=+3` → `[0,0,0,0,1,2,...]`、`shift=-3` → `[3,4,...,9,9,9,9]`、`shift=0` 不變）。
- 500 個真實 beat × 4 種 shift：輸出形狀維持 `(1,150)`、**R 峰位移量 100% 正確**。
- 接縫處跳變實測：`torch.roll` 中位數 0.0043 / 最大 0.0430，**邊緣補值 100% 為 0**。

（順帶一提，第一次驗證時用「全訊號最大逐點跳變」當指標，兩種做法都得到 1.0 —— 因為最大跳變一定是 QRS，直接蓋過接縫。要量接縫就必須指定接縫位置去量，這個指標不能用全域最大值。）

## 理論上更正確、但這次沒做的做法
這 150 點是從連續 ECG 上切下來的，所以時間平移最正確的實作是**在切 beat 的時候就把 shift 加進去**，兩端補的會是真實的鄰近取樣點，既不循環也不用假的平坦段。但那要動到資料處理管線（回頭重新 parse srj），以目前 ±5 點的幅度來說不值得為此重跑整個資料處理。

## 順帶更新：行號連結
這次 `augment_signal` 從 3 行變成 14 行(+11)，插入點之後的行號全部位移，前面幾節引用的 `M5_DataAug.py#L...` / `Model_Builder_Predictor_Belle.py#L...` 已一併更新（插入點之前的不受影響）。

另外發現 2026/09/08 那節指向 `ReduceLROnPlateau`（拿掉 `verbose=1`）的兩個連結 **在這次改動之前就已經失效**了，順手一起修正成 [L1522](Model_Builder_Predictor_Belle.py#L1522)、[L1942](Model_Builder_Predictor_Belle.py#L1942)。所有連結都逐一對照過實際程式碼內容確認無誤。

## 尚未驗證
改完還沒在真實訓練中跑過。

# 2026/09/11 評估 Time Warping：結論是不要做（只做分析，未改程式）

## 問題
考慮在 `augment_signal` 加入 Time Warping（時間軸局部拉伸／壓縮但保持波形形狀），疑慮是會不會破壞波形。

## 結論：不要做，但風險不是「破壞波形」
插值後波形形狀其實保留得很好。真正的問題嚴重得多：**time warping 擾動的正是帶有標籤資訊的量，等於把標籤本身抹掉。**

## 證據一：時間軸資訊就是判別依據
量測每個 uuid 各類別的 T 波峰位置（R 峰固定在 idx 50，所以這是 R-T 間期，QT 的代理指標；`fs=250`，1 idx = 4 ms）：

| uuid | High − Normal | Low − Normal |
|---|---|---|
| 2197 | −32 ms | — |
| 2199 | −12 ms | — |
| 2204 | −32 ms | — |
| 2206 | −20 ms | — |
| 2208 | −12 ms | −16 ms |
| 2210 | −8 ms | +4 ms |
| 2215 | −24 ms | **+20 ms** |
| 2216 | −28 ms | **+56 ms** |
| 2223 | −24 ms | −8 ms |
| 2249 | −24 ms | — |

**High 相對 Normal 的 T 波提前，10/10 個 uuid 方向一致**，且類別內 IQR 只有 2–12 idx，比類別間差距還小 —— 是個乾淨、跨受試者一致的特徵。Low 那邊 2215／2216 出現 +20／+56 ms 的延後，方向符合**低血糖造成 QT 延長**這個已知生理現象。

## 證據二：模擬 warp 對可分性的破壞
以「T 波峰位置」單一特徵區分 Normal vs High 的 AUC（warp 保持 R 峰不動、長度仍 150）：

| | 無 warp | ±5% | ±10% | ±20% |
|---|---|---|---|---|
| 10 個 uuid 平均 AUC | **0.813** | 0.787 | 0.751 | 0.660 |

±20% 吃掉約一半可用的類別可分性（0.313 → 0.160）。

原因很直觀：±10% 的 warp 讓 T 峰移動 ±(120−50)×0.10 ≈ **±7 idx ≈ ±28 ms**，而 High 對 Normal 的整體差距只有 8–32 ms。**擾動幅度跟訊號本身同量級。**

## 為什麼現有的 ±5 點平移沒有這個問題
這是關鍵對比：

- **平移是剛性的**：整個視窗一起移動，R-T 間期、QRS 寬度、所有時間間隔完全不變。改變的只是波形在視窗裡的絕對位置，而那本來就不該是判別依據。
- **warp 改變間期本身**：它擾動的正是帶有標籤資訊的量。

同理，現有的**全域振幅縮放也是安全的**：訊號已 min-max 正規化（R 峰 = 1.0），全域乘 0.9~1.1 保持所有比例不變，T/R 振幅比原封不動。

順帶量了 T 波振幅，High 比 Normal 低（9/10 個 uuid），代表**振幅同樣編碼了標籤** —— 所以也**不要加「只縮放某一段」的振幅增強**，那會跟 warp 犯一樣的錯。

## 一條更好的增強方向（尚未實作）
真正的限制是資料量。[Data_Parsing_Belle.py L407-412](Data_Parsing_Belle.py#L407-L412) 顯示每個 .txt 是一段 10 秒 ECG 裡所有心跳平均後的 mean wave（跳過第一拍和最後一拍）。10 秒在 250 Hz 下約有 10–17 拍。

可以**對這些心跳做 bootstrap：每次隨機抽一個子集（例如 70%）去平均，產生多個不同的 mean wave**。這樣得到的是真實、生理上正確、標籤完全正確的新樣本，沒有任何合成失真。代價是要動資料處理管線並重跑 parsing。

## 給之後的判準
這份資料的標籤同時編碼在**時間間期**和**振幅比例**上。任何新增的 augmentation 都要先問一句：**它會不會改動這兩者？** 會的話就是在抹掉標籤，不管它在一般 ECG 任務上多標準。

# 2026/09/11 加入頻域平滑幅度擾動，並改用開關單獨測試（只改 M5_DataAug.py）

## 問題
承上一節：time warping 不能用，因為它擾動的是帶標籤資訊的時間間期。那**把訊號轉到頻域加擾動再轉回時域**可行嗎？

## 可行，但只有一種變體能用
把幾種做法放在同一個尺度上比 —— **換到多少多樣性（RMS 變化）vs 賠掉多少標籤（T 峰位置的 Normal/High 可分性 AUC 損失）**：

| 方法 | 參數 | 多樣性 | 標籤損失 | 效率 |
|---|---|---|---|---|
| **頻域 平滑幅度包絡** | ±40% | 10.1% | **1.1%** | **9.6** |
| **頻域 平滑幅度包絡** | ±60% | 15.3% | **1.1%** | **14.2** |
| 頻域 per-bin 幅度 | ±40% | 12.5% | 21.7% | 0.6 |
| 頻域 相位 | ±0.4 rad | 12.5% | 26.4% | 0.5 |
| 時域 高斯雜訊（現有） | std 0.02 | 6.7% | 12.6% | 0.5 |
| 時域 time warp | ±10% | 9.5% | 24.2% | 0.4 |

**平滑幅度包絡的效率比其他所有方法高 15~25 倍。** 其餘全部落在 0.4~0.6，等於「加多少多樣性就賠掉多少標籤」。

## 為什麼差這麼多
關鍵：**相位攜帶時間資訊，而時間資訊就是標籤**（上一節已證明 T 波間期是判別依據）。

- **相位擾動**：直接攻擊時序，跟 time warping 同一類錯誤。
- **per-bin 隨機幅度**：雖然沒碰相位，但用白雜訊乘頻譜＝時域跟隨機 kernel 卷積，產生 ringing 把波峰位置抹糊，**間接**破壞時序。
- **平滑幅度包絡**：緩慢變化的頻譜傾斜，時域上等同輕微的濾波器響應變化 —— 正是不同電極接觸、皮膚阻抗、裝置差異造成的真實變異。生理上合理且完全不動相位。

另一個好處：前處理已做過 `nk.ecg_clean` 和 `baseline_remove`，平滑包絡不會引入頻帶外內容；而高斯白雜訊會加入前處理本來會濾掉的高頻，造成訓練與推論的分布落差。

## 頻譜結構（150 點 @ 250 Hz）
rfft 只有 76 個 bin，解析度 1.67 Hz。能量分布：0–5 Hz 佔 23.5%、5–10 Hz 佔 20.9%、10–20 Hz 佔 31.3%、20–40 Hz 佔 19.9%，**40 Hz 以上只剩 4.4%**。所以「只擾動高頻」這種保守做法幾乎不會產生任何多樣性，沒有意義。

## 實作：改用開關，這次只測頻域
為了單獨測試頻域擾動，**沒有把舊的增強註解掉，改成在檔案開頭加開關**（[L57-64](M5_DataAug.py#L57-L64)），之後要評估組合時只要改這幾個值，不必再動 `augment_signal`：

```python
AUG_GAUSSIAN_NOISE   = False   ##加高斯雜訊
AUG_AMPLITUDE_SCALE  = False   ##振幅隨機縮放 0.9~1.1 倍
AUG_TIME_SHIFT       = False   ##時間軸剛性平移(邊緣補值)
AUG_FREQ_MAGNITUDE   = True    ##頻域平滑幅度擾動(只動幅度包絡，相位不動)

FREQ_AUG_AMPLITUDE   = 0.4     ##增益包絡的擾動幅度(±40%)
FREQ_AUG_CONTROL_PTS = 5       ##控制點數，越少包絡越平滑、越不易產生 ringing
```

新增的擾動在 [L138](M5_DataAug.py#L138)：

```python
sig_len = signal.shape[-1]
spec = torch.fft.rfft(signal, dim=-1)
ctrl = 1.0 + (torch.rand(FREQ_AUG_CONTROL_PTS, device=signal.device) - 0.5) * 2 * FREQ_AUG_AMPLITUDE
gain = F.interpolate(ctrl.view(1, 1, -1), size=spec.shape[-1],
                     mode='linear', align_corners=True).view(-1).clone()
gain[0] = 1.0   ###DC 不動，避免整體基線飄移
signal = torch.fft.irfft(spec * gain, n=sig_len, dim=-1)
```

**踩到的坑**：頻譜變數不能命名為 `F` —— 本檔開頭已經有 `import torch.nn.functional as F`（[L25](M5_DataAug.py#L25)），命名成 `F` 會在函式內遮蔽掉這個模組，讓下一行的 `F.interpolate` 直接壞掉。所以用 `spec`。

## 驗證（透過真正的 `ECGDataset` 端對端跑過）
用 uuid 2197 的 Train 資料（3713 筆，`method='time'`, `classes=2`, `augment=True`）：

- 實際被增強的比例 51%（開關機率 0.5，符合預期）
- 輸出 shape `(1,150)`、dtype float32 不變，無 NaN/Inf
- RMS 變化中位數 **7.9%**（有效的多樣性）
- **R 峰位置 100% 完全不變**
- **T 峰位移 ≤1 idx 的比例 98.0%**（標籤資訊保住了）
- 四個開關全部關掉時，訊號完全不變（確認開關真的有效）
- `DataLoader` 走一遍正常，batch `(32, 1, 150)`

## 三個要注意的限制
1. **現有的高斯雜訊效率是 0.5，跟 time warping 同一級。** 但這不代表它沒用 —— 雜訊除了多樣性還有抗噪穩健性的價值，那不會反映在這個指標上。所以是**新增**頻域擾動，不是拿它替換雜訊。
2. **「標籤損失」只量了 T 峰位置這一個特徵。** 它確實有判別力（基準 AUC 0.80），但模型可能還用了別的特徵。保護它是必要條件，不是充分條件。
3. **「多樣性」用 RMS 變化衡量對剛性平移不公平** —— 平移會產生很大的逐點差異但波形沒變，所以剛性平移的效率數字是被高估的，不要拿它跟頻域方法直接比。

**效率排序不等於準確率會提升**，最終仍需實際訓練驗證。

## 這次只改了 M5_DataAug.py
`Model_Builder_Predictor_Belle.py` 的 `augment_signal`（[L866](Model_Builder_Predictor_Belle.py#L866)）**沒有跟著改**，仍是原本三種時域增強、且沒有開關。兩個檔案的 augmentation 邏輯從這次開始分岔了，之後如果要讓 MBP 跟上，記得一併移植開關與頻域區塊。

## 尚未驗證
還沒實際訓練過。下次跑的時候這一版等於「只有頻域擾動」的 WithAug，可以直接跟 NoAug 比。

# 2026/09/11 Performance_Matrix 改存到 perf_Matrix/，並修好 NoAug/WithAug 互相覆蓋的問題

## 發現的問題
混淆矩陣的輸出路徑是：

```python
output_excel = os.path.join(basepath, uuid+"_Performance_Matrix.xlsx")
```

直接寫在 `basepath`（= `Model/`）根目錄，**路徑裡完全沒有 `splitting_ratio` 也沒有 `model_subdir`** —— 跟 `Best_*_Model`、`Temp_*_Model`、`GlucoseData` 等其他所有輸出都不一樣。結果就是不同模式跑完會互相覆蓋同名檔案。

**這不是理論問題，已經實際發生了。** 比對時間戳可以確認：

| 資料夾 | Performance_*.txt 時間範圍 |
|---|---|
| NoAug | 2026-09-09 09:28 ~ 10:02 |
| WithAug_1 | 2026-09-09 10:19 ~ 11:31 |
| WithAug_2 | 2026-09-11 01:52 ~ 03:02 |

而 `Model/` 根目錄那 10 個 xlsx 的時間是 **2026-09-11 01:57 ~ 03:05**，對應的是 WithAug_2。也就是說 **NoAug 與 WithAug_1 的混淆矩陣都已經被 WithAug_2 蓋掉，救不回來了**（`.pth` 和 `Performance_*.txt` 因為路徑有分模式，都還在）。

## 改法
三分類（[L1387](M5_DataAug.py#L1387)）與二分類（[L1929](M5_DataAug.py#L1929)）兩處都改成：

```python
output_excel = os.path.join(basepath,"perf_Matrix",model_subdir,uuid+"_Performance_Matrix.xlsx")
os.makedirs(os.path.dirname(output_excel), exist_ok=True)
```

輸出結構變成：

```
Model/perf_Matrix/
├── NoAug/       <uuid>_Performance_Matrix.xlsx
├── WithAug_1/   <uuid>_Performance_Matrix.xlsx
└── WithAug_2/   <uuid>_Performance_Matrix.xlsx
```

`model_subdir` 是空字串時 `os.path.join` 會自動略過該層，退回 `perf_Matrix/` 根目錄，不會產生空目錄或路徑錯誤。加了 `os.makedirs(..., exist_ok=True)`，因為這個資料夾不像 `save_path` 在訓練前就被建好。

## 既有檔案的處理
`Model/` 根目錄原本那 10 個 xlsx 已依時間戳判定屬於 WithAug_2，搬到 `Model/perf_Matrix/WithAug_2/`。`Model/` 根目錄現在沒有殘留的 xlsx。

（xlsx 有被 .gitignore 排除，所以這個搬移只影響本機，不進版控。）

## 驗證
用暫存目錄實測三種 `model_subdir` 的路徑產生結果：

| model_subdir | 產生的路徑 |
|---|---|
| `'WithAug_2'` | `perf_Matrix/WithAug_2/2197_Performance_Matrix.xlsx` |
| `'NoAug'` | `perf_Matrix/NoAug/2208_Performance_Matrix.xlsx` |
| `''`（預設值） | `perf_Matrix/2249_Performance_Matrix.xlsx` |

三種都能正確建立目錄並寫出 xlsx。

## 這次只改了 M5_DataAug.py
`Model_Builder_Predictor_Belle.py` 沒有這段 Excel 匯出程式碼，所以不受影響。

## 給之後的提醒
**任何會依 NoAug/WithAug 分別產生的輸出，路徑裡一定要帶 `model_subdir`。** 這次是靠時間戳才反推出那批 xlsx 屬於哪一輪；如果當時多跑幾輪，就完全分不出來了。
