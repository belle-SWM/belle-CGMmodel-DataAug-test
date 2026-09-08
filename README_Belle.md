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
1. **`ReduceLROnPlateau(..., verbose=1, ...)` 在目前環境會崩潰**：目前 venv 裝的是 PyTorch 2.8，`verbose` 參數已經被移除，`BuildModel_ThreeClasses`/`BuildModel_TwoClasses` 原本一啟動建立 scheduler 就會丟 `TypeError`，跟這次的 augmentation 修改無關，是既有程式與新版 PyTorch 不相容。已把兩處（[L1430](Model_Builder_Predictor_Belle.py#L1430) 三分類、[L1844](Model_Builder_Predictor_Belle.py#L1844) 二分類）的 `verbose=1` 拿掉。
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
