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

## 小提醒
- 一定要用 Python 3.9，不要用 3.12，目前鎖定的 neurokit2/nolds 版本在 3.12 上沒驗證過。
- 之後若要升級 neurokit2、nolds、scipy 版本，記得先 `import SWMlib` 測試一次確認不會炸，再更新 requirements.txt。
