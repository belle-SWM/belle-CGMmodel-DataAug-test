# -*- coding: utf-8 -*-
"""
compare_txt_outputs.py

用途：比對舊版與新版 pipeline 產出的 ECG txt 檔案內容是否完全一致。
      對應 _write_ecg_data() 產出的檔案，命名格式：
      {uuid}_{time_str}_{flag}_{index}_{glucose_value}.txt
      內容：每行一個 ECG 數值。

用多核心 (multiprocessing.Pool) 平行比對，加快大量檔案的驗證速度。

使用方式：
    直接修改下方 CONFIG 區塊的路徑，然後執行：
    python compare_txt_outputs.py
"""

import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import csv
import time


# =========================== CONFIG ===========================
# 依實際環境修改以下路徑與參數

OLD_DIR = r'C:\Users\belle\Documents\algorithm_blood_glucose-main\Model\RawData\1726'   # 舊版輸出資料夾 (RawData 根目錄)
NEW_DIR = r'C:\Users\belle\Documents\algorithm_blood_glucose-main\Model_fromNEW\RawData\1726'   # 新版輸出資料夾 (RawData 根目錄)

WORKERS = cpu_count()      # 平行處理程序數量，預設用滿所有CPU核心clear
TOLERANCE = 0               # 數值比對容忍誤差，0代表完全字串比對
                             # 若懷疑因浮點數表示法差異(如 1.0 vs 1)誤判不一致，可設定例如 1e-9

REPORT_CSV = r'C:\Users\belle\Documents\algorithm_blood_glucose-main\compare_report.csv'  # 詳細比對結果輸出路徑
                             # 若不需要輸出報告，設為 None 即可
# =================================================================


def collect_txt_files(root_dir):
    """
    遞迴收集 root_dir 底下所有 .txt 檔案，回傳 {相對路徑: 絕對路徑} 的 dict。
    相對路徑作為比對兩邊檔案是否存在的 key（通常結構是 uuid/xxx.txt）。
    """
    file_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.txt'):
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root_dir)
                file_map[rel_path] = abs_path
    return file_map


def compare_single_file(args, tolerance=0):
    """
    比對單一檔案內容是否一致。這個 function 會被 multiprocessing 平行呼叫。

    input ---
    args: (rel_path, old_abs_path, new_abs_path)
    tolerance: 數值容忍誤差，0代表完全字串比對

    output ---
    (rel_path, status, detail)
    status: 'match' / 'mismatch' / 'error'
    """
    rel_path, old_abs_path, new_abs_path = args

    try:
        with open(old_abs_path, 'r') as f:
            old_lines = [line.rstrip('\n') for line in f.readlines()]
        with open(new_abs_path, 'r') as f:
            new_lines = [line.rstrip('\n') for line in f.readlines()]
    except Exception as e:
        return (rel_path, 'error', f'讀檔失敗: {e}')

    if len(old_lines) != len(new_lines):
        return (rel_path, 'mismatch',
                f'行數不同: old={len(old_lines)} 行, new={len(new_lines)} 行')

    if tolerance == 0:
        # 完全字串比對
        for i, (old_v, new_v) in enumerate(zip(old_lines, new_lines)):
            if old_v != new_v:
                return (rel_path, 'mismatch',
                        f'第 {i+1} 行不同: old="{old_v}" new="{new_v}"')
    else:
        # 數值容忍誤差比對 (避免浮點數表示法差異誤判)
        for i, (old_v, new_v) in enumerate(zip(old_lines, new_lines)):
            try:
                old_f = float(old_v)
                new_f = float(new_v)
            except ValueError:
                if old_v != new_v:
                    return (rel_path, 'mismatch',
                            f'第 {i+1} 行無法轉為數值且不同: old="{old_v}" new="{new_v}"')
                continue
            if abs(old_f - new_f) > tolerance:
                return (rel_path, 'mismatch',
                        f'第 {i+1} 行數值差異超過容忍值: old={old_f} new={new_f} '
                        f'diff={abs(old_f - new_f)}')

    return (rel_path, 'match', '')


def main():
    old_dir = OLD_DIR
    new_dir = NEW_DIR
    workers = WORKERS
    tolerance = TOLERANCE
    report_csv = REPORT_CSV

    if not os.path.isdir(old_dir):
        print(f'[錯誤] 找不到舊版資料夾: {old_dir}')
        sys.exit(1)
    if not os.path.isdir(new_dir):
        print(f'[錯誤] 找不到新版資料夾: {new_dir}')
        sys.exit(1)

    print(f'掃描舊版檔案: {old_dir}')
    old_files = collect_txt_files(old_dir)
    print(f'  共找到 {len(old_files)} 個 txt 檔')

    print(f'掃描新版檔案: {new_dir}')
    new_files = collect_txt_files(new_dir)
    print(f'  共找到 {len(new_files)} 個 txt 檔')

    old_keys = set(old_files.keys())
    new_keys = set(new_files.keys())

    only_in_old = old_keys - new_keys
    only_in_new = new_keys - old_keys
    common_keys = old_keys & new_keys

    print(f'\n兩邊都有的檔案: {len(common_keys)}')
    print(f'只在舊版存在 (新版缺少): {len(only_in_old)}')
    print(f'只在新版存在 (舊版沒有，新版多產出): {len(only_in_new)}')

    if only_in_old:
        print('\n[只在舊版存在的檔案清單] (最多列出前20筆)')
        for k in sorted(only_in_old)[:20]:
            print(f'  {k}')
        if len(only_in_old) > 20:
            print(f'  ...(還有 {len(only_in_old) - 20} 筆未列出)')

    if only_in_new:
        print('\n[只在新版存在的檔案清單] (最多列出前20筆)')
        for k in sorted(only_in_new)[:20]:
            print(f'  {k}')
        if len(only_in_new) > 20:
            print(f'  ...(還有 {len(only_in_new) - 20} 筆未列出)')

    # ---- 用 multiprocessing 平行比對兩邊都有的檔案內容 ----
    compare_args = [
        (rel_path, old_files[rel_path], new_files[rel_path])
        for rel_path in common_keys
    ]

    print(f'\n開始平行比對 {len(compare_args)} 個檔案內容 (使用 {workers} 個處理程序)...')
    t0 = time.perf_counter()

    compare_func = partial(compare_single_file, tolerance=tolerance)
    with Pool(processes=workers) as pool:
        results = pool.map(compare_func, compare_args)

    elapsed = time.perf_counter() - t0
    print(f'比對完成，耗時 {elapsed:.2f} 秒')

    match_count = sum(1 for _, status, _ in results if status == 'match')
    mismatch_results = [r for r in results if r[1] == 'mismatch']
    error_results = [r for r in results if r[1] == 'error']

    print(f'\n===== 內容比對結果 =====')
    print(f'一致 (match): {match_count}')
    print(f'不一致 (mismatch): {len(mismatch_results)}')
    print(f'讀取錯誤 (error): {len(error_results)}')

    if mismatch_results:
        print('\n[不一致檔案明細] (最多列出前20筆)')
        for rel_path, status, detail in mismatch_results[:20]:
            print(f'  {rel_path}: {detail}')
        if len(mismatch_results) > 20:
            print(f'  ...(還有 {len(mismatch_results) - 20} 筆未列出，詳見 report_csv)')

    if error_results:
        print('\n[讀取錯誤檔案明細]')
        for rel_path, status, detail in error_results:
            print(f'  {rel_path}: {detail}')

    # ---- 輸出詳細報告 csv (可選) ----
    if report_csv:
        with open(report_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['relative_path', 'status', 'detail'])
            for k in sorted(only_in_old):
                writer.writerow([k, 'missing_in_new', ''])
            for k in sorted(only_in_new):
                writer.writerow([k, 'missing_in_old', ''])
            for rel_path, status, detail in results:
                writer.writerow([rel_path, status, detail])
        print(f'\n詳細報告已輸出: {report_csv}')

    # ---- 最終結論 ----
    all_pass = (len(only_in_old) == 0 and len(only_in_new) == 0
                and len(mismatch_results) == 0 and len(error_results) == 0)

    print('\n===== 最終結論 =====')
    if all_pass:
        print('✓ 新舊版本輸出完全一致，可以進行替換。')
    else:
        print('✗ 新舊版本輸出有差異，請勿替換正式程式碼，先排查上述明細。')

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
