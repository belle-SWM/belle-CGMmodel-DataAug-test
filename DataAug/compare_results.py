# -*- coding: utf-8 -*-
# ============================================================
# 比較不同 augmentation 模式訓練出來的結果
#
# 掃描 Model/<splitting_ratio>/<mode>/Best_{Two,Three}Classes_Model/<uuid>/Performance_*.txt，
# 把每個 (uuid, 分類數, 模式) 的指標整理成一張大表，以 baseline 模式為基準顯示 Δ。
#
# 用法：
#     python compare_results.py
#     python compare_results.py --baseline NoAug
#     python compare_results.py --source historic      ##改用 Historic_Best_Performance.txt
#     python compare_results.py --no-plot              ##不畫圖
#
# 輸出：
#     terminal 大表
#     Model/<splitting_ratio>/comparison/comparison.csv
#     Model/<splitting_ratio>/comparison/comparison_TwoClasses.png
#     Model/<splitting_ratio>/comparison/comparison_ThreeClasses.png
# ============================================================

import os
import re
import sys
import csv
import glob
import argparse

###主表格要顯示的四個指標（CSV 會收錄檔案裡所有欄位，不只這四個）
HEADLINE_METRICS = ['Sensitivity', 'Specificity', 'F1', 'Acc']

###每個指標在表頭要顯示的短名稱
SHORT_NAME = {'Sensitivity': 'Sens', 'Specificity': 'Spec', 'F1': 'F1', 'Acc': 'Acc'}

CLASS_DIRS = [('TwoClasses', 'Best_TwoClasses_Model'),
              ('ThreeClasses', 'Best_ThreeClasses_Model')]

###取自 dataviz reference palette 的 categorical slot 1~3（blue / orange / aqua）。
###這三個 slot 在文件裡已驗證過 all-pairs（light: CVD ΔE 9.2、normal-vision 24.0），
###順序固定不可循環使用；超過 8 個模式就不該再配新顏色，應改成分面或歸入 Other。
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#d9d8d4'


def parse_performance_file(path):
    """把 Performance_*.txt / Historic_Best_Performance.txt 解析成 dict。

    檔案長這樣（分隔線與逗號分隔的多欄位都要處理）：
        Sensitivity:52.09
        -----------------------
        TP_Normal:304, FP_Normal:632, FN_Normal:164, TN_Normal:771
        Model_Name:BestModel_007_2026_09_09_0938.pth
    """
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or set(line) <= set('-'):   ###跳過分隔線
                continue
            for part in line.split(','):            ###TP/FP/FN/TN 那行一列有多個欄位
                if ':' not in part:
                    continue
                key, _, value = part.partition(':')
                ###M5 寫佔位檔時有一行漏了 \n，會變成 "-----High_Sensitivity:0"，
                ###key 會被分隔線汙染，所以要把前後的 '-' 一併去掉
                key, value = key.strip().strip('-').strip(), value.strip()
                if not key:
                    continue
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value                ###例如 Model_Name
    return out


def is_placeholder(row):
    """判斷是不是「還沒有結果」的佔位檔。

    M5 在訓練開始前就會先寫一份全 0 的 Performance_*.txt(見 BuildModel_* 開頭)，
    等訓練跑完才覆蓋。那份檔案不是 0 分成績，而是「尚未產生結果」，
    直接當成 0 拿去平均會嚴重拉低該模式的數字。
    判準：四個主指標全為 0，且所有 TP/FP/FN/TN 也全為 0。
    """
    metrics = [row.get(m) for m in HEADLINE_METRICS]
    if any(not isinstance(v, float) or v != 0 for v in metrics):
        return False
    counts = [v for k, v in row.items()
              if re.match(r'^(TP|FP|FN|TN)(_|$)', k) and isinstance(v, float)]
    return bool(counts) and all(v == 0 for v in counts)


def pick_source_file(uuid_dir, source):
    """挑出要用的績效檔。

    latest   : 最新一份 Performance_*.txt（同一個模式可能跑過多次，各留一份）
    historic : Historic_Best_Performance.txt（該模式歷來最好的一次）
    """
    if source == 'historic':
        path = os.path.join(uuid_dir, 'Historic_Best_Performance.txt')
        return path if os.path.exists(path) else None

    files = sorted(glob.glob(os.path.join(uuid_dir, 'Performance_*.txt')))
    ###檔名格式 Performance_<version>_<YYYY_MM_DD_HHMM>.txt，字典序即時間序
    return files[-1] if files else None


def discover_modes(ratio_dir):
    """找出所有模式資料夾（底下要有 Best_*Classes_Model 才算）"""
    modes = []
    if not os.path.isdir(ratio_dir):
        return modes
    for name in sorted(os.listdir(ratio_dir)):
        d = os.path.join(ratio_dir, name)
        if not os.path.isdir(d):
            continue
        if any(os.path.isdir(os.path.join(d, sub)) for _, sub in CLASS_DIRS):
            modes.append(name)
    return modes


def collect(basepath, ratio, source):
    """回傳 rows: list of dict(uuid, classes, mode, <metrics...>)"""
    ratio_dir = os.path.join(basepath, ratio)
    modes = discover_modes(ratio_dir)
    rows = []
    pending = []      ###只有佔位檔、還沒有實際結果的（通常代表正在訓練）
    for mode in modes:
        for classes, subdir in CLASS_DIRS:
            model_dir = os.path.join(ratio_dir, mode, subdir)
            if not os.path.isdir(model_dir):
                continue
            for uuid in sorted(os.listdir(model_dir)):
                uuid_dir = os.path.join(model_dir, uuid)
                if not os.path.isdir(uuid_dir):
                    continue
                path = pick_source_file(uuid_dir, source)
                if path is None:
                    print('  (略過，找不到績效檔) %s' % uuid_dir)
                    continue
                row = {'uuid': uuid, 'classes': classes, 'mode': mode,
                       'source_file': os.path.relpath(path, basepath)}
                row.update(parse_performance_file(path))
                if is_placeholder(row):
                    pending.append('%s/%s/%s' % (mode, classes, uuid))
                    continue
                rows.append(row)
    return rows, modes, pending


def fmt(value, width=9):
    return ('%*.2f' % (width, value)) if isinstance(value, float) else ('%*s' % (width, '-'))


def print_table(rows, modes, baseline):
    """一張大表 + Δ 欄：每個 uuid 依模式列出，非基準模式下方多一列相對基準的增減"""
    order = {m: i for i, m in enumerate(modes)}
    for classes, _ in CLASS_DIRS:
        sub = [r for r in rows if r['classes'] == classes]
        if not sub:
            continue
        uuids = sorted({r['uuid'] for r in sub})
        mode_w = max([len(m) for m in modes] + [9])

        print()
        print('=' * (12 + mode_w + 10 * len(HEADLINE_METRICS)))
        print('%s   (baseline: %s)' % (classes, baseline))
        print('=' * (12 + mode_w + 10 * len(HEADLINE_METRICS)))
        header = '%-6s %-*s' % ('uuid', mode_w, 'mode')
        for m in HEADLINE_METRICS:
            header += '%9s ' % SHORT_NAME.get(m, m)
        print(header)
        print('-' * len(header))

        for uuid in uuids:
            by_mode = {r['mode']: r for r in sub if r['uuid'] == uuid}
            base_row = by_mode.get(baseline)
            for mode in sorted(by_mode, key=lambda m: order.get(m, 99)):
                row = by_mode[mode]
                line = '%-6s %-*s' % (uuid, mode_w, mode)
                for m in HEADLINE_METRICS:
                    line += fmt(row.get(m)) + ' '
                print(line)

                ###非基準模式，下面補一列 Δ
                if base_row is not None and mode != baseline:
                    delta = ' ' * 6 + ' ' + '%-*s' % (mode_w, 'Δ')
                    for m in HEADLINE_METRICS:
                        a, b = row.get(m), base_row.get(m)
                        if isinstance(a, float) and isinstance(b, float):
                            delta += '%+9.2f ' % (a - b)
                        else:
                            delta += '%9s ' % '-'
                    print(delta)
            print('-' * len(header))

        ###各模式的平均。務必連 n 一起印：訓練可能還在跑，各模式涵蓋的 uuid 數不一定相同，
        ###把 n=1 的模式跟 n=5 的模式放在同一欄比較會嚴重誤導。
        print('%-6s %-*s' % ('MEAN', mode_w, ''))
        coverage = {}
        for mode in modes:
            vals = [r for r in sub if r['mode'] == mode]
            if not vals:
                continue
            coverage[mode] = {r['uuid'] for r in vals}
            line = '%-6s %-*s' % ('', mode_w, mode)
            for m in HEADLINE_METRICS:
                nums = [r[m] for r in vals if isinstance(r.get(m), float)]
                line += fmt(sum(nums) / len(nums) if nums else None) + ' '
            line += '  n=%d' % len(vals)
            print(line)

        ###涵蓋範圍不一致就明講，並補一組只用共同 uuid 的平均
        full = max((len(v) for v in coverage.values()), default=0)
        partial = sorted(m for m, v in coverage.items() if len(v) < full)
        if partial:
            common = set.intersection(*coverage.values()) if coverage else set()
            print()
            print('  !! 這些模式尚未跑完 %s 全部 %d 個 uuid：%s'
                  % (classes, full, ', '.join('%s(%d)' % (m, len(coverage[m])) for m in partial)))
            print('     上面的 MEAN 各自只平均自己有的 uuid，模式之間不可直接比較。')
            if common:
                print('     以下改用所有模式都有的 %d 個 uuid（%s）重算：'
                      % (len(common), ', '.join(sorted(common))))
                for mode in modes:
                    if mode not in coverage:
                        continue
                    vals = [r for r in sub if r['mode'] == mode and r['uuid'] in common]
                    line = '     %-*s' % (mode_w, mode)
                    for m in HEADLINE_METRICS:
                        nums = [r[m] for r in vals if isinstance(r.get(m), float)]
                        line += fmt(sum(nums) / len(nums) if nums else None) + ' '
                    print(line)
            else:
                print('     沒有任何 uuid 是所有模式都有的，無法算出可比較的平均。')


def write_csv(rows, out_path):
    """收錄解析到的所有欄位，不只主表那四個"""
    fixed = ['uuid', 'classes', 'mode', 'source_file']
    extra = sorted({k for r in rows for k in r} - set(fixed))
    ###主指標排前面，其餘（各類別 sensitivity、TP/FP/FN/TN、Model_Name）跟在後面
    ordered = [m for m in HEADLINE_METRICS if m in extra] + \
              [k for k in extra if k not in HEADLINE_METRICS]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=fixed + ordered)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r['classes'], r['uuid'], r['mode'])):
            writer.writerow(r)
    print('\nCSV 已輸出: %s' % out_path)


def plot(rows, modes, out_dir, ratio):
    """每個分類一張圖，2x2 子圖對應四個主指標；x 軸 uuid，每組 N 根長條代表 N 個模式"""
    try:
        import matplotlib
        matplotlib.use('Agg')       ###容器沒有顯示器
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print('\n(略過畫圖：%s)' % e)
        return

    if len(modes) > len(SERIES_COLORS):
        print('\n(略過畫圖：模式數 %d 超過調色盤的 %d 個 slot，'
              '再配新顏色會失去辨識度，請改用分面呈現)' % (len(modes), len(SERIES_COLORS)))
        return

    os.makedirs(out_dir, exist_ok=True)
    for classes, _ in CLASS_DIRS:
        sub = [r for r in rows if r['classes'] == classes]
        if not sub:
            continue
        uuids = sorted({r['uuid'] for r in sub})
        present = [m for m in modes if any(r['mode'] == m for r in sub)]

        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
        x = np.arange(len(uuids))
        ###留 2px 視覺間隙：總寬度只用到 0.82，再由各模式均分
        total_w = 0.82
        bar_w = total_w / len(present)

        for ax, metric in zip(axes.ravel(), HEADLINE_METRICS):
            ax.set_facecolor(SURFACE)
            for i, mode in enumerate(present):
                vals = []
                for u in uuids:
                    hit = [r for r in sub if r['uuid'] == u and r['mode'] == mode]
                    v = hit[0].get(metric) if hit else None
                    vals.append(v if isinstance(v, float) else np.nan)
                ax.bar(x + (i - (len(present) - 1) / 2) * bar_w, vals,
                       width=bar_w * 0.92,          ###0.92 留出相鄰長條之間的底色間隙
                       color=SERIES_COLORS[i], label=mode,
                       edgecolor=SURFACE, linewidth=1.0)

            ax.set_title(metric, color=TEXT_PRIMARY, fontsize=11, loc='left')
            ax.set_xticks(x)
            ax.set_xticklabels(uuids, color=TEXT_SECONDARY, fontsize=9)
            ax.set_ylim(0, 100)
            ax.tick_params(axis='y', colors=TEXT_SECONDARY, labelsize=9)
            ax.set_ylabel('%', color=TEXT_SECONDARY, fontsize=9)
            ###格線退到背景，只留水平線
            ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)
            for side in ['top', 'right', 'left']:
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(GRID_COLOR)

        ###兩個以上的序列一律要有圖例，識別不能只靠顏色
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(present),
                   frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)
        fig.suptitle('%s  ·  %s' % (classes, ratio), color=TEXT_PRIMARY,
                     fontsize=13, x=0.01, ha='left')
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.subplots_adjust(hspace=0.42)   ###避免下排子圖標題貼到上排的 x 軸標籤

        out = os.path.join(out_dir, 'comparison_%s.png' % classes)
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        print('圖表已輸出: %s' % out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description='比較不同 augmentation 模式的訓練結果')
    ap.add_argument('--basepath', default=os.path.join(here, 'Model'),
                    help='預設 <本檔所在目錄>/Model')
    ap.add_argument('--splitting-ratio', default='70_30')
    ap.add_argument('--baseline', default=None,
                    help='作為 Δ 基準的模式，預設自動選 NoAug，沒有的話用第一個')
    ap.add_argument('--source', choices=['latest', 'historic'], default='latest',
                    help='latest=最新的 Performance_*.txt；historic=Historic_Best_Performance.txt')
    ap.add_argument('--out-dir', default=None,
                    help='預設 <basepath>/<splitting_ratio>/comparison')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    rows, modes, pending = collect(args.basepath, args.splitting_ratio, args.source)
    if not rows:
        print('找不到任何績效檔。檢查 --basepath / --splitting-ratio 是否正確：%s'
              % os.path.join(args.basepath, args.splitting_ratio))
        return 1

    baseline = args.baseline or ('NoAug' if 'NoAug' in modes else modes[0])
    if baseline not in modes:
        print('指定的 baseline "%s" 不在找到的模式裡：%s' % (baseline, modes))
        return 1
    ###基準模式排最前面，Δ 才好對照
    modes = [baseline] + [m for m in modes if m != baseline]

    print('basepath        : %s' % args.basepath)
    print('splitting_ratio : %s' % args.splitting_ratio)
    print('找到的模式      : %s' % ', '.join(modes))
    print('績效來源        : %s' % args.source)
    if pending:
        print('尚無結果(佔位檔): %s' % ', '.join(pending))
        print('                  這些是 M5 在訓練前先建立的全 0 檔案，代表還在訓練中，已排除不計入。')

    print_table(rows, modes, baseline)

    out_dir = args.out_dir or os.path.join(args.basepath, args.splitting_ratio, 'comparison')
    write_csv(rows, os.path.join(out_dir, 'comparison.csv'))
    if not args.no_plot:
        plot(rows, modes, out_dir, args.splitting_ratio)
    return 0


if __name__ == '__main__':
    sys.exit(main())
