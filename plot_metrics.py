import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 設定エリア
# ==========================================
# 実験結果のルートディレクトリ
ROOT_DIR = "results_experiment_text2img_high_snr"

# グラフの出力先
OUTPUT_IMAGE = "metrics_summary_plot.png"

# 1. プロットしたいSNRのリスト
# None または空リスト [] にすると、すべてのSNRをプロットします。
TARGET_SNRS = [1.0, 2.0, 3.0, 4.0, 5.0]
TARGET_SNRS = [-5.0,-4.0, -3.0,-2.0]
# TARGET_SNRS = None 

# 2. プロットしたい手法のリスト
# 必要なものだけコメントアウトを外して選んでください。
TARGET_METHODS = [
    #"pass1",
    "pass2_structural",
    "pass2_raw",
    "pass2_random",       # <--- 追加しました
    # "pass2_edge_rec",
    "pass2_edge_gt",
    # "pass2_hybrid",
    "pass2_smart_hybrid",
    "pass2_sbf"
    #"pass2_oracle"
]
# TARGET_METHODS = None  # 全てプロットする場合はこちら

# プロット対象の指標
METRICS = ["psnr", "lpips", "dists", "clip", "fid"]

# 表示名マッピング (グラフの凡例に使われます)
METHOD_LABELS = {
    "pass1": "Pass 1 (Initial)",
    "pass2_structural": "Struct",
    "pass2_raw": "Raw",
    "pass2_random": "Random",  # <--- 追加しました
    "pass2_edge_rec": "Edge (Rec)",
    "pass2_edge_gt": "Edge (GT)",
    "pass2_hybrid": "Hybrid",
    "pass2_smart_hybrid": "Smart Hybrid",
    "pass2_sbf": "SBF (Proposed)",
    "pass2_oracle": "Oracle"
}

# 線のスタイル設定
STYLE_CONFIG = {
    "pass1": {"color": "black", "linestyle": "--", "marker": "x", "linewidth": 1.5},
    "pass2_sbf": {"color": "red", "linestyle": "-", "marker": "o", "linewidth": 2.5},
    "pass2_smart_hybrid": {"color": "orange", "linestyle": "-", "marker": "s", "linewidth": 2.0},
    "pass2_oracle": {"color": "blue", "linestyle": ":", "marker": "*", "linewidth": 1.5},
    "pass2_structural": {"color": "green", "linestyle": "-.", "marker": "^", "linewidth": 1.5},
    "pass2_hybrid": {"color": "purple", "linestyle": "-.", "marker": "v", "linewidth": 1.5},
    "pass2_random": {"color": "gray", "linestyle": "--", "marker": "d", "linewidth": 1.5}, # <--- 追加 (グレーの点線)
    "pass2_raw": {"color": "brown", "linestyle": ":", "marker": ".", "linewidth": 1.5},
    "pass2_edge_gt": {"color": "teal", "linestyle": "-.", "marker": "p", "linewidth": 1.5},
}

# ==========================================
# データ読み込み関数
# ==========================================
def load_data(root_dir):
    search_pattern = os.path.join(root_dir, "**", "metrics_*.json")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"Error: No json files found in {root_dir}")
        return None

    data_store = {} 
    print(f"Found {len(files)} metric files.")

    for fpath in files:
        # SNR抽出
        match = re.search(r"snr(-?\d+\.?\d*)", fpath)
        if not match:
            match = re.search(r"snr(-?\d+\.?\d*)", os.path.dirname(fpath))
        
        if match:
            snr = float(match.group(1))
        else:
            continue

        try:
            with open(fpath, 'r') as f:
                content = json.load(f)
                
            summary = content.get("summary", {})
            avgs = summary.get("averages", {})
            fids = summary.get("fid", {})

            if snr not in data_store:
                data_store[snr] = {}

            all_methods = set(list(avgs.keys()) + list(fids.keys()))

            for method in all_methods:
                if method not in data_store[snr]:
                    data_store[snr][method] = {}
                
                if method in avgs:
                    for m, v in avgs[method].items():
                        data_store[snr][method][m] = v
                
                if method in fids and fids[method] is not None:
                    data_store[snr][method]["fid"] = fids[method]

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    return data_store

# ==========================================
# プロット関数
# ==========================================
def plot_results(data_store):
    if not data_store:
        return

    # --- 1. SNRのフィルタリング ---
    available_snrs = sorted(data_store.keys())
    
    if TARGET_SNRS and len(TARGET_SNRS) > 0:
        snr_list = [s for s in available_snrs if s in TARGET_SNRS]
        missing_snr = set(TARGET_SNRS) - set(snr_list)
        if missing_snr:
            print(f"Warning: Data for SNRs {missing_snr} not found.")
    else:
        snr_list = available_snrs

    if not snr_list:
        print("No SNRs to plot.")
        return

    print(f"Plotting for SNRs: {snr_list}")

    # --- 2. 手法(Method)のフィルタリング ---
    available_methods = set()
    for snr in snr_list:
        available_methods.update(data_store[snr].keys())

    if TARGET_METHODS and len(TARGET_METHODS) > 0:
        plot_methods = [m for m in TARGET_METHODS if m in available_methods]
        missing_methods = set(TARGET_METHODS) - set(available_methods)
        if missing_methods:
            print(f"Warning: Data for Methods {missing_methods} not found or not in selected SNRs.")
    else:
        # 指定がない場合は自動整列
        sorted_methods = sorted(list(available_methods))
        priority_order = ["pass1", "pass2_sbf", "pass2_smart_hybrid", "pass2_oracle"]
        remaining = [m for m in sorted_methods if m not in priority_order]
        plot_methods = priority_order + remaining

    print(f"Plotting Methods: {plot_methods}")

    # --- グラフ作成 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        for method in plot_methods:
            x_vals = []
            y_vals = []
            
            for snr in snr_list:
                if method in data_store[snr] and metric in data_store[snr][method]:
                    val = data_store[snr][method][metric]
                    if val is not None:
                        x_vals.append(snr)
                        y_vals.append(val)
            
            if len(x_vals) > 0:
                style = STYLE_CONFIG.get(method, {})
                label_text = METHOD_LABELS.get(method, method)
                
                if not style:
                    # デフォルトスタイル
                    ax.plot(x_vals, y_vals, label=label_text, marker='.', alpha=0.5)
                else:
                    ax.plot(x_vals, y_vals, label=label_text, **style)

        ax.set_title(metric.upper())
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(metric)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        if idx == 0:
            ax.legend(fontsize='small', loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Graph saved to: {os.path.abspath(OUTPUT_IMAGE)}")
    plt.show()

if __name__ == "__main__":
    data = load_data(ROOT_DIR)
    plot_results(data)