#!/bin/bash

# 出力ディレクトリの作成
mkdir -p results_experiment_text2img_high_snr

# 共通設定
# 注意: ディレクトリ構成が scripts/img2img.py となっていることを前提としています
PY_CMD="python -m scripts.img2img --config models/ldm/text2img256/config.yaml --ckpt models/ldm/text2img256/model.ckpt --output_dir results_experiment_text2img_high_snr"

# 手法の定義
# 既存手法
METHODS_BASE="structural raw random edge_rec edge_gt"
# 新規追加手法 (提案手法 Smart Hybrid と 理論上限 Oracle)
METHODS_PROPOSED="smart_hybrid oracle"

# 共通パラメータ
SA_VAL="0.0"
RATE="0.1"  # 必要に応じて変更してください（例: 0.2）
SNR="-1"    # 修正: =の前後スペース削除

echo "================================================================="
echo " STARTING RUN (Text2Img): SNR Range -5 to 5, Rate ${RATE}"
echo "================================================================="
echo "  > Running Baselines..."

# 修正: --target_methods の重複を削除
$PY_CMD --snr -3 -r ${RATE} --struct_alpha ${SA_VAL} --target_methods smart_hybrid oracle > log_snr-3_proposed.txt 2>&1

# SNR -5, 0, 5 でループを実行

echo "================================================================="
echo " All experiments completed."
echo "================================================================="