#!/bin/bash

# =================================================================
#  PARTIAL NOISE EXPERIMENT RUNNER (Matched with run.sh)
# =================================================================

# 新しい出力ディレクトリの作成
OUT_DIR="results_partial_noise_experiment"
mkdir -p $OUT_DIR

# 共通設定 (scripts.img2img_partial_noise を使用)
PY_CMD="python -m scripts.img2img_partial_noise --config models/ldm/text2img256/config.yaml --ckpt models/ldm/text2img256/model.ckpt --output_dir $OUT_DIR"

# 手法の定義 (run.sh と完全に一致させました)
# 既存手法: 構造, Raw, ランダム, エッジ再構成, エッジGT
METHODS_BASE="structural raw random edge_rec edge_gt"

# 新規追加手法: Smart Hybrid, Oracle, SBF
METHODS_PROPOSED="smart_hybrid oracle sbf"

# 共通パラメータ
SA_VAL="0.0"
RATE="0.1" 

echo "================================================================="
echo " STARTING PARTIAL NOISE RUN: SNR Range -5 to 0, Rate ${RATE}"
echo "================================================================="

# SNR -5 から 0 dB で実行
for SNR in {-5..0}; do

    echo "-----------------------------------------------------------------"
    echo " Processing Partial Noise SNR: ${SNR} dB (Rate: ${RATE})"
    echo "-----------------------------------------------------------------"

    # 1. Baseline Methods (structural, raw, random, edge_rec, edge_gt)
    echo "  > Running Baselines..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --target_methods $METHODS_BASE \
        > log_partial_snr${SNR}_baseline.txt 2>&1

    # 2. Hybrid Balanced (0.5 / 0.5)
    echo "  > Running Hybrid Balanced (0.5/0.5)..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --hybrid_alpha 0.5 --hybrid_beta 0.5 \
        --target_methods hybrid \
        > log_partial_snr${SNR}_h05_05.txt 2>&1

    # 3. Hybrid Edge-Bias (0.3 / 0.7)
    echo "  > Running Hybrid Edge-Bias (0.3/0.7)..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --hybrid_alpha 0.3 --hybrid_beta 0.7 \
        --target_methods hybrid \
        > log_partial_snr${SNR}_h03_07.txt 2>&1

    # 4. Proposed & Oracle Methods (Smart Hybrid, SBF, Oracle)
    echo "  > Running Proposed (Smart Hybrid, SBF) & Oracle..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --target_methods $METHODS_PROPOSED \
        > log_partial_snr${SNR}_proposed.txt 2>&1

done

echo "================================================================="
echo " All partial noise experiments completed."
echo " Results are in: $OUT_DIR"
echo "================================================================="