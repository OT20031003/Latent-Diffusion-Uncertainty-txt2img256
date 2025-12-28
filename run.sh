#!/bin/bash

# 出力ディレクトリの作成
mkdir -p results_experiment_text2img_high_snr

# 共通設定
PY_CMD="python -m scripts.img2img --config models/ldm/text2img256/config.yaml --ckpt models/ldm/text2img256/model.ckpt --output_dir results_experiment_text2img_high_snr"

# 手法の定義
# 既存手法
METHODS_BASE="structural raw random edge_rec edge_gt"

# 新規追加手法 (提案手法 Smart Hybrid, SBF と 理論上限 Oracle)
# ここに 'sbf' を追加しました
METHODS_PROPOSED="smart_hybrid oracle sbf"

# 共通パラメータ
SA_VAL="0.0"
RATE="0.1"  # 必要に応じて変更してください（例: 0.2）

echo "================================================================="
echo " STARTING RUN (Text2Img): SNR Range -5 to 5, Rate ${RATE}"
echo "================================================================="

# SNR 1, 2, 3, 4, 5 でループを実行 (元のスクリプトに合わせて {1..5})
for SNR in {-5..0}; do

    echo "-----------------------------------------------------------------"
    echo " Processing SNR: ${SNR} dB (Rate: ${RATE})"
    echo "-----------------------------------------------------------------"

    # 1. Baseline Methods (structural, raw, etc.)
    echo "  > Running Baselines..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --target_methods $METHODS_BASE \
        > log_snr${SNR}_baseline.txt 2>&1

    # 2. Hybrid Balanced (0.5 / 0.5)
    echo "  > Running Hybrid Balanced (0.5/0.5)..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --hybrid_alpha 0.5 --hybrid_beta 0.5 \
        --target_methods hybrid \
        > log_snr${SNR}_h05_05.txt 2>&1

    # 3. Hybrid Edge-Bias (0.3 / 0.7)
    echo "  > Running Hybrid Edge-Bias (0.3/0.7)..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --hybrid_alpha 0.3 --hybrid_beta 0.7 \
        --target_methods hybrid \
        > log_snr${SNR}_h03_07.txt 2>&1

    # 4. Proposed & Oracle Methods (新規追加 + SBF)
    # Smart Hybrid, Oracle, SBF を実行
    echo "  > Running Proposed (Smart Hybrid, SBF) & Oracle..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --target_methods $METHODS_PROPOSED \
        > log_snr${SNR}_proposed.txt 2>&1

done

echo "================================================================="
echo " All experiments completed."
echo "================================================================="