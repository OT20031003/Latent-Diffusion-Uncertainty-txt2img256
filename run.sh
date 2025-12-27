#!/bin/bash

# 出力ディレクトリの作成
mkdir -p results_experiment_text2img_high_snr

# 共通設定
PY_CMD="python -m scripts.img2img --config models/ldm/text2img256/config.yaml --ckpt models/ldm/text2img256/model.ckpt --output_dir results_experiment_text2img_high_snr"

# ベースライン手法
METHODS_BASE="structural raw random edge_rec edge_gt"

# 共通パラメータ
SA_VAL="0.0"
RATE="0.1"  # 最後の行の指定に合わせて0.1に設定（必要なら0.2に変更してください）

echo "================================================================="
echo " STARTING RUN (Text2Img): SNR Range -5 to 5, Rate ${RATE}"
echo "================================================================="

# SNR -5, 0, 5 でループを実行
# もし全ての整数(-5,-4...5)を実行したい場合は {-5..5} に変更してください
for SNR in -5 0 5; do

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

done

echo "================================================================="
echo " All experiments for SNR -5 to 5 completed."
echo "================================================================="