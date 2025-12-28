#!/bin/bash

# 出力ディレクトリの作成
mkdir -p results_experiment_text2img_high_snr

# 共通設定
PY_CMD="python -m scripts.img2img --config models/ldm/text2img256/config.yaml --ckpt models/ldm/text2img256/model.ckpt --output_dir results_experiment_text2img_high_snr"

# =================================================================
#  変更点: 実行対象を新規手法 'clip_seg' のみに限定
# =================================================================
METHODS_NEW="clip_seg"

# 共通パラメータ
SA_VAL="0.0"
RATE="0.1"

echo "================================================================="
echo " STARTING ADDITIONAL RUN (CLIP Seg): SNR Range -5 to 0, Rate ${RATE}"
echo "================================================================="

# SNR -5 から 0 でループを実行
for SNR in {-5..0}; do

    echo "-----------------------------------------------------------------"
    echo " Processing SNR: ${SNR} dB (Rate: ${RATE})"
    echo "-----------------------------------------------------------------"

    # ---------------------------------------------------------------
    # 既存手法群はすべてコメントアウト (実行済みのため)
    # ---------------------------------------------------------------
    
    # echo "  > [Skipping] Baselines..."
    # $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
    #     --target_methods structural raw random edge_rec edge_gt \
    #     > log_snr${SNR}_baseline.txt 2>&1

    # echo "  > [Skipping] Hybrid Balanced..."
    # $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
    #     --hybrid_alpha 0.5 --hybrid_beta 0.5 \
    #     --target_methods hybrid \
    #     > log_snr${SNR}_h05_05.txt 2>&1

    # echo "  > [Skipping] Hybrid Edge-Bias..."
    # $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
    #     --hybrid_alpha 0.3 --hybrid_beta 0.7 \
    #     --target_methods hybrid \
    #     > log_snr${SNR}_h03_07.txt 2>&1

    # echo "  > [Skipping] Proposed & Oracle..."
    # $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
    #     --target_methods smart_hybrid oracle sbf \
    #     > log_snr${SNR}_proposed.txt 2>&1

    # ---------------------------------------------------------------
    # 新規追加手法 (CLIP Seg) のみ実行
    # ---------------------------------------------------------------
    echo "  > Running New Method: CLIP Segmentation (clip_seg)..."
    $PY_CMD --snr ${SNR} -r ${RATE} --struct_alpha ${SA_VAL} \
        --target_methods $METHODS_NEW \
        > log_snr${SNR}_clip_seg.txt 2>&1

done

echo "================================================================="
echo " Additional experiments completed."
echo "================================================================="