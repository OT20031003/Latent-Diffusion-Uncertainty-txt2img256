#!/bin/bash

# =================================================================
# Experiment 1: Alpha Sensitivity (パラメータ調整)
# 新定義のAlpha: 0.0(不確実性のみ) -> 1.0(完全ランダム)
# 狙い目: 0.3 付近でバランスが取れるはずです
# =================================================================
echo "--- Exp1: Alpha Sensitivity (SNR -15dB, r=0.2) ---"
# 軽いノイズ (-10dB): ここでRandomに勝つのが難しい。不確実性の精度は高いはずなので0.3で勝負
python -m scripts.img2img --snr -10 -r 0.05 --struct_alpha 0.3 > log_snr-10_r0.05_a0.3.txt 2>&1

# ベースライン (純粋な平滑化不確実性)
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.0 > log_snr-15_r0.2_a0.0.txt 2>&1

# 【本命】バランス型 (不確実性70% + ランダム30%)
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 > log_snr-15_r0.2_a0.3.txt 2>&1

# ランダム寄り (比較用。以前の0.6はここではランダム性が強すぎるかも)
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.5 > log_snr-15_r0.2_a0.5.txt 2>&1


# =================================================================
# Experiment 2: Robustness (ロバスト性検証)
# 変更点: 推奨Alphaを 0.6 -> 0.3 に変更して検証します
# =================================================================
echo "--- Exp2: Robustness (Alpha 0.3) ---"

# 激しいノイズ (-20dB): 不確実性推定が乱れやすいため、ランダム混ぜ(0.3)が効くはず
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.3 > log_snr-20_r0.2_a0.3.txt 2>&1


python -m scripts.img2img --snr 0 -r 0.05 --struct_alpha 0.1 > log_snr0_r0.05_a0.1.txt 2>&1

# =================================================================
# Experiment 3: Low Rate Test (低レート検証)
# 変更点: ここも Alpha 0.3 で統一
# =================================================================
echo "--- Exp3: Low Retransmission Rate ---"

python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.3 > log_snr-15_r0.1_a0.3.txt 2>&1

echo "All experiments completed."

#tail -f log_snr-10_r0.05_a0.3.txt
# nohup ./run.sh > main_process.log 2>&1 &
#python -m scripts.img2img --snr -13 -r 0.1