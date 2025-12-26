#!/bin/bash

# =================================================================
# Experiment 1: Alpha Sensitivity Analysis (Alphaの感度分析)
# 目的: 構造情報(エッジ)の重みを変えることで、指標がどう変化するか見る。
# 固定条件: SNR=-15dB (中程度のノイズ), Rate=0.2 (十分な再送量)
# =================================================================
echo "--- Exp1: Alpha Sensitivity (SNR -15dB, r=0.2) ---"

# Alpha 0.0: 従来のUncertaintyのみ (ベースライン)
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.0 > log_snr-15_r0.2_a0.0.txt 2>&1

# Alpha 0.3: 構造情報少し
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 > log_snr-15_r0.2_a0.3.txt 2>&1

# Alpha 0.6: 構造情報重視 (推奨値)
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.6 > log_snr-15_r0.2_a0.6.txt 2>&1

# Alpha 0.9: ほぼエッジのみ再送
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.9 > log_snr-15_r0.2_a0.9.txt 2>&1


# =================================================================
# Experiment 2: Robustness across SNR (異なるノイズ環境での検証)
# 目的: 推奨Alpha(例: 0.6)が、激しいノイズや軽いノイズでもRandomに勝てるか確認。
# =================================================================
echo "--- Exp2: Robustness (Alpha 0.6) ---"

# SNR -20dB (激しいノイズ): ID Loss改善のチャンス
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.6 > log_snr-20_r0.2_a0.6.txt 2>&1

# SNR -10dB (軽いノイズ): Randomに勝ちにくい環境。ここでの勝利が重要。
# ※再送率は少し下げて 0.1 で検証
python -m scripts.img2img --snr -10 -r 0.1 --struct_alpha 0.6 > log_snr-10_r0.1_a0.6.txt 2>&1


# =================================================================
# Experiment 3: Low Rate Test (少ない再送量での効率性)
# 目的: 再送率が低くても効果が出るか？ (0.1 = 10%再送)
# =================================================================
echo "--- Exp3: Low Retransmission Rate ---"

python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.6 > log_snr-15_r0.1_a0.6.txt 2>&1

echo "All experiments completed."

#tail -f log_snr-10_r0.1_a0.6.txt
# nohup ./run.sh > main_process.log 2>&1 &
#python -m scripts.img2img --snr -13 -r 0.1 