#!/bin/bash

# 仮想環境のアクティベート（必要なら）
# source /path/to/venv/bin/activate

# 1つずつ実行するため、行末の '&' と行頭の 'nohup' は削除します
# ログファイル名を変えておくと後で分析しやすいです

echo "--- Job 1 start ---"
python -m scripts.img2img --snr -15 -r 0.1 > log_snr-15_r0.1.txt 2>&1

echo "--- Job 2 start ---"
python -m scripts.img2img --snr -20 -r 0.2 > log_snr-10_r0.2.txt 2>&1

echo "--- Job 3 start ---"
python -m scripts.img2img --snr -7 -r 0.2 > log_snr-7_r0.2.txt 2>&1

echo "--- Job 4 start ---"
python -m scripts.img2img --snr -5 -r 0.1 > log_snr-5_r0.1.txt 2>&1

echo "--- Job 5 start ---"
python -m scripts.img2img --snr -10 -r 0.1 > log_snr-10_r0.1.txt 2>&1

echo "--- Job 6 start ---"
python -m scripts.img2img --snr -10 -r 0.3 > log_snr-10_r0.3.txt 2>&1

echo "--- Job 7 start ---"
python -m scripts.img2img --snr 0 -r 0.1 > log_snr-0_r0.1.txt 2>&1

echo "--- Job 8 start ---"
python -m scripts.img2img --snr 0 -r 0.2 > log_snr-0_r0.2.txt 2>&1

echo "complete"

#tail -f log_snr-15_r0.2.txt