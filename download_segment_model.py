from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import os

# 保存先のディレクトリ名
save_directory = "./weights/mask2former"

# モデルID
model_id = "facebook/mask2former-swin-base-coco-panoptic"

print(f"ダウンロード開始: {model_id}")

# 1. モデルとプロセッサをネットから読み込む
processor = Mask2FormerImageProcessor.from_pretrained(model_id)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

# 2. ローカルに保存する
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

processor.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"保存完了: {save_directory}")