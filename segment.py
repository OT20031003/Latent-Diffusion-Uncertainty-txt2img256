from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. 画像の準備（ネット上のサンプル画像をロード）
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)



# 【変更点】モデルIDではなく、ローカルのパスを指定
local_model_path = "./weights/mask2former" 

# ローカルパスから読み込み
processor = Mask2FormerImageProcessor.from_pretrained(local_model_path)
model = Mask2FormerForUniversalSegmentation.from_pretrained(local_model_path)

# 3. 前処理と推論
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 4. 結果の変換（パノプティック・セグメンテーション）
# target_sizesには元の画像サイズを指定して、リサイズ後のマスクを取得します
result = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]

# 結果の取り出し
segmentation_map = result["segmentation"] # (H, W) のIDマップ
segments_info = result["segments_info"]   # 各IDが何のクラスかという情報

# --- 以下、可視化のための処理 ---

# ラベルIDを可視化用にランダムな色に変換するヘルパー関数
def visualize_map(segmentation_map, segments_info):
    # マップをnumpy化
    seg_np = segmentation_map.cpu().numpy()
    
    # 色付け用のパレット作成
    color_mask = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
    
    # 検出された各セグメントに色を塗る
    for info in segments_info:
        segment_id = info['id']
        label_id = info['label_id']
        label_name = model.config.id2label[label_id]
        
        print(f"ID: {segment_id}, Label: {label_name} (Class ID: {label_id})")
        
        # ランダムな色を生成 (固定シードで再現性確保)
        np.random.seed(label_id)
        color = np.random.randint(0, 255, size=3)
        
        # マスク部分を塗る
        color_mask[seg_np == segment_id] = color

    # 表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Map")
    plt.imshow(color_mask)
    plt.axis('off')
    plt.show()

visualize_map(segmentation_map, segments_info)