import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# 1. 画像の準備
url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 2. モデルとプロセッサのロード
local_model_path = "./weights/mask2former" 
try:
    processor = Mask2FormerImageProcessor.from_pretrained(local_model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(local_model_path)
except:
    model_id = "facebook/mask2former-swin-tiny-coco-panoptic"
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

# 【重要】モデルの設定でアテンション出力を確実に有効化する
model.config.output_attentions = True

# 3. 推論
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    # output_attentions=True を明示的に渡す
    outputs = model(**inputs, output_attentions=True)

# 4. ポストプロセス（セグメンテーション結果の取得）
result = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]

segmentation_map = result["segmentation"].cpu().numpy()
segments_info = result["segments_info"]

# --- 5. 重要度（Attention Importance）の抽出ロジック ---

# 属性名の確認とアテンションの取得
# Mask2Formerの場合、decoder_attentions に格納されます
attentions = getattr(outputs, "decoder_attentions", None)

if attentions is None:
    # 属性名が 'attentions' の場合や、辞書形式の場合のフォールバック
    attentions = getattr(outputs, "attentions", None)

if attentions is None:
    print("Available attributes in outputs:", dir(outputs))
    raise AttributeError("モデルからアテンションを取得できませんでした。transformersのバージョンを確認してください。")

# 最終層のセルフアテンションを取得
# Mask2Formerのデコーダー層は (Self-Attention) の形式で返される
last_layer_attn = attentions[-1] # (Batch, Heads, Queries, Queries)

# ヘッドの平均をとる
# avg_attn shape: (Queries, Queries) -> (100, 100)
avg_attn = last_layer_attn.mean(dim=1)[0]

# 重要度の計算: 各クエリ（列）が他の全クエリから受けたアテンションの合計値
# これにより「その物体がどれだけシーン解析の根拠にされたか」を測る
importance_scores = avg_attn.sum(dim=0).cpu().numpy()

# クエリとセグメントの紐付け
class_queries_logits = outputs.class_queries_logits[0]
query_probs = class_queries_logits.softmax(-1)
query_scores, query_labels = torch.max(query_probs, dim=-1)

# 重要度マップの初期化（画像と同じサイズ）
importance_map = np.zeros_like(segmentation_map, dtype=np.float32)

print(f"\n{'Label':<15} | {'Importance Score':<20}")
print("-" * 40)

for info in segments_info:
    seg_id = info['id']
    label_id = info['label_id']
    label_name = model.config.id2label[label_id]
    
    # このセグメントに対応する最適なクエリIndexを探す
    # (対象のラベルを予測しているクエリの中で、最もスコアが高いもの)
    mask_indices = (query_labels == label_id).nonzero(as_tuple=True)[0]
    
    if len(mask_indices) > 0:
        best_q_idx = mask_indices[torch.argmax(query_scores[mask_indices])].item()
        score = importance_scores[best_q_idx]
        
        # マスク領域にスコアを代入
        importance_map[segmentation_map == seg_id] = score
        print(f"{label_name:<15} | {score:.4f}")

# --- 6. 可視化 ---

def visualize_importance(image, seg_map, imp_map, segments_info):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 元画像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 通常のセグメンテーション
    color_mask = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for info in segments_info:
        np.random.seed(info['label_id'])
        color_mask[seg_map == info['id']] = np.random.randint(0, 255, size=3)
    axes[1].imshow(color_mask)
    axes[1].set_title("Segmentation Map")
    axes[1].axis('off')
    
    # 重要度ヒートマップ
    # 正規化してジェットカラーマップを適用
    if imp_map.max() > 0:
        imp_map_norm = (imp_map - imp_map.min()) / (imp_map.max() - imp_map.min() + 1e-8)
        axes[2].imshow(image)
        im = axes[2].imshow(imp_map_norm, cmap='jet', alpha=0.6)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].imshow(image)
        
    axes[2].set_title("Segment Importance (Self-Attention)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_importance(image, segmentation_map, importance_map, segments_info)