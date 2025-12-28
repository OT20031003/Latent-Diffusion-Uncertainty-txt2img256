import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# 1. 画像の準備
url = "http://images.cocodataset.org/val2017/000000000802.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 2. モデルとプロセッサのロード
# 一般的なViTモデル (patch size 16, image size 224)
model_id = "google/vit-base-patch16-224"
try:
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTForImageClassification.from_pretrained(model_id)
except Exception as e:
    print(f"モデルのロードに失敗しました: {e}")
    exit()

# 【重要】モデルの設定でアテンション出力を確実に有効化する
model.config.output_attentions = True

# 3. 推論
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    # output_attentions=True を明示的に渡す
    outputs = model(**inputs, output_attentions=True)

# 予測結果（クラスラベル）の取得
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
print(f"Predicted Class: {predicted_label}")

# --- 4. 重要度（Attention Importance）の抽出ロジック ---

# アテンションの取得
# ViTの場合、outputs.attentions はタプルで、各要素は (Batch, Heads, Seq_Len, Seq_Len)
attentions = outputs.attentions

if attentions is None:
    raise AttributeError("モデルからアテンションを取得できませんでした。")

# 最終層のアテンションを取得
last_layer_attn = attentions[-1]  # Shape: (1, 12, 197, 197) ※197 = 1([CLS]) + 14*14(Patches)

# バッチ次元を削除
attn_tensor = last_layer_attn[0]  # Shape: (12, 197, 197)

# ヘッドの平均をとる
# これにより、複数のヘッドが注目している平均的な箇所を抽出
avg_attn = torch.mean(attn_tensor, dim=0)  # Shape: (197, 197)

# 【ViT特有の処理】
# index 0 は [CLS] トークン（分類に使われる特殊トークン）。
# index 1 以降が画像パッチに対応するトークン。
# 「[CLS]トークンが、画像パッチのどこを強く見ていたか」を重要度とする。
# つまり、0行目の、1列目以降を取得する。
cls_attention = avg_attn[0, 1:]  # Shape: (196,)

# 画像パッチのグリッドサイズを計算 (例: 196 = 14 * 14)
grid_size = int(np.sqrt(cls_attention.shape[0]))
print(f"Attention Grid Size: {grid_size}x{grid_size}")

# 重要度マップをグリッド状に変形
importance_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

# --- 5. 可視化 ---

def visualize_vit_importance(image, imp_map, label_name):
    # 画像のリサイズ（モデル入力サイズに合わせる必要はないが、見やすくするためオリジナルを使用）
    # ヒートマップを元画像のサイズに拡大するためにOpenCVやscipyを使う手もあるが、
    # ここではmatplotlibのimshow(extent=...)や補間を利用して簡易に行う
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 元画像
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nPred: {label_name}")
    axes[0].axis('off')
    
    # 重要度ヒートマップのオーバーレイ
    # 重要度を正規化
    imp_map_norm = (imp_map - imp_map.min()) / (imp_map.max() - imp_map.min() + 1e-8)
    
    axes[1].imshow(image)
    # bicubic補間を使って、14x14の粗いマップを滑らかに画像全体に引き伸ばす
    im = axes[1].imshow(imp_map_norm, cmap='jet', alpha=0.6, interpolation='bicubic')
    axes[1].set_title("Attention Map ([CLS] token to patches)")
    axes[1].axis('off')
    
    # カラーバー
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

visualize_vit_importance(image, importance_map, predicted_label)