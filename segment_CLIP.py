import torch
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    Mask2FormerImageProcessor, 
    Mask2FormerForUniversalSegmentation,
    CLIPProcessor, 
    CLIPModel
)

def main():
    # ==========================================
    # 1. デバイスとモデルの準備
    # ==========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Mask2Former (セグメンテーション用) ---
    print("Loading Mask2Former...")
    seg_model_id = "facebook/mask2former-swin-tiny-coco-panoptic"
    seg_processor = Mask2FormerImageProcessor.from_pretrained(seg_model_id)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(seg_model_id).to(device)
    seg_model.eval()

    # --- CLIP (スコア計算用) ---
    print("Loading CLIP...")
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_model.eval()

    # ==========================================
    # 2. 画像のロードと前処理
    # ==========================================
    url = "http://images.cocodataset.org/val2017/000000001993.jpg"
    print(f"Loading image from {url}...")
    image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image_np = np.array(image_pil)

    # ==========================================
    # 3. セグメンテーションの実行
    # ==========================================
    print("Running Segmentation...")
    inputs = seg_processor(images=image_pil, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = seg_model(**inputs)
    
    # ポストプロセスでマップを取得
    result = seg_processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image_pil.size[::-1]]
    )[0]
    
    segmentation_map = result["segmentation"].cpu().numpy() # [H, W] 各ピクセルにID
    segments_info = result["segments_info"]

    # ==========================================
    # 4. ベースライン(元画像)のCLIP特徴量取得
    # ==========================================
    def get_clip_embedding(img):
        """CLIPモデルを使って画像埋め込みベクトルを取得"""
        inputs = clip_processor(images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        # 正規化
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs

    print("Calculating baseline CLIP score...")
    emb_original = get_clip_embedding(image_pil)

    # ==========================================
    # 5. 各セグメントをノイズ化して変化量を計算（面積正規化付き）
    # ==========================================
    importance_map = np.zeros_like(segmentation_map, dtype=np.float32)
    
    print(f"\n{'Label':<20} | {'Area (px)':<10} | {'Raw Diff':<10} | {'Norm Importance':<15}")
    print("-" * 65)

    for info in segments_info:
        seg_id = info['id']
        label_id = info['label_id']
        label_name = seg_model.config.id2label[label_id]
        
        # マスクの作成 (対象セグメントのみTrue)
        mask_bool = (segmentation_map == seg_id)
        
        # 面積（ピクセル数）を取得
        area = np.sum(mask_bool)
        if area == 0:
            continue

        # ノイズ画像の作成
        noise = np.random.randint(0, 255, image_np.shape, dtype=np.uint8)
        masked_img_np = image_np.copy()
        masked_img_np[mask_bool] = noise[mask_bool]
        
        masked_pil = Image.fromarray(masked_img_np)
        
        # ノイズ適用画像のCLIP特徴量
        emb_masked = get_clip_embedding(masked_pil)
        
        # コサイン類似度計算
        similarity = (emb_original @ emb_masked.T).item()
        
        # 生の変化量 (1.0 - 類似度)
        raw_importance = 1.0 - similarity
        
        # --- 【修正ポイント】 面積による正規化 (バランス型) ---
        # 変化量を「面積の平方根」で割ることで、極端に大きい領域のスコアを抑制しつつ
        # 極小領域のノイズ耐性も持たせる
        normalized_importance = raw_importance / (np.sqrt(area) + 1e-6)
        
        # マップには正規化後の値を格納
        importance_map[mask_bool] = normalized_importance
        
        print(f"{label_name:<20} | {area:<10} | {raw_importance:.4f}     | {normalized_importance:.6f}")

    # ==========================================
    # 6. 可視化
    # ==========================================
    visualize_results(image_pil, segmentation_map, importance_map, segments_info)

def visualize_results(image, seg_map, imp_map, segments_info):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 1. 元画像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 2. セグメンテーションマップ
    color_mask = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for info in segments_info:
        np.random.seed(info['id'] + 1)
        color_mask[seg_map == info['id']] = np.random.randint(50, 255, size=3)
    axes[1].imshow(color_mask)
    axes[1].set_title("Segmentation Map")
    axes[1].axis('off')
    
    # 3. CLIP重要度ヒートマップ (正規化済み)
    axes[2].imshow(image)
    
    # 正規化して表示 (0〜Max)
    if imp_map.max() > imp_map.min():
        norm_imp = (imp_map - imp_map.min()) / (imp_map.max() - imp_map.min() + 1e-8)
    else:
        norm_imp = imp_map
        
    im = axes[2].imshow(norm_imp, cmap='jet', alpha=0.6)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Normalized CLIP Importance\n(Diff / sqrt(Area))")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()