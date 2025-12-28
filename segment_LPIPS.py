import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lpips  # LPIPSライブラリ
from torchvision import transforms
from transformers import (
    Mask2FormerImageProcessor, 
    Mask2FormerForUniversalSegmentation
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

    # --- LPIPS (スコア計算用) ---
    print("Loading LPIPS (AlexNet)...")
    # LPIPSは距離指標です (0に近いほど似ている、値が大きいほど違う)
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    # ==========================================
    # 2. 画像のロードと前処理
    # ==========================================
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    print(f"Loading image from {url}...")
    image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image_np = np.array(image_pil)

    # LPIPS用の前処理関数: [0, 1] -> [-1, 1] に正規化してTensor化
    def to_lpips_tensor(img_pil):
        t = transforms.ToTensor()(img_pil) # [0, 1]
        t = (t * 2.0) - 1.0 # [-1, 1]
        return t.unsqueeze(0).to(device) # Batch dimension追加

    # ベースライン（元画像）のTensor作成
    tensor_original = to_lpips_tensor(image_pil)

    # ==========================================
    # 3. セグメンテーションの実行
    # ==========================================
    print("Running Segmentation...")
    inputs = seg_processor(images=image_pil, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = seg_model(**inputs)
    
    # ポストプロセス
    result = seg_processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image_pil.size[::-1]]
    )[0]
    
    segmentation_map = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    # ==========================================
    # 4. 各セグメントをノイズ化してLPIPS距離を計算
    # ==========================================
    importance_map = np.zeros_like(segmentation_map, dtype=np.float32)
    
    print(f"\n{'Label':<20} | {'Area (px)':<10} | {'LPIPS Dist':<10} | {'Norm Importance':<15}")
    print("-" * 65)

    for info in segments_info:
        seg_id = info['id']
        label_id = info['label_id']
        label_name = seg_model.config.id2label[label_id]
        
        # マスク作成
        mask_bool = (segmentation_map == seg_id)
        area = np.sum(mask_bool)
        if area == 0:
            continue

        # ノイズ画像の作成
        noise = np.random.randint(0, 255, image_np.shape, dtype=np.uint8)
        masked_img_np = image_np.copy()
        masked_img_np[mask_bool] = noise[mask_bool]
        
        masked_pil = Image.fromarray(masked_img_np)
        
        # ノイズ適用画像をTensor化
        tensor_masked = to_lpips_tensor(masked_pil)
        
        # --- LPIPS計算 ---
        # 元画像とノイズ画像の「距離」を計算します
        with torch.no_grad():
            dist = loss_fn(tensor_original, tensor_masked)
        
        # LPIPSは距離そのものなので、値が大きいほど「変化が大きい（重要）」
        raw_importance = dist.item()
        
        # --- 面積による正規化 (バランス型) ---
        # LPIPSも面積に依存するため、平方根で割って補正します
        normalized_importance = raw_importance / (np.sqrt(area) + 1e-6)
        
        importance_map[mask_bool] = normalized_importance
        
        print(f"{label_name:<20} | {area:<10} | {raw_importance:.4f}     | {normalized_importance:.6f}")

    # ==========================================
    # 5. 可視化
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
    
    # 3. LPIPS重要度ヒートマップ
    axes[2].imshow(image)
    
    if imp_map.max() > imp_map.min():
        norm_imp = (imp_map - imp_map.min()) / (imp_map.max() - imp_map.min() + 1e-8)
    else:
        norm_imp = imp_map
        
    im = axes[2].imshow(norm_imp, cmap='jet', alpha=0.6)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Normalized LPIPS Importance\n(Distance / sqrt(Area))")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()