import os
import time
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageStat
from tqdm import tqdm

def is_valid_image(img, brightness_thr=15, std_thr=5):
    """
    画像が適切か判定するフィルタリング関数
    - brightness_thr: 平均輝度の下限（これより暗ければ真っ黒とみなす）
    - std_thr: 色のばらつき（標準偏差）の下限（これが低いと単色・グレー画像）
    """
    try:
        # グレースケールで輝度を確認
        gray_img = img.convert("L")
        stat = ImageStat.Stat(gray_img)
        mean_brightness = stat.mean[0]
        std_dev = stat.stddev[0]

        # 条件: 「明るさが一定以上」かつ「単色ではない（情報がある）」
        if mean_brightness < brightness_thr:
            return False, f"暗すぎます (輝度: {mean_brightness:.1f})"
        
        if std_dev < std_thr:
            return False, f"情報量が少なすぎます (Std: {std_dev:.1f})"

        return True, "OK"
    except Exception as e:
        return False, f"判定エラー: {e}"

def download_and_process_clean_bdd100k(output_dir, num_images=100, target_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"BDD100K(train)から品質チェックを行いながら {num_images} 枚を取得します...")

    try:
        # ストリーミングモードでロード
        dataset = load_dataset("dgural/bdd100k", split="train", streaming=True)
        # ランダムシャッフル
        dataset = dataset.shuffle(seed=int(time.time()), buffer_size=1000)
    except Exception as e:
        print(f"データセット設定エラー: {e}")
        return

    count = 0
    skipped_count = 0
    
    # プログレスバー
    pbar = tqdm(total=num_images)

    for i, sample in enumerate(dataset):
        if count >= num_images:
            break

        try:
            img = sample['image']
            
            # --- 1. 品質チェック (ここを追加) ---
            valid, reason = is_valid_image(img)
            if not valid:
                skipped_count += 1
                # ログが多すぎる場合はコメントアウトしてください
                # print(f"Skip: {reason}")
                continue

            # --- 2. 画像加工 ---
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 高品質リサイズ
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

            # --- 3. 保存 ---
            filename = f"bdd100k_clean_{count:04d}.png"
            save_path = os.path.join(output_dir, filename)
            
            img_resized.save(save_path, "PNG")
            
            count += 1
            pbar.update(1)

        except Exception as e:
            print(f"処理エラー: {e}")

    pbar.close()
    print(f"--- 完了 ---")
    print(f"保存枚数: {count}枚")
    print(f"除外枚数: {skipped_count}枚 (暗い/破損)")
    print(f"保存先: {output_dir}")

# --- 設定 ---
OUTPUT_DIR = "data/bdd100k_clean_256"

if __name__ == "__main__":
    download_and_process_clean_bdd100k(OUTPUT_DIR, num_images=100)