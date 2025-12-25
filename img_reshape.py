import os
import glob
from PIL import Image

def process_images(input_dir, output_dir, target_size=(256, 256)):
    """
    指定ディレクトリの画像(jpg, png等)を256x256にリサイズし、
    PNG形式で別ディレクトリに保存する関数
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 対象とする拡張子のリスト（必要に応じて追加してください）
    # 大文字・小文字を区別しないロジックにするため、ここでは小文字で定義します
    target_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # ディレクトリ内の全ファイルを走査
    files = glob.glob(os.path.join(input_dir, "*"))
    
    count = 0
    print("処理を開始します...")

    for file_path in files:
        # 拡張子を取得してチェック
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in target_extensions:
            continue  # 画像以外のファイルはスキップ

        try:
            with Image.open(file_path) as img:
                # --- 色モードの変換処理 ---
                # PNGはCMYKなどをサポートしていないため、RGBに変換します。
                # ※透過情報を維持したい場合は "RGBA" に書き換えてください。
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # リサイズ (高品質フィルタ使用)
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

                # --- 保存ファイル名の作成 ---
                # 元のファイル名(拡張子なし)を取得
                filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
                # 拡張子を .png に固定してパスを作成
                new_filename = filename_no_ext + ".png"
                save_path = os.path.join(output_dir, new_filename)

                # PNGとして保存
                img_resized.save(save_path, "PNG")
                
                print(f"変換完了: {os.path.basename(file_path)} -> {new_filename}")
                count += 1

        except Exception as e:
            print(f"エラーが発生しました ({file_path}): {e}")

    print(f"--- 完了: 合計 {count} 枚の画像を処理しました ---")

# --- 設定 ---
# 実際のパスに書き換えてください
INPUT_DIR = "inp"   # 元画像のフォルダ
OUTPUT_DIR = "input_dir" # 保存先のフォルダ

if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)