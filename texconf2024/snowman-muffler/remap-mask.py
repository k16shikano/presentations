import os
import cv2
import numpy as np

# マスク画像フォルダ
masks_dir = "masks"
output_masks_dir = "masks"

# 保存先フォルダを作成
os.makedirs(output_masks_dir, exist_ok=True)

def process_mask(mask):
    """
    マスクを0と1のラベルにマッピングする
    """
    # 0より大きい値をすべて1に変換
    processed_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    return processed_mask

def process_all_masks(input_dir, output_dir):
    """
    指定されたディレクトリ内のマスクを処理して保存
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # PNGファイルのみを対象
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # マスク画像を読み込む
            mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to read mask: {input_path}")
                continue

            # マスクを0と1にリマッピング
            processed_mask = process_mask(mask)

            # 処理済みマスクを保存
            cv2.imwrite(output_path, processed_mask)
            print(f"Processed and saved mask: {output_path}")

# 実行
process_all_masks(masks_dir, output_masks_dir)
print("All masks have been processed and remapped to [0, 1].")
