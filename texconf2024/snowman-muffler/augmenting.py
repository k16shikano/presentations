import cv2
import numpy as np
import albumentations as A
from albumentations.core.composition import OneOf
from albumentations import HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate

# オーグメンテーションの設定
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),  # 左右反転
    A.RandomBrightnessContrast(p=0.2),  # 明るさとコントラスト調整
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 平行移動、回転
    A.ElasticTransform(p=0.2, alpha=120, sigma=12),  # 弾性変形（修正済み）
])

# 画像とマスクを読み込む
image = cv2.imread("images/kozuka.png")
mask = cv2.imread("masks/kozuka.png", cv2.IMREAD_GRAYSCALE)

# Albumentationsでオーグメンテーションを適用
augmented = augmentation(image=image, mask=mask)
augmented_image = augmented['image']
augmented_mask = augmented['mask']

# オーグメンテーション後の結果を保存（または可視化）
cv2.imwrite("augmented_image.jpg", augmented_image)
cv2.imwrite("augmented_mask.png", augmented_mask)

for i in range(10):
    augmented = augmentation(image=image, mask=mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    # 生成されたデータを保存
    cv2.imwrite(f"images/kozuka_{i}.png", augmented_image)
    cv2.imwrite(f"masks/kozuka_{i}.png", augmented_mask)


