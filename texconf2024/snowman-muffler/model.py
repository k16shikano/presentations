import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss

class SnowmanDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.transform = transform
        self.target_size = target_size  # 目標サイズ

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像とマスクの読み込み
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # リサイズ
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)

        # 正規化
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # バイナリ値の想定 (0~1)

        return image, mask

# 訓練関数
def train_model(images_dir, masks_dir, save_path, num_epochs=10, batch_size=4, lr=0.001):
    # データセットとデータローダー
    train_dataset = SnowmanDataset(images_dir=images_dir, masks_dir=masks_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # SMPのU-Netモデル
    model = Unet(
        encoder_name="resnet34",       # エンコーダ
        encoder_weights="imagenet",   # 事前学習済み重み
        in_channels=3,                # 入力チャンネル
        classes=1                     # 出力クラス
    )

    # 損失関数と最適化
    dice_loss = DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 訓練ループ
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # モデル保存
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# メイン関数
if __name__ == '__main__':
    train_model(images_dir='images', masks_dir='masks', save_path='unet_model_5.pth', num_epochs=5)
