# score_folder.py

import os
import glob
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from unet import UNet
import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 때와 동일한 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# AE 로드
def load_ae(ckpt_name="model20.pth"):
    ckpt_path = os.path.join(cfg.models_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = UNet(in_channels=3, n_classes=3, depth=cfg.depth, padding=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def compute_error_map(model, x):
    x = x.to(device)
    with torch.no_grad():
        recon = model(x)

    # [-1,1] 기준이면 그대로 차이
    err = (x - recon).pow(2).mean(dim=1)   # [B,H,W] → 채널 평균
    return err.squeeze(0).cpu().numpy()    # [H,W]

def img_to_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(img).unsqueeze(0)  # [1,3,H,W]
    return x

def score_folder(folder, model):
    paths = sorted(
        [p for p in glob.glob(os.path.join(folder, "*")) 
         if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    )
    scores_mean = []
    scores_p95 = []

    for p in paths:
        x = img_to_tensor(p)
        err_map = compute_error_map(model, x)
        scores_mean.append(err_map.mean())
        scores_p95.append(np.percentile(err_map, 95))

    return paths, np.array(scores_mean), np.array(scores_p95)

if __name__ == "__main__":
    model = load_ae("model20.pth")

    # Real / Fake 폴더 경로 (네가 만든 구조에 맞게 바꿔)
    real_dir = "test_images/real"
    fake_dir = "test_images/fake"

    real_paths, real_mean, real_p95 = score_folder(real_dir, model)
    fake_paths, fake_mean, fake_p95 = score_folder(fake_dir, model)

    print("Real  mean err:", real_mean.mean(), "p95:", real_p95.mean())
    print("Fake  mean err:", fake_mean.mean(), "p95:", fake_p95.mean())

    # 분포 시각화
    plt.hist(real_p95, bins=30, alpha=0.5, label="real (p95)")
    plt.hist(fake_p95, bins=30, alpha=0.5, label="fake (p95)")
    plt.legend()
    plt.title("AE error (95th percentile)")
    plt.show()
