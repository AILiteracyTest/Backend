import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from .unet import UNet
from . import config as cfg

# -----------------------------
# 설정
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 학습 때랑 똑같은 전처리 (반드시 동일해야 함!)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

# 결과 저장 폴더
os.makedirs(cfg.res_dir, exist_ok=True)

# -----------------------------
# 1) 모델 로드
# -----------------------------
def load_ae_model(ckpt_name: str = "model20.pth"):
    model_path = os.path.join(cfg.models_dir, ckpt_name)    
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)

    model = UNet(
        in_channels=3,
        n_classes=3,
        depth=cfg.depth,
        padding=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {model_path}")
    return model


# -----------------------------
# 2) 이미지 로드 + 전처리
# -----------------------------
def load_image_as_tensor(img_path):
    # BGR로 읽고 → RGB로 바꾸기 (train 때랑 동일)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # transform 적용
    x = transform(img)          # [3, H, W], [-1,1] 범위
    x = x.unsqueeze(0)          # [1, 3, H, W]
    return x, img               # 텐서 + 원본 RGB numpy


# -----------------------------
# 3) Error map 계산
# -----------------------------
def compute_error_map(model, x):
    """
    x: [1, 3, H, W] normalized [-1,1]
    """
    x = x.to(device)
    with torch.no_grad():
        recon = model(x)        # [1,3,H,W]

    # L2 error per pixel (채널 평균)
    error = (x - recon) ** 2        # [1,3,H,W]
    error = error.mean(dim=1)       # [1,H,W]
    error_map = error.squeeze(0)    # [H,W]

    return recon.cpu(), error_map.cpu()


# -----------------------------
# 4) 시각화: 원본 / 재구성 / 에러 heatmap
# -----------------------------
def visualize_result(orig_img_rgb, x, recon, error_map, save_path):
    """
    orig_img_rgb: 원래 cv2 읽은 RGB (numpy)
    x: 입력 텐서 [1,3,H,W] (normalized)
    recon: 재구성 텐서 [1,3,H,W]
    error_map: [H,W]
    """
    # 입력/재구성 텐서를 [0,1] 이미지로 되돌리기 (Normalize 역변환)
    x_img = x.squeeze(0).cpu()         # [3,H,W]
    recon_img = recon.squeeze(0)       # [3,H,W]

    # [-1,1] → [0,1]
    def denorm(t):
        t = t * 0.5 + 0.5
        return torch.clamp(t, 0.0, 1.0)

    x_vis = denorm(x_img).permute(1, 2, 0).numpy()       # [H,W,3]
    recon_vis = denorm(recon_img).permute(1, 2, 0).numpy()

    # error map 정규화 (0~1)
    err = error_map.numpy()
    err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)

    # heatmap (matplotlib colormap 사용)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original (resized)")
    plt.imshow(x_vis)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Reconstruction")
    plt.imshow(recon_vis)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Error Map")
    plt.imshow(err_norm, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    # 원본 위에 heatmap overlay
    # 원본도 256x256으로 resize해서 overlay
    orig_resized = cv2.resize(orig_img_rgb, (x_vis.shape[1], x_vis.shape[0]))
    orig_resized = orig_resized.astype(np.float32) / 255.0

    # err_norm을 3채널로 확장
    err_color = plt.get_cmap("jet")(err_norm)[..., :3]   # [H,W,3]

    alpha = 0.5
    overlay = (1 - alpha) * orig_resized + alpha * err_color

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved visualization to {save_path}")

# -----------------------------
# 5) 하나의 이미지에 대해 전체 파이프라인 실행
# -----------------------------
def run_ae(img_path, ckpt_name="model20.pth"):
    """
    img_path: 분석할 이미지 경로 (로컬 파일)
    return: dict 형태로 결과 반환
    """
    model = load_ae_model(ckpt_name)
    x, orig = load_image_as_tensor(img_path)
    recon, err_map = compute_error_map(model, x)

    err = err_map.numpy()
    mean_err = err.mean()
    p95_err = np.percentile(err, 95)
    max_err = err.max()

    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(cfg.res_dir, f"{base}_ae_heatmap.png")
    visualize_result(orig, x, recon, err_map, save_path)

    return {
        "overlay_path": save_path,
        "mean_err": float(mean_err),
        "p95_err": float(p95_err),
        "max_err": float(max_err),
    }