import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from unet import UNet
import config as cfg

import base64    

from openai import OpenAI


def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_vlm_explanation(overlay_path, mean_err, p95_err, max_err):
    img_b64 = load_image_base64(overlay_path)

    system_prompt = """
    당신의 역할은 '이미지가 왜 AI가 생성한 이미지처럼 보이는지'
    Autoencoder 재구성 오차(heatmap)를 근거로 명확하게 설명하는 것입니다.

    제공되는 합성 이미지에는 다음 네 가지가 포함됩니다:
    1) 원본 이미지
    2) Autoencoder 재구성 이미지
    3) 재구성 오차 히트맵 (파랑=오차 낮음, 빨강=오차 큼)
    4) 히트맵을 원본 위에 덮은 오버레이 이미지

    Autoencoder는 '실제 사진(real-world photos)'만을 학습한 모델이며,
    특정 영역에서 재구성 오차가 크다는 것은
    그 구역이 실제 사진 분포에서 벗어난 비정상적 패턴을 포함한다는 것을 의미합니다.

    당신의 답변에서는 다음을 지켜주세요:
    - 구체적인 위치를 언급하세요.
    - '왜 해당 영역이 비정상적인지'를 사진적 관점(텍스처, 형태, 조명, 구조 등)에서 설명하세요.
    - 출력 언어는 반드시 한국어로 하세요.
    """

    user_text = f"""
    아래 이미지는 원본/재구성/오차 히트맵/오버레이를 하나로 합친 이미지입니다.

    Autoencoder 재구성 오차 통계값은 다음과 같습니다:
    - 평균 오차(mean error): {mean_err:.6f}
    - 95퍼센타일 오차(p95 error): {p95_err:.6f}
    - 최대 오차(max error): {max_err:.6f}

    다음을 설명해주세요:
    1) 이미지에서 재구성 오차가 큰(빨간색) 영역이 구체적으로 어디인지.
    2) 그 영역들이 왜 AI가 생성한 이미지인지.
    """

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",   
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                ],
            },
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content

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
def load_ae_model(ckpt_name="model20.pth"):
    model_path = os.path.join(cfg.models_dir, ckpt_name)
    ckpt = torch.load(model_path, map_location=device)

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
def run_on_image(img_path, ckpt_name="model20.pth"):
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

    # VLM 호출
    explanation = ask_vlm_explanation(save_path, mean_err, p95_err, max_err)
    return {
        "overlay_path": save_path,
        "mean_err": float(mean_err),
        "p95_err": float(p95_err),
        "max_err": float(max_err),
        "explanation": explanation,
    }


if __name__ == "__main__":
    # 여기에 테스트해보고 싶은 이미지 경로 넣기
    test_img_path = "test/fake_04.png"  # 예시
    result = run_on_image(test_img_path, ckpt_name="model20.pth")
    print("=== VLM explanation ===")
    print(result["explanation"])