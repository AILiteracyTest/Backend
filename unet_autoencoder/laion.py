# laion.py

from datasets import load_dataset
from PIL import Image
import os
import io
import numpy as np
import torch

# ----------------------
# 설정
# ----------------------
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
N_TRAIN = 20000   # 학습용 이미지 개수
N_VAL = 2000      # 검증용 이미지 개수

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


def decode_image(byte_data, shape=(3, 384, 384)):
    """
    ccvl/LAION-High-Qualtiy-Pro-6M-VLV 의 'image' 컬럼은
    raw uint8 buffer (3x384x384) 또는 npy로 저장되어 있음.
    → torch.Tensor(C,H,W) + PIL.Image 로 변환
    """
    # 1) raw uint8 buffer로 해석 시도
    try:
        arr = np.frombuffer(byte_data, dtype=np.uint8)
        arr = arr.reshape(shape)  # (C,H,W)
        tensor = torch.from_numpy(arr.copy())
    except ValueError:
        # 2) np.save 형식일 수 있으므로 numpy load로 시도
        with io.BytesIO(byte_data) as buf:
            try:
                arr = np.load(buf, allow_pickle=True)
                if isinstance(arr, np.ndarray):
                    tensor = torch.from_numpy(arr)
                else:
                    raise ValueError("Loaded object is not ndarray")
            except Exception as e:
                print("Warning: could not deserialize bytes:", e)
                # 실패하면 그냥 검은 이미지로 대체
                tensor = torch.zeros(shape, dtype=torch.uint8)

    # CHW 형태 보장
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
        chw = tensor
    elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:
        chw = tensor.permute(2, 0, 1)  # HWC → CHW
    else:
        raise ValueError(f"Unexpected tensor shape {tensor.shape}")

    # PIL 이미지로 변환
    hwc = chw.permute(1, 2, 0).numpy().astype(np.uint8)  # (H,W,C)
    pil_img = Image.fromarray(hwc)

    return pil_img


# ----------------------
# 데이터셋 스트리밍 로드
# ----------------------
ds = load_dataset(
    "ccvl/LAION-High-Qualtiy-Pro-6M-VLV",
    split="train",
    streaming=True,
)

train_count = 0
val_count = 0

for sample in ds:
    if train_count >= N_TRAIN and val_count >= N_VAL:
        break

    try:
        byte_data = sample["image"]
        img = decode_image(byte_data)  # PIL.Image
    except Exception as e:
        print("skip sample, decode error:", e)
        continue

    # train / val 나누어서 저장
    if train_count < N_TRAIN:
        save_path = os.path.join(TRAIN_DIR, f"{train_count:06d}.jpg")
        train_count += 1
    else:
        save_path = os.path.join(VAL_DIR, f"{val_count:06d}.jpg")
        val_count += 1

    img.save(save_path)

print("Done!")
print("train images:", train_count)
print("val images:", val_count)
