import os
import glob
import cv2
import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset


class DAE_dataset(Dataset):
    """
    LAION Aesthetics v2+ 같은 '일반 이미지 폴더'에서
    Autoencoder 학습용 (input == target) 컬러 이미지를 로드하는 Dataset.
    """

    def __init__(self, data_dir, transform=None,
                 exts=(".jpg", ".jpeg", ".png", ".webp")):
        """
        data_dir: 이미지들이 들어 있는 루트 폴더
                  (하위에 여러 서브폴더가 있어도 recursive=True로 전부 탐색)
        transform: torchvision.transforms.Compose([...]) 등
        exts: 허용할 이미지 확장자 튜플
        """
        self.data_dir = data_dir
        self.transform = transform

        # data_dir 아래 모든 하위 폴더까지 포함해서 이미지 경로 수집
        self.imgs_data = []
        for ext in exts:
            self.imgs_data.extend(
                glob.glob(os.path.join(self.data_dir, "**", f"*{ext}"),
                          recursive=True)
            )
        self.imgs_data = sorted(self.imgs_data)

        if len(self.imgs_data) == 0:
            raise RuntimeError(f"No images found in {data_dir} with extensions {exts}")

    def __len__(self):
        return len(self.imgs_data)

    def __getitem__(self, index):
        img_path = self.imgs_data[index]

        # 컬러(BGR)로 읽고 → RGB로 변환
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [H, W, 3]

        # transform 적용 (Resize, ToTensor, Normalize 등)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # transform이 없다면 최소한 텐서 모양으로라도 맞춰주기 (fallback)
            img = img.astype(np.float32) / 255.0        # 0~1
            img = np.transpose(img, (2, 0, 1))          # [3, H, W]

        # AE이므로 입력과 타깃이 동일
        return img, img


class custom_test_dataset(Dataset):
    """
    학습된 AE에 이미지를 넣어보고 싶을 때 쓰는 테스트용 Dataset.
    이미지 1장만 반환 (input == 이미지).
    out_size로 리사이즈 + 패딩해서 고정 크기로 맞추는 구조는 유지.
    """

    def __init__(self, data_dir, transform=None, out_size=(256, 256)):
        assert out_size[0] <= out_size[1], \
            "height/width of the output image shouldn't be greater than 1"
        self.data_dir = data_dir
        self.transform = transform
        self.out_size = out_size
        self.imgs_data = self.get_data(self.data_dir)

    def get_data(self, data_path):
        data = []
        for img_path in glob.glob(data_path + os.sep + '*'):
            data.append(img_path)
        return sorted(data)

    def __len__(self):
        return len(self.imgs_data)

    def __getitem__(self, index):
        img_path = self.imgs_data[index]

        # 컬러로 읽고 → RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [H, W, 3]

        h, w, _ = img.shape
        H_out, W_out = self.out_size

        # 높이가 out_size 보다 크면 비율 유지한 채로 축소
        if h > H_out:
            resize_factor = H_out / h
            img = cv2.resize(img, (0, 0),
                             fx=resize_factor, fy=resize_factor,
                             interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape

        # 너비가 out_size 보다 크면 비율 유지한 채로 축소
        if w > W_out:
            resize_factor = W_out / w
            img = cv2.resize(img, (0, 0),
                             fx=resize_factor, fy=resize_factor,
                             interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape

        # 패딩 계산 (위/아래, 좌/우)
        pad_height = H_out - h
        pad_top = int(pad_height / 2)
        pad_bottom = H_out - h - pad_top

        pad_width = W_out - w
        pad_left = int(pad_width / 2)
        pad_right = W_out - w - pad_left

        # 3채널이므로 채널별로 같은 값(검정)으로 패딩
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # [3, H, W]

        return img
