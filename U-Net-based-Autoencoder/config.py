import os

# -----------------------------
# 경로 관련
# -----------------------------

# 모델 저장 폴더
models_dir = 'models'

# loss curve 저장 폴더
losses_dir = 'losses'

# reconstruction 결과 (예: fake 이미지 heatmap 테스트용) 저장 폴더
res_dir = 'results'

# 데이터 루트 폴더 (여기 아래에 train/val 폴더가 있다고 가정)
data_dir = 'data'          # ./data
train_dir = 'train'        # ./data/train
val_dir = 'val'            # ./data/val

# 나중에 train.py에서:
# train_data_dir = os.path.join(data_dir, train_dir)
# val_data_dir   = os.path.join(data_dir, val_dir)

# -----------------------------
# UNet AE 모델 설정
# -----------------------------

# UNet depth (인코더/디코더 단계 수)
depth = 4  # 메모리 부족하면 3으로 줄여도 됨

# -----------------------------
# 학습 관련 설정
# -----------------------------

# 처음에는 항상 False (처음부터 학습)
resume = False              # True면 저장된 weight에서 이어서 학습

# resume=True일 때 불러올 checkpoint 파일 이름
ckpt = 'model01.pth'

# learning rate
lr = 1e-4        # AE 학습이면 1e-4 정도가 보통 더 잘 맞음 (1e-5는 약간 느림)

# epoch 수
epochs = 20      # AE는 12보다 조금 더 돌려도 괜찮음. 필요에 따라 조정

# batch size
batch_size = 32  # 메모리 보고 조정 (터지면 16, 8로 줄이기)

# train/val log 출력 간격 (iteration 단위)
log_interval = 25

# -----------------------------
# 테스트 관련 (옵션)
# -----------------------------

# test용 이미지를 따로 둘 거면 이런 식으로 쓸 수 있음 (원하면 유지)
test_dir = os.path.join(data_dir, val_dir)  # 검증 폴더 재사용해도 됨
test_bs = 64
