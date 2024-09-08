import torch

# CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    print("CUDA 사용 가능합니다.")
else:
    print("CUDA 사용 불가능합니다.")