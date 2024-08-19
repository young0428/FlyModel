import torch
import numpy as np
# 두 개의 텐서 생성
tensor1 = torch.randn(3,50,64,128,3)
a = np.array(list(tensor1))
b = a[0, 0:10,::2,::2,1:2]
print(np.shape(b))

# 첫 번째 차원(axis=0)으로 이어붙이기
