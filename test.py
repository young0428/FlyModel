import torch

# 두 개의 텐서 생성
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

# 첫 번째 차원(axis=0)으로 이어붙이기
result = torch.cat((tensor1, tensor2), dim=-1)
print(result)

# 두 번째 차원(axis=1)으로 이어붙이기
result = torch.cat((tensor1, tensor2), dim=1)
print(result)
