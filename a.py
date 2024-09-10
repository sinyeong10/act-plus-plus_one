
import torch
print(torch.version.cuda)  # CUDA 버전 확인
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인

print(torch.version)
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA 사용 가능")
else:
    device = torch.device("cpu")    
    print("CUDA를 사용할 수 없습니다")