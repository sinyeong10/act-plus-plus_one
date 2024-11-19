import torch
import torchvision.models as models

# ResNet-18 모델 로드
model = models.resnet18(pretrained=True)

# 모델 구조 출력
print(model)
