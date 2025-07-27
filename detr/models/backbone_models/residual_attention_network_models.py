import torch
import torch.nn as nn
from .basic_layers import ResidualBlock
from .attention_module import AttentionModuleStage0, AttentionModuleStage1, AttentionModuleStage2, AttentionModuleStage3
from .attention_module import AttentionModuleStage1Cifar, AttentionModuleStage2Cifar, AttentionModuleStage3Cifar
from torchvision.models._utils import IntermediateLayerGetter


class ResidualAttentionModel448(nn.Module):
    # for input size 448
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # tbq add
        # 112*112
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModuleStage0(128, 128)
        # tbq add end
        self.residual_block1 = ResidualBlock(128, 256, 2)
        # 56*56
        self.attention_module1 = AttentionModuleStage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModuleStage2(512, 512)
        self.attention_module2_2 = AttentionModuleStage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModuleStage3(1024, 1024)
        self.attention_module3_2 = AttentionModuleStage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModuleStage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel92(nn.Module):
    # for input size 224
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModuleStage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModuleStage2(512, 512)
        self.attention_module2_2 = AttentionModuleStage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModuleStage3(1024, 1024)
        self.attention_module3_2 = AttentionModuleStage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModuleStage3(1024, 1024)  # tbq add

        

        #주석 내용을 연결하여 수정
        self.layer4 = nn.Sequential(
            ResidualBlock(1024, 2048, 2),
            ResidualBlock(2048, 2048),
            ResidualBlock(2048, 2048),
            nn.BatchNorm2d(2048)
        )
        self.last = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )

        # self.residual_block4 = ResidualBlock(1024, 2048, 2)
        # self.residual_block5 = ResidualBlock(2048, 2048)
        # self.residual_block6 = ResidualBlock(2048, 2048)
        # self.mpool2 = nn.Sequential(
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=7, stride=1)
        # )
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        print("attention92 input shape", x.shape)
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)

        #수정
        layer4_output = self.layer4(out)
        print("layer4_output shape", layer4_output.shape)
        layer4_output = self.last(layer4_output)
        out = layer4_output.view(layer4_output.size(0), -1)

        
        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)

        # out = self.mpool2(out)
        # out = out.view(out.size(0), -1)
        
        out = self.fc(out)

        return out


class ResidualAttentionModel56(nn.Module):
    # for input size 224
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModuleStage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModuleStage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModuleStage3(1024, 1024)
        
        
        
        #주석 내용을 연결하여 수정
        self.layer4 = nn.Sequential(
            ResidualBlock(1024, 2048, 2),
            ResidualBlock(2048, 2048),
            ResidualBlock(2048, 2048),
            nn.BatchNorm2d(2048)
        )
        self.last = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )

        # self.residual_block4 = ResidualBlock(1024, 2048, 2)
        # self.residual_block5 = ResidualBlock(2048, 2048)
        # self.residual_block6 = ResidualBlock(2048, 2048)
        # self.mpool2 = nn.Sequential(
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=7, stride=1)
        # )
        
        
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)

        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)


        
        #수정
        layer4_output = self.layer4(out)
        print("layer4_output shape", layer4_output.shape)
        layer4_output = self.last(layer4_output)
        out = layer4_output.view(layer4_output.size(0), -1)



        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)
        # out = self.mpool2(out)
        # out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


class ResidualAttentionModel92Cifar10(nn.Module):
    # for input size 32
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 16*16
        self.attention_module1 = AttentionModuleStage1Cifar(128, 128)  # 16*16
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 8*8
        self.attention_module2 = AttentionModuleStage2Cifar(256, 256)  # 8*8
        self.attention_module2_2 = AttentionModuleStage2Cifar(256, 256)  # 8*8 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModuleStage3Cifar(512, 512)  # 4*4
        self.attention_module3_2 = AttentionModuleStage3Cifar(512, 512)  # 4*4 # tbq add
        self.attention_module3_3 = AttentionModuleStage3Cifar(512, 512)  # 4*4 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 4*4
        self.residual_block5 = ResidualBlock(1024, 1024)  # 4*4
        self.residual_block6 = ResidualBlock(1024, 1024)  # 4*4
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4, stride=1)
        )
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel92U(nn.Module):
    # for input size 32
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        # self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32

        self.attention_module1 = AttentionModuleStage1Cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModuleStage2Cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModuleStage2Cifar(256, 256, size=(16, 16))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModuleStage3Cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModuleStage3Cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModuleStage3Cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class Residual18_AttentionModel(nn.Module):
    """residual_attention_18"""
    # for input size 224
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.residual_block1 = ResidualBlock(64, 64)
        self.attention_module1 = AttentionModuleStage1(64, 64)
        self.residual_block2 = ResidualBlock(64, 128, 2)
        self.attention_module2 = AttentionModuleStage2(128, 128)
        self.residual_block3 = ResidualBlock(128, 256, 2)
        self.attention_module3 = AttentionModuleStage3(256, 256)
        
        
        
        #주석 내용을 연결하여 수정
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.BatchNorm2d(512)
        )
        self.last = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )

        # self.residual_block4 = ResidualBlock(1024, 2048, 2)
        # self.residual_block5 = ResidualBlock(2048, 2048)
        # self.residual_block6 = ResidualBlock(2048, 2048)
        # self.mpool2 = nn.Sequential(
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=7, stride=1)
        # )
        
        
        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)

        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)


        
        #수정
        layer4_output = self.layer4(out)
        print("layer4_output shape", layer4_output.shape)
        layer4_output = self.last(layer4_output)
        out = layer4_output.view(layer4_output.size(0), -1)


        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)
        # out = self.mpool2(out)
        # out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


from .backbone_registry import register_backbone, BackboneBase

@register_backbone("residual_attention_18")
class residual_attention_Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 check_featuremap: bool):
        backbone = Residual18_AttentionModel()
        # getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 #512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, check_featuremap)
        return_layers = {'layer4': "0", "residual_block1": "1", "attention_module1": "2", "residual_block2": "3", "attention_module2": "4", "residual_block3": "5", "attention_module3": "6"} #숫자는 출력에 접근하기 위한 키값
        print(return_interm_layers, return_layers) #False
        #backbone은 resnet18임

        #todo 아래 에러 발생함
#           File "/home/robo/workspace/act-plus-plus_one/detr/models/backbone_models/residual_attention_network_models.py", line 430, in __init__
#     self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
# NameError: name 'IntermediateLayerGetter' is not defined

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        dummy_input = torch.randn(1, 3, 480, 640)
        output = self.body(dummy_input)
        for key, value in output.items():
            print(f"{key}: {value.shape}")

        print("\n\n\nself.body",self.body)
        
        #고정된 카메라로 인해 ROI의 특정 영역에 가중치 부여
        self.weights = torch.full((3, 480, 640), 0.5).to('cuda')
        x1, y1 = 230, 350
        x2, y2 = 412, 400
        self.weights[:, y1:y2, x1:x2] = 1.5
    
    def forward(self, tensor):
        # print("\n\npre tensor.shape",tensor.shape, tensor[0][0][0][0])
        # 텐서에 가중치 행렬 곱하기
        tensor = tensor * self.weights
        # print("\n\nlast tensor.shape",tensor.shape, tensor[0][0][0][0])
        #주요 포인트!!
        # #featuremap 분석을 위함 나중에 주석처리해야함
        if self.check_featuremap:
            # print(tensor.shape) #torch.Size([1, 3, 480, 640])

            def denormalize(tensor, mean, std):
                device = tensor.device
                # print(device) #cuda:0
                mean = torch.tensor(mean, device=device).view(3, 1, 1)
                std = torch.tensor(std, device=device).view(3, 1, 1)
                return tensor * std + mean
            
            # 예제 값
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # 정규화된 텐서를 비정규화
            print("prev backbone forward :",tensor[0][0][0][:10])
            first_image_tensor = denormalize(tensor, mean, std)
            print("next backbone forward :",first_image_tensor[0][0][0][:10])

            import matplotlib.pyplot as plt
            import torchvision.transforms as transforms
            print(first_image_tensor.shape, first_image_tensor.shape[0])

            for idx in range(first_image_tensor.shape[0]):
                rgb_tensor = first_image_tensor[idx][[2, 1, 0], :, :]  #.permute(1, 2, 0)  # [3, 480, 640] -> [480, 640, 3]
                to_pil = transforms.ToPILImage() #[C, H, W] 형태(채널, 높이, 너비)의 3차원 텐서를 [H, W, C] 형태의 PIL 이미지로 변환
                image = to_pil(rgb_tensor.cpu())  # GPU에 있으면 CPU로 옮긴 후 변환

                # 이미지 시각화
                plt.imshow(image)
                plt.axis('off')
                # plt.show()
                plt.savefig(f"tmp//1cam_first_image_{self.key}_{idx}.png", bbox_inches='tight')
                plt.close()
            self.key += 1

            outputs = []
            names = []
            featuremap = self.featuremap(tensor)
            for a,b in featuremap.items():
                print(a, b.shape)
                outputs.append(b)
                names.append(str(a))

            print(len(outputs)) #17개
            #print feature_maps
            for feature_map in outputs:
                print(feature_map.shape)

            for idx in range(feature_map.shape[0]):
                processed = []
                for feature_map in outputs:
                    feature_map = feature_map[idx].squeeze(0)
                    feature_map = feature_map[[2, 1, 0], :, :]
                    gray_scale = torch.sum(feature_map,0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    processed.append(gray_scale.data.cpu().numpy())
                for fm in processed:
                    print(fm.shape)

                fig = plt.figure(figsize=(30, 50))
                for i in range(len(processed)):
                    a = fig.add_subplot(5, 4, i+1)
                    imgplot = plt.imshow(processed[i])
                    a.axis("off")
                    a.set_title(names[i].split('(')[0], fontsize=10)
                plt.savefig(f"tmp//1cam_featuremap_{self.key}_{idx}.png", bbox_inches='tight')

                # plt.show()
                plt.close()

            # if self.key >= 5:
            #     sys.exit()
            # print("\n\ntensor")
            # print(tensor.shape) #torch.Size([8, 3, 480, 640])
        #self.body는 resnet18에서 layer4까지만 계산한 것
        xs = self.body(tensor)
        # print("\n\nxs")
        # for a, b in xs.items():
        #     print(a,b.shape) #0 torch.Size([8, 512, 15, 20])
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out



