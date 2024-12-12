import torch.nn as nn
from .basic_layers import ResidualBlock
from .attention_module import AttentionModuleStage0, AttentionModuleStage1, AttentionModuleStage2, AttentionModuleStage3
from .attention_module import AttentionModuleStage1Cifar, AttentionModuleStage2Cifar, AttentionModuleStage3Cifar


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
