# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

import sys
project_dir = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\detr'
sys.path.append(project_dir)
project_dir = r'C:\Users\cbrnt\OneDrive\문서\act-plus-plus\detr\models'
sys.path.append(project_dir)
print(sys.path)

from .residual_attention_network import Residual18_AttentionModel as ResidualAttentionModel
        
from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

check_featuremap = True #False #True #False #True #False #True

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        #기본적으로 backbone.named_parameters()는 True임
        
        # conv1.weight torch.Size([64, 3, 7, 7]) True
        # layer1.0.conv1.weight torch.Size([64, 64, 3, 3]) True
        # layer1.0.conv2.weight torch.Size([64, 64, 3, 3]) True
        # layer1.1.conv1.weight torch.Size([64, 64, 3, 3]) True
        # layer1.1.conv2.weight torch.Size([64, 64, 3, 3]) True
        # layer2.0.conv1.weight torch.Size([128, 64, 3, 3]) True
        # layer2.0.conv2.weight torch.Size([128, 128, 3, 3]) True
        # layer2.0.downsample.0.weight torch.Size([128, 64, 1, 1]) True
        # layer2.1.conv1.weight torch.Size([128, 128, 3, 3]) True
        # layer2.1.conv2.weight torch.Size([128, 128, 3, 3]) True
        # layer3.0.conv1.weight torch.Size([256, 128, 3, 3]) True
        # layer3.0.conv2.weight torch.Size([256, 256, 3, 3]) True
        # layer3.0.downsample.0.weight torch.Size([256, 128, 1, 1]) True
        # layer3.1.conv1.weight torch.Size([256, 256, 3, 3]) True
        # layer3.1.conv2.weight torch.Size([256, 256, 3, 3]) True
        # layer4.0.conv1.weight torch.Size([512, 256, 3, 3]) True
        # layer4.0.conv2.weight torch.Size([512, 512, 3, 3]) True
        # layer4.0.downsample.0.weight torch.Size([512, 256, 1, 1]) True
        # layer4.1.conv1.weight torch.Size([512, 512, 3, 3]) True
        # layer4.1.conv2.weight torch.Size([512, 512, 3, 3]) True
        # fc.weight torch.Size([1000, 512]) True
        # fc.bias torch.Size([1000]) True
        
        #처음부터 주석처리 되어 있었음
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     print(name, parameter.shape, parameter.requires_grad)
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"} #숫자는 출력에 접근하기 위한 키값
        # print(return_interm_layers, return_layers) #False
        #backbone은 resnet18임
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # print("self.body", self.body) #backbone에서 layer4까지만 계산하여 전달함
        self.num_channels = num_channels

        # self.a = backbone
        #주요 포인트!!
        # #featuremap 분석을 위함 나중에 주석처리해야함
        if check_featuremap:
        #     import copy
        #     a = copy.deepcopy(backbone)
        #     self.featuremap = IntermediateLayerGetter(a, return_layers={'layer4': "1"})#{"layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"})
            self.key = 0

    def forward(self, tensor):
        #주요 포인트!!
        # #featuremap 분석을 위함 나중에 주석처리해야함
        if check_featuremap:
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


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        #torchvision.models 모듈에서 name으로 지정된 ResNet 모델을 동적으로 가져옴
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        #주어진 모듈의 순차적 실행

    def forward(self, tensor_list: NestedTensor):
        # print(tensor_list.shape) #torch.Size([8, 3, 480, 640])
        xs = self[0](tensor_list) 
        # print("\n\nxs",type(xs)) #xs <class 'collections.OrderedDict'>
        # for key, value in xs.items():
        #     print(f"{key}: {value.shape}") #0: torch.Size([8, 512, 15, 20])
        #0: torch.Size([8, 512, 15, 20]) : batchsize, num_channels, height, width
        out: List[NestedTensor] = [] #자료형의 명시적 표현
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            # print("\n\nposition encoding", type(self[1](x).to(x.dtype))) #<class 'torch.Tensor'>
            # print(self[1](x).to(x.dtype).shape) #torch.Size([1, 512, 15, 20])
            pos.append(self[1](x).to(x.dtype)) #torch.Size([1, 512, 15, 20])

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    print("\n\nargs.backbone", args.backbone)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class background_Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = ResidualAttentionModel()
        # getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 #512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        return_layers = {'layer4': "0", "residual_block1": "1", "attention_module1": "2", "residual_block2": "3", "attention_module2": "4", "residual_block3": "5", "attention_module3": "6"} #숫자는 출력에 접근하기 위한 키값
        print(return_interm_layers, return_layers) #False
        #backbone은 resnet18임
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        dummy_input = torch.randn(1, 3, 480, 640)
        output = self.body(dummy_input)
        for key, value in output.items():
            print(f"{key}: {value.shape}")

        print("\n\n\nself.body",self.body)
        
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
        if check_featuremap:
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



def build_background_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = background_Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model