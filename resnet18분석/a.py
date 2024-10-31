import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

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

from util.misc import NestedTensor, is_main_process

from models.position_encoding import build_position_encoding
import sys

import IPython
e = IPython.embed

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
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        print("\n\ntensor")
        print(tensor)
        xs = self.body(tensor)
        print(xs)
        sys.exit()
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
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main(args):
    
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class'] #이 값에 따라서 모델에 따른 설정값 설정
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'

    
    if is_sim or task_name == 'all':
    #'dataset_dir', 'num_episodes', 'episode_len' 'camera_names' 설정 값을 가져옴
    #이건 시뮬레이션 용?
        # from constants import SIM_TASK_CONFIGS
        task_config = {
        'dataset_dir': 'scr/tonyzhao/datasets/next_sim_mycobot_320' + '/next_sim_mycobot_320',
        'num_episodes': 50,
        'episode_len': 115,
        'camera_names': ['right_wrist', 'top']#, 'left_wrist', 'right_wrist']
    }
    else: #이건 하드웨어 용?
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    #.get() 없으면 뒤의 값을 가져옴
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    #모델의 설정 값
    # fixed parameters
    state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args['lr'],
                'num_queries': args['chunk_size'],
                'kl_weight': args['kl_weight'],
                'hidden_dim': args['hidden_dim'],
                'dim_feedforward': args['dim_feedforward'],
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': camera_names,
                'vq': args['use_vq'],
                'vq_class': args['vq_class'],
                'vq_dim': args['vq_dim'],
                'action_dim': 8,
                'no_encoder': args['no_encoder'],
                'one_arm_policy_config' : True,
                }
    
    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    
    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'actuator_config': actuator_config,
        'one_arm_config' : False,
    }

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None


    # 이미지 전처리를 위한 Transform 정의
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    # HDF5 파일 경로 설정
    dataset_path = r"C:\Users\cbrnt\OneDrive\문서\act-plus-plus\mycobot320_script\all_data\cur\twocam_nextstep5단위\two_cam_episode_5.hdf5"

    # HDF5 파일
    with h5py.File(dataset_path, 'r') as file:
        # 저장된 카메라 이름들 (여기서는 가정)
        camera_names = ['right_wrist', "top"]
        
        for i in range(115):
            fig, axes = plt.subplots(1, len(camera_names), figsize=(12, 6))  # 카메라 수에 따라 그림판을 생성

            for idx, cam_name in enumerate(camera_names):
                print(f"Loading images from {cam_name}")
                images = file[f'observations/images/{cam_name}'][i]  # 이미지를 읽음

                # 이미지를 matplotlib를 이용해 출력
                axes[idx].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(f'Frame {i} from {cam_name}')

            plt.tight_layout()
            plt.show()

            # 예제 이미지 로드 및 전처리
            img = Image.fromarray(images)  # 이미지를 불러옵니다 (example.jpg는 이미지 파일 경로)
            img_tensor = preprocess(img).unsqueeze(0)  # 배치 차원 추가

            # # 모델 및 백본 로드
            # args = {
            #     'backbone': 'resnet18',
            #     'lr_backbone': 0.0,
            #     'masks': True,
            #     'dilation': False
            # }

            # class Args:
            #     backbone = 'resnet18'
            #     lr_backbone = 0.0
            #     masks = True
            #     dilation = False

            # args = Args()

            # Backbone 빌드
            model = build_backbone(args)
            model.eval()

            # 이미지 처리
            with torch.no_grad():
                features, positions = model(NestedTensor(img_tensor, None))

            # 시각화를 위한 헬퍼 함수 정의
            def show_feature_map(tensor, layer_name):
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(tensor[0, 0].cpu().numpy(), cmap='viridis')  # 첫 번째 채널만 시각화
                ax.axis('off')
                ax.set_title(f"Layer: {layer_name}")
                plt.show()

            # 각 레이어 출력 시각화
            for i, feature_map in enumerate(features):
                show_feature_map(feature_map.tensors, f"Layer {i+1}")

import argparse
#명령줄의 인자를 파싱해서 main함수에 전달
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    #action='store_true'는 --eval 옵션이 명령줄에 포함되지 않으면 args['eval']을 False로 설정
    #action='store'은 저장
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    parser.add_argument('--model', action='store', type=str, help='model', default='policy_last.ckpt')

    sys.argv = [
        'auto_run.py',
        '--ckpt_dir', 'scr/mycobot320_data/next_twocam_mycobot320_chunk100_biased_first',
        '--policy_class', 'ACT',
        '--task_name', 'sim_mycobot320',
        '--batch_size', '8',
        '--seed', '0',
        '--num_steps', '5',
        '--lr', '1e-05',
        '--eval_every', '500',
        '--validate_every', '500',
        '--save_every', '500',
        '--kl_weight', '10',
        '--chunk_size', '100',
        '--hidden_dim', '512',
        '--dim_feedforward', '3200',
        '--model', 'best_policy_step_50000_seed_0.ckpt',
        '--eval'#,
        # '--temporal_agg'
    ]
    main(vars(parser.parse_args()))