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

# from constants import FPS #50
# from constants import PUPPET_GRIPPER_JOINT_OPEN #그리퍼 한계 값
from utils import load_data, load_data_one # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer

# from sim_env import BOX_POSE

import IPython
e = IPython.embed

FPS = 10
PUPPET_GRIPPER_JOINT_OPEN = 100
check_featuremap = True#False #True #False

#dataset_dir 경로에서 함수내 max_idx 값까지 f"qpos_{i}.npy"파일이 있는 지 찾아보고 없으면 i를 반환함
#인덱스를 순차적으로 생성하는 데 씀?
def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

#인자 받아 설정하고 task_name의 앞의 4글자가 sim_인지 체크, policy에 따른 모델 설정, 평가 모드에 따라 평가하거나 학습함
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
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
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
    state_dim = 14
    if task_name == 'sim_move_cube_scripted' or task_name == 'sim_mycobot320': #one arm
        state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
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
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         'one_arm_policy_config' : False,
                         }
        if task_name == 'sim_move_cube_scripted' or task_name == 'sim_mycobot320': #one arm
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
    elif policy_class == 'Diffusion':
        #Diffusion시 설정값
        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':
        #CNNMLP설정값
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError
    #actuator 설정, eval_bc에서 활용됨
    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }
    #모델 관련 설정
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
        'one_arm_config' : True,
    }
    if task_name == 'sim_move_cube_scripted' or task_name == 'sim_mycobot320': #one arm
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
    #ckpt_dir 폴더가 없으면 폴더 생성
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    #설정 값을 저장할 경로인 config.pkl 생성
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1] #생성한 마지막 경로가 expr_name이 됨
    print(expr_name)
    # expr_name = "cbrnt1210"
    if not is_eval: #--eval값이 없으면 False로 되어 있음, wandb 설정함
        wandb.init(project="mobile-aloha2", reinit=True, entity="cbrnt1210", name=expr_name)
        wandb.config.update(config)
    #설정 값 config.pkl 저장함
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    if is_eval: #평가모드 True로 값을 줬을 경우
        
        ckpt_names = [f'best_policy_step_5000_seed_0.ckpt'] #f'policy_last.ckpt',  #마지막 체크포인트 파일을 의미
        results = []
        for ckpt_name in ckpt_names: #하나만 있으니 하나에 대해서 실행함
            #eval_bc 함수를 통해 지정한 모델(ACT)을 가져와서 돌려보고 성공확률과 평균보상을 반환 함
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=2)
            # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
            results.append([ckpt_name, success_rate, avg_return])
        #결과 출력
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit() #종료


    
    #utils.load_data ?를 함
    if "sim_move_cube" in task_name or 'sim_mycobot320' in task_name:
        train_dataloader, val_dataloader, stats, _ = load_data_one(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)
    else:
        train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)

    #['action_mean', 'action_std', 'action_min', 'action_max', 'qpos_mean', 'qpos_std', 'example_qpos']가 들어가 있는 stats변수를 저장함
    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    #trian_bc : 모델을 학습시키고 (평가로 중간 결과도 보며) 가장 좋은 모델 정보를 반환 함
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    #가장 좋았던 모델 저장하고 loss 출력하며 몇번 step이 가장 좋았는 지 출력
    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'last_{best_step}_policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    wandb.finish()

#입력받는 값인 policy_class와 그에 따라 설정된 policy_config를 받아 모델을 가져옴
def make_policy(policy_class, policy_config):
    #policy.ACTPolicy 클래스를 설정에 따라 생성하여 반환함
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

#policy별로 optimizer가져옴
def make_optimizer(policy_class, policy):
    #객체의 optimizer인 torch.optim.AdamW를 가져옴
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

#카메라별 이미지를 가져와서 정규화하고 첫 차원을 추가함
def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        #환경에서 카메라별 이미지를 가져옴 채널을 앞으로 가져옴
        print(len(ts.observation['images'][cam_name]))
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')#ts.observation['images'][cam_name], 'h w c -> c h w') #복원필요
        curr_images.append(curr_image)
    #리스트에 여러 값이 있는 걸 차원으로 쌓음 (2,2) 4개가 (4,2,2)로!
    curr_image = np.stack(curr_images, axis=0)
    #정규화 및 첫 차원 추가
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    #일단 넘어감?
    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
        print(curr_image.shape)
    return curr_image

#가상환경 설정하고 지정한 모델(ACT)을 가져와서 돌려보고 성공확률과 평균보상을 반환
def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=50, dir_step = 0):
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

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name) #경로랑 파일이름을 결합
    policy = make_policy(policy_class, policy_config) #make_policy ?를 함
    print("ckpt_path", ckpt_path)
    loading_status = policy.deserialize(torch.load(ckpt_path)) #ACT기준 ckpt_path 경로에서 상태 정보를 가져와 / 모델에 로드함
    print("loading_status",loading_status)
    if check_featuremap:
        print(policy.model.backbones[0][0].body) #2번 카메라 policy.model.backbones[1][0]
        policy.model.backbones[0][0].featuremap = policy.model.backbones[0][0].body
        policy.model.backbones[1][0].featuremap = policy.model.backbones[1][0].body
        
    policy.cuda() #GPU로 옮김
    policy.eval() #평가모드 설정(학습과 다름!)
    if vq: #vq를 cmd에서 파일 실행시켰을 때 값을 준 경우 Latent_Model_Transformer ?을 로드함
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')
    #['action_mean', 'action_std', 'action_min', 'action_max', 'qpos_mean', 'qpos_std', 'example_qpos']를 로드함
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print("stats", stats)
    # if use_actuator_net:
    #     prediction_len = actuator_config['prediction_len']
    #     future_len = actuator_config['future_len']
    #     history_len = actuator_config['history_len']
    #     actuator_network_dir = actuator_config['actuator_network_dir']

    #     from act.train_actuator_network import ActuatorNetwork
    #     actuator_network = ActuatorNetwork(prediction_len)
    #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
    #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
    #     actuator_network.eval()
    #     actuator_network.cuda()
    #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

    #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
    #     with open(actuator_stats_path, 'rb') as f:
    #         actuator_stats = pickle.load(f)
        
    #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
    #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
    #     def collect_base_action(all_actions, norm_episode_all_base_actions):
    #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
    #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

    #s_qpos를 정규화 함
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else: #Diffusion이 아니면 역정규화함
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # # load environment
    # if real_robot: #mobile_aloha라이브러리 참고
    #     from aloha_scripts.robot_utils import move_grippers # requires aloha
    #     from aloha_scripts.real_env import make_real_env # requires aloha
    #     env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
    #     env_max_reward = 0
    # else: #make_sim_env ?함
    #     from sim_env import make_sim_env #이거 내가 짜야함....
    #     env = make_sim_env(task_name) #sim_transfer_cube, sim_insertion 이게 아니면 에러남....
    #     env_max_reward = env.task.max_reward

    from aloha_scripts.real_env import make_real_env # requires aloha
    env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
    env_max_reward = 0

    #query의 빈도를 설정?
    query_frequency = policy_config['num_queries'] #ACT기준 chunk_size만큼 생성됨
    if temporal_agg: #매 시간 단계마다 액션을 쿼리?, 이때 다수를 처리?
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot: #실제 로봇에서 지연을 고려?
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    #episode_len에 해당하는 값
    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        if real_robot:
            e()
        rollout_id += 0
        #시뮬에이션에서 해야할 일에 해당하는 물건 크기 설정
        ### set task
        # if 'sim_transfer_cube' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        # elif 'sim_insertion' in task_name:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        # elif 'sim_move' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        #환경 초기화
        ts = env.reset()

        ### evaluation loop
        if temporal_agg: #모든 타임스텝에 대한 작업 데이터 저장?
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 8]).cuda() #16

        #시뮬레이션이 끝난 후 각 스텝별 상태(위치 데이터)를 저장?
        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []

        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode(): #모델을 추론모드로 설정 : 자동 미분 비활성화, 메모리 사용 최적화, 모델의 forward 동작만 실행
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render:
                    print("fps_rendering", onscreen_render)
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.draw()  # 화면 갱신 대신 draw 사용
                    plt.savefig(f'scr/image/{dir_step}_{rollout_id}/rendered_image_{t}_{max_timesteps}.png')  # 파일로 저장
                    
                    # image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    # plt_img.set_data(image)
                    # plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                #ts = env.reset()임
                obs = ts.observation
                #여기선 카메라별 이미지가 아님! 그냥 usb 카메라 한개이미지!
                # print(type(obs['images']))
                # print(obs['images'].keys())
                #아마 image파일 경로에 따라 다를 수 있어서 그거 처리?
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                #14축에 대한 위치값?
                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy) #qpos_numpy를 정규화 함
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0) #첫번째 차원 추가
                # qpos_history[:, t] = qpos
                if t % query_frequency == 0: #지정된 query_frequency마다 get_image ?를 함
                    curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))
                # print('get image: ', time.time() - time2)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        #__call__가 불러지는 듯?
                        policy(qpos, curr_image)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        if vq: #일단 패스?
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # e()
                            all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot: #일단 패스?
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    #일단 패스?
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                #일단 패스?
                elif config['policy_class'] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    raw_action = all_actions[:, t % query_frequency]
                #일단 패스?
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                else:
                    raise NotImplementedError
                # print('query policy: ', time.time() - time3)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy() #첫번째차원제거
                action = post_process(raw_action) #아마 역정규화함?
                #ACT기준 16 action을 14, 2로 나눔
                target_qpos = action[:-2]

                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)
                if "sim_move_cube" in task_name or 'sim_mycobot320' in task_name: #수정
                    target_qpos = action[:-1]
                    base_action = action[-1:]

                ### step the environment
                time5 = time.time()
                #control까지 들어가야 함수가 존재, 이전,물리환경,이후로 동작하는 듯?
                if real_robot:
                    ts = env.step(target_qpos, base_action)
                else:
                    # print(target_qpos)
                    ts = env.step(target_qpos)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                #프레임 속도 조절
                duration = time.time() - time1
                sleep_time = max(0, DT - duration) #기다려야할 시간 계산
                # print(sleep_time)
                time.sleep(sleep_time) #기다림
                # time.sleep(max(0, DT - duration - culmulated_delay))
                #한단계 처리시간이 설정 시간 DT(1/FPS)보다 김
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))
            #평균 프레임 속도 계산 및 출력
            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close() #시각화창 종료
        #일단 넘어감?
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i+1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()
        
        print("그리퍼 열고 종료")
        env.mycobot.set_gripper_value(100,20,1)
        time.sleep(2)
        
        # env.save("scr/mycobot320_data/twocam_mycobot320_chunk20_1", 1)

        #반환된 보상의 총합, 최대 보상, 성공 여부 계산해서 출력
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}, Max : {episode_highest_reward}')

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
    #여러번 시도한 후 성공률 계산, 평균 보상 계산
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    #최대 보상까지 모든 보상 수준에 대해
    for r in range(env_max_reward+1):
        #그 이상 받은 에피소드의 수와 비율을 계산해 출력
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    #최대 보상까지 모든 보상 수준에 대해 그 이상 받은 에피소드의 수와 비율,
    #모든 보상의 합, 가장 큰 보상에 대한 결과 저장
    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))
    #성공률과 평균보상 반환
    return success_rate, avg_return

#data(image_data, qpos_data, action_data, is_pad)를 gpu로 올려서 policy.__call__정책 실행
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    # print("imitate_episodes 640line action_data.size() : ", action_data.size()) # torch.Size([8, 100, 8])
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

#모델을 학습시키며 (중간 평가도 함) 가장 좋은 결과를 반환함
def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    eval_every = 1000
    validate_every = 1000
    save_every = 1000

    set_seed(seed)
    #make_policy ?를 함
    policy = make_policy(policy_class, policy_config)
    #쓰지 않는 듯, 모델을 로드해서 상태를 가져옴
    print(config['load_pretrain'], config['resume_ckpt_path'])
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    
    policy.cuda()
    #옵티마이저 설정
    optimizer = make_optimizer(policy_class, policy)
    #초기값은 무한대
    min_val_loss = np.inf
    best_ckpt_info = None
    #계속 값을 가져올 이터레이터 생성
    train_dataloader = repeater(train_dataloader)
    #num_step만큼 반복하며 진행 상황을 표시
    for step in tqdm(range(num_steps+1)):
        #validate_every마다 검증함
        # validation


        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode(): #연산을 추론 모드 설정
                policy.eval() #모델을 평가모드로
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    # print("imitate_episodes 684line data : ", type(data), data[2].size()) #<class 'list'> torch.Size([8, 100, 8])
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    #50개 넘어가면 멈춤, 너무 많이 다 돌리지 않음!
                    if batch_idx > 50:
                        break
                #compute_dict_mean ?를 함
                validation_summary = compute_dict_mean(validation_dicts)

                #epoch_val_loss값이 min_val_loss값보다 작아질 때마다 갱신하고 모델 정보 기록
                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize())) #모델 상태를 직렬화? 함
            
            #계산되어 모든 값 저장하는 듯?
            print("\n\nvalidation_summary.keys() :",validation_summary.keys())
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
            #

        #중간평가는 실제라 위험함
        # #모델을 저장하고, 평가시켜 성공률을 출력
        # # evaluation
        # if (step > 0) and (step % eval_every == 0):
        #     # first save then eval
        #     ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
        #     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        #     torch.save(policy.serialize(), ckpt_path)
        #     success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10, dir_step=step)
        #     wandb.log({'success': success}, step=step)


        # training
        policy.train() #모델 학습 모드 설정
        optimizer.zero_grad() #모든 파라미터의 그래디언트를 0으로 초기화
        data = next(train_dataloader) #데이터 가져옴
        forward_dict = forward_pass(data, policy) #ACT의 __call__ ?를 실행
        # backward
        loss = forward_dict['loss'] #loss만 가져와
        loss.backward() #역전파
        optimizer.step() #optimizer로 모델 업데이트
        wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        #save_every마다 모델 저장
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

            best_step, min_val_loss, best_state_dict = best_ckpt_info
            ckpt_path = os.path.join(ckpt_dir, f'best_policy_step_{best_step}_seed_{seed}.ckpt')
            torch.save(best_state_dict, ckpt_path)

    #마지막 모델 저장
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    #가장 loss가 낮은 상태, 저장
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'best_policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

#data_loader을 계속 호출하며 한 번 거기서 데이터를 다 출력시 epoch 정보를 출력하고 증가시킴
def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader): #매번 loader에 data_loader이 전달됨
        for data in loader:
            yield data
        print(f'Epoch {epoch} done') #모든 데이터를 다 전달해서 epoch정보 출력함
        epoch += 1

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
    
    main(vars(parser.parse_args())) #명령줄의 인자를 파싱해서 main함수에 전달
