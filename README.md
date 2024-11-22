# Imitation Learning algorithms and Co-training for Mobile ALOHA


#### Project Website: https://mobile-aloha.github.io/

This repo contains the implementation of ACT, Diffusion Policy and VINN, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate them in sim or real.
For real, you would also need to install [Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha). This repo is forked from the [ACT repo](https://github.com/tonyzhaozh/act).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

- also need to install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch) for Diffusion Policy by `pip install -e .`

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments (LEGACY table-top ALOHA environments)

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <data save dir> --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the simulated episodes after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

Note: to visualize data from the mobile-aloha hardware, use the visualize_episodes.py from https://github.com/MarkFzp/mobile-aloha

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt dir> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

### [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

<br>

aloha 오픈 소스에 코드가 바로 돌아가지 않음, 따라서 깃허브 이슈를 통해 해결해야 함

dm_control이 리눅스 환경에서만 지원됨 따라서 리눅스 환경에서 코드를 실행해야하며 WSL 환경에서 돌릴 경우 backbone.py에서 utils.py와 detr의 파일 import 주소를 못찾아서 강제로 할당해줘야 함

데이터 셋을 살펴보면 script는 80~90%의 성공률을 보이고, human_data는 10%의 성공률을 보임

단, human_data는 실패 과정을 포함하여 모두 다르게 움직이기 때문에 성공하는 것에 대한 의미가 있음!

여러 데이터 중 성공 시점을 기준으로 잡아 시간축을 보정하고 성공하는 위치 값을 찾아낼 수 있게 한다면 더 높은 성공율을 보일 것 같음..

<br>

constants.py에서 데이터와 관련된 변수 설정, 특히 카메라 이름의 갯수가 중요

sim_env.py에서 기존의 형태를 기준으로 새로운 에피소드 설정, scripted_policy.py에서 x,y,z의 기준으로 이동할 값을 설정, record_sim_episodes.py에서 scripted_policy에서 실행할 클래스 지정

가상환경에서 동작시킨 후 프레임마다 평가 기준을 설정함

기존의 xml 파일을 수정해서 박스 객체를 추가함

기존의 xml 파일을 수정해서 팔하나를 제거함

카메라가 많아지면 더 빠른 epoch만에 성공을 확인할 수 있으나 학습 시간이 10배 더 걸림!

chunk_size가 커질 수록 모델 학습 시간이 더 걸림

chunk_size가 커질 수록 현재 입력 이미지를 기준으로 더 많은 프레임의 이동 위치를 계산

중간에 물체를 놓는 경우가 발생하면 시간적 앙상블을 적용하여 매 프레임마다 계산하면 미세한 오차를 다른 시점에서의 예측 값이 보정할 수 있음!

단, 중간 시점에 예측할 때 오차가 크다면 그로 인한 영향이 전체 동작에 영향을 미칠 수 있음!

랜덤 위치인 A1, A2, A3...에서 B로 성공함

<br>

실제 환경에서 모델을 동작시키기 위해 aloha_script 폴더에서 make_real_env클래스의 동작을 정의해줘야함

imitete_episodes.py를 기준으로 mycobot320 로봇암을 돌리기 위한 model_runs.py를 작성함

데이터 생성을 위해 로봇암의 기준 값을 계산함, 이후 다음 값까지 이동할 프레임을 할당하고 선형보간으로 각 프레임당 위치를 계산함

실제로 script로 로봇암을 동작시키며 동시에 쓰레딩으로 현재 모터의 값과 카메라의 값을 가져옴

그리퍼의 값을 읽을 때 에러가 나는 오류가 발생했기 때문에 이전에 명령 내린 값을 현재 읽는 값으로 가져옴

mycobot320은 비동기적 프로그래밍 방식으로 작동하며, 그리퍼와 구동부가 서로 다른 모터로 작동하나 동시에 명령이 주어질 경우 혼선이 일어나 값이 바껴 신호 전달 시간이 보장되어야 함!

이 때 모델이 매 프레임마다 계산하고 명령을 줄 수 있는 시간이 지연되지 않도록 FPS를 조정해야 함, 여기서는 처음에 50이었지만 10까지 떨어짐

이렇게 A에서 B로 동작 재현이 성공함

<br>

이후 imitation learning의 script 데이터의 특성상 물체가 없이 동작시키고 그 때의 데이터를 생성할 수 있음

따라서 박스만 남기고 다 마스킹하고 학습하니 위치 정보를 판단하기가 어려워 잘 동작하지 못하는 것으로 보임..

이미지에서 1~2%의 부분을 기준으로 상태를 판단해야 함, 따라서 crop을 통해 놓는 위치를 제외하고 모두 마스킹

카메라 하나로도 정상 작동을 확인할 수 있고, 하나의 카메라를 상태를 파악하는 용도로 사용해도 판단하고 정상 작동을 할 수 있음

단, chunk_size가 20은 실패했으며 첫번째 카메라도 상태를 인식하는 초기 시점을 기준으로 잡은 75의 경우 성공했으며 행동이 교차되는 곳에서 지연과 급격한 움직임이 보임

A에서 B1으로, B1에 물건이 있다면 B2로 판단하여 성공함

"Residual Attention Network for Image Classification"논문을 기준으로 전체 이미지를 대상으로 attention으로 처리할 수 있는 지 시도 중

모델 크기가 resnet152X2, resnet50+residualattention92는 180MB가 넘어 전용 GPU 메모리 4GB로는 부족함

따라서 resnet18을 바탕으로 attention module를 넣어 모델 경량화하여 모델 학습 중

<br>

imitation을 실제 환경에서 하고 싶다면!

1. script data로 원하는 기능을 할 수 있어야 함

2. 설정한 프레임마다 모터의 위치 값 등을 알 수 있어야 함

3. 다시 같은 값으로 입력했을 때 정상 동작을 확인해야 함

4. 혹시 모를 모델의 판단 오류로 인한 문제를 예방하기 위해 상한, 하한 값을 통해 동작하게 해야하며, 프레임당 최대 동작 각도 등을 알면 검증하기 편리

5. 모델을 넣고 학습할 수 있는 컴퓨터 성능이 보장되어야 함

6. 학습 후 결과 확인, 프레임 지연 시 조정 등

7. 입력 이미지가 어떻게 전달되는 지 backbone의 feature map을 확인

8. 가상 환경으로 디지털 트윈을 할 수 있다면 모델이 판단하여 이동한 궤적을 생성하여 검증

9. 점점 더 어려운 목표를 시도해보기

<br>

### 8. [진행중] 2024년 mycobot320을 이용한 imitation learning
#### 목표 : mycobot320을 imitation learning으로 동작
[관련코드](https://github.com/sinyeong10/act-plus-plus_one)

#### 진행 과정
2024.7.31 : [aloha코드](https://github.com/MarkFzp/act-plus-plus)를 기반으로 예시를 돌릴 수 있게 환경 설정 및 오타(?) 해결

2024.8.02 : [aloha코드](https://github.com/MarkFzp/act-plus-plus)에서 imitate_episode.py 이해

2024.8.04 : [aloha코드](https://github.com/MarkFzp/act-plus-plus)에서 policy.py 이해

2024.8.13 : sim_move_cube_scripted라는 에피소드 설정

2024.8.14 : sim_move_cube_scripted 에피소드를 위해 xml 파일 설정, 돌려보고 [관련 내용](https://github.com/sinyeong10/act-plus-plus_one/blob/main/scr_readme.md)로 정리

2024.8.16 : sim_move_cube_scripted 에피소드를 one arm으로 설정(모델 입출력 수정)

2024.8.30 : sim_move_cube_scripted 에피소드를 one arm으로 50개의 데이터를 100000번 학습 시켰으나 일정 수준 이상 성공률이 올라가지 않음

\[카메라를 2개로 하면 학습 시간이 카메라 1개에 비해 10배 늘어남, 더 작은 epoch로 성공하는 시점이 발생하나 시간상으로는 더 느림\]

2024.9.07 : mycobot320으로 sim_move_cube_scripted 에피소드 구현

2024.9.12 : mycobot320으로 10개의 데이터를 5000번 학습 후 결과 확인

2024.9.20 : 카메라가 목표를 보고 있는 상황에서 시도, chunk를 1초, 2초 단위의 연산량으로 잡고 처리

\[chunk가 작을 수록 모델 학습 및 추론 시간 감소, ~~해당 DT(1/FPS)만에 갈 수 없는 위치를 계산하는 경우 존재~~\]

\[고정 카메라에 그리퍼가 보이지 않아 물체를 잡아야하는 순간과 잡고 있는 순간을 구분 못하는 듯?\]

2024.9.23 : mycobot320 조작하는 코드도 라이브러리에 추가, 폴더 정리

2024.9.25 : 에피소드를 더 잘게 쪼개서 기능별 시도, 하드웨어 문제 파악 및 해결! 성공!! 만세!

[진행 과정](https://youtu.be/Ph2mDUV5k8M)

2024.9.27 : 상황별로 다른 행동을 하도록 에피소드 구현

2024.9.30 : 확률적으로 동작하거나, 한 동작만 하는 문제 발생

\[chunk가 클수록 현재 상태에서 다음 상태를 더 고려하기에 행동이 부드러워지지만 좀 더 줄여야 할 듯?\]

2024.10.31 : 상황에 따른 판단 분석을 위해 resnet18의 feature map을 살펴봄, imitation의 특성에 따라 간단한 masking하는 코드 생성, chunk size 75로 결정

<img src="https://github.com/user-attachments/assets/2076220c-08f8-42ce-8c72-4b3e363e9cd6" height="300"/>
<img src="https://github.com/user-attachments/assets/50f0c9a6-5eb7-497c-a6f5-660ebb5d3bfd" height="300"/>

masking

<img src="https://github.com/user-attachments/assets/cdd663e9-9594-4edb-9be0-e053b48dee7f" height="300"/>
<img src="https://github.com/user-attachments/assets/b34490df-a604-4403-b183-114c5e3c1e33" height="300"/>

\[imitation의 특성에 따라 배경을 분류해낼 수 있고 카메라의 노이즈는 회전변환행렬을 통해서 매칭 시킬 수 있을 것임\]

\[하지만 mycobot320의 로봇암과 연결된 카메라 선의 위치가 매번 바껴서 배경 제거할 때 고려해야 함\]

\[하지만 주요 박스를 제외하고 모두 마스킹하면 공간적 정보를 인식하지 못해서 제대로 행동하지 못하는 듯!\]

2024.11.14 : 이전 실패한 모델의 feature map 분석, 데이터 전처리를 통해 2가지 행동 중 무엇을 해야할 지 판단하여 동작 성공!!

첫번째 행동으로 동작한 2번째 카메라의 값이 모두 0일 때의 이미지와 feature map

<img src="https://github.com/user-attachments/assets/ae7e85c9-8baf-40cb-a12f-381857840ea4" height="250">
<img src="https://github.com/user-attachments/assets/beeb68b3-7ada-4075-9fe8-61dafef65ff8" height="250">

첫번째 행동으로 동작해야 하지만 두번째 행동을 할 때의 2번째 카메라의 이미지와 feature map

<img src="https://github.com/user-attachments/assets/3e71bdb9-e3fb-4042-a395-53cbd5a8fd06" height="250">
<img src="https://github.com/user-attachments/assets/38f5b733-0437-4d37-aa26-7ad53c70c6eb" height="250">

[하나의 모델로 2가지의 행동을 판단하여 동작](https://youtu.be/vXW5mIuQnWM)

2024.11.19 : [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)논문과 [구현 링크](https://github.com/Necas209/ResidualAttentionNetwork-PyTorch?tab=readme-ov-file)를 참고하여 backbone의 차원을 맞춰 교체 시도, 모델의 경량화 필요성을 확인, Optimizer를 SGD, Adam로 설정하는 것에 따른 성능 차이를 확인

\[resent151x2 혹은 resnet50, residualattentionmodel92는 모델 크기가 180M 넘어가서 gpu 메모리 한계가 걸림.. resnet18 기준으로 새로 모델을 짜서 경량화 시도 중\]

2024.11.21 : [구현 링크](https://github.com/Necas209/ResidualAttentionNetwork-PyTorch?tab=readme-ov-file)를 참고하여 ResidualAttentionModel56U의 성능 테스트, resnet18과 ResidualAttentionModel을 바탕으로 새롭게 작성한 모델(Resent18AttentionModelU) 성능 테스트, CIFAR-10 데이터셋으로 평가

\[ResidualAttentionModel56U : Total params : 7,202,794, Estimated Total Size (MB) : 107.90, Number of model parameters : 11413482, epoch 100 Test Accuracy : 93.86%\]

\[Resent18AttentionModelU : Total params : 459.114, Estimated Total Size (MB) : 7.59, Number of model parameters : 723306, epoch 100 Test Accuracy : 88.01%\]

