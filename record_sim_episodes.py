import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, PickAndMovePolicy

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy

    elif task_name == 'sim_move_cube_scripted':
        policy_cls = PickAndMovePolicy
    else:
        raise NotImplementedError

    success = []
    episode_idx = 0
    cnt = 0
    # for episode_idx in range(num_episodes):
    while episode_idx < num_episodes:
        cnt += 1
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise) #False면 노이즈 없음
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            # print("ts.observation", ts.observation)
            action = policy(ts)
            # print("action", action) #여기 가려함 #[0.21718881 0.49999888 0.29525084 1 0. 0. 0. 0.]형태

            ts = env.step(action) #환경에서 적용해 봄
            # print("72line : ts.observation", ts.observation) #(7,7,7,image,7,2)
                # self._task.before_step(action, self._physics)
                # self._physics.step(self._n_sub_steps)
                # self._task.after_step(self._physics)
            # # 속성에 접근
            # print(ts.step_type)   # dm_env.StepType.FIRST
            # print(ts.reward)      # None
            # print(ts.discount)    # None    
            # print(ts.observation.keys()) # observation 값
            # odict_keys(['qpos', 'qvel', 'env_state', 'images', 'mocap_pose_left', 'mocap_pose_right', 'gripper_ctrl'])

            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")
            # clear unused variables
            del env
            del episode
            del policy
            continue

        joint_traj = [ts.observation['qpos'] for ts in episode] # 401 14
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode] #401 4 #팔하나 날리니 401 2 됨
        # print("joint_traj",joint_traj[0], type(joint_traj), type(joint_traj[0]))
        # print("::", len(joint_traj),joint_traj[0].size)
        # print("gripper_ctrl_traj", len(gripper_ctrl_traj), gripper_ctrl_traj[0].size, "::",gripper_ctrl_traj[0])
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            # left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            # right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            # print("joint, ctrl", joint, ctrl)
            right_crtl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            # print("right_crtl",right_crtl)
            # joint[6] = left_ctrl
            # joint[6+7] = right_ctrl
            joint[6] = right_crtl #엄밀한 값이 아니라서 함수 처리해서 하는 듯?
            # print("joint", joint)

        # print("subtask_info", episode[0].observation['env_state'])
        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0
        print("subtask_info", subtask_info)
        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        # print("subtask_info",subtask_info)
        # #모든 박스의 정보가 옴
        BOX_POSE[0] = subtask_info[:7] # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            # print(joint_traj[t], joint_traj[t][:7])
            action = joint_traj[t][:7] #앞만 가져옴
            ts = env.step(action)
            # print("140line : ts.observation", ts.observation) #(14,14,6) #one (7,7,6)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        if task_name == 'sim_move_cube_scripted': #one arm
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                #왜 qpos, qvel이 14지?
                qpos = obs.create_dataset('qpos', (max_timesteps, 7))
                qvel = obs.create_dataset('qvel', (max_timesteps, 7))
                action = root.create_dataset('action', (max_timesteps, 7))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
            episode_idx += 1
        else:
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
            episode_idx += 1

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')
    print(f"cnt : {cnt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

