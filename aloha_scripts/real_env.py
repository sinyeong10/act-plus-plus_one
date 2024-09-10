
import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
from pyquaternion import Quaternion


# 컴 세팅
from pymycobot.mycobot import MyCobot 
import time
import cv2
import numpy as np

import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, setup_robots=True, setup_base=False):
        
        self.cap = cv2.VideoCapture(1)
        
        self.mycobot = MyCobot('COM7', 115200) #('/dev/ttyACM0',115200)
        start_time = time.time()
        self.mycobot.set_gripper_mode(0)
        print(self.mycobot.get_coords())
        print(self.mycobot.get_angles())
        print(self.mycobot.get_gripper_value())

        self.mycobot.init_eletric_gripper()
        self.mycobot.set_gripper_calibration()

        self._reset_joints()
        self._reset_gripper()
        

    def get_qpos(self):
        print(self.mycobot.get_angles(), self.mycobot.get_gripper_value())
        return np.concatenate([self.mycobot.get_angles(), [self.mycobot.get_gripper_value()]])

    def get_qvel(self):
        return np.concatenate([[20]*6,[10]*1])

    def get_images(self):
        _, cur_frame = self.cap.read()
        print(np.array(cur_frame).shape)
        return np.array(cur_frame)

    def _reset_joints(self):
        self.mycobot.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
        time.sleep(5)

    def _reset_gripper(self):
        self.mycobot.set_gripper_value(70, 20, 1)
        time.sleep(1)

    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action, base_action=None, get_tracer_vel=False, get_obs=True):
        print("action.shape", action.shape)
        action = list(action)
        print("action", action)
        state_len = 6
        self.mycobot.send_angles(action[:state_len], 20)
        self.mycobot.set_gripper_value(int(action[-1]), 20, 1)
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

def make_real_env(init_node, setup_robots=True, setup_base=False):
    env = RealEnv(init_node, setup_robots, setup_base)
    return env