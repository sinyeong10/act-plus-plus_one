
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

masking_input = False #True #False #True

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
        
        self.cap0 = cv2.VideoCapture(0)
        
        self.cap1 = cv2.VideoCapture(1)        

        if setup_robots:        
            while True:
                _, frame = self.cap0.read()
                
                cv2.imshow("right_wrist", frame)
                key = cv2.waitKey(10)
                if key == ord("q"):
                    break

            while True:
                _, frame = self.cap1.read()

                if True: #masking_input:
                    x1, y1 = 230, 350
                    x2, y2 = 412, 400
                    zero_image = frame.copy()
                    zero_image[:,:] = 0
                    x1, x2 = max(0, min(x1, x2)), max(x1, x2)
                    y1, y2 = max(0, min(y1, y2)), max(y1, y2)
                    print(x1,y1, "\\", x2, y2)
                    zero_image[y1:y2+1, x1:x2+1] = frame[y1:y2+1, x1:x2+1]
                    cv2.imshow("top", zero_image)
                else:
                    cv2.imshow("top", frame)
                    
                key = cv2.waitKey(10)
                if key == ord("q"):
                    break
        

        self.mycobot = MyCobot('COM11', 115200) #('/dev/ttyACM0',115200)
        start_time = time.time()
        self.mycobot.set_gripper_mode(0)
        print(self.mycobot.get_coords())
        print(self.mycobot.get_angles())
        print(self.mycobot.get_gripper_value())

        self.mycobot.init_eletric_gripper()
        self.mycobot.set_gripper_calibration()

        self._reset_joints()
        self._reset_gripper()

        self.cnt = 0
        self.prev_gripper = 70
        # self.gripper_value = [70]*40+list(np.linspace(70, 25, 10))+[25]*85+list(np.linspace(25,70,10))+[70]*100

        # self.force_actions = [[-3.07, 33.39, 32.78, 6.94, -88.33, -1.58], [-3.158, 36.704, 30.319000000000003, 6.878, -88.356, -1.606],\
        #     [-3.246, 40.018, 27.858, 6.816000000000001, -88.382, -1.6320000000000001], [-3.334, 43.332, 25.397000000000002, 6.7540000000000004, -88.408, -1.6580000000000001],\
        #     [-3.422, 46.646, 22.936, 6.692, -88.434, -1.6840000000000002], [-3.51, 49.96, 20.475, 6.630000000000001, -88.46000000000001, -1.71],\
        #     [-3.598, 53.274, 18.014000000000003, 6.5680000000000005, -88.486, -1.7360000000000002], [-3.686, 56.588, 15.553, 6.506, -88.512, -1.762],\
        #     [-3.774, 59.902, 13.092000000000002, 6.444, -88.538, -1.788], [-3.862, 63.216, 10.631000000000004, 6.382000000000001, -88.56400000000001, -1.814],\
        #     [-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],\
        #     [-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],[-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],\
        #     [-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],[-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],\
        #     [-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],[-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],\
        #     [-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],[-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],\
        #     [-3.95, 66.53, 8.17, 6.32, -88.59, -1.84],[-3.95, 66.53, 8.17, 6.32, -88.59, -1.84]]

        self.backupdata = {"top":[], "right_wrist":[]}

    def save(self, path, idx):
        print("\n\nsave\n\n")
        max_timesteps = len(self.backupdata["right_wrist"])
        import os
        import h5py
        t0 = time.time()
        dataset_dir = path
        dataset_path = os.path.join(dataset_dir, f'mycobot320_model_run_image_{idx}')
        # if task_name == 'sim_move_cube_scripted': #one arm
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in ["top", "right_wrist"]:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3), )
        
            for name, array in self.backupdata.items():
                name = f"/observations/images/"+name
                print(name)
                root[name][...] = array

    def get_qpos(self):
        print(self.cnt, "qpos :", self.mycobot.get_angles(), self.mycobot.get_gripper_value(), self.prev_gripper)#self.gripper_value[self.cnt])
        gripper = self.mycobot.get_gripper_value()
        if self.mycobot.get_gripper_value() == 255:
            # if 41<= self.cnt <= 50 or 135<= self.cnt:
            # gripper = self.gripper_value[self.cnt] #명령순서 정확하게
            gripper = self.prev_gripper #이전 action기준으로 계산된 값
        self.cnt += 1
        return np.concatenate([self.mycobot.get_angles(), [gripper]])

    def get_qvel(self):
        return np.concatenate([[20]*6,[10]*1])

    def get_images(self):
        image_dict = dict()
        _, cur_frame0 = self.cap1.read()
        image_dict["top"] = cur_frame0
        # frame_height, frame_width, channels = cur_frame0.shape
        # zero_frame = np.zeros((frame_height, frame_width, channels), dtype=np.uint8)
        # print(zero_frame.shape, cur_frame0.shape)
        #강제 마스킹
        # cur_frame0 = np.zeros(cur_frame0.shape, dtype=np.uint8)

        if masking_input:
            x1, y1 = 230, 350
            x2, y2 = 412, 400
            zero_image = cur_frame0.copy()
            zero_image[:,:] = 0
            x1, x2 = max(0, min(x1, x2)), max(x1, x2)
            y1, y2 = max(0, min(y1, y2)), max(y1, y2)
            print(x1,y1, "\\", x2, y2)
            zero_image[y1:y2+1, x1:x2+1] = cur_frame0[y1:y2+1, x1:x2+1]
            image_dict["top"] = zero_image

        _, cur_frame1 = self.cap0.read()
        #강제 마스킹
        # cur_frame1 = np.zeros(cur_frame1.shape, dtype=np.uint8)
        image_dict["right_wrist"] = cur_frame1
        print(self.cnt, "cur_frame", np.array(cur_frame0).shape, np.array(cur_frame1).shape)

        # self.backupdata["top"].append(cur_frame0)
        # self.backupdata["right_wrist"].append(cur_frame1)
        return image_dict

    def _reset_joints(self):
        self.mycobot.send_angles([(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], 20) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
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
        print("환경 초기화")
        if not fake:
            self._reset_joints()
            self._reset_gripper()
            self.prev_gripper = 70
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action, base_action=None, get_tracer_vel=False, get_obs=True):
        print("action.shape", action.shape)
        action = list(action)
        print(self.cnt, "action :", action, "이전 값 :", self.prev_gripper)
        #강제로 되는 값을 넣어 시도하면서 오류를 발견! 각도 주고 바로 그리퍼 값 주면, 각도 값에 혼선이 오는 듯!
        # if self.cnt < len(self.force_actions):
        #     action[:-1] = self.force_actions[self.cnt]
        #     print(self.cnt, "force action : ", action)
        state_len = 6
        # print(action[:state_len], "이 들어감!!\n\n")
        self.mycobot.send_angles(action[:state_len], 20)
        # time.sleep(0.04)
        # print("first :", action[-1], end=" // ")
        diff = int(action[-1])-self.prev_gripper
        self.prev_gripper = self.prev_gripper+min(5, abs(diff)) if  diff > 0 else self.prev_gripper-min(5, abs(diff))
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        #뒤에서 처리하는 경우
        if self.prev_gripper > 100:
            print("\n\n\n그리퍼 100이상의 값이 계산됨..")
            self.prev_gripper = 100
        if self.prev_gripper < 0:
            print("\n\n\n그리퍼 0이하의 값이 계산됨..")
            self.prev_gripper = 0
        self.mycobot.set_gripper_value(self.prev_gripper, 20, 1)
        # print("back :", action[-1])
        time.sleep(0.02)
            
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

def make_real_env(init_node, setup_robots=True, setup_base=False):
    env = RealEnv(init_node, setup_robots, setup_base)
    return env