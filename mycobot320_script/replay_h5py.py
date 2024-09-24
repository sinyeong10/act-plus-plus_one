import os
import h5py
import numpy as np
import time
dataset_path = r"C:/Users/cbrnt/OneDrive/문서/mycobot320/episode_0.hdf5"
if not os.path.isfile(dataset_path):
    print(f'Dataset does not exist at \n{dataset_path}\n')
    exit()

with h5py.File(dataset_path, 'r') as root:
    actions = root['/action'][()]

FPS = 15
time0 = time.time()
DT = 1/FPS
for action in zip(actions):
    time1 = time.time()


import cv2
from pymycobot.mycobot import MyCobot 
mc = MyCobot('COM7', 115200)
mc.set_gripper_mode(0)

print(mc.get_coords())
print(mc.get_angles())

 
print(1)

#벌리고 초기 이동
mc.init_eletric_gripper()
mc.set_gripper_calibration()
# mc.set_eletric_gripper(0)
mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
start_time = time.time()
cap = cv2.VideoCapture(1)
end_time = time.time()
# time.sleep(5-(start_time-end_time))
mc.set_gripper_value(70, 20, 1)
time.sleep(1)