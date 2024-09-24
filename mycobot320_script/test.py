# 컴 세팅
from pymycobot.mycobot import MyCobot 
import time

mc = MyCobot('COM11', 115200)
mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
time.sleep(3)
mc.init_eletric_gripper()
mc.set_gripper_calibration()
for _ in range(3):
    # mc.set_gripper_mode(0)
    print(mc.get_gripper_value())
    time.sleep(0.3)
    #벌리고 초기 이동)
    # mc.set_eletric_gripper(0)
    mc.set_gripper_value(20, 20, 1)
    # print(mc.get_gripper_value())
    time.sleep(1)
    print('close')
    print(mc.get_gripper_value())
    time.sleep(0.3)
    # mc.set_eletric_gripper(1)
    mc.set_gripper_value(100, 20, 1)
    # print(mc.get_gripper_value())
    time.sleep(1)
    print(mc.get_gripper_value())
    time.sleep(0.3)
    print("end")
