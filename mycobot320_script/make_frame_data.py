# 컴 세팅
from pymycobot.mycobot import MyCobot 
import time
import cv2
import numpy as np
import sys

episode_idx = 0

mc = MyCobot('COM11', 115200) #('/dev/ttyACM0', 115200)
mc.set_gripper_mode(0)

start_time = time.time()
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)#("/dev/bus/usb/001/005")
# cap2 = cv2.VideoCapture(2)

# 카메라 장치가 제대로 열렸는지 확인
if not cap0.isOpened():
    print("cap0 카메라를 열 수 없습니다.")
    exit()
if not cap1.isOpened():
    print("cap1 카메라를 열 수 없습니다.")
    exit()

    
while True:
    _, frame = cap0.read()
    
    cv2.imshow("cap0", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

while True:
    _, frame = cap1.read()
    
    cv2.imshow("cap1", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cv2.destroyAllWindows()

end_time = time.time()



base = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], \
        [(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)], [(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)],\
            [(-4.65), 56.6, 16.17, -7.47, (-91.4), (-3.95)], [(-2.54),29.44,23.81,17.92,(-88.33),(-84.81)],\
                [24.16,23.55,37.96,10.63,(-91.58),(-84.81)], [23.81,85.42,(-50.71),44.64,(-89.38),(-71.01)], [23.81,85.42,(-50.71),44.64,(-89.38),(-71.01)]]
print(len(base))
# timestep = [3.1, 1.1, 1,  1.5, 3,  1.5, 3,  1]
timestep = [3.1, 0.9, 1,  1.5, 3,  1.5, 2.5, 1]
framestep = [31, 9, 10, 15, 30, 15, 25, 10]
# framestep=[45,   10,  10, 21,  45, 21,  45, 10]
print(sum(timestep))
print(sum(framestep))

import numpy as np
def cal(a,b,framestep, FPS = 10):
    a = np.array(a)
    b = np.array(b)

    step_time = 1 / FPS
    # 나눌 갯수
    # num_steps = int(total_time / step_time)

    # 각도 간격을 나누는 함수 (등간격)
    angles_over_time = np.linspace(a, b, framestep)
    return angles_over_time[1:]
    # print(len(angles_over_time), angles_over_time)

pre_base = np.array(base[0])
frame = [pre_base[np.newaxis, :]]
# print(frame[0].shape)
for cur_base, cur_time in zip(base[1:], framestep):
    cur_value = cal(pre_base, cur_base, cur_time+1)
    # print(cur_value.shape)
    frame.append(cur_value)
    pre_base = cur_base
angles_data = list(np.concatenate(frame, axis=0))

gripper_data = [70]*40+list(np.linspace(70, 25, 10))+[25]*85+list(np.linspace(25,70,10))
print(len(gripper_data), len(angles_data))

cnt = 0
for a, b in zip(gripper_data, angles_data):
    # print(cnt, a,b)
    cnt += 1

print(angles_data[-2:])
print(gripper_data[-2:])

while episode_idx <= 0:
    print(episode_idx)
    # a = time.time()
    print(mc.get_coords())
    # b = time.time()
    print(mc.get_angles())
    # c = time.time()
    print(mc.get_gripper_value())
    # d = time.time()

    # print(a,b,c,d)
    # print(f"전체 완 : {d-a}") #0.05초
    # print(b-a, c-b, d-c) #0.02초 0.03초 0.01초

    print(1)

    #벌리고 초기 이동
    mc.init_eletric_gripper()
    mc.set_gripper_calibration()
    # mc.set_eletric_gripper(0)
    # mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
    mc.send_angles([(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], 20)
    time.sleep(2)
    # time.sleep(5-(start_time-end_time))
    mc.set_gripper_value(70, 20, 1)
    time.sleep(1)

    cnt = 0
    qpos = []
    # action = []
    frame0 = []
    frame1 = []
    gripper_state = []
    # try:
    for elem, gripper_value in zip(angles_data[:-1], gripper_data):
        print(cnt)
        start_time = time.time()
        if 41<= cnt <= 50 or 135<= cnt:
            print(cnt, "그리퍼 작동중 :", gripper_value)
            mc.set_gripper_value(int(gripper_value), 20, 1)
        else:
            mc.send_angles(elem, 20)
        time.sleep(2)
        print(elem, gripper_value)
        _, cur_frame0=cap0.read()
        frame0.append(cur_frame0)
        _, cur_frame1=cap1.read()
        frame1.append(cur_frame1)
        # print(frame.shape)
        # 현재 모터 각도 출력
        angles = mc.get_angles()
        # coords = mc.get_coords()
        gripper = mc.get_gripper_value()
        # print(f"{cnt} : 현재 모터 각도, 좌표: {angles}, {coords}")
        qpos.append(angles)
        # action.append(elem)
        gripper_state.append(gripper)
        # 0.1초간 대기
        cnt += 1
        end_time = time.time()
        sleep_time = 1.0/10-(end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print("\n\n", cnt, sleep_time)


    print("order_end")
    mc.set_gripper_value(100, 20, 1)
    time.sleep(2)


    with open(f'mycobot320_script/qpos_{episode_idx}.txt', 'w') as file:
        for line in qpos:
            file.write(str(line) + '\n')
    # with open(f'mycobot320_script/action_{episode_idx}.txt', 'w') as file:
    #     for line in action:
    #         file.write(str(line) + '\n')
    with open(f'mycobot320_script/frame0_{episode_idx}.txt', 'w') as file:
        for line in frame0:
            file.write(str(line) + '\n')
    with open(f'mycobot320_script/gripper_{episode_idx}.txt', 'w') as file:
        for line in gripper_state:
            file.write(str(line) + '\n')
    print(np.stack(frame0, axis=0).shape)
    print(np.stack(frame1, axis=0).shape)
    print(np.array(qpos).shape)
    # print(np.array(action).shape)
    print(np.array(gripper_state).shape)

    break

    data_dict = {
        '/observations/qpos': [],
        # '/observations/qvel': [],
        '/action': [],
    }

    camera_names = ['right_wrist', 'top']
    num_episodes = 50
    episode_len = 200

    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # frame = frame[:-1]
    # qpos = qpos[:-1] #가는 중
    # action = action[:-1] #여기로 가라
    # gripper_state = gripper_state[:-1]

    max_timesteps = len(qpos)
    for elem, gripper_value in zip(angles_data, gripper_data):
        tmp = list(elem)
        tmp.append(gripper_value)
        cur_action = tmp
        # print(cur_action)
        cur_qpos = qpos.pop(0)
        add_gripper = gripper_state.pop(0)
        if 0 <= add_gripper <= 100:
            cur_qpos.append(add_gripper)
        else:
            print(f"\n\n정상데이터가 아님 {add_gripper}\n\n")
            cur_qpos.append(gripper_value)
        data_dict['/observations/qpos'].append(cur_qpos)
        # data_dict['/observations/qvel'].append(ts.observation['qvel']) #속도 값이 없음
        data_dict['/action'].append(cur_action)

        # for cam_name in camera_names:
        cur_frame0 = frame0.pop(0)
        data_dict[f'/observations/images/right_wrist'].append(cur_frame0)
        cur_frame1 = frame1.pop(0)
        data_dict[f'/observations/images/top'].append(cur_frame1)

    import os
    import h5py
    t0 = time.time()
    dataset_dir = r"C:/Users/cbrnt/OneDrive/문서/mycobot320/twocamtmp"
    dataset_path = os.path.join(dataset_dir, f'two_cam_episode_{episode_idx}')
    # if task_name == 'sim_move_cube_scripted': #one arm
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        #그리퍼 정보가 마지막에 들어감! 근데 그리퍼 정보를 읽을 방법이 없음
        qpos = obs.create_dataset('qpos', (max_timesteps, 7))
        # qvel = obs.create_dataset('qvel', (max_timesteps, 6))
        action = root.create_dataset('action', (max_timesteps, 7))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')
    episode_idx += 1
    print(f'Saved to {dataset_dir}')
    print(f"cnt : {cnt}")

    print("end")

    mc.set_gripper_value(100, 20, 1)
    time.sleep(2)


