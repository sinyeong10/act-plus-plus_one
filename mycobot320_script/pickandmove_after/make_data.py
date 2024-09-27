# 컴 세팅
from pymycobot.mycobot import MyCobot 
import time
import cv2
import numpy as np

episode_idx = 0

def end_gripper(mc):
    mc.set_gripper_value(100, 20, 1)
    time.sleep(1)
    exit()

def print_motor_angles(mc, FPS=10):
    global episode_idx
    cnt = 0
    qpos = []
    action = []
    frame = []
    gripper_state = []
    # try:
    while cnt <= 160:
        start_time = time.time()
        _, cur_frame=cap.read()
        frame.append(cur_frame)
        # print(frame.shape)
        # 현재 모터 각도 출력
        angles = mc.get_angles()
        coords = mc.get_coords()
        gripper = mc.get_gripper_value()
        # print(f"{cnt} : 현재 모터 각도, 좌표: {angles}, {coords}")
        qpos.append(angles)
        action.append(coords)
        gripper_state.append(gripper)
        # 1초간 대기
        end_time = time.time()
        sleep_time = 1.0/FPS-(end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print("\n\n", cnt, sleep_time)
        cnt += 1

    with open(f'qpos_{episode_idx}.txt', 'w') as file:
        for line in qpos:
            file.write(str(line) + '\n')
    with open(f'action_{episode_idx}.txt', 'w') as file:
        for line in action:
            file.write(str(line) + '\n')
    with open(f'frame_{episode_idx}.txt', 'w') as file:
        for line in frame:
            file.write(str(line) + '\n')
    with open(f'gripper_{episode_idx}.txt', 'w') as file:
        for line in gripper_state:
            file.write(str(line) + '\n')
    print(np.stack(frame, axis=0).shape)
    print(np.array(qpos).shape)
    print(np.array(action).shape)
    print(np.array(gripper_state).shape)

    
    data_dict = {
        '/observations/qpos': [],
        # '/observations/qvel': [],
        '/action': [],
    }

    camera_names = ['right_wrist']
    num_episodes = 50
    episode_len = 200
    
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    
    frame = frame[:-1]
    qpos = qpos[:-1] #가는 중
    action = action[:-1] #여기로 가라
    gripper_state = gripper_state[:-1]

    max_timesteps = len(qpos)
    print(max_timesteps)
    while qpos:
        cur_action = action.pop(0)
        cur_qpos = qpos.pop(0)
        cur_frame = frame.pop(0)
        add_gripper = gripper_state.pop(0)
        # print(cur_qpos, add_gripper)
        cur_qpos = cur_qpos+[add_gripper]
        print(cur_qpos)
        data_dict['/observations/qpos'].append(cur_qpos)
        # data_dict['/observations/qvel'].append(ts.observation['qvel']) #속도 값이 없음
        data_dict['/action'].append(cur_action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(cur_frame)

    import os
    import h5py
    t0 = time.time()
    dataset_dir = r"C:/Users/cbrnt/OneDrive/문서/mycobot320"
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
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
        action = root.create_dataset('action', (max_timesteps, 6))

        for name, array in data_dict.items():
            # print(name, array[0])
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')
    episode_idx += 1
    print(f'Saved to {dataset_dir}')
    print(f"cnt : {cnt}")

mc = MyCobot('COM11', 115200)
mc.set_gripper_mode(0)

print(mc.get_coords())
print(mc.get_angles())
print(mc.get_gripper_value())

print(1)

try:
    #벌리고 초기 이동
    mc.init_eletric_gripper()
    mc.set_gripper_calibration()
    # mc.set_eletric_gripper(0)
    # mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
    mc.send_angles([(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], 20)
    start_time = time.time()
    cap = cv2.VideoCapture(1)
    end_time = time.time()
    time.sleep(5-(start_time-end_time))
    mc.set_gripper_value(70, 20, 1)
    time.sleep(1)

    # import threading
    # thread = threading.Thread(target=print_motor_angles, args=(mc,))
    # thread.start()

    print(2)

    #피킹 위치로 이동 후 집기
    mc.send_angles([(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], 20)
    time.sleep(3.1)
    mc.send_angles([(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)], 20) #
    time.sleep(0.9)

    # mc.set_eletric_gripper(1)
    mc.set_gripper_value(20, 20, 1)
    time.sleep(1)

    print(3)

    #중간 위치로 이동
    mc.send_angles([(-4.65), 56.6, 16.17, -7.47, (-91.4), (-3.95)], 20)
    time.sleep(1.5)

    
    print(4)
    #물체 놓을 위치로 이동
    mc.send_angles([(-2.54),29.44,23.81,17.92,(-88.33),(-84.81)],20)
    time.sleep(3)

    print(5)

    #여기서부터 다른 행동을 하게 함
    # mc.send_angles([24.16,23.55,37.96,10.63,(-91.58),(-84.81)],20)
    # time.sleep(1.5)

    # mc.send_angles([24.08, 47.89, 0.52, 21.89, -89.03, -50.18], 20)
    # time.sleep(2)



    mc.send_angles([34.08, 47.89, 5.52, 0.61, -65.03, -49.83], 20)
    time.sleep(2.5)

    mc.send_angles([30.08, 55.86, 13.72, 3.61, -65.03, -57.83], 20)
    time.sleep(1.5)

    print(6)

    exit()
    #초기 설정값
    # mc.send_angles([23.81,85.42,(-50.71),44.64,(-89.38),(-71.01)],20)
    # time.sleep(3)
    # print("a")

    #수정함 #여기서 조금만 더 위로
    # mc.send_angles([24.08, 61.52, (-15.73), 35.5, (-89.12), (-71.19)], 20)
    # time.sleep(2)

    
    #물체 놓기
    # mc.set_eletric_gripper(0)
    mc.set_gripper_value(80, 20, 1)
    time.sleep(1)


    #초기 위치로
    # mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
    # time.sleep(10)
    mc.set_gripper_value(100, 20, 1)
    time.sleep(1)

    print("end")
except:
    end_gripper(mc)