#모터 값 넣기
#수정
base = [[(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], \
        [(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)], [(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)],\
        [(-4.65), 56.6, 16.17, -7.47, (-91.4), (-3.95)], [(-2.54),29.44,23.81,17.92,(-88.33),(-84.81)],\
        # [24.08, 47.89, 0.52, 21.89, -89.03, -50.18],\
        [34.08, 47.89, 5.52, 0.61, -65.03, -49.83], [30.08, 55.86, 13.72, 3.61, -65.03, -57.83],\
        [30.08, 55.86, 13.72, 3.61, -65.03, -57.83]]

#초기
# base = [[(-3.07), 33.39, 32.78, 6.94, (-88.33), (-1.58)], \
#         [(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)], [(-3.95), 66.53, 8.17, 6.32, (-88.59), (-1.84)],\
#             [(-4.65), 56.6, 16.17, -7.47, (-91.4), (-3.95)], [(-2.54),29.44,23.81,17.92,(-88.33),(-84.81)],\
#                 [24.16,23.55,37.96,10.63,(-91.58),(-84.81)], [23.81,85.42,(-50.71),44.64,(-89.38),(-71.01)], [23.81,85.42,(-50.71),44.64,(-89.38),(-71.01)]]

print(len(base))
#10FPS기준
timestep = [1, 1,  1.5, 3,  2.5, 1.5, 1]
framestep = [10, 10, 15, 30, 25, 15, 10]
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
# print(frame)
# print(frame[0].shape)
for cur_base, cur_time in zip(base[1:], framestep):
    cur_value = cal(pre_base, cur_base, cur_time+1)
    # print(cur_value)
    frame.append(cur_value)
    pre_base = cur_base
angles_data = list(np.concatenate(frame, axis=0))

#그리퍼 값 넣기
gripper_data = [70]*10+list(np.linspace(70, 20, 10))+[20]*85+list(np.linspace(20,70,10))
print(len(gripper_data), len(angles_data)) #그리퍼가 한개 작음, qpos에서 끝난 값이 다음 값과 연동되어 겹치는 값을 삭제하고 처음에 값을 추가시켰기 때문

cnt = 0
filename = r"mycobot320_script\frame_base_action.txt"
with open(filename, 'w') as f:
    for a, b in zip(gripper_data, angles_data):
        print(cnt, a,b)
        # 각 행의 데이터를 문자열로 변환하고 저장
        f.write(str(cnt)+" : "+str(a)+" : "+str(list(b))+'\n')
        cnt += 1
