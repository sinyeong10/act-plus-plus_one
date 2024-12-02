import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 크기 지정
size = (480, 640, 3)

# 리스트 생성 및 초기화
array = [[[{ } for _ in range(size[2])] for _ in range(size[1])] for _ in range(size[0])]

# 확인
print(len(array), len(array[0]), len(array[0][0]))  # 출력: 480 640 3
print(array[0][0][0])  # 출력: {}

# HDF5 파일 경로 설정
dataset_path = r"scr\tonyzhao\datasets\next_sim_mycobot_320\zval2.hdf5"
# HDF5 파일
with h5py.File(dataset_path, 'r') as file:
    # 저장된 카메라 이름들 (여기서는 가정)
    camera_names = ['right_wrist', "top"]
    
    for i in range(105):
        # fig, axes = plt.subplots(1, len(camera_names), figsize=(12, 6))  # 카메라 수에 따라 그림판을 생성

        # print(f"Loading images from {camera_names[0]}")
        # images = file[f'observations/images/{camera_names[0]}'][i]  # 이미지를 읽음

        # 이미지를 matplotlib를 이용해 출력
        # axes[0].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
        # axes[0].set_title(f'Frame {i} from {camera_names[0]}')

        print(f"Loading images from {camera_names[1]}")
        images = file[f'observations/images/{camera_names[1]}'][i]  # 이미지를 읽음

        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                for k in range(images.shape[2]):
                    # print(images[i][j][k])
                    if images[i][j][k] in array[i][j][k]:
                        array[i][j][k][images[i][j][k]] += 1
                    else:
                        array[i][j][k][images[i][j][k]] = 1

        # 이미지를 matplotlib를 이용해 출력
        # axes[1].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
        # axes[1].set_title(f'Frame {i} from {camera_names[1]}')

        # print(images)

        # plt.tight_layout()
        # plt.show()
        # plt.close()

print(array)

import json
# 저장할 파일 경로
output_file_path = "array_output.txt"

# 데이터 저장
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(str(array))  # array를 문자열로 변환하여 한 줄로 저장

print(f"Data has been saved to {output_file_path}")
