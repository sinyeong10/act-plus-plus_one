import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

# HDF5 파일 경로 설정
dataset_path = r"scr\tonyzhao\datasets\next_sim_mycobot_320\zval1.hdf5"

# HDF5 파일
with h5py.File(dataset_path, 'r') as file:
    # 저장된 카메라 이름들 (여기서는 가정)
    camera_names = ['right_wrist', "top"]
    for i in range(115):
        fig, axes = plt.subplots(1, len(camera_names), figsize=(12, 6))  # 카메라 수에 따라 그림판을 생성

        for idx, cam_name in enumerate(camera_names):
            print(f"Loading images from {cam_name}")
            images = file[f'observations/images/{cam_name}'][i]  # 이미지를 읽음
            print(images.shape)

            # 이미지를 matplotlib를 이용해 출력
            axes[idx].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f'Frame {i} from {cam_name}')

        plt.tight_layout()
        plt.show()
