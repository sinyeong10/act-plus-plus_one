import h5py
import numpy as np
import cv2
import os
#scr\tonyzhao\datasets\next_sim_mycobot_320\zval1.hdf5
for k in range(1, 2+1):
    filename = f"zval{k}" #f"two_cam_episode_{k}" #1~48
    # HDF5 파일 경로 설정
    dataset_path = f"scr\\tonyzhao\\datasets\\next_sim_mycobot_320\\{filename}.hdf5"

    # 이미지 저장 경로 설정
    output_folder = "scr\\mask_data\\"+filename+"_image"
    os.makedirs(output_folder, exist_ok=True)

    # HDF5 파일
    with h5py.File(dataset_path, 'r') as file:
        # 저장된 카메라 이름들 (여기서는 가정)
        camera_names = ['right_wrist', "top"]
        
        # 각 이미지 프레임을 순회하며 JPG로 저장
        for i in range(115):
            for cam_name in camera_names:
                # 이미지를 읽어옴
                images = file[f'observations/images/{cam_name}'][i]

                # 이미지를 그대로 JPG로 저장 (RGB 형식 유지)
                output_path = os.path.join(output_folder, f'{cam_name}_frame_{i}.jpg')
                cv2.imwrite(output_path, images)
                print(f"Saved {output_path}")

    print("All images have been saved as JPG files.")
