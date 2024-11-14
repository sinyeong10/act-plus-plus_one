import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

# HDF5 파일 경로 설정
episode_name = "two_cam_episode_19"
dataset_path = f"scr\\tonyzhao\\datasets\\next_sim_mycobot_320_action_mask\\{episode_name}.hdf5"

remake_path = f"scr\\mask_data\\{episode_name}_image_mask"

# HDF5 파일
with h5py.File(dataset_path, 'r') as file:
    # 저장된 카메라 이름들 (여기서는 가정)
    camera_names = ['right_wrist', "top"]
    
    for i in range(115):
        fig, axes = plt.subplots(2, len(camera_names), figsize=(30, 30))  # 카메라 수에 따라 그림판을 생성
        filename = [f"right_wrist_frame_{i}.jpg", f"top_frame_{i}.jpg"]

        mng = plt.get_current_fig_manager()
        try:
            mng.full_screen_toggle()  # Matplotlib 3.3 이상에서 지원
        except AttributeError:
            # Matplotlib 3.2 이하 버전용
            mng.window.state('zoomed')  # Windows에서 최대화 (MacOS/Linux는 다른 방법이 필요할 수 있음)

        for idx, cam_name in enumerate(camera_names):
            print(f"Loading images from {dataset_path} {cam_name}")
            images = file[f'observations/images/{cam_name}'][i]  # 이미지를 읽음

            # 이미지를 matplotlib를 이용해 출력
            axes[0, idx].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
            axes[0, idx].set_title(f'Frame {i} from {cam_name}')
        
        for idx, file_name in enumerate(filename):
            # print(f"Loading images from {file_name}")
            print(f"{remake_path}\\{file_name}")
            images = cv2.imread(f"{remake_path}\\{file_name}")

            axes[1, idx].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
            axes[1, idx].set_title(f'Frame {i} from {file_name}')

        plt.tight_layout()
        plt.show()
