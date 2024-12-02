#.hdf5파일을 찾아 지정한 length만큼 갯수를 합쳐서 하나의 이미지로 만듦
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

for number in range(1, 49):
    filename = f"two_cam_episode_{number}" #f"zval{number}"
    import os
    folder_path = f"mycobot320_script\\image_mean\\{filename}"
    os.makedirs(folder_path, exist_ok=True)
    # HDF5 파일 경로 설정
    dataset_path = f"scr\\tonyzhao\\datasets\\next_sim_mycobot_320\\{filename}.hdf5"
    # HDF5 파일
    with h5py.File(dataset_path, 'r') as file:
        # 저장된 카메라 이름들 (여기서는 가정)
        camera_names = ['right_wrist', "top"]
        for length in range(115, 116):#1, 115+1):
            # 크기 지정
            size = (480, 640, 3)

            # 모든 값을 0으로 초기화
            array = np.zeros(size, dtype=np.float64)

            print(length, array.shape)  # 출력: (480, 640, 3)
            for i in range(int(length)):
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
                            array[i][j][k] += images[i][j][k] / float(length)
                            

                # 이미지를 matplotlib를 이용해 출력
                # axes[1].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
                # axes[1].set_title(f'Frame {i} from {camera_names[1]}')

                # print(images)

                # plt.tight_layout()
                # plt.show()
                # plt.close()

            # print(array)
            array = array.astype(np.uint8)

            # 이미지를 matplotlib로 시각화
            plt.imshow(array[:, :, ::-1])
            # plt.title("Mean Image")
            plt.axis("off")  # 축 숨기기
            plt.savefig(f"mycobot320_script\\image_mean\\{filename}\\mean_{length}_one.png", bbox_inches='tight', pad_inches=0)  # 파일 저장
            # plt.show()
            plt.close()
            cv2.imwrite(f"mycobot320_script\\image_mean\\{filename}\\cv_mean_{length}_one.png", array)

            