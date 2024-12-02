#f"mycobot320_script\\image_mean\\{folder_name}\\cv_mean_1_one.png"을 합침

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# 크기 지정
size = (480, 640, 3)

# 모든 값을 0으로 초기화
array = np.zeros(size, dtype=np.float64)

length = 25
for number in range(1, length+1): #(length, 49):
    folder_name = f"two_cam_episode_{number}"

    # HDF5 파일 경로
    file_path = f"mycobot320_script\\image_mean\\{folder_name}\\cv_mean_1_one.png"

    images = Image.open(file_path)

    images = np.array(images)

    # 결과 확인
    print(f"Image shape: {images.shape}")  # 이미지 크기 출력
    print(images[200][200])

    # fig, axes = plt.subplots(1, len(camera_names), figsize=(12, 6))  # 카메라 수에 따라 그림판을 생성

    # print(f"Loading images from {camera_names[0]}")
    # images = file[f'observations/images/{camera_names[0]}'][i]  # 이미지를 읽음

    # 이미지를 matplotlib를 이용해 출력
    # axes[0].imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
    # axes[0].set_title(f'Frame {i} from {camera_names[0]}')

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
plt.imshow(array)
array = array[:, :, ::-1]
plt.title("Mean Image")
plt.axis("off")  # 축 숨기기
plt.savefig(f"mycobot320_script\\image_mean\\mean_{length}_one.png", bbox_inches='tight', pad_inches=0)  # 파일 저장
# plt.show()
plt.close()
cv2.imwrite(f"mycobot320_script\\image_mean\\cv_mean_{length}_one.png", array)

            