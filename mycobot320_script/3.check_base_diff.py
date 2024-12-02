#초기 값 기준 1번행동은 거의 오차가 없음, 2번 행동은 많음
#주로 카메라 선, 로봇암에서 오차가 발생

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 저장된 PNG 파일 읽기
# NumPy 배열로 변환

for i in range(1, 49):
    # HDF5 파일 경로
    file_path1 = r"mycobot320_script\image_mean\two_cam_episode_25\cv_mean_1_one.png"
    file_path2 = f"mycobot320_script\\image_mean\\two_cam_episode_{i}\\cv_mean_1_one.png"

    image1 = Image.open(file_path1)
    image2 = Image.open(file_path2)

    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # 결과 확인
    print(f"Image shape: {image1_array.shape}, {image2_array.shape}")  # 이미지 크기 출력
    print(image1_array[200][200], image2_array[200][200])

    # NumPy 배열로 변환 후 차이 계산 (절댓값)
    diff = np.abs(image1_array[:,:,:3].astype(np.int16) - image2_array[:,:,:3].astype(np.int16))  # overflow 방지

    print(diff.shape, diff[200][200])

    # 좌표 정의
    x1, y1 = 345, 350  # 시작 좌표
    x2, y2 = 395, 395  # 끝 좌표
    color = [255, 255, 255]  # 흰색 (RGB 형식)

    # 상단 가로선
    diff[y1, x1:x2] = color
    # 하단 가로선
    diff[y2, x1:x2] = color
    # 왼쪽 세로선
    diff[y1:y2, x1] = color
    # 오른쪽 세로선
    diff[y1:y2, x2] = color

    # 차이 이미지를 uint8로 변환
    # diff = (diff * 10)
    print(diff.min(), diff.max(), diff[200][200])

    # 차이 이미지 출력
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title(f'{i}_Absolute Difference Between Images')
    plt.imshow(diff)  # HDF5의 데이터 포맷에 맞게 이미 RGB 또는 그레이스케일로 처리됨
    plt.show()
    plt.close()


    # RGB를 그레이스케일로 변환 (가중합 방식)
    diff_gray = np.dot(diff[..., :3], [1/3, 1/3, 1/3])#[0.2989, 0.587, 0.114])  # 가중치: R, G, B
    # diff_gray *= 10

    # 결과 확인
    print(f"{i}_Gray difference shape: {diff_gray.shape}")
    print(diff_gray.min(), diff_gray.max(), diff_gray[200, 200])

    # # 차이 이미지 출력
    # plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.title(f'Absolute Difference (Grayscale)')
    # plt.imshow(diff_gray, cmap='gray')  # 그레이스케일로 출력
    # plt.colorbar()  # 색상 막대 추가
    # plt.show()
    # plt.close()

    # episode_len = 105
    # diff_gray = diff_gray / episode_len * 50
    # print(diff_gray.min(), diff_gray.max(), diff_gray[200, 200])
    # # 차이 이미지 출력
    # plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.title('105frame Absolute Difference (Grayscale)')
    # plt.imshow(diff_gray, cmap='gray', vmin=0, vmax=255)  # 그레이스케일로 출력
    # plt.colorbar()  # 색상 막대 추가
    # plt.show()
    # plt.close()

    # filtered_diff = np.where(diff_gray >= 40, diff_gray, 0)*2
    # print(filtered_diff.min(), filtered_diff.max(), filtered_diff[200, 200])

    # # 차이 이미지 출력
    # plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.title('filter frame Absolute Difference (Grayscale)')
    # plt.imshow(filtered_diff, cmap='gray', vmin=0, vmax=255)  # 그레이스케일로 출력
    # plt.colorbar()  # 색상 막대 추가
    # plt.show()
    # plt.close()