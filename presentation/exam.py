import numpy as np
import matplotlib.pyplot as plt

# 초기 설정
start_point = np.array([90, 240])  # 시작 좌표
length = 200  # 선의 길이
num_iterations = 8  # 최대 반복 횟수

# 배열을 그릴 Figure와 Axis를 생성합니다.
fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)

ax.scatter(start_point[0], start_point[1], color='red', s=200)  

start_point = np.array([90, 179])

# 첫 번째 선의 각도
current_angle = np.pi / 4 * 3  # 초기 각도는 0 (수평 오른쪽)
angle_increment = np.pi / 4

# 반복하여 선을 그리기
for i in range(num_iterations):
    current_angle -= angle_increment
    # 끝점 계산
    end_point = start_point + length * np.array([np.cos(current_angle), np.sin(current_angle)])
    
    # 선을 그립니다.
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='black')
    
    # 길이 20마다 새로운 랜덤 각도로 다시 선을 그림
    for j in range(1, 11):  # 100/20 = 5, 총 5번 반복
        for _ in range(3):
            # 작은 선의 끝점 계산
            small_end = start_point + (length / 10 * j) * np.array([np.cos(current_angle), np.sin(current_angle)])
            new_angle = np.random.uniform(0, 2 * np.pi)  # 랜덤 각도 생성
            new_end = small_end + length * np.array([np.cos(new_angle), np.sin(new_angle)])
            
            # 작은 선을 그립니다.
            ax.plot([small_end[0], new_end[0]], [small_end[1], new_end[1]], color='gray', linestyle='dashed', alpha = 0.3)
        
    # 다음 선의 시작점을 현재 끝점으로 설정
    start_point = start_point + 124 * np.array([np.cos(current_angle), np.sin(current_angle)])
    # 각도를 변경
    # current_angle = np.random.uniform(0, 2 * np.pi)  # 다음 반복을 위한 새로운 랜덤 각도

# 이미지 출력 설정
ax.set_xlim(0, 480)
ax.set_ylim(0, 480)
ax.set_aspect('equal')
plt.axis('off')  # 축을 숨깁니다.
plt.show()
