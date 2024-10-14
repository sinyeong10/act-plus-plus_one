import numpy as np
import matplotlib.pyplot as plt

# 480x480x3 크기의 배열을 만들고 모든 값을 255로 초기화합니다. (흰색 배경)
image = np.full((480, 480, 3), 255, dtype=np.uint8)

# 배열을 그릴 Figure와 Axis를 생성합니다.
fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)

# 원의 중심 좌표와 반지름
center = (240, 240)
radius = 150

# 원을 그립니다.
circle = plt.Circle(center, radius, color='black', fill=False, linestyle='dashed')
ax.add_artist(circle)

ax.scatter(90.0, 240.0, color='red', s=200)  

# 접선의 alpha 값을 설정합니다.
alpha_values = [1, 0.5, 0.3, 0.2, 0.1]

# 여러 개의 접선을 그리기 위해 contact_angle을 4에서 6까지 0.5씩 증가
for i, alpha in zip(np.arange(4.0, 1.5, -0.5), alpha_values):
    contact_angle = np.pi * i / 4  # 각도를 라디안으로 변환

    # 접점 계산
    x_contact = center[0] + radius * np.cos(contact_angle)
    y_contact = center[1] + radius * np.sin(contact_angle)

    # 접선의 기울기 계산 (접점에서의 법선의 기울기의 역수, 음의 역수로 접선 기울기 계산)
    normal_slope = (y_contact - center[1]) / (x_contact - center[0])
    tangent_slope = -1 / normal_slope

    # 접선의 시작과 끝 점 계산
    length = 200  # 선의 길이
    x1 = x_contact - length / np.sqrt(1 + tangent_slope**2)
    y1 = y_contact - tangent_slope * (length / np.sqrt(1 + tangent_slope**2))
    x2 = x_contact + length / np.sqrt(1 + tangent_slope**2)
    y2 = y_contact + tangent_slope * (length / np.sqrt(1 + tangent_slope**2))

    # 접선을 그립니다.
    ax.plot([x1, x2], [y1, y2], color='black', alpha=alpha)
    print(x1, x2, (x1+x2)//2)
    print(y1,y2, (y1+y2)//2)

# 이미지 출력 설정
# ax.imshow(image)
ax.set_xlim(0, 480)
ax.set_ylim(0, 480)
plt.axis('off')  # 축을 숨깁니다.

# 그림을 화면에 표시합니다.
plt.show()
