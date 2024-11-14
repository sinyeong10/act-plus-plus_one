import cv2


filename = "two_cam_episode_1_image"
framename = "right_wrist_frame_0"
# 이미지 파일 경로
image_path = f"scr\\mask_data\\{filename}\\{framename}.jpg"

# 이미지 로드
image = cv2.imread(image_path)
clone = image.copy()  # 원본 이미지를 복사해둠

# 전역 변수 설정
ref_point = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    # 마우스 왼쪽 버튼 클릭 시 시작점 기록
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # 마우스 이동 중이라면 현재 위치 업데이트
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        image[:] = clone.copy()  # 원본 이미지에서 다시 복사
        cv2.rectangle(image, ref_point[0], (x, y), (0, 255, 0), 1)

    # 마우스 왼쪽 버튼을 놓으면 끝 지점 기록 및 영역 설정
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 1)
        image[:] = clone.copy()  # 원본 이미지에서 다시 복사
        cv2.imshow("image", image)

# 윈도우 생성 및 마우스 콜백 설정
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# 이미지 표시 및 영역 선택
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # 'r' 키를 누르면 원본 이미지로 재설정
    if key == ord("r"):
        image = clone.copy()
        print("r")

    # 'c' 키를 누르면 현재 선택된 영역을 검정색으로 설정하고 다음 영역 선택 가능
    elif key == ord("c"):
        print("c")
        # 선택된 영역을 검정색으로 설정
        if len(ref_point) == 2:
            print(ref_point) #기존에는 0에서 1이었음 ex) [(33, 75), (116, 120)]
            #처리해야할 값 예시 [(418, 96), (219, 165)]


            # top_left = ref_point[0]
            # bottom_right = ref_point[1]
            # image[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1] = 0

            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]
            x1, x2 = max(0, min(x1, x2)), max(x1, x2)
            y1, y2 = max(0, min(y1, y2)), max(y1, y2)
            image[y1:y2+1, x1:x2+1] = 0
            print(x1,y1, "\\", x2, y2)

            clone = image.copy()  # 업데이트된 이미지를 복사해둠
            ref_point = []  # ref_point 초기화

    # 'q' 키를 누르면 루프를 종료
    elif key == ord("q"):
        break

# 결과 이미지 저장 및 표시
cv2.imwrite(f"scr\\mask_data\\{filename}_mask\\{filename}.jpg", image)
cv2.destroyAllWindows()
