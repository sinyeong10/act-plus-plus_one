import cv2
import os

for i in range(1, 48+1):#49):
    for j in range(115):
        camera_name = [f"top_frame_{j}", f"right_wrist_frame_{j}"]
        camera_idx = 0
        filename = f"two_cam_episode_{i}_image"
        framename = camera_name[camera_idx]
        # 이미지 파일 경로
        image_path = f"scr\\mask_data\\{filename}\\{framename}.jpg"
        print(image_path)
        os.makedirs(f"scr\\mask_data\\{filename}_mask", exist_ok=True)  # 폴더가 없으면 생성
        # 이미지 로드
        image = cv2.imread(image_path)
        clone = image.copy()  # 원본 이미지를 복사해둠

        # # 이미지 표시 및 영역 선택
        # while True:
        #     cv2.imshow("image", image)
        #     key = cv2.waitKey(1) & 0xFF

        #     # 'r' 키를 누르면 원본 이미지로 재설정
        #     if key == ord("r"):
        #         image = clone.copy()
        #         print("r")

        #     # 'c' 키를 누르면 현재 선택된 영역을 검정색으로 설정하고 다음 영역 선택 가능
        #     elif key == ord("c"):
        #         print("c")
        # 선택된 영역을 검정색으로 설정
        x1, y1 = 230, 350
        x2, y2 = 412, 400
        zero_image = image.copy()
        zero_image[:,:] = 0

        x1, x2 = max(0, min(x1, x2)), max(x1, x2)
        y1, y2 = max(0, min(y1, y2)), max(y1, y2)
        # image[y1:y2+1, x1:x2+1] = 0
        print(x1,y1, "\\", x2, y2)
        zero_image[y1:y2+1, x1:x2+1] = image[y1:y2+1, x1:x2+1]
        image = zero_image.copy()

        clone = image.copy()  # 업데이트된 이미지를 복사해둠
        ref_point = []  # ref_point 초기화

            # # 'q' 키를 누르면 루프를 종료
            # elif key == ord("q"):
            #     break

        # 결과 이미지 저장 및 표시
        print(f"scr\\mask_data\\{filename}_mask\\{framename}.jpg")
        cv2.imwrite(f"scr\\mask_data\\{filename}_mask\\{framename}.jpg", image)
        cv2.destroyAllWindows()

        
        camera_idx = 1
        framename = camera_name[camera_idx]
        # 이미지 파일 경로
        image_path = f"scr\\mask_data\\{filename}\\{framename}.jpg"
        print(image_path)
        os.makedirs(f"scr\\mask_data\\{filename}_mask", exist_ok=True)  # 폴더가 없으면 생성
        # 이미지 로드
        image = cv2.imread(image_path)
        # clone = image.copy()  # 원본 이미지를 복사해둠

        # x1, y1 = 230, 350
        # x2, y2 = 412, 400
        # zero_image = image.copy()
        # zero_image[:,:] = 0

        # x1, x2 = max(0, min(x1, x2)), max(x1, x2)
        # y1, y2 = max(0, min(y1, y2)), max(y1, y2)
        # # image[y1:y2+1, x1:x2+1] = 0
        # print(x1,y1, "\\", x2, y2)
        # zero_image[y1:y2+1, x1:x2+1] = image[y1:y2+1, x1:x2+1]
        # image = zero_image.copy()

        # 결과 이미지 저장 및 표시
        print(f"scr\\mask_data\\{filename}_mask\\{framename}.jpg")
        cv2.imwrite(f"scr\\mask_data\\{filename}_mask\\{framename}.jpg", image)
        cv2.destroyAllWindows()
