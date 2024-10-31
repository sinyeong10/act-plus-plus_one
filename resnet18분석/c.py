import cv2
import sys

# 웹캠에서 영상을 가져오기 위해 VideoCapture 객체 생성
cap = cv2.VideoCapture(0)

# 캡처가 정상적으로 초기화되었는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    sys.exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 여기서 프레임 처리를 할 수 있습니다. 예를 들어, 화면에 표시하거나 다른 작업 수행 가능
    
    # 화면에 프레임 표시
    cv2.imshow('frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
