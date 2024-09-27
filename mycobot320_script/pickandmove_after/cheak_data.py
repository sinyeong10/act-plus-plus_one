import threading
import time
from pymycobot.mycobot import MyCobot
import cv2

def print_motor_angles(mc):
    try:
        while True:
            # 현재 모터 각도 출력
            angles = mc.get_angles()
            print(f"현재 모터 각도: {angles}")

            # 1초간 대기
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n스레드가 사용자에 의해 종료됨")

# MyCobot 객체 초기화
mc = MyCobot('COM7', 115200)  # 포트와 속도에 맞게 수정

# MyCobot 초기 설정
# mc.init_eletric_gripper()
# mc.set_eletric_gripper(0)
# mc.set_gripper_value(100, 20, 1)
mc.set_gripper_mode(0)
mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)

# 모터 각도 출력을 담당할 스레드 생성 및 실행
thread = threading.Thread(target=print_motor_angles, args=(mc,))
thread.start()


#벌리고 초기 이동
mc.init_eletric_gripper()
mc.set_eletric_gripper(0)
mc.set_gripper_value(100, 20, 1)
time.sleep(1)
mc.send_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 20)
time.sleep(5)

cv2.waitKey(0)
print(1)

try:
    # 5초간 기다림
    time.sleep(5)

except KeyboardInterrupt:
    print("\n메인 프로그램이 사용자에 의해 종료됨")

finally:
    # 스레드 종료
    thread.join()

    # 리소스 정리
    mc.release_all_servos()
    mc.disconnect()
