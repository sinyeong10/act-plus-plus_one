import cv2
import numpy as np


cap0 = cv2.VideoCapture(0)

# cap1 = cv2.VideoCapture(1)        

while True:
    _, frame = cap0.read()
    
    cv2.imshow("right_wrist", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

# while True:
#     _, frame = cap1.read()
        
#     cv2.imshow("top", frame)
#     key = cv2.waitKey(10)
#     if key == ord("q"):
#         break
    
