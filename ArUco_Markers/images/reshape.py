from PIL import Image
import pillow_heif
import os
import glob
import pathlib
import cv2
import time

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "code", "calibration_data", "video2.MOV")
SAVE_PATH = os.path.join(BASE_PATH, "code", "Initialization")


cap = cv2.VideoCapture(TASK_PATH)

if not cap.isOpened():
    print(f"Error: Unable to open video file: {TASK_PATH}")
    exit()
    
i = 0

# Read and display frames in a loop
while True:
    ret, frame = cap.read()
    
    # Check if the frame was successfully grabbed
    if not ret or frame is None:
        print("End of video or error in reading the frame.")
        break
    
    # Resize frame to 640x480
    color_image = cv2.resize(frame, (640, 480))
    
    # Display the frame
    # cv2.imshow('RGB Image', color_image)
    
    i += 1
    
    if i == 100:
        cv2.imwrite(os.path.join(SAVE_PATH, "init_img1.jpg"), frame)

        
    if i == 150:
        cv2.imwrite(os.path.join(SAVE_PATH, "init_img2.jpg"), frame)

    if i == 600:
        cv2.imwrite(os.path.join(SAVE_PATH, "init_img3.jpg"), frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    