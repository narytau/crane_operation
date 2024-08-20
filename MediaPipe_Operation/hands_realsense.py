import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

# Path for model file of MediaPipe
base_path = os.path.dirname(__file__)
task_path = base_path + "\\recognizer\\" + "hand_landmarker.task"
model_path = base_path + '\\theta_data\\'

# Import classes of MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Setting
handedness = None
hand_landmarks = None
hand_world_landmarks = None	

# Callback function to show the results
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global handedness, hand_landmarks, hand_world_landmarks
    handedness = result.handedness
    hand_landmarks = result.hand_landmarks
    hand_world_landmarks = result.hand_world_landmarks
    return handedness, hand_landmarks, hand_world_landmarks

# Set for options of Gesture Recognizer
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

joint_array = np.zeros((4,2))

# Create instance of Gesture Recognizer
with HandLandmarker.create_from_options(options) as landmarker:
    # Set pipeline of Realsense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    
    # Initial value of timestamp
    frame_timestamp_ms = 0
    while True:
        # Wait for frames and get color frame
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
            
        # Convert RGB into numpy array (480x640x3)
        color_image = np.asanyarray(color_frame.get_data())
        image_shape = color_image.shape

        # Convert RGB into MediaPipe Image Object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

        # Recognition by Gesture recognizer
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1       
        
        # Flip the image
        flip_color_image = cv2.flip(color_image, 1)    
        
        if hand_world_landmarks:
            thumb = np.array([hand_landmarks[0][4].x,
                              hand_landmarks[0][4].y,
                              hand_landmarks[0][4].z])
            index = np.array([hand_landmarks[0][8].x,
                              hand_landmarks[0][8].y,
                              hand_landmarks[0][8].z])
            wrist = np.array([hand_landmarks[0][0].x,
                              hand_landmarks[0][0].y,
                              hand_landmarks[0][0].z])
            thumb_mcp = np.array([hand_landmarks[0][1].x,
                                  hand_landmarks[0][1].y,
                                  hand_landmarks[0][1].z])
            
            thumb_world = np.array([hand_world_landmarks[0][4].x,
                                    hand_world_landmarks[0][4].y,
                                    hand_world_landmarks[0][4].z])
            index_world = np.array([hand_world_landmarks[0][8].x,
                                    hand_world_landmarks[0][8].y,
                                    hand_world_landmarks[0][8].z])
            wrist_world = np.array([hand_world_landmarks[0][0].x,
                                    hand_world_landmarks[0][0].y,
                                    hand_world_landmarks[0][0].z])
            thumb_mcp_world = np.array([hand_world_landmarks[0][1].x,
                                        hand_world_landmarks[0][1].y,
                                        hand_world_landmarks[0][1].z])

            print("detect!")
            print(hand_world_landmarks[0])
            joint_array[0, :] = [(1 - thumb[0])*image_shape[1], thumb[1]*image_shape[0]]
            joint_array[1, :] = [(1 - index[0])*image_shape[1], index[1]*image_shape[0]]
            joint_array[2, :] = [(1 - wrist[0])*image_shape[1], wrist[1]*image_shape[0]]
            joint_array[3, :] = [(1 - thumb_mcp[0])*image_shape[1], thumb_mcp[1]*image_shape[0]]
            joint_array = joint_array.astype(int)
            
            ratio = np.linalg.norm(thumb - index) / np.linalg.norm(thumb_mcp - wrist)
            lower_limit = 0.5
            upper_limit = 4.8
            ratio_open = (ratio - lower_limit) / (upper_limit - lower_limit) * 100
            ratio_open = max(0, min(ratio_open, 100))

            
            cv2.putText(flip_color_image, str(round(np.linalg.norm(thumb_world - index_world), 2)), 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(flip_color_image, str(round(np.linalg.norm(thumb_mcp_world - wrist_world), 2)), 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(flip_color_image, str(round(ratio_open,4)), 
                        (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            for i in range(4):
                cv2.circle(flip_color_image, (joint_array[i,0], joint_array[i,1]), 5, (0, 255, 0), -1)  # Draw a green circle

 
        # Show image
        cv2.imshow('RGB Image', flip_color_image)
        
        # If the escape button is pressed, exit the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break
      
# Stop pipeline
pipe.stop()
cv2.destroyAllWindows()