import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from sklearn.preprocessing import StandardScaler

'''
Deal with both hands
(Right hand has a priority)
'''

# Path for model file of MediaPipe
base_path = os.path.dirname(__file__)
task_path = base_path + "\\recognizer\\" + "pose_landmarker_full.task"
task_hand_path = base_path + "\\recognizer\\" + "hand_landmarker.task"

model_path = base_path + '\\theta_data\\'

# Import classes of MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Setting
pose_landmarks = None
pose_world_landmarks = None	
segmentation_masks = None

# Setting
handedness = None
hand_landmarks = None
hand_world_landmarks = None	

# Callback function to show the results
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('pose landmarker result: {}'.format(result))
    global pose_landmarks, pose_world_landmarks, segmentation_masks
    pose_landmarks = result.pose_landmarks
    pose_world_landmarks = result.pose_world_landmarks
    segmentation_masks = result.segmentation_masks
    return pose_landmarks, pose_world_landmarks, segmentation_masks

# Callback function to show the results
def print_hand_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global handedness, hand_landmarks, hand_world_landmarks
    handedness = result.handedness
    hand_landmarks = result.hand_landmarks
    hand_world_landmarks = result.hand_world_landmarks
    return handedness, hand_landmarks, hand_world_landmarks



# Set for options of Gesture Recognizer
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

options_hand = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_hand_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_hand_result)

# [shoulder, elbow, wrist]
right_body_index = [12, 14, 16]    
left_body_index = [11, 13, 15]
center_body_index = [11, 12, 23, 24]

joint_array = np.zeros((3, 2, 2))
right_body_position_array = np.zeros((len(right_body_index),3))
left_body_position_array = np.zeros((len(left_body_index),3))
body_position_array = np.zeros((len(right_body_index), 3, 2))
body_center_array = np.zeros((len(center_body_index), 3))
vector_array = np.zeros((len(right_body_index),3,2))
theta_array = np.zeros((2, len(right_body_index)))

array_size = 15

iter = 0

# Threshold
distance_threshold = 0.8
recog_threshold = 0.7

# Open the model
with open(model_path+'model2.pickle', mode='rb') as f:
    svm_model = pickle.load(f)
save_scaler = pickle.load(open(model_path+'scaler.sav', 'rb'))

# Create instance of Gesture Recognizer
with PoseLandmarker.create_from_options(options) as landmarker:
    with HandLandmarker.create_from_options(options_hand) as landmarker_hand:
    
        # Set pipeline of Realsense
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipe.start(cfg)
        
        # Initial value of timestamp
        frame_timestamp_ms = 0
        while True:
            # Wait for frames and get color frame
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame:
                continue
                
            # Distance
            depth_data = depth_frame.get_distance(100,100)
            # print(depth_data)
            
            # Convert RGB into numpy array (480x640x3)
            color_image = np.asanyarray(color_frame.get_data())
            image_shape = color_image.shape

            # Convert RGB into  MediaPipe Image Object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

            # Recognition by Gesture recognizer
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            landmarker_hand.detect_async(mp_image, frame_timestamp_ms)
            
            frame_timestamp_ms += 1      
                    
            # Flip the image
            flip_color_image = cv2.flip(color_image, 1)
            
            # Check if variables is not empty
            if pose_landmarks:
                if hand_world_landmarks:
                    for i, ind in enumerate(right_body_index):
                        body_position_array[i, :, 0] = [pose_landmarks[0][ind].x,
                                                        pose_landmarks[0][ind].y,
                                                        pose_landmarks[0][ind].z]
                        
                    for i, ind in enumerate(left_body_index):
                        body_position_array[i, :, 1] = [pose_landmarks[0][ind].x,
                                                        pose_landmarks[0][ind].y,
                                                        pose_landmarks[0][ind].z]
                    thumb = np.array([hand_landmarks[0][8].x,
                                    hand_landmarks[0][8].y,
                                    hand_landmarks[0][8].z])
                    
                    # print(body_position_array[0, :, :])
                    print("thumb", thumb)
                

            # Show image
            cv2.imshow('RGB Image', flip_color_image)
            
            # If the escape button is pressed, exit the loop
            if cv2.waitKey(5) & 0xFF == 27:
                print("Stop")
                break
      
# Stop pipeline
pipe.stop()
cv2.destroyAllWindows()