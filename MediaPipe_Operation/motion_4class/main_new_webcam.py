import os
import cv2
import sys
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import tensorflow as tf

from joblib import Memory
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.ndimage import gaussian_filter1d


# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)

TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "motion_model_transformer")
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier.keras")

PACKAGE_PATH = os.path.join(BASE_PATH, "my_module")
sys.path.append(PACKAGE_PATH)

import function_math

# MediaPipe Imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Body Index Constants
RIGHT_BODY_INDEX = [12, 14, 16, 18, 20, 22]
LEFT_BODY_INDEX = [11, 13, 15, 17, 19, 21]
CENTER_BODY_INDEX = [11, 12, 23, 24]

# Data Collection Constants
WINDOW_SIZE = 100
MOTION_SPEED = ['HIGH', 'MIDDLE', 'LOW']
DATA_DEPTH = 15
SIGMA_ARRAY = [1, 1, 10]

# Initialize Variables
frame_time_stamp = 0
pose_landmarks = None
motion_array = np.zeros((DATA_DEPTH, 2*3)) 

# Open the model
model_NN = tf.keras.models.load_model(MODEL_SAVE_PATH)
model_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'SVM_model2.sav'), 'rb'))
scaler_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler_SVM.sav'), 'rb'))


# Callback Function for PoseLandmarker
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global pose_landmarks
    pose_landmarks = result.pose_landmarks

# Create PoseLandmarker Options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Apply Gauss kernel to data
def apply_gaussian_smoothing(data, sigma):
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=0)
    return smoothed_data

def process_data_i(measurement, data_window, sigma):
    if data_window.shape[0] < WINDOW_SIZE:
        data_window = np.vstack([data_window, measurement])
    else:
        data_window = np.roll(data_window, -1, axis=0)
        data_window[-1] = measurement

    smoothed_data = apply_gaussian_smoothing(data_window, sigma)
    current_smoothed_value = smoothed_data[-1]
    
    return data_window, current_smoothed_value

def process_data(body_position_array, data_window, sigma_array):
    body_position_filtered = np.zeros_like(body_position_array)
    # x,y,z
    for i in range(3):
        measurement = body_position_array[:,i,:].ravel()
        data_window[i], current_smoothed_value = process_data_i(measurement, data_window[i], sigma_array[i])
        body_position_filtered[:,i,:] = current_smoothed_value.reshape(body_position_array.shape[0], 
                                                                       body_position_array.shape[2])        
    return data_window, body_position_filtered


# Function to process frames
def process_frame_main(frame, motion_array, landmarker, sigma_array):
    global pose_landmarks, frame_time_stamp, body_position_past
    global data_window

    body_position_array = np.zeros((len(RIGHT_BODY_INDEX), 3, 2))
    
    # Get depth information
    # depth_data = depth_frame.get_distance(100, 100)
    
    # Convert to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Perform pose detection
    landmarker.detect_async(mp_image, frame_time_stamp)
    frame_time_stamp += 1
    
    # Shift the array
    motion_array = np.roll(motion_array, -1, axis=0)
    
    # Check if landmarks are available
    if pose_landmarks:
        for i, ind in enumerate(RIGHT_BODY_INDEX):
            body_position_array[i, :, 0] = [pose_landmarks[0][ind].x,
                                            pose_landmarks[0][ind].y,
                                            pose_landmarks[0][ind].z]
        for i, ind in enumerate(LEFT_BODY_INDEX):
            body_position_array[i, :, 1] = [pose_landmarks[0][ind].x,
                                            pose_landmarks[0][ind].y,
                                            pose_landmarks[0][ind].z]

        # # Gaussian Filter
        # data_window, body_position_filtered = process_data(body_position_array, 
        #                                                    data_window, 
        #                                                    sigma_array)
        # body_position_array = body_position_filtered
        
        # Normalize based on shoulder width
        shoulder_to_elbow = function_math.calculate_unit_vector(body_position_array[1,:,1] - body_position_array[0,:,1])
        elbow_to_wrist    = function_math.calculate_unit_vector(body_position_array[2,:,1] - body_position_array[1,:,1])
        wrist_to_thumb    = function_math.calculate_unit_vector(body_position_array[5,:,1] - body_position_array[2,:,1])

        
        # Renew the data
        motion_array[-1, :] = np.concatenate([shoulder_to_elbow[:2], 
                                            elbow_to_wrist[:2],
                                            wrist_to_thumb[:2]])

        return True, body_position_array, motion_array
    return False, body_position_array, motion_array

def display_data(body_position_array, flip_color_image, motion_pred):
    '''
    DISPLAY
    Draw circle: 
        cv2.circle(image, center_coordinates, radius, color, thickness)
    Draw line:   
        cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
    '''
    joint_array = np.zeros((3, 2, 2))
    for i in range(joint_array.shape[0]):
        circle_x = (1 - body_position_array[i, 0, 0]) * flip_color_image.shape[1]
        circle_y = body_position_array[i, 1, 0] * flip_color_image.shape[0]
        joint_array[i, :, 0] = [circle_x, circle_y]
        
        circle_x = (1 - body_position_array[i, 0, 1]) * flip_color_image.shape[1]
        circle_y = body_position_array[i, 1, 1] * flip_color_image.shape[0]
        joint_array[i, :, 1] = [circle_x, circle_y]
        
    joint_array = joint_array.astype(int)
    
    # circles    
    for i in range(joint_array.shape[0]):
        cv2.circle(flip_color_image, (joint_array[i,0,0], joint_array[i,1,0]), 5, (2, 127, 0), -1)  # Draw a green circle
        cv2.circle(flip_color_image, (joint_array[i,0,1], joint_array[i,1,1]), 5, (255, 39, 0), -1)  # Draw a blue circle
        
    # lines
    cv2.line(flip_color_image, (joint_array[1,0,0], joint_array[1,1,0]), (joint_array[2,0,0], joint_array[2,1,0]),
            (2, 127, 0), thickness=2, lineType=cv2.LINE_8, shift=0)
    cv2.line(flip_color_image, (joint_array[1,0,1], joint_array[1,1,1]), (joint_array[2,0,1], joint_array[2,1,1]),
            (255, 39, 0), thickness=2, lineType=cv2.LINE_8, shift=0)
        
    cv2.putText(flip_color_image, motion_pred, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    ################################################################

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 数値安定性のために最大値を引きます
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def predict_motion(motion_array, model):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    motion_prob = softmax(np.squeeze(model.predict(flatten_motion_array)))
    if max(motion_prob) > 0.5:
        motion_pred = np.argmax(motion_prob)
    else:
        motion_pred = -1
    return motion_pred, motion_prob

def predict_motion_SVM(motion_array, model, scaler):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)

    scaled_flatten_motion_array = scaler.transform(flatten_motion_array)
    
    motion_prob = model.predict_proba(scaled_flatten_motion_array)
    motion_pred = model.predict(scaled_flatten_motion_array)
    
    if max(motion_prob[0]) < 0.75:
        motion_pred = [-1]
    # print(motion_pred, motion_prob)
    
    pred_num = motion_pred[0]
    if pred_num == 0:
        pred = "High"
    elif pred_num == 1:
        pred = "Middle"
    elif pred_num == 2:
        pred = "Low"
    else:
        pred = "Unclassified"
    
    return pred, motion_prob

# Data Collection
def run():
    global motion_array
    iter = 0
    cap = cv2.VideoCapture(0)
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            color_image = image
            flip_color_image = cv2.flip(color_image, 1)

            processed, body_position_array, motion_array = process_frame_main(color_image, motion_array, landmarker, SIGMA_ARRAY)
            if processed:
                iter += 1
                
            motion_pred, motion_prob = predict_motion_SVM(motion_array, model_SVM, scaler_SVM)
            # motion_pred, motion_prob = predict_motion(motion_array, model_NN)

            display_data(body_position_array, flip_color_image, motion_pred)

            cv2.putText(flip_color_image, str(np.round(motion_prob[0] * 100)),  
                            (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('RGB Image', flip_color_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    run()