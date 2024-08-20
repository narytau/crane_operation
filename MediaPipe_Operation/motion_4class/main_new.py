import os
import cv2
import sys
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_selection import SelectFromModel

import torch
import torch.nn.functional as F
from class_NN import RegularizedNN, ComplexNN

# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)

TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")

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
DATA_DEPTH = 15
SIGMA_ARRAY = [1, 1, 10]

# Initialize Variables
frame_time_stamp = 0
pose_landmarks = None
motion_array = np.zeros((DATA_DEPTH, 2*3)) 
theta_array = np.zeros((2, len(RIGHT_BODY_INDEX[:3])))
vector_array = np.zeros((len(RIGHT_BODY_INDEX),3,2))
body_center_array = np.zeros((len(CENTER_BODY_INDEX), 3))


# Open the model
# Rotate
scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler3.sav'), 'rb'))
model_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'SVM_model3.sav'), 'rb'))
# forest_model = pickle.load(open(os.path.join(SAVE_PATH, 'forest_model4_without.sav'), 'rb'))
forest_model = None

# Theta
scaler_theta = pickle.load(open(os.path.join(MODEL_PATH, 'scaler.sav'), 'rb'))
model_theta  = pickle.load(open(os.path.join(MODEL_PATH, 'model2.pickle'), 'rb'))


input_size = 90  # 入力ベクトルの長さ
num_classes = 4  # クラスの数
# model_NN = RegularizedNN(input_size, num_classes)
model_NN = ComplexNN(input_size, num_classes)
model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier4_long.pth')))
model_NN.eval()

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

def process_data_i(measurement, data_window, sigma):
    if data_window.shape[0] < WINDOW_SIZE:
        data_window = np.vstack([data_window, measurement])
    else:
        data_window = np.roll(data_window, -1, axis=0)
        data_window[-1] = measurement

    smoothed_data = function_math.apply_gaussian_smoothing(data_window, sigma)
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
def process_frame_main(frame, motion_array, theta_array, landmarker, sigma_array):
    global pose_landmarks, frame_time_stamp
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
                                            
                                            
        vector_array[0,:,0] = body_position_array[2,:,0] - body_position_array[1,:,0]
        vector_array[1,:,0] = body_position_array[2,:,0] - body_position_array[0,:,0]
        vector_array[2,:,0] = body_position_array[1,:,0] - body_position_array[0,:,0]
        vector_array[0,:,1] = body_position_array[2,:,1] - body_position_array[1,:,1]
        vector_array[1,:,1] = body_position_array[2,:,1] - body_position_array[0,:,1]
        vector_array[2,:,1] = body_position_array[1,:,1] - body_position_array[0,:,1]

        # Right            
        for i in range(len(RIGHT_BODY_INDEX[:3])):
            vec = - vector_array[i, 0:2, 0]
            angle = np.arctan2(vec[1], vec[0])
            theta_array[0, i] = 180 * function_math.normalize_angle(angle=angle, origin=110*np.pi/180) / np.pi
            
        # Left
        for i in range(len(LEFT_BODY_INDEX[:3])):
            vec = - vector_array[i, 0:2, 1]
            angle = np.arctan2(vec[1], - vec[0])
            theta_array[1, i] = 180 * function_math.normalize_angle(angle=angle, origin=110*np.pi/180) / np.pi

        # Array for center of the body  
        for ind, body_ind in enumerate(CENTER_BODY_INDEX):
            body_center_array[ind, :] = np.array([pose_landmarks[0][body_ind].x,
                                                    pose_landmarks[0][body_ind].y,
                                                    pose_landmarks[0][body_ind].z])


        # Compensation for human tilt and camera tilt
        vec1 = body_center_array[0,:2] - body_center_array[3,:2]
        vec2 = body_center_array[1,:2] - body_center_array[2,:2]
        vec = - (vec1 + vec2) / 2
        tilt = (np.pi / 2 - np.arctan2(vec[1], vec[0])) * 180 / np.pi
        theta_array += tilt
        
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

        return True, body_position_array, motion_array, theta_array
    return False, body_position_array, motion_array, theta_array

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
        
    if motion_pred is not None:
        cv2.putText(flip_color_image, str(motion_pred), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def predict_motion_NN(motion_array, model, scaler):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    scaled_flatten_motion_array = scaler.transform(flatten_motion_array)

    input_tensor = torch.tensor(scaled_flatten_motion_array, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # バッチ次元の追加

    with torch.no_grad():  # 勾配計算を無効にする
        output = model(input_tensor)
    
    # softmax
    motion_prob = F.softmax(output.squeeze(), dim=-1).numpy()
    motion_pred = process_predcition(motion_prob)
    
    return motion_pred, motion_prob


def predict_motion_SVM(motion_array, model, scaler, forest_model=None):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    scaled_flatten_motion_array = scaler.transform(flatten_motion_array)

    if forest_model is not None:
        selector = SelectFromModel(forest_model, threshold="median")
        scaled_flatten_motion_array = selector.transform(scaled_flatten_motion_array)
    
    # motion_pred = model.predict(scaled_flatten_motion_array)
    motion_prob = model.predict_proba(scaled_flatten_motion_array)
    
    motion_pred = process_predcition(motion_prob[0])
    return motion_pred, motion_prob[0]

def process_predcition(probability):
    threshold = 0.7
    prediction_num = np.argmax(probability)
    
    if max(probability) < threshold or prediction_num == 3:    
        prediction = "Unclassified"
    elif prediction_num == 0:
        prediction = "High"
    elif prediction_num == 1:
        prediction = "Middle"
    elif prediction_num == 2:
        prediction = "Low"
    else:
        prediction = "Error"

    return prediction

# Data Collection
def run():
    iter = 0
    global motion_array, theta_array
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipe.start(cfg)
        
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()
            
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            flip_color_image = cv2.flip(color_image, 1)

            processed, body_position_array, motion_array, theta_array = process_frame_main(color_image, motion_array, theta_array, landmarker, SIGMA_ARRAY)

            if processed:
                iter += 1
                
            # # Prepare for using model
            # theta_scaled = scaler_theta.transform(theta_array)
            # theta_prob = model_theta.predict_proba(theta_scaled)
            # theta_pred = model_theta.predict(theta_scaled)
            
            # if theta_prob[1,-1] > 0.7: 
            #     # motion_pred, motion_prob = predict_motion_SVM(motion_array, model_SVM, scaler, forest_model)
            #     motion_pred, motion_prob = predict_motion_NN(motion_array, model_NN, scaler)
            #     cv2.putText(flip_color_image, str(np.round(motion_prob * 100)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # else:
            #     motion_pred = None

            # motion_pred, motion_prob = predict_motion_SVM(motion_array, model_SVM, scaler, forest_model)
            motion_pred, motion_prob = predict_motion_NN(motion_array, model_NN, scaler)

            display_data(body_position_array, flip_color_image, motion_pred)

            cv2.putText(flip_color_image, str(np.round(motion_prob * 100)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('RGB Image', flip_color_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        pipe.stop()
        cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    run()