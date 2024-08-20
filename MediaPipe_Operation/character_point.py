import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from joblib import Memory
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from scipy.ndimage import gaussian_filter1d


# Constants and Paths
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "motion_model")

# MediaPipe Imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Body Index Constants
# shoulder, elbow, wrist, pinky, index, thumb
RIGHT_BODY_INDEX = [12, 14, 16, 18, 20, 22] 
LEFT_BODY_INDEX = [11, 13, 15, 17, 19, 21]
CENTER_BODY_INDEX = [11, 12, 23, 24]

# Data Collection Constants
WINDOW_SIZE = 100
MOTION_SPEED = ['HIGH', 'MIDDLE', 'LOW']
DATA_DEPTH = 30
SIGMA_ARRAY = [1, 1, 10]

# Initialize Variables
frame_time_stamp = 0
pose_landmarks = None
motion_array = np.zeros((DATA_DEPTH, 3*4)) 
past_bool_array = np.zeros(DATA_DEPTH)
data_window = [np.zeros((0, len(RIGHT_BODY_INDEX) * 2))] * 3   
body_position_past = np.zeros((len(RIGHT_BODY_INDEX), 3, 2))
theta_array = []
                                          
# Open the model
save_motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler_30.sav'), 'rb'))
save_motion_model  = pickle.load(open(os.path.join(SAVE_PATH, 'motion_model_30.sav'), 'rb'))

# Open the model
save_motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler_30_gaussian.sav'), 'rb'))
save_motion_pca    = pickle.load(open(os.path.join(SAVE_PATH, 'pca_model_30_gaussian.sav'), 'rb'))
save_motion_forest = pickle.load(open(os.path.join(SAVE_PATH, 'forest_model_30_gaussian.sav'), 'rb'))
save_motion_model  = pickle.load(open(os.path.join(SAVE_PATH, 'motion_model_30_gaussian.sav'), 'rb'))

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
def process_frame_main(frame, depth_frame, motion_array, landmarker, sigma_array):
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

        # Gaussian Filter
        data_window, body_position_filtered = process_data(body_position_array, 
                                                           data_window, 
                                                           sigma_array)
        body_position_array = body_position_filtered
        
        # Normalize based on shoulder width
        shoulder_width = np.linalg.norm(body_position_array[0,:,0] - body_position_array[0,:,1])
        shoulder_to_elbow = body_position_array[1,:,1] - body_position_array[0,:,1]
        shoulder_to_wrist = body_position_array[2,:,1] - body_position_array[0,:,1]
        shoulder_to_thumb = body_position_array[5,:,1] - body_position_array[0,:,1]
        shoulder_to_pinky = body_position_array[3,:,1] - body_position_array[0,:,1]
        
        
        # Renew the data
        motion_array[-1, :] = np.concatenate((shoulder_to_elbow, 
                                              shoulder_to_wrist, 
                                              shoulder_to_thumb,
                                              shoulder_to_pinky)) / shoulder_width
        
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
        
    cv2.putText(flip_color_image, motion_pred[0], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    
def predict_motion(motion_array):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    
    # Prepare for using model
    flatten_motion_array_scaled = save_motion_scaler.transform(flatten_motion_array)
    motion_prob = save_motion_model.predict_proba(flatten_motion_array_scaled)
    motion_pred = save_motion_model.predict(flatten_motion_array_scaled)
    
    # motion_probの返り値しだい
    if max(motion_prob[0]) < 0.8:
        motion_pred = ['NA']
    print(motion_pred, motion_prob)    
    # Renew bool array
    
    return motion_pred
    
    
def predict_motion_reduced(motion_array):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    
    # Prepare for using model
    flatten_motion_array_scaled = save_motion_scaler.transform(flatten_motion_array)
    flatten_motion_array_reduced = save_motion_pca.transform(flatten_motion_array_scaled)
    
    selector = SelectFromModel(save_motion_forest, threshold="median")
    X_important = selector.transform(flatten_motion_array_reduced)

    motion_prob = save_motion_model.predict_proba(X_important)
    motion_pred = save_motion_model.predict(X_important)
    
    # motion_probの返り値しだい
    if max(motion_prob[0]) < 0.8:
        motion_pred = ['NA']
    print(motion_pred, motion_prob)    
    # Renew bool array
    
    return motion_pred

memory = Memory("cachedir")

@memory.cache
def cached_predict_motion(flatten_motion_array_scaled):
    motion_prob = save_motion_model.predict_proba(flatten_motion_array_scaled)
    motion_pred = save_motion_model.predict(flatten_motion_array_scaled)
    return motion_pred, motion_prob

def predict_motion_reduced(motion_array):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    flatten_motion_array_scaled = save_motion_scaler.transform(flatten_motion_array)
    flatten_motion_array_reduced = save_motion_pca.transform(flatten_motion_array_scaled)
    
    selector = SelectFromModel(save_motion_forest, threshold="median")
    X_important = selector.transform(flatten_motion_array_reduced)
    
    motion_pred, motion_prob = cached_predict_motion(X_important)
    
    if max(motion_prob[0]) < 0.85:
        motion_pred = ['NA']
    
    return motion_pred

def calculate_angle(vec1, vec2):
    x = np.inner(vec1, vec2)
    theta = np.arccos(x / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return theta

def normalize_angle(angle, origin):
    """Function to normalize an angle to the range [-origin, origin + 2pi]"""
    return (angle + origin) % (2 * np.pi) - origin


# Data Collection
def run():
    global motion_array
    iter = 0
    
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipe.start(cfg)
        
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            flip_color_image = cv2.flip(color_image, 1)

            processed, body_position_array, motion_array = process_frame_main(color_image, depth_frame, motion_array, landmarker, SIGMA_ARRAY)
            if processed:
                iter += 1
                
            wrist_to_pinky = body_position_array[2,:,1] - body_position_array[1,:,1]    
            wrist_to_pinky = wrist_to_pinky[:2] / np.linalg.norm(wrist_to_pinky)
            angle = np.arctan2(wrist_to_pinky[1], wrist_to_pinky[0])
            angle = angle + np.pi / 2
            theta = 180 * normalize_angle(angle=angle, origin=np.pi) / np.pi
            
            theta_array.append(theta)
                
            motion_pred = predict_motion_reduced(motion_array)

            display_data(body_position_array, flip_color_image, motion_pred)
            
             ################################################################
            cv2.putText(flip_color_image, str(np.round(theta,4)),  
                    (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('RGB Image', flip_color_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        pipe.stop()
        cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    run()
    
    plt.plot(theta_array)
    plt.show()