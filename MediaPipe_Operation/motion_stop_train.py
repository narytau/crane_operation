import os
import cv2
import time
import random
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from my_module import function_math

# Constants and Paths
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

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
DATA_SIZE = 5000

MOTION_SPEED = [0]
SIGMA_ARRAY = [1, 1, 10]

# Initialize Variables
frame_time_stamp = 0
pose_landmarks = None
motion_array = np.zeros((DATA_SIZE, 2*4, len(MOTION_SPEED))) # DATA_SIZE x (3 xyz x 4 points) x 3 speeds
position_array = np.zeros((DATA_SIZE, 3, len(MOTION_SPEED))) # DATA_SIZE x (3 xyz x 4 points) x 3 speeds
data_window = [np.zeros((0, len(RIGHT_BODY_INDEX) * 2))] * 3   
body_position_past = np.zeros((len(RIGHT_BODY_INDEX), 3, 2))

# Callback Function for PoseLandmarker
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global pose_landmarks, pose_world_landmarks
    pose_landmarks = result.pose_landmarks
    # pose_world_landmarks = result.pose_world_landmarks

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
        body_position_filtered[:,i,:] = current_smoothed_value.reshape(body_position_array.shape[0], body_position_array.shape[2])        
    return data_window, body_position_filtered

# Function to process frames
def process_frame_train(frame, depth_frame, iter, speed_mode, landmarker, sigma_array):
    global pose_landmarks, pose_world_landmarks
    global data_window, motion_array, frame_time_stamp, body_position_past, position_array

    body_position_array = np.zeros((len(RIGHT_BODY_INDEX), 3, 2))
    
    # Get depth information
    # depth_data = depth_frame.get_distance(100, 100)
    
    # Convert to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Perform pose detection
    landmarker.detect_async(mp_image, frame_time_stamp)
    frame_time_stamp += 1
    
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
            
        # # Detect
        # if iter == 0:
        #     body_position_past = body_position_array
        # bool_outlier = np.abs(body_position_past - body_position_array) > 0.5
        # body_position_array[bool_outlier] = body_position_past[bool_outlier]
        
        # body_position_past = body_position_array            
            
        # Gaussian Filter
        # data_window, body_position_filtered = process_data(body_position_array, 
        #                                                    data_window, 
        #                                                    sigma_array)
        # body_position_array = body_position_filtered

        # Normalize based on shoulder width
        shoulder_to_elbow_right = function_math.calculate_unit_vector(body_position_array[1,:,0] - body_position_array[0,:,0])
        shoulder_to_elbow_left = function_math.calculate_unit_vector(body_position_array[1,:,1] - body_position_array[0,:,1])
        elbow_to_wrist_right    = function_math.calculate_unit_vector(body_position_array[2,:,0] - body_position_array[1,:,0])
        elbow_to_wrist_left    = function_math.calculate_unit_vector(body_position_array[2,:,1] - body_position_array[1,:,1])

        # Renew the data
        motion_array[iter, :, speed_mode] = np.concatenate([shoulder_to_elbow_right[:2],
                                                            shoulder_to_elbow_left[:2], 
                                                            elbow_to_wrist_right[:2],
                                                            elbow_to_wrist_left[:2]])
        
        position_array[iter, :, speed_mode] = body_position_array[5,:,1]
        
        return True, body_position_array
    return False, body_position_array

def display_data(body_position_array, flip_color_image):
    '''
    DISPLAY
    Draw circle: 
        cv2.circle(image, center_coordinates, radius, color, thickness)
    Draw line:   
        cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
    '''
    joint_array = np.zeros((5, 2, 2))
    ary = [0, 1, 2, 3, 5]
    for i, ind in enumerate(ary):
        circle_x = (1 - body_position_array[ind, 0, 0]) * flip_color_image.shape[1]
        circle_y = body_position_array[i, 1, 0] * flip_color_image.shape[0]
        joint_array[i, :, 0] = [circle_x, circle_y]
        
        circle_x = (1 - body_position_array[ind, 0, 1]) * flip_color_image.shape[1]
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
        
# Data Collection
def collect_data():
    iter = 0
    speed_mode = 0
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
            
            if iter % 50 == 0:
                # cv2.putText(flip_color_image, "stop 2 second", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                time.sleep(3)
                # cv2.putText(flip_color_image, "start", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            print(iter)
            processed, body_position_array = process_frame_train(color_image, depth_frame, iter, speed_mode, landmarker, SIGMA_ARRAY)
            if processed:
                iter += 1

            if iter == DATA_SIZE:
                iter = 0
                speed_mode += 1
                if speed_mode == len(MOTION_SPEED):
                    break
                time.sleep(5)
                print(f"{MOTION_SPEED[speed_mode]} start!")
                
            display_data(body_position_array, flip_color_image)

            cv2.imshow('RGB Image', flip_color_image)
            
            # cv2.putText(flip_color_image, str(motion_array[iter-1, [1,5], speed_mode]),  
            #     (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        pipe.stop()
        cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    # Collect data
    collect_data()

    # Reshape and prepare data for training
    # flip_motion_array = np.flip(motion_array, axis=0)
    # data = np.concatenate((motion_array, flip_motion_array), axis=0)
    
    data = motion_array[:,:,0]
    
    # data = np.concatenate((data[:,:,0], data[:,:,1], data[:,:,2]), axis=0)
    

    np.savetxt(os.path.join(DATA_SAVE_PATH, 'stop_gesture1.txt'), data)
    
    
    plt.plot(np.arange(1, DATA_SIZE+1), data[:, -1])
    plt.title("z axis of thumb (when not rotated)")
    plt.show()
    
    # frame_num = 30
    # data_shape = data.shape
    # data_array = data.reshape(-1, data.shape[1] * frame_num)
    # label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_shape[0] / frame_num / len(MOTION_SPEED)))])
    
    # scaler = StandardScaler()
    # motion_scale = scaler.fit_transform(data_array)
    
    # # reduce 
    # pca = PCA(n_components=0.95)
    # motion_reduced = pca.fit_transform(motion_scale)

    # # Train SVM model with GridSearchCV
    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    # X_train, X_test, y_train, y_test = train_test_split(motion_reduced, label_array, random_state=0)

    # # grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    # # grid_search.fit(X_train, y_train)
    
    # # best_model = grid_search.best_estimator_
    # # score = best_model.score(X_test, y_test)
    # # print(f'Best model accuracy: {score}')
    # # print(best_model)

    # # Random Forest
    # forest = RandomForestClassifier(n_estimators=100)
    # forest.fit(X_train, y_train)
    # selector = SelectFromModel(forest, threshold="median")
    # X_important_train = selector.fit_transform(X_train, y_train)
    # X_important_test = selector.transform(X_test)

    # # Grid Search
    # grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    # grid_search.fit(X_important_train, y_train)

    # best_model = grid_search.best_estimator_
    # score = best_model.score(X_important_test, y_test)

    # print("X test, y test")
    # print(X_test.shape, y_test.shape)
    # print(f'Best model accuracy: {score}')
    # print(best_model)
    
    # # Save
    # pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_30_angle_900.sav'), 'wb'))

    # with open(os.path.join(SAVE_PATH, 'pca_model_30_angle.sav'), mode='wb') as f:
    #     pickle.dump(pca, f)
        
    # with open(os.path.join(SAVE_PATH, 'forest_model_30_angle.sav'), mode='wb') as f:
    #     pickle.dump(forest, f)
        
    # with open(os.path.join(SAVE_PATH, 'motion_model_30_angle.sav'), mode='wb') as f:
    #     pickle.dump(best_model, f, protocol=2)
