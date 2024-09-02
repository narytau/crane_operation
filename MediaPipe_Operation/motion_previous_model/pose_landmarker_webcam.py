import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from sklearn.preprocessing import StandardScaler


# Path for model file of MediaPipe
base_path = os.path.dirname(__file__)
task_path = os.path.join(base_path, "recognizer", "pose_landmarker_full.task")
model_path = os.path.join(base_path, "theta_data")

# Import classes of MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Setting
pose_landmarks = None
pose_world_landmarks = None	
segmentation_masks = None

# Callback function to show the results
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('pose landmarker result: {}'.format(result))
    global pose_landmarks, pose_world_landmarks, segmentation_masks
    pose_landmarks = result.pose_landmarks
    pose_world_landmarks = result.pose_world_landmarks
    segmentation_masks = result.segmentation_masks
    return pose_landmarks, pose_world_landmarks, segmentation_masks

def calculate_angle(vec1, vec2):
    x = np.inner(vec1, vec2)
    theta = np.arccos(x / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return 180 * theta / np.pi

def normalize_angle(angle):
    """Function to normalize an angle to the range [-3pi/4, 5pi/4]"""
    return (angle + 3 *np.pi / 4) % (2 * np.pi) - 3 * np.pi / 4

# Set for options of Gesture Recognizer
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

right_body_index = [12, 14, 16]    # [shoulder, elbow, wrist]
left_body_index = [11, 13, 15]
horizontal_axis = [-1, 0, 0]

body_position_array = np.zeros((len(right_body_index), 3, 2))
vector_array = np.zeros((len(right_body_index),3,2))
theta_array = np.zeros((2, len(right_body_index))) # 

array_size = 15
theta_data = np.zeros((array_size,2,4))
theta_hands_up = np.zeros((array_size, 2))
theta_hands_down = np.zeros((array_size, 2))
theta_right = np.zeros((array_size, 2))
theta_left = np.zeros((array_size, 2))

iter = 0

# Open the model
with open(model_path+'model.pickle', mode='rb') as f:
    svm_model = pickle.load(f)
save_scaler = pickle.load(open(model_path+'scaler.sav', 'rb'))


# Initial value of timestamp
frame_timestamp_ms = 0

cap = cv2.VideoCapture(0)
# Create instance of Gesture Recognizer
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

            
        # Convert RGB into numpy array (480x640x3)
        color_image = image
        image_shape = color_image.shape

        # Convert RGB into  MediaPipe Image Object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

        # Recognition by Gesture recognizer
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1      
                
        # Flip the image
        flip_color_image = cv2.flip(color_image, 1)
        
        # Check if variables is not empty
        if pose_landmarks:
            
            # cv2.circle(image, center_coordinates, radius, color, thickness)  # Draw a green circle
            for i, ind in enumerate(right_body_index):
                body_position_array[i, :, 0] = [pose_landmarks[0][ind].x,
                                                pose_landmarks[0][ind].y,
                                                pose_landmarks[0][ind].z]
                
                cv2.circle(color_image, (660, 200), 10, (0, 255, 0), -1)  # Draw a green circle
                
            for i, ind in enumerate(left_body_index):
                body_position_array[i, :, 1] = [pose_landmarks[0][ind].x,
                                                pose_landmarks[0][ind].y,
                                                pose_landmarks[0][ind].z]
                
            # Display
            for i in range(len(right_body_index)):
                circle_x = int(body_position_array[i, 0, 0] * image_shape[1])
                circle_y = int(body_position_array[i, 1, 0] * image_shape[0])
                cv2.circle(color_image, (circle_x, circle_y), 8, (0, 255, 0), -1)  # Draw a green circle
                
                circle_x = int(body_position_array[i, 0, 1] * image_shape[1])
                circle_y = int(body_position_array[i, 1, 1] * image_shape[0])
                cv2.circle(color_image, (circle_x, circle_y), 8, (255, 0, 0), -1)  # Draw a blue circle
                
                
            # Calculate angle (elbow_to_wrist, shoulder_to_wrist, shoulder_to_elbow
            vector_array[0,:,0] = body_position_array[2,:,0] - body_position_array[1,:,0]
            vector_array[1,:,0] = body_position_array[2,:,0] - body_position_array[0,:,0]
            vector_array[2,:,0] = body_position_array[1,:,0] - body_position_array[0,:,0]
            vector_array[0,:,1] = body_position_array[2,:,1] - body_position_array[1,:,1]
            vector_array[1,:,1] = body_position_array[2,:,1] - body_position_array[0,:,1]
            vector_array[2,:,1] = body_position_array[1,:,1] - body_position_array[0,:,1]

            # Right            
            for i in range(len(right_body_index)):
                vec = - vector_array[i, 0:2, 0]
                angle = np.arctan2(vec[1], vec[0])
                theta_array[0, i] = 180 * normalize_angle(angle=angle) / np.pi
                
            # Left
            for i in range(len(left_body_index)):
                vec = - vector_array[i, 0:2, 1]
                angle = np.arctan2(vec[1], - vec[0])
                theta_array[1, i] = 180 * normalize_angle(angle=angle) / np.pi
                
        color_image = cv2.flip(color_image, 1)
        
        # Up-Down-Right-Left
        theta_scaled = save_scaler.transform(theta_array)
        theta_prob = svm_model.predict_proba(theta_scaled)
        theta_pred = svm_model.predict(theta_scaled)
        
        # Threshold
        threshold = 0.75
        print(theta_pred)
        
        if max(theta_prob[0,:]) < threshold:
            ans = 'unclassified'
        elif theta_pred[0] == 'Down':
            if max(theta_prob[1,:]) < threshold:
                ans = 'unclassified'
            else:
                if theta_pred[1] == 'Left':
                    ans = 'Right'
                elif theta_pred[1] == 'Right':
                    ans = 'Left'
                else:
                    ans = theta_pred[1]
        else:
            ans = theta_pred[0]
            
        cv2.putText(color_image, ans, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
   
        # Show image
        cv2.imshow('RGB Image', flip_color_image)
        
        # If the escape button is pressed, exit the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
            
# Stop pipeline
cap.release()
cv2.destroyAllWindows()