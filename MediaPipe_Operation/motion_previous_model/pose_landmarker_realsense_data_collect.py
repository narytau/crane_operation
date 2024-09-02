import os
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

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
    origin = 110 * np.pi / 180
    return (angle + origin) % (2 * np.pi) - origin

# Set for options of Gesture Recognizer
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

right_body_index = [12, 14, 16]    # [shoulder, elbow, wrist]
left_body_index = [11, 13, 15]

right_body_position_array = np.zeros((len(right_body_index),3))
left_body_position_array = np.zeros((len(left_body_index),3))
body_position_array = np.zeros((len(right_body_index), 3, 2))
vector_array = np.zeros((len(right_body_index),3,2))
theta_array = np.zeros((len(right_body_index), 2))


array_size = 15
theta_data = np.zeros((array_size, 3, 4))
labels = ['Up', 'Down', 'Right', 'Left']

iter = 0
label_iter = 0

# # Open the model
# with open(base_path+'model.pickle', mode='rb') as f:
#     svm_model = pickle.load(f)
# save_scaler = pickle.load(open(base_path+'scaler.sav', 'rb'))


cap = cv2.VideoCapture(0)
# Create instance of Gesture Recognizer
with PoseLandmarker.create_from_options(options) as landmarker:
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

        # Convert RGB into  MediaPipe Image Object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

        # Recognition by Gesture recognizer
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1      
                
        
        # Check if variables is not empty
        if pose_landmarks:
            
            # cv2.circle(image, center_coordinates, radius, color, thickness)  # Draw a green circle
            for i, ind in enumerate(right_body_index):
                # right_body_position_array[i, :] = [pose_landmarks[0][ind].x,
                #                                    pose_landmarks[0][ind].y,
                #                                    pose_landmarks[0][ind].z]
                
                body_position_array[i, :, 0] = [pose_landmarks[0][ind].x,
                                                pose_landmarks[0][ind].y,
                                                pose_landmarks[0][ind].z]
                
                cv2.circle(color_image, (660, 200), 10, (0, 255, 0), -1)  # Draw a green circle
                
                
            for i, ind in enumerate(left_body_index):
                # left_body_position_array[i, :] = [pose_landmarks[0][ind].x,
                #                                    pose_landmarks[0][ind].y,
                #                                    pose_landmarks[0][ind].z]
                
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
            
            for i in range(len(right_body_index)):
                for j in range(2):
                    # theta_array[i, j] = calculate_angle(vector_array[i, 0:2, j], horizontal_axis[0:2])
                    vec = - vector_array[i, 0:2, j]  # Direction is changed for theta
                    angle = np.arctan2(vec[1]/np.linalg.norm(vec), vec[0]/np.linalg.norm(vec))
                    theta_array[i, j] = 180 * normalize_angle(angle=angle) / np.pi
            
        color_image = cv2.flip(color_image, 1)
        
                
        # Show image
        cv2.imshow('RGB Image', color_image)
        
        # If the escape button is pressed, exit the loop
        if cv2.waitKey(5) & 0xFF == 27:
            print("%s: %d" % (labels[label_iter], iter))
            theta_data[iter, :, label_iter] = theta_array[:, 0]
            print(angle)
            print(theta_array[:,0])
            iter += 1
            if iter == array_size:
                iter = 0
                label_iter += 1
                if label_iter == 4:
                    np.savetxt('C:\\Users\\nakamura\\Downloads\\Stuttgart_git\\MediaPipe_Operation\\theta_data\\theta_data_ver2.txt', 
                               theta_data.flatten())
                    print("C:\\Users\\nakamura\\Downloads\\Stuttgart_git\\MediaPipe_Operation\\theta_data\\theta_data_ver2.txt")
                    break
            
             
            
# Stop pipeline
pipe.stop()
cv2.destroyAllWindows()