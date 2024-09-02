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
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(CURRENT_PATH, "theta_data")

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

# def normalize_angle(angle):
#     """Function to normalize an angle to the range [-3pi/4, 5pi/4]"""
#     return (angle + 3 *np.pi / 4) % (2 * np.pi) - 3 * np.pi / 4

def normalize_angle(angle):
    """Function to normalize an angle to the range [-3pi/4, 5pi/4]"""
    origin = 110 * np.pi / 180
    return (angle + origin) % (2 * np.pi) - origin

def calculate_percentage(start, end, position):
    return max(0, min(100, ((position - start) / (end - start)) * 100))

# Set for options of Gesture Recognizer
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

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
with open(os.path.join(MODEL_PATH, "model2.pickle"), mode='rb') as f:
    svm_model = pickle.load(f)
save_scaler = pickle.load(open(os.path.join(MODEL_PATH, "scaler.sav"), 'rb'))

# Create instance of Gesture Recognizer
with PoseLandmarker.create_from_options(options) as landmarker:
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
        frame_timestamp_ms += 1      
                
        # Flip the image
        flip_color_image = cv2.flip(color_image, 1)
        
        # Check if variables is not empty
        if pose_landmarks:
            for i, ind in enumerate(right_body_index):
                body_position_array[i, :, 0] = [pose_landmarks[0][ind].x,
                                                pose_landmarks[0][ind].y,
                                                pose_landmarks[0][ind].z]
                
            for i, ind in enumerate(left_body_index):
                body_position_array[i, :, 1] = [pose_landmarks[0][ind].x,
                                                pose_landmarks[0][ind].y,
                                                pose_landmarks[0][ind].z]
                
            '''
            DISPLAY
            Draw circle: 
                cv2.circle(image, center_coordinates, radius, color, thickness)
            Draw line:   
                cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
            '''

            for i in range(len(right_body_index)):
                circle_x = (1 - body_position_array[i, 0, 0]) * image_shape[1]
                circle_y = body_position_array[i, 1, 0] * image_shape[0]
                joint_array[i, :, 0] = [circle_x, circle_y]
                
                circle_x = (1 - body_position_array[i, 0, 1]) * image_shape[1]
                circle_y = body_position_array[i, 1, 1] * image_shape[0]
                joint_array[i, :, 1] = [circle_x, circle_y]
                
            joint_array = joint_array.astype(int)
            
            # circles    
            for i in range(len(right_body_index)):
                cv2.circle(flip_color_image, (joint_array[i,0,0], joint_array[i,1,0]), 5, (2, 127, 0), -1)  # Draw a green circle
                cv2.circle(flip_color_image, (joint_array[i,0,1], joint_array[i,1,1]), 5, (255, 39, 0), -1)  # Draw a blue circle
                
            # lines
            cv2.line(flip_color_image, (joint_array[1,0,0], joint_array[1,1,0]), (joint_array[2,0,0], joint_array[2,1,0]),
                     (2, 127, 0), thickness=2, lineType=cv2.LINE_8, shift=0)
            cv2.line(flip_color_image, (joint_array[1,0,1], joint_array[1,1,1]), (joint_array[2,0,1], joint_array[2,1,1]),
                     (255, 39, 0), thickness=2, lineType=cv2.LINE_8, shift=0)
                
            # coordinates
            cv2.putText(flip_color_image, str(np.round(body_position_array[1,:,0],2)),  
                        (joint_array[0,0,0], joint_array[0,1,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(flip_color_image, str(np.round(body_position_array[2,:,0],2)),  
                        (joint_array[1,0,0], joint_array[1,1,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(flip_color_image, str(np.round(body_position_array[1,:,1],2)),  
                        (joint_array[0,0,1], joint_array[0,1,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(flip_color_image, str(np.round(body_position_array[2,:,1],2)),  
                        (joint_array[1,0,1], joint_array[1,1,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            
            # Calculate angle (elbow_to_wrist, shoulder_to_wrist, shoulder_to_elbow)
            #                 (beta, gamma, alpha)
            vector_array[0,:,0] = body_position_array[2,:,0] - body_position_array[1,:,0]
            vector_array[1,:,0] = body_position_array[2,:,0] - body_position_array[0,:,0]
            vector_array[2,:,0] = body_position_array[1,:,0] - body_position_array[0,:,0]
            vector_array[0,:,1] = body_position_array[2,:,1] - body_position_array[1,:,1]
            vector_array[1,:,1] = body_position_array[2,:,1] - body_position_array[0,:,1]
            vector_array[2,:,1] = body_position_array[1,:,1] - body_position_array[0,:,1]
            
            # for i in range(len(right_body_index)):
            #     for j in range(2):
            #         # theta_array[i, j] = calculate_angle(vector_array[i, 0:2, j], horizontal_axis[0:2])
            #         vec = - vector_array[i, 0:2, j]  # Direction is changed for theta
            #         angle = np.arctan2(vec[1]/np.linalg.norm(vec), vec[0]/np.linalg.norm(vec))
            #         theta_array[i, j] = 180 * normalize_angle(angle=angle) / np.pi

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
            
            # Array for center of the body  
            for ind, body_ind in enumerate(center_body_index):
                body_center_array[ind, :] = np.array([pose_landmarks[0][body_ind].x,
                                                    pose_landmarks[0][body_ind].y,
                                                    pose_landmarks[0][body_ind].z])
                                                    
            # Compensation for human tilt and camera tilt
            vec1 = body_center_array[0,:2] - body_center_array[3,:2]
            vec2 = body_center_array[1,:2] - body_center_array[2,:2]
            vec = - (vec1 + vec2) / 2
            tilt = (np.pi / 2 - np.arctan2(vec[1], vec[0])) * 180 / np.pi
            theta_array += tilt
            
            # Warning when human is close to camera
            
            # print(np.array(body_center_array[:,:2]) > 0 , np.array(body_center_array[:,:2]) < 1) 
            center_point = np.mean(body_center_array[:,:2], 0)
            center_x = (1 - center_point[0]) * image_shape[1]
            center_y = center_point[1] * image_shape[0]
            # depth_data = depth_frame.get_distance(int(center_x), int(center_y))

            # if np.mean(body_center_array[:,2]) < distance_threshold:
            #     cv2.putText(flip_color_image, str(round(np.mean(body_center_array[:,2]), 4)), (30, 80), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(flip_color_image, str(round(pose_world_landmarks[0][15].z, 4)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 39, 0), 2)

        # Prepare for using model
        theta_scaled = save_scaler.transform(theta_array)
        theta_prob = svm_model.predict_proba(theta_scaled)
        theta_pred = svm_model.predict(theta_scaled)
        
        # Right-hand recognition
        if max(theta_prob[0,:]) < recog_threshold:
            ans_right = 'unclassified'
        elif theta_pred[0] == 'Up':
            ans_right = 'Up'
            perc_right =  calculate_percentage(0, 75, theta_array[0,2])
        elif theta_pred[0] == 'Right':
            ans_right = 'Right'
            perc_right =  calculate_percentage(-60, 0, theta_array[0,2])
        elif theta_pred[0] == 'Left':
            ans_right = 'Left'
            perc_right =  calculate_percentage(240, 200, theta_array[0,2])
        elif theta_pred[0] == 'Down':
            ans_right = 'Down'
            perc_right =  calculate_percentage(240, 200, theta_array[0,2])
        else:
            print("Error (Right hand)")
            
        # Left-hand recognition
        if max(theta_prob[1,:]) < recog_threshold:
            ans_left = 'unclassified'
        elif theta_pred[1] == 'Up':
            ans_left = 'Up'
            perc_left =  calculate_percentage(0, 75, theta_array[1,2])
        elif theta_pred[1] == 'Right':
            ans_left = 'Left'
            perc_left =  calculate_percentage(-60, 0, theta_array[1,2])
        elif theta_pred[1] == 'Left':
            ans_left = 'Right'
            perc_left =  calculate_percentage(240, 200, theta_array[1,2])
        elif theta_pred[1] == 'Down':
            ans_left = 'Down'
            perc_left =  calculate_percentage(240, 200, theta_array[1,2])        
        else:
            print("Error (Left hand)")   
            

        # if max(theta_prob[0,:]) < recog_threshold:
        #     ans = 'unclassified'
        # elif theta_pred[0] == 'Up':
        #     ans = 'Up'
        #     perc = calculate_percentage(0, 75, theta_array[0,2])
        # elif theta_pred[0] == 'Right':
        #     ans = 'Right'
        #     perc = calculate_percentage(-60, 0, theta_array[0,2])
        # elif theta_pred[0] == 'Left':
        #     ans = 'Left'
        #     perc = calculate_percentage(240, 200, theta_array[0,2])
        # elif theta_pred[0] == 'Down':
        #     if max(theta_prob[1,:]) < recog_threshold:
        #         ans = 'unclassified'
        #     else:
        #         if theta_pred[1] == 'Right':
        #             ans = 'Left'
        #             perc = calculate_percentage(-60, 0, theta_array[1,2])
        #         elif theta_pred[1] == 'Left':
        #             ans = 'Right'
        #             perc = calculate_percentage(240, 200, theta_array[1,2])
        #         elif theta_pred[1] == 'Up':                    
        #             ans = 'Up'
        #             perc = calculate_percentage(0, 75, theta_array[1,2])
        #         elif theta_pred[1] == 'Down':
        #             ans = 'Down'
        #             perc = calculate_percentage(-65, -85, theta_array[0,2])
        #         else:
        #             print("Error!")
        # else:
        #     print("Error!")
            
        # if ans != 'unclassified':
        #     cv2.putText(flip_color_image, str(round(perc)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 39, 0), 2)
        # cv2.putText(flip_color_image, ans, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 39, 0), 2)
        # cv2.putText(flip_color_image, str(round(depth_data,4)), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 39, 0), 2)
        
        
        # Right hand
        if ans_right == 'unclassified':
            cv2.putText(flip_color_image, ans_right, (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 127, 0), 2)
        else:
            cv2.putText(flip_color_image, str(round(perc_right)), (520, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 127, 0), 2)
            cv2.putText(flip_color_image, ans_right, (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 127, 0), 2)

        # Left hand
        if ans_left != 'unclassified':
            cv2.putText(flip_color_image, str(round(perc_left)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 39, 0), 2)
        cv2.putText(flip_color_image, ans_left, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 39, 0), 2)
        
        
        
        # Show image
        cv2.imshow('RGB Image', flip_color_image)
        
        # If the escape button is pressed, exit the loop
        if cv2.waitKey(5) & 0xFF == 27:
            print("Stop")
            break

# Stop pipeline
pipe.stop()
cv2.destroyAllWindows()