import os
import cv2
import sys
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from sklearn.feature_selection import SelectFromModel
from my_module.function_math import apply_gaussian_smoothing, normalize_angle, calculate_unit_vector

def process_data_i(measurement, data_window, filter_size, sigma):
    if data_window.shape[0] < filter_size:
        data_window = np.vstack([data_window, measurement])
    else:
        data_window = np.roll(data_window, -1, axis=0)
        data_window[-1] = measurement

    smoothed_data = apply_gaussian_smoothing(data_window, sigma)
    current_smoothed_value = smoothed_data[-1]
    
    return data_window, current_smoothed_value

def process_data(body_position_array, data_window, filter_size, sigma_array):
    body_position_filtered = np.zeros_like(body_position_array)
    # x,y,z
    for i in range(3):
        measurement = body_position_array[:,i,:].ravel()
        data_window[i], current_smoothed_value = process_data_i(measurement, data_window[i], filter_size, sigma_array[i])
        body_position_filtered[:,i,:] = current_smoothed_value.reshape(body_position_array.shape[0], 
                                                                    body_position_array.shape[2])
        
    return data_window, body_position_filtered

def process_theta_array(body_position_array, pose_landmarks, RIGHT_BODY_INDEX, LEFT_BODY_INDEX, CENTER_BODY_INDEX):
    # Initialize
    vector_array = np.zeros_like(body_position_array)
    body_center_array = np.zeros((len(CENTER_BODY_INDEX), 3))
    theta_array = np.zeros((2, len(RIGHT_BODY_INDEX[:3])))

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
        theta_array[0, i] = 180 * normalize_angle(angle=angle, origin=110*np.pi/180) / np.pi
        
    # Left
    for i in range(len(LEFT_BODY_INDEX[:3])):
        vec = - vector_array[i, 0:2, 1]
        angle = np.arctan2(vec[1], - vec[0])
        theta_array[1, i] = 180 * normalize_angle(angle=angle, origin=110*np.pi/180) / np.pi

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
    
    # The direction crane should move (Right hand)
    direction = calculate_unit_vector(vector_array[0, :, 0])
    
    return theta_array, direction

def predict_motion_NN(motion_array, model, scaler):
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    scaled_flatten_motion_array = scaler.transform(flatten_motion_array)

    input_tensor = torch.tensor(scaled_flatten_motion_array, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)

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

def predict_motion_transformer(motion_array, model, scaler, device):
    scaled_motion_array = scaler.transform(motion_array)

    # arrange shape of array
    motion_array_tensor = np.zeros((1, motion_array.shape[0], 1, motion_array.shape[1]))
    motion_array_tensor[0, :, 0, :] = scaled_motion_array
    motion_array_tensor = torch.tensor(motion_array_tensor, dtype=torch.float32).to(device) # (1, frame_num, feature)
    with torch.no_grad():
        output = model(motion_array_tensor)
    
    motion_prob = F.softmax(output.squeeze(), dim=-1).numpy()[:4]
    motion_pred = process_predcition(motion_prob)
    return motion_pred, motion_prob
    

def detect_stop_motion_SVM(motion_array, model, scaler):
    stop_signal = False    
    
    flatten_motion_array = motion_array.reshape(1, motion_array.size)
    scaled_flatten_motion_array = scaler.transform(flatten_motion_array)

    # motion_pred = model.predict(scaled_flatten_motion_array)
    motion_prob = model.predict_proba(scaled_flatten_motion_array)
    print(motion_prob)
    if motion_prob[0][0] > 0.8:
        stop_signal = True    
    return stop_signal

def process_predcition(probability):
    threshold = 0.75
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

class BaseMotionRecognition():
    # Body Index Constants
    RIGHT_BODY_INDEX  = [12, 14, 16, 18, 20, 22]
    LEFT_BODY_INDEX   = [11, 13, 15, 17, 19, 21]
    CENTER_BODY_INDEX = [11, 12, 23, 24]

    # MediaPipe Imports
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    def __init__(self, data_depth=15, filter_size=None, stop_data_depth=50):
        """_summary_

        Args:
            data_depth (int, optional): Same as frame_num. Defaults to 15.
            filter_size (int, optional): To use Gaussian Filter, input the size. Defaults to None.
        """
        # Initialize Variables
        self.data_depth = data_depth
        self.filter_size = filter_size
        
        self.sigma_array = [1, 1, 10]
        self.frame_time_stamp = 0
        self.pose_landmarks = None
        self.last_pred = 3
        
        self.data_window = [np.zeros((0, len(self.RIGHT_BODY_INDEX) * 2))] * 3     
        self.motion_array = np.zeros((self.data_depth, 2*3)) 
        self.stop_motion_array = np.zeros((stop_data_depth, 2*4))
        self.body_position_array = np.zeros((len(self.RIGHT_BODY_INDEX), 3, 2))
        self.theta_array = np.zeros((2, len(self.RIGHT_BODY_INDEX[:3])))
        
        self.motion_scaler = None
        self.model_SVM = None
        self.model_NN = None
        self.forest_model = None
        
        self.model_theta = None
        self.theta_scaler = None
        
        self.motion_array_all = np.zeros(6)
        self.pred_array_all = np.zeros(4)
        self.pred_array2 = np.ones(10) * 3

    # Callback Function for PoseLandmarker
    def print_result(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.pose_landmarks = result.pose_landmarks
        self.pose_world_landmarks = result.pose_world_landmarks
        
    def display_data(self, flip_color_image, motion_pred=None):
        '''
        DISPLAY
        Draw circle: 
            cv2.circle(image, center_coordinates, radius, color, thickness)
        Draw line:   
            cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
        '''
        joint_array = np.zeros((3, 2, 2))
        for i in range(joint_array.shape[0]):
            circle_x = (1 - self.body_position_array[i, 0, 0]) * flip_color_image.shape[1]
            circle_y = self.body_position_array[i, 1, 0] * flip_color_image.shape[0]
            joint_array[i, :, 0] = [circle_x, circle_y]
            
            circle_x = (1 - self.body_position_array[i, 0, 1]) * flip_color_image.shape[1]
            circle_y = self.body_position_array[i, 1, 1] * flip_color_image.shape[0]
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
    
    # Function to process frames
    def process_frame_main(self, mp_image, landmarker):
        """2 Angles and Gaussian Filter

        Args:
            frame (_type_): _description_
            landmarker (_type_): _description_
            depth_frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get depth information
        # depth_data = depth_frame.get_distance(100, 100)
        
        # Perform pose detection
        landmarker.detect_async(mp_image, self.frame_time_stamp)
        self.frame_time_stamp += 1
        
        # Shift the array
        self.motion_array = np.roll(self.motion_array, -1, axis=0)        
        
        # Check if landmarks are available
        if self.pose_landmarks:
            for i, ind in enumerate(self.RIGHT_BODY_INDEX):
                self.body_position_array[i, :, 0] = [self.pose_landmarks[0][ind].x,
                                                    self.pose_landmarks[0][ind].y,
                                                    self.pose_landmarks[0][ind].z]
                
            for i, ind in enumerate(self.LEFT_BODY_INDEX):
                self.body_position_array[i, :, 1] = [self.pose_landmarks[0][ind].x,
                                                    self.pose_landmarks[0][ind].y,
                                                    self.pose_landmarks[0][ind].z]

            # # # Gaussian Filter
            # if self.filter_size is not None:
            #     self.data_window, body_position_filtered = process_data(self.body_position_array,
            #                                                             self.data_window,
            #                                                             self.filter_size,
            #                                                             self.sigma_array)
            #     self.body_position_array = body_position_filtered
                
            # Normalize based on shoulder width
            shoulder_to_elbow = calculate_unit_vector(self.body_position_array[1,:,1] - self.body_position_array[0,:,1])
            elbow_to_wrist    = calculate_unit_vector(self.body_position_array[2,:,1] - self.body_position_array[1,:,1])
            wrist_to_thumb    = calculate_unit_vector(self.body_position_array[5,:,1] - self.body_position_array[2,:,1])
                                    
            # Renew the data
            if self.frame_time_stamp % self.skip_num == 0:
                self.motion_array[-1, :] = np.concatenate([shoulder_to_elbow[:2], 
                                                            elbow_to_wrist[:2],
                                                            wrist_to_thumb[:2]])
                
                self.motion_array_all = np.vstack([self.motion_array_all, self.motion_array[-1, :]])
            
            self.theta_array, self.direction = process_theta_array(self.body_position_array, self.pose_landmarks, 
                                                self.RIGHT_BODY_INDEX, self.LEFT_BODY_INDEX, self.CENTER_BODY_INDEX)
            return True
        return False
    
        # Function to process frames
    def process_frame_main_with_stop(self, mp_image, landmarker):
        """2 Angles and Gaussian Filter

        Args:
            frame (_type_): _description_
            landmarker (_type_): _description_
            depth_frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get depth information
        # depth_data = depth_frame.get_distance(100, 100)
        
        # Perform pose detection
        landmarker.detect_async(mp_image, self.frame_time_stamp)
        self.frame_time_stamp += 1
        
        # Shift the array
        self.motion_array = np.roll(self.motion_array, -1, axis=0)        
        self.stop_motion_array = np.roll(self.stop_motion_array, -1, axis=0)        
        
        # Check if landmarks are available
        if self.pose_landmarks:
            for i, ind in enumerate(self.RIGHT_BODY_INDEX):
                self.body_position_array[i, :, 0] = [self.pose_landmarks[0][ind].x,
                                                    self.pose_landmarks[0][ind].y,
                                                    self.pose_landmarks[0][ind].z]
                
            for i, ind in enumerate(self.LEFT_BODY_INDEX):
                self.body_position_array[i, :, 1] = [self.pose_landmarks[0][ind].x,
                                                    self.pose_landmarks[0][ind].y,
                                                    self.pose_landmarks[0][ind].z]

            # # # Gaussian Filter
            # if self.filter_size is not None:
            #     self.data_window, body_position_filtered = process_data(self.body_position_array,
            #                                                             self.data_window,
            #                                                             self.filter_size,
            #                                                             self.sigma_array)
            #     self.body_position_array = body_position_filtered
                
            # Normalize based on shoulder width
            shoulder_to_elbow_right = calculate_unit_vector(self.body_position_array[1,:,0] - self.body_position_array[0,:,0])
            shoulder_to_elbow_left  = calculate_unit_vector(self.body_position_array[1,:,1] - self.body_position_array[0,:,1])
            elbow_to_wrist_right    = calculate_unit_vector(self.body_position_array[2,:,0] - self.body_position_array[1,:,0])
            elbow_to_wrist_left     = calculate_unit_vector(self.body_position_array[2,:,1] - self.body_position_array[1,:,1])
            wrist_to_thumb_right    = calculate_unit_vector(self.body_position_array[5,:,0] - self.body_position_array[2,:,0])
            wrist_to_thumb_left     = calculate_unit_vector(self.body_position_array[5,:,1] - self.body_position_array[2,:,1])
                                    
            # Renew the data
            if self.frame_time_stamp % (self.skip_num + 1) == 0:
                self.motion_array[-1, :] = np.concatenate([shoulder_to_elbow_left[:2], 
                                                            elbow_to_wrist_left[:2],
                                                            wrist_to_thumb_left[:2]])
                
                self.stop_motion_array[-1, :] = np.concatenate([shoulder_to_elbow_right[:2],
                                                            shoulder_to_elbow_left[:2], 
                                                            elbow_to_wrist_right[:2],
                                                            elbow_to_wrist_left[:2]])
                
                self.motion_array_all = np.vstack([self.motion_array_all, self.motion_array[-1, :]])
            
            self.theta_array, self.direction = process_theta_array(self.body_position_array, self.pose_landmarks, 
                                                self.RIGHT_BODY_INDEX, self.LEFT_BODY_INDEX, self.CENTER_BODY_INDEX)
            return True
        return False
    
    def predict_motion(self):
        if self.model_SVM is not None and self.model_NN is None and self.model_transformer is None:
            self.motion_pred, self.motion_prob = predict_motion_SVM(motion_array=self.motion_array, 
                                                        model=self.model_SVM, 
                                                        scaler=self.motion_scaler, 
                                                        forest_model=self.forest_model)
            if self.motion_pred != "Unclassified":
                self.last_pred = self.motion_pred
                
        elif self.model_SVM is None and self.model_NN is not None and self.model_transformer is None:
            self.motion_pred, self.motion_prob = predict_motion_NN(motion_array=self.motion_array,
                                                        model=self.model_NN,
                                                        scaler=self.motion_scaler)

            self.pred_array_all = np.vstack([self.pred_array_all, self.motion_prob])
            
        elif self.model_SVM is None and self.model_NN is None and self.model_transformer is not None:
            self.motion_pred, self.motion_prob = predict_motion_transformer(motion_array=self.motion_array,
                                                        model=self.model_transformer,
                                                        scaler=self.motion_scaler,
                                                        device=self.device)            
            
            print("before")
            print(self.pred_array2)
            self.pred_array2 = np.roll(self.pred_array2, -1)
            self.pred_array2[-1] = np.argmax(self.motion_prob)
            
            # when all elements are same and they are different from last pred
            print(self.pred_array2)
            
            if len(np.unique(self.pred_array2)) == 1 and self.last_pred != self.pred_array2[0]:
                if self.pred_array2[0] != 3:
                    self.last_pred = self.pred_array2[0]
                
            if self.motion_pred != "Unclassified":
                self.last_pred = np.argmax(self.motion_prob)
        else:
            print("Please input the model.")
            
    def detect_stop(self):
        if self.model_stop is not None:
            self.stop_signal = detect_stop_motion_SVM(self.stop_motion_array, self.model_stop, self.stop_scaler)
        else:
            print("Please input the stop model")
        
    def set_task_path(self, TASK_PATH):
        self.task_path = TASK_PATH
    
        # Create PoseLandmarker Options
        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.task_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )
        
    def set_motion_model(self, motion_scaler, model_SVM=None, model_NN=None, model_transformer=None, forest_model=None):
        if model_SVM is None and model_NN is None and model_transformer is None:
            print("Please input the model.")
        else:
            self.motion_scaler = motion_scaler
            self.model_SVM = model_SVM
            self.model_NN = model_NN
            self.model_transformer = model_transformer
            self.forest_model = forest_model
            
    def set_theta_model(self, theta_scaler, model_theta):
        self.theta_scaler = theta_scaler
        self.model_theta = model_theta
    
    def set_stop_model(self, stop_scaler, model_stop):
        self.stop_scaler = stop_scaler
        self.model_stop = model_stop
        
    def set_skip_num(self, skip_num=1):
        self.skip_num = skip_num

    def get_body_position_array(self):
        return self.body_position_array
        
    def get_motion_array(self):
        return self.motion_array