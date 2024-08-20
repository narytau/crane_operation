import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from my_module import function_math

def process_data_i(measurement, data_window, filter_size, sigma):
    if data_window.shape[0] < filter_size:
        data_window = np.vstack([data_window, measurement])
    else:
        data_window = np.roll(data_window, -1, axis=0)
        data_window[-1] = measurement

    smoothed_data = function_math.apply_gaussian_smoothing(data_window, sigma)
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
    
    def __init__(self, window_size, filter_size):
        # Initialize Variables
        self.window_size = window_size
        self.filter_size = filter_size
        
        self.sigma_array = [1, 1, 10]
        self.frame_time_stamp = 0
        self.pose_landmarks = None
        
        self.data_window = [np.zeros((0, len(self.RIGHT_BODY_INDEX) * 2))] * 3     
        self.motion_array = np.zeros((self.window_size, 2*4)) 
        self.body_position_array = np.zeros((len(self.RIGHT_BODY_INDEX), 3, 2))
               
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
            if motion_pred == 0:
                text_pred = "HIGH"
            elif motion_pred == 1:
                text_pred = "MIDDLE"
            elif motion_pred == 2:
                text_pred = "LOW"
            elif motion_pred == -1:
                text_pred = "Unclassified"
            else:
                text_pred = "Error!"
                
            cv2.putText(flip_color_image, text_pred, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Function to process frames
    def process_frame_main(self, frame, landmarker, depth_frame):
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
        
        # Convert to MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
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

            # Gaussian Filter
            self.data_window, body_position_filtered = process_data(self.body_position_array,
                                                                    self.data_window,
                                                                    self.filter_size,
                                                                    self.sigma_array)
            
            self.body_position_array = body_position_filtered
            
            # Normalize based on shoulder width
            shoulder_to_elbow = function_math.calculate_unit_vector(self.body_position_array[1,:,1] - self.body_position_array[0,:,1])
            elbow_to_wrist    = function_math.calculate_unit_vector(self.body_position_array[2,:,1] - self.body_position_array[1,:,1])
            wrist_to_thumb    = function_math.calculate_unit_vector(self.body_position_array[5,:,1] - self.body_position_array[2,:,1])
            wrist_to_pinky    = function_math.calculate_unit_vector(self.body_position_array[3,:,1] - self.body_position_array[2,:,1])
                                    
            # Renew the data
            self.motion_array[-1, :] = np.array([function_math.calculate_theta(shoulder_to_elbow),
                                                 function_math.calculate_theta(elbow_to_wrist),
                                                 function_math.calculate_theta(wrist_to_thumb),
                                                 function_math.calculate_theta(wrist_to_pinky),
                                                 function_math.calculate_phi(shoulder_to_elbow),
                                                 function_math.calculate_phi(elbow_to_wrist),
                                                 function_math.calculate_phi(wrist_to_thumb),
                                                 function_math.calculate_phi(wrist_to_pinky)])
            return True
        return False
        
    def set_task_path(self, TASK_PATH):
        self.task_path = TASK_PATH
    
        # Create PoseLandmarker Options
        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.task_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )
        
    def set_motion_scaler(self, motion_scaler):
        self.motion_scaler = motion_scaler
    
    def get_body_position_array(self):
        return self.body_position_array
        
    def get_motion_array(self):
        return self.motion_array
    
class NewMotionRecognition(BaseMotionRecognition):
    def __init__(self, window_size, filter_size, data_size=None):
        super().__init__(window_size, filter_size)
        self.motion_array = np.zeros((self.window_size, 2*3)) 
        if data_size is not None:
            self.motion_array_train = np.zeros(data_size, 2*3, 3)
        
        # Function to process frames
    def process_frame_unitvec(self, frame, landmarker, iter=None, speed_mode=None, depth_frame=None):
        """Compressed unit vector

        Args:
            frame (_type_): _description_
            landmarker (_type_): _description_
            depth_frame (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get depth information
        # depth_data = depth_frame.get_distance(100, 100)
        
        # Convert to MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Perform pose detection
        landmarker.detect_async(mp_image, self.frame_time_stamp)
        self.frame_time_stamp += 1
        
        # Shift the array
        self.motion_array = np.roll(self.motion_array, -1, axis=0)
        
        # Check if landmarks are available
        if self.pose_landmarks:
            for i, ind in enumerate(self.RIGHT_BODY_INDEX):
                self.body_position_array[i, :, 0] = [self.pose_world_landmarks[0][ind].x,
                                                     self.pose_world_landmarks[0][ind].y,
                                                     self.pose_world_landmarks[0][ind].z]
            for i, ind in enumerate(self.LEFT_BODY_INDEX):
                self.body_position_array[i, :, 1] = [self.pose_world_landmarks[0][ind].x,
                                                     self.pose_world_landmarks[0][ind].y,
                                                     self.pose_world_landmarks[0][ind].z]

            # # Gaussian Filter
            # self.data_window, body_position_filtered = process_data(self.body_position_array,
            #                                                         self.data_window,
            #                                                         self.filter_size,
            #                                                         self.sigma_array)
            
            # self.body_position_array = body_position_filtered
            
            # Normalize based on shoulder width
            shoulder_to_elbow = function_math.calculate_unit_vector(self.body_position_array[1,:,1] - self.body_position_array[0,:,1])
            elbow_to_wrist    = function_math.calculate_unit_vector(self.body_position_array[2,:,1] - self.body_position_array[1,:,1])
            wrist_to_thumb    = function_math.calculate_unit_vector(self.body_position_array[5,:,1] - self.body_position_array[2,:,1])
            # wrist_to_pinky    = function_math.calculate_unit_vector(self.body_position_array[3,:,1] - self.body_position_array[2,:,1])
                                    
            # Renew the data
            self.motion_array[-1, :] = np.array([shoulder_to_elbow[:2], 
                                                 elbow_to_wrist[:2],
                                                 wrist_to_thumb[:2]])
            
            if iter is not None:
                self.motion_array_train[iter, :, speed_mode] = np.array([shoulder_to_elbow[:2], 
                                                                         elbow_to_wrist[:2],
                                                                         wrist_to_thumb[:2]])
            return True
        return False   
        
    