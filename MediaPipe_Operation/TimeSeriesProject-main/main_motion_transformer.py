import os
import cv2
import sys
import torch
import copy
from torch import nn,optim 
from models.model.transformer import Transformer 
from ecg_dataset import myDataLoader
import numpy as np
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_module.BaseMotionRecognition import BaseMotionRecognition

def find_most_frequent_element(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    
    max_elements = unique_elements[counts == max_count]
    
    for elem in arr:
        if elem in max_elements:
            return elem

def num_to_class(num):
    if num == 0:
        pred = 'High'
    elif num == 1:
        pred = 'Middle'
    elif num == 2:
        pred = 'Low'
    elif num == 3:
        pred = 'Unclassified'
    else:
        pred = 'Error'
    return pred        
        
def predicition_handler(arr, pred):
    if arr[0] == 3:
        if np.all(arr == 3):
            pred = 'Unclassified'
        else:
            for num in arr:
                if num != 3:
                    pred = num_to_class(num)
                    break
    return pred
        
class RotateDetectionwithGesture(BaseMotionRecognition):
    # Import classes of MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    
    def __init__(self, data_depth, filter_size=None):
        super().__init__(data_depth, filter_size)
        self.pred_array = (np.ones(10) * 3).astype(int)
        self.handedness = None
        self.handgestures = None


    def decide_pred(self):
        self.pred_array = np.roll(self.pred_array, 1)
        self.pred_array[0] = np.argmax(self.motion_prob)
        self.handled_pred = predicition_handler(self.pred_array, self.motion_pred)
        
    # Create a gesture recognizer instance with the live stream mode:
    def print_hand_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.handedness = result.handedness
        self.handgestures = result.gestures
        
    def set_hand_task_path(self, HAND_TASK_PATH):
        self.hand_task_path = HAND_TASK_PATH
    
        # Create PoseLandmarker Options
        self.hand_options = self.GestureRecognizerOptions(
            base_options=self.BaseOptions(model_asset_path=self.hand_task_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_hand_result)
    
    def set_device(self, device):
        self.device = device            
    
    def run(self):
        is_started = False
        
        with self.PoseLandmarker.create_from_options(self.options) as landmarker, self.GestureRecognizer.create_from_options(self.hand_options) as handlandmarker:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipe.start(cfg)
            
            while True:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                # depth_frame = frames.get_depth_frame()
                color_image = np.asanyarray(color_frame.get_data())
                flip_color_image = cv2.flip(color_image, 1)
                
                # Convert to MediaPipe image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)

                handlandmarker.recognize_async(mp_image, self.frame_time_stamp)

                processed = super().process_frame_main_with_stop(mp_image, landmarker)
                
                
                # # Prepare for using model
                # theta_scaled = self.theta_scaler.transform(self.theta_scaler)
                # theta_prob = model_theta.predict_proba(theta_scaled)
                # theta_pred = model_theta.predict(theta_scaled)
                
                # if theta_prob[1,-1] > 0.7: 
                #     # motion_pred, motion_prob = predict_motion_SVM(motion_array, model_SVM, scaler, forest_model)
                #     motion_pred, motion_prob = predict_motion_NN(motion_array, model_NN, scaler)
                #     cv2.putText(flip_color_image, str(np.round(motion_prob * 100)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # else:
                #     motion_pred = None
                
                # Before start
                if is_started == False:
                    cv2.putText(flip_color_image, "Start operation", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if processed == True:
                    is_started = True
                    super().predict_motion()
                    # super().display_data(flip_color_image, self.motion_pred)
                    super().display_data(flip_color_image)
                    cv2.putText(flip_color_image, str(np.round(self.motion_prob * 100)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(flip_color_image, "Direction", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.putText(flip_color_image, str(np.round(self.direction, 2)), (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    self.decide_pred()
                    print(self.pred_array, self.handled_pred)
                    cv2.putText(flip_color_image, self.handled_pred, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # print(self.motion_pred_with_array)

                elif is_started and processed and self.theta_array[1, -1] <= -30:
                    self.motion_pred = 'Unclassified'
                    super().display_data(flip_color_image, self.motion_pred)
                
                cv2.imshow('RGB Image', flip_color_image)

                self.detect_stop()
                print(self.stop_signal)
                if self.handedness and self.handgestures:
                    if self.handedness[0][0].category_name == 'Right' and self.handgestures[0][0].category_name == 'Thumb_Up':
                        break
                
                if self.stop_signal:
                    print("----------------------------------------------------------------")
                    print("The system has detected a motion to stop the operation.")
                    print("Finished!")
                    break

                if cv2.waitKey(5) & 0xFF == 27:
                    print("----------------------------------------------------------------")
                    print("The Escape key has been pressed, and the operation will stop.")
                    print("Finished!")
                    break
                
            pipe.stop()
            cv2.destroyAllWindows()


# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
HAND_TASK_PATH = os.path.join(BASE_PATH, "recognizer", "gesture_recognizer.task")
MODEL_PATH = os.path.join(BASE_PATH, "motion_previous_model", "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")

frame_num = 30

stop_motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'stop_scaler_SVM.sav'), 'rb'))
stop_model_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'SVM_model_stop.sav'), 'rb'))

rotate_detection_with_gesture = RotateDetectionwithGesture(data_depth=frame_num)
rotate_detection_with_gesture.set_task_path(TASK_PATH)
rotate_detection_with_gesture.set_hand_task_path(HAND_TASK_PATH)
rotate_detection_with_gesture.set_stop_model(stop_scaler=stop_motion_scaler, model_stop=stop_model_SVM)

# definition of model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_len = frame_num # sequence length of time series
max_len = 1000 # max time series sequence length 
n_head = 4 # number of attention head
n_layer = 3 # number of encoder layer
drop_prob = 0.1
d_model = 128 # number of dimension (for positional embedding)
ffn_hidden = 512 # size of hidden layer before classification 
feature = 6 # for univariate time series (1d), it must be adjusted for 1. 
batch_size = 1

model = Transformer(d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, feature=feature, details=False, device=device).to(device=device)
model.load_state_dict(torch.load('myModel'))
model.eval()  # 評価モードに設定


motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler_transformer_slice.sav'), 'rb'))
rotate_detection_with_gesture.set_device(device=device)
rotate_detection_with_gesture.set_motion_model(motion_scaler=motion_scaler, model_transformer=model)



##############################################################################################
theta_scaler = pickle.load(open(os.path.join(MODEL_PATH, 'scaler.sav'), 'rb'))
model_theta  = pickle.load(open(os.path.join(MODEL_PATH, 'model2.pickle'), 'rb'))
rotate_detection_with_gesture.set_theta_model(theta_scaler=theta_scaler, model_theta=model_theta)
rotate_detection_with_gesture.set_skip_num(skip_num=0)
rotate_detection_with_gesture.run()

# DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")
# np.savetxt(os.path.join(DATA_SAVE_PATH, 'motion_array_all.txt'), rotate_detection_with_gesture.motion_array_all)
# np.savetxt(os.path.join(DATA_SAVE_PATH, 'pred_array_all.txt'), rotate_detection_with_gesture.pred_array_all)
