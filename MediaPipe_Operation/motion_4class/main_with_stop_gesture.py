import os
import cv2
import sys
import torch
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from class_NN import ComplexNN, ComplexNN2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_module.BaseMotionRecognition import BaseMotionRecognition

def find_most_frequent_element(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    
    max_elements = unique_elements[counts == max_count]
    
    for elem in arr:
        if elem in max_elements:
            return elem
        
def update_judge(pred, judge):
    if not pred:
        return None  

    # 初期のjudgeを設定
    judge = pred[0]

    # 状態を追跡するための変数
    counter = [0, 0, 0, 0]  # 0から3の数の出現回数を記録

    # predの値を順に処理
    for i in range(1, len(pred)):
        current = pred[i]

        # judgeとは異なる数のカウントを増やす
        if current != judge:
            counter[current] += 1

            # 異なる数が6回中4回以上現れた場合、judgeを更新
            if counter[current] >= 4:
                judge = current
                counter = [0, 0, 0, 0]  # カウンターをリセット
                counter[judge] = 1      # 新しいjudgeのカウントを初期化

    return judge

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
        
class RotateDetectionwithGesture(BaseMotionRecognition):
    # Import classes of MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    def __init__(self, data_depth, filter_size=None):
        super().__init__(data_depth, filter_size)
        self.pred_array = (np.ones(5) * 3).astype(int)
        self.handedness = None
        self.handgestures = None

    def decide_pred(self):
        self.pred_array = np.roll(self.pred_array, 1)
        self.pred_array[0] = np.argmax(self.motion_prob)
        self.motion_pred_with_array = find_most_frequent_element(self.pred_array)
        
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
                    cv2.putText(flip_color_image, "Determine the speed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if processed == True and self.theta_array[1, -1] > -30:
                    is_started = True
                    super().predict_motion()
                    super().display_data(flip_color_image, self.motion_pred)
                    cv2.putText(flip_color_image, str(np.round(self.motion_prob * 100)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(flip_color_image, "Direction", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.putText(flip_color_image, str(np.round(self.direction, 2)), (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    self.decide_pred()
                    # print(self.motion_pred_with_array)

                elif is_started and processed and self.theta_array[1, -1] <= -30:
                    self.motion_pred = 'Unclassified'
                    super().display_data(flip_color_image, self.motion_pred)
                    
                cv2.putText(flip_color_image, num_to_class(self.last_pred), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('RGB Image', flip_color_image)

                self.detect_stop()
                print(self.stop_signal)
                if self.handedness and self.handgestures:
                    if self.handedness[0][0].category_name == 'Right' and self.handgestures[0][0].category_name == 'Thumb_Up':
                        break
                
                if self.stop_signal:
                    break

                if cv2.waitKey(5) & 0xFF == 27:
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

"""
# SVM data4 (LOW cannot be detected)
motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler4.sav'), 'rb'))
model_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'SVM_model4.sav'), 'rb'))
# forest_model = pickle.load(open(os.path.join(SAVE_PATH, 'forest_model4_without.sav'), 'rb'))
rotate_detection_with_gesture.set_motion_model(motion_scaler=motion_scaler, model_SVM=model_SVM)
"""

"""
# 3 classes NN (cannot recognize the rotation)
motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler4_without.sav'), 'rb'))
input_size = 90  # 入力ベクトルの長さ
num_classes = 3  # クラスの数
# model_NN = RegularizedNN(input_size, num_classes)
model_NN = ComplexNN(input_size, num_classes, num1=76, num2=43)
model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier4_without.pth')))
model_NN.eval()
rotate_detection_with_gesture.set_motion_model(motion_scaler=motion_scaler, model_NN=model_NN)
"""


# BEST 4 classes NN (6 or 9) 6: frame 25, skip 1 || 10: frame 30, skip 1 
# motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler11.sav'), 'rb'))
motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler_with_unclass_data2.sav'), 'rb'))
input_size = 6 * frame_num  # 入力ベクトルの長さ
num_classes = 4             # クラスの数
# model_NN = RegularizedNN(input_size, num_classes)
# model_NN = ComplexNN(input_size, num_classes, num1=76, num2=43)
model_NN = ComplexNN(input_size, num_classes, num1=126, num2=75)
# model_NN = ComplexNN2(input_size, num_classes, 130, 80, 50)

# model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier11.pth')))
model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier_with_unclass_data2.pth')))
model_NN.eval()
rotate_detection_with_gesture.set_motion_model(motion_scaler=motion_scaler, model_NN=model_NN)


"""
motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler7.sav'), 'rb'))
input_size = 6 * frame_num  # 入力ベクトルの長さ
num_classes = 4             # クラスの数
# model_NN = RegularizedNN(input_size, num_classes)
model_NN = ComplexNN(input_size, num_classes, num1=76, num2=43)
model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier7.pth')))
model_NN.eval()
rotate_detection_with_gesture.set_motion_model(motion_scaler=motion_scaler, model_NN=model_NN)
"""

"""
# 3 classes --> The action without move is categorized as LOW 
motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler8.sav'), 'rb'))
input_size = 6 * frame_num  # 入力ベクトルの長さ
num_classes = 3             # クラスの数
# model_NN = RegularizedNN(input_size, num_classes)
model_NN = ComplexNN(input_size, num_classes, num1=76, num2=43)
model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier8.pth')))
model_NN.eval()
rotate_detection_with_gesture.set_motion_model(motion_scaler=motion_scaler, model_NN=model_NN)
"""

##############################################################################################
theta_scaler = pickle.load(open(os.path.join(MODEL_PATH, 'scaler.sav'), 'rb'))
model_theta  = pickle.load(open(os.path.join(MODEL_PATH, 'model2.pickle'), 'rb'))
rotate_detection_with_gesture.set_theta_model(theta_scaler=theta_scaler, model_theta=model_theta)
rotate_detection_with_gesture.set_skip_num(skip_num=1)
rotate_detection_with_gesture.run()


# # Open the model
# # Rotate
# motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler5.sav'), 'rb'))
# model_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'SVM_model3.sav'), 'rb'))
# # forest_model = pickle.load(open(os.path.join(SAVE_PATH, 'forest_model4_without.sav'), 'rb'))



# input_size = 90  # 入力ベクトルの長さ
# num_classes = 4  # クラスの数
# # model_NN = RegularizedNN(input_size, num_classes)
# model_NN = ComplexNN(input_size, num_classes, num1=76, num2=43)
# model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier5.pth')))
# model_NN.eval()

DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")
np.savetxt(os.path.join(DATA_SAVE_PATH, 'motion_array_all.txt'), rotate_detection_with_gesture.motion_array_all)
np.savetxt(os.path.join(DATA_SAVE_PATH, 'pred_array_all.txt'), rotate_detection_with_gesture.pred_array_all)
