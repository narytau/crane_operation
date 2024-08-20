import os
import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn
import mediapipe as mp
import pyrealsense2 as rs
from BaseMotionRecognition import BaseMotionRecognition
from class_NN import ComplexNN

def find_most_frequent_element(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    
    max_elements = unique_elements[counts == max_count]
    
    for elem in arr:
        if elem in max_elements:
            return elem
        
# Function to classify with variable seq_length
def classify_with_variable_seq_length(model, data):
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add batch dimension
    with torch.no_grad():
        inputs = tensor_data  # Use the first seq_length samples
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()
        
# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.fc(output)
        return output


class RotateDetection(BaseMotionRecognition):
    def __init__(self, data_depth, filter_size=None):
        super().__init__(data_depth, filter_size)
        self.pred_array = (np.ones(5) * 3).astype(int)
        
    def decide_pred(self):
        self.pred_array = np.roll(self.pred_array, 1)
        self.pred_array[0] = np.argmax(self.motion_prob)
        self.motion_pred_with_array = find_most_frequent_element(self.pred_array)
        
        
    def run(self):
        iter = 0
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
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

                processed = super().process_frame_main(mp_image, landmarker)
                self.frame_time_stamp += 1
                
                if processed:
                    iter += 1
                    
                
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

                if self.theta_array[1, -1] > 0:
                    predicted_class = classify_with_variable_seq_length(model, self.motion_array)
                    cv2.putText(flip_color_image, str(predicted_class), (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # super().predict_motion()
                    # super().display_data(flip_color_image, self.motion_pred)
                    # cv2.putText(flip_color_image, str(np.round(self.motion_prob * 100)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    # cv2.putText(flip_color_image, str(np.round(self.direction, 2)), (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # self.decide_pred()
                    # cv2.putText(flip_color_image, str(self.motion_pred_with_array), (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    
                cv2.imshow('RGB Image', flip_color_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                
            pipe.stop()
            cv2.destroyAllWindows()

# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")


rotatedetection = RotateDetection(data_depth=20)
rotatedetection.set_task_path(TASK_PATH)

# Open the model
# Rotate
motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler5.sav'), 'rb'))
model_SVM = pickle.load(open(os.path.join(SAVE_PATH, 'SVM_model3.sav'), 'rb'))
# # forest_model = pickle.load(open(os.path.join(SAVE_PATH, 'forest_model4_without.sav'), 'rb'))

# theta_scaler = pickle.load(open(os.path.join(MODEL_PATH, 'scaler.sav'), 'rb'))
# model_theta  = pickle.load(open(os.path.join(MODEL_PATH, 'model2.pickle'), 'rb'))

# input_size = 90  # 入力ベクトルの長さ
# num_classes = 4  # クラスの数
# # model_NN = RegularizedNN(input_size, num_classes)
# model_NN = ComplexNN(input_size, num_classes, num1=76, num2=43)
# model_NN.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier5.pth')))
# model_NN.eval()

# Load the saved model
input_dim = 6
nhead = 2
num_encoder_layers = 3
dim_feedforward = 128
output_dim = 3

MODEL_SAVE_PATH = os.path.join(SAVE_PATH, 'gesture_classifier_transformer.pth')
model = TransformerModel(input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()


# rotatedetection.set_motion_model(motion_scaler=motion_scaler, model_NN=model_NN)
# rotatedetection.set_theta_model(theta_scaler=theta_scaler, model_theta=model_theta)
rotatedetection.run()