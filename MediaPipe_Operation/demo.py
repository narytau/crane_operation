import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from BaseMotionRecognition import BaseMotionRecognition

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig

def real_time_inference(model, scaler, data, window_size=30):
    data = scaler.transform(data)
    dataset = RealTimeTimeSeriesDataset(data, window_size=window_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # predictions = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            probs = softmax(outputs)
            probabilities = probs.squeeze().tolist()
            if max(probabilities) > 0.8:
                predicted = np.argmax(probabilities)
            else:
                predicted = -1
            # predictions.append(predicted.item())
    return predicted

class RealTimeTimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=30):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_length=30):
        super(TimeSeriesTransformer, self).__init__()
        config = BertConfig(
            vocab_size=1,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=seq_length,  # 30の長さのシーケンス
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.linear = nn.Linear(input_dim, config.hidden_size)
        self.transformer = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, num_classes)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.transformer(inputs_embeds=x).last_hidden_state
        x = x.mean(dim=1)  # (batch_size, hidden_size) 時系列データ全体の平均をとる
        output = self.fc(x)  # (batch_size, num_classes)
        return output

class MotionRecognition(BaseMotionRecognition):
    def __init__(self, window_size, filter_size, model=None):
        super().__init__(window_size, filter_size)
        self.iter = 0
        self.model = model
    
    def run(self):
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipe.start(cfg)
            
            while True:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                color_image = np.asanyarray(color_frame.get_data())
                flip_color_image = cv2.flip(color_image, 1)

                processed = self.process_frame_main(color_image, landmarker, depth_frame)
                
                if processed:
                    self.iter += 1

                if self.model is not None and len(self.motion_array) >= self.window_size:
                    predictions = real_time_inference(self.model, self.motion_scaler, self.motion_array, self.window_size)
                    
                self.display_data(flip_color_image, predictions)
                
                cv2.imshow('RGB Image', flip_color_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            pipe.stop()
            cv2.destroyAllWindows()
            
            
def main():
    # Constants and Paths
    BASE_PATH = os.path.dirname(__file__)
    TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
    MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
    SAVE_PATH = os.path.join(BASE_PATH, "motion_model_transformer")
    MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "transformer_model.pth")

    WINDOW_SIZE = 50
    FILTER_SIZE = 100

    # モデルの初期化とロード
    model = TimeSeriesTransformer(input_dim=8, num_classes=3, seq_length=30)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print("Model loaded successfully")

    motion_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler.sav'), 'rb'))

    motion_recognition = MotionRecognition(WINDOW_SIZE, FILTER_SIZE, model=model)
    motion_recognition.set_task_path(TASK_PATH)
    motion_recognition.set_motion_scaler(motion_scaler=motion_scaler)
    
    motion_recognition.run()
    
if __name__ == "__main__":
    main()