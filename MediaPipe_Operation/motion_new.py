import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from BaseMotionRecognition import NewMotionRecognition
from my_module import function_math


class MotionRecognition(NewMotionRecognition):
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

                # if self.model is not None and len(self.motion_array) >= self.window_size:
                #     predictions = real_time_inference(self.model, self.motion_scaler, self.motion_array, self.window_size)
                    
                self.display_data(flip_color_image)
                
                cv2.imshow('RGB Image', flip_color_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            pipe.stop()
            cv2.destroyAllWindows()
            
            
def main():
    # Constants and Paths
    BASE_PATH = os.path.dirname(__file__)
    TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
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