import os
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Constants and Paths
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "motion_model")

data = np.loadtxt(os.path.join(SAVE_PATH, 'motion_data.txt'))
print(data[0:10,:])

MOTION_SPEED = ['HIGH', 'MIDDLE', 'LOW']

print(data.shape)
data1 = np.roll(data[0:1000,:], -2, axis=0)
print(data1.shape)
plt.plot(data1[:,::3])
plt.legend(['HIGH', 'MIDDLE', 'LOW'])
# plt.show()


frame_num_array = [1, 2, 5, 10, 20, 25, 50, 100]
score_array = []

for frame_num in frame_num_array:
    data_shape = data.shape
    data_array = data.reshape(-1, data.shape[1] * frame_num)
    label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_shape[0] / frame_num / len(MOTION_SPEED)))])

    scaler = StandardScaler()
    motion_scale = scaler.fit_transform(data_array)

    # Train SVM model with GridSearchCV
    ##################
    # frame = 30(1s)くらいにしてみる、そのためにデータ数を増やす
    
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    X_train, X_test, y_train, y_test = train_test_split(motion_scale, label_array, random_state=0)

    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    score_array.append(score)
    print("X test, y test")
    print(X_test.shape, y_test.shape)
    print(f'Best model accuracy: {score}')
    print(best_model)
    
print(frame_num_array)
print(score_array)