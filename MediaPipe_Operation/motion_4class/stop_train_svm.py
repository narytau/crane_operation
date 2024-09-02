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
from sklearn.decomposition import PCA
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report

# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

category = [0, 1] # 0->stop_ges, 1->non_ges

data_array0 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'stop_gesture1.txt'))
data_array1 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'non_gesture.txt'))

frame_num = 50

data_array0 = data_array0.reshape(-1, data_array0.shape[1] * frame_num)
data_array1 = data_array1.reshape(-1, data_array1.shape[1] * frame_num)
data_array1 = data_array1[:100, :]

data_array = np.vstack((data_array0, data_array1))
print(data_array.shape)

scaler = StandardScaler()
stop_scale = scaler.fit_transform(data_array)

label_array = np.array([i for i in category for _ in range(int(data_array.shape[0] / len(category)))])

# Train SVM model with GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(stop_scale, label_array, random_state=0)

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
score = best_model.score(X_test, y_test)

print(X_test.shape)
print(f'Best model accuracy: {score}')
print(best_model)

pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'stop_scaler_SVM.sav'), 'wb'))

with open(os.path.join(SAVE_PATH, 'SVM_model_stop.sav'), mode='wb') as f:
    pickle.dump(best_model, f, protocol=2)





