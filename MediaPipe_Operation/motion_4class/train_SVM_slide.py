import os
import cv2
import sys
import time
import pickle
import random
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_module.sliding_time_window import sliding_time_window


def create_data_array(data, frame_num, skip_num, is_stop):
    data_new = data[::skip_num, :]
    data_new = data_new.reshape(-1, data_new.shape[1] * frame_num)

    
    if is_stop:
        random_integers = [random.randint(0, data.shape[0]-1) for _ in range(int(data_new.shape[0] / 3))]
        data_stop = data[random_integers, :]
        data_stop = np.hstack([data_stop] * frame_num)
        gauss_matrix = np.random.randn(data_stop.shape[0], data_stop.shape[1]) / 500
        data_stop += gauss_matrix
        
        data_new = np.vstack((data_new, data_stop))


    return data_new 


# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(CURRENT_PATH)
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier_with_unclass_data2.pth")

MOTION_SPEED = [0, 1, 2, 3]
is_saved = True
frame_num = 30

# Data preparation
data_array = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_original10.txt'))
data_unclass1 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass10.txt'))
data_unclass2 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass11.txt'))
data_unclass3 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass12.txt'))
# data_unclass = np.vstack((data_unclass1.reshape(-1, data_unclass1.shape[1]*frame_num),
#                         data_unclass2.reshape(-1, data_unclass2.shape[1]*frame_num),
#                         data_unclass3.reshape(-1, data_unclass3.shape[1]*frame_num)))

data_unclass = np.vstack((data_unclass2, data_unclass3))

random_integers = [random.randint(0, data_array.shape[0]-1)  for _ in range(int(data_array.shape[0] / 3 / frame_num))]
random_integers = [i for i in random_integers for _ in range(frame_num)]

data_stop = data_array[random_integers, :]

data_array_new = np.vstack((data_array, data_stop))
data_array_new[-data_unclass.shape[0]:, :] = data_unclass

gauss_matrix = np.random.randn(data_array_new.shape[0], data_array_new.shape[1]) / 500
data_array_new += gauss_matrix
# data_array_new = data_array_new[::2, :]


# sliding time window
data_each_num = int(data_array_new.shape[0] / 4)
for i in range(4):
    data_each = data_array_new[i*data_each_num:(i+1)*data_each_num, :]
    data_window_each = sliding_time_window(array=data_each, window_size=frame_num, step_size=20)
    if i == 0:
        data_window_array = data_window_each
    else:
        data_window_array = np.concatenate([data_window_array, data_window_each], axis=0)


# Reshape: data_num x frame_num x features --> data_num x (frame_num x features)
data_window_array = np.reshape(np.transpose(data_window_array, (0,2,1)), 
                            (data_window_array.shape[0],-1))


# Standard scaler
scaler = StandardScaler()
print(data_window_array.shape)
data_window_array = scaler.fit_transform(data_window_array)

pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_SVM_slide.sav'), 'wb'))


label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_window_array.shape[0] / len(MOTION_SPEED)))])

X_train, X_test, y_train, y_test = train_test_split(data_window_array, label_array, train_size=0.75, random_state=42)
param_grid = {'C': [1,10,100], 'gamma': [0.01,0.1,1]}
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
score = best_model.score(X_test, y_test)

print(f'Best model accuracy: {score}')
print(best_model)

with open(os.path.join(SAVE_PATH, 'model_SVM_slide.sav'), mode='wb') as f:
    pickle.dump(best_model, f)