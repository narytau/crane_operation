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

# Constants and Paths
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "motion_model")

MOTION_SPEED = ['HIGH', 'MIDDLE', 'LOW']

data = np.loadtxt(os.path.join(SAVE_PATH, 'motion_data_30_angle.txt'))

frame_num = 30
data_shape = data.shape
data_array = data.reshape(-1, data.shape[1] * frame_num)
label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_shape[0] / frame_num / len(MOTION_SPEED)))])
print(data_array.shape, label_array.shape)

scaler = StandardScaler()
motion_scale = scaler.fit_transform(data_array)


# reduce 
pca = PCA(n_components=0.95)
motion_reduced = pca.fit_transform(motion_scale)
print("motion_pca")
print(motion_reduced.shape)

# Train SVM model with GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(motion_scale, label_array, random_state=0)

forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
selector = SelectFromModel(forest, threshold="median")
X_important_train = selector.fit_transform(X_train, y_train)
X_important_test = selector.transform(X_test)

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
# grid_search.fit(X_important_train, y_train)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
# score = best_model.score(X_important_test, y_test)
score = best_model.score(X_test, y_test)

print("X important test")
# print(X_test.shape)
print(f'Best model accuracy: {score}')
print(best_model)



# Save
pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_30_angle.sav'), 'wb'))
with open(os.path.join(SAVE_PATH, 'pca_model_30_angle.sav'), mode='wb') as f:
    pickle.dump(pca, f)
with open(os.path.join(SAVE_PATH, 'forest_model_30_angle.sav'), mode='wb') as f:
    pickle.dump(forest, f)
with open(os.path.join(SAVE_PATH, 'motion_model_30_angle.sav'), mode='wb') as f:
    pickle.dump(best_model, f, protocol=2)
