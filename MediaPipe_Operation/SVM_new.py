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
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler5.sav'), 'rb'))

MOTION_SPEED = [0, 1, 2, 3]

data_array = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data5.txt'))

EACH_DATA = 800
frame_num = 15
# data_shape = data.shape
# data_array = data.reshape(-1, data.shape[1] * frame_num)
label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_array.shape[0] / len(MOTION_SPEED)))])
# label_array = np.hstack(([0]*EACH_DATA, [1]*EACH_DATA, [2]*2*EACH_DATA, [3]*2*EACH_DATA))

print(data_array.shape, label_array.shape)

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
grid_search.fit(X_important_train, y_train)
# grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
score = best_model.score(X_important_test, y_test)
# score = best_model.score(X_test, y_test)

print("X important test")
# print(X_test.shape)
print(f'Best model accuracy: {score}')
print(best_model)

# with open(os.path.join(SAVE_PATH, 'forest_model5.sav'), mode='wb') as f:
#     pickle.dump(forest, f)

# with open(os.path.join(SAVE_PATH, 'SVM_model3.sav'), mode='wb') as f:
#     pickle.dump(best_model, f, protocol=2)


# Print grid search results
print("Grid search results:")
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print(f"Mean: {mean:.3f}, Std: {std:.3f}, Params: {param}")

best_model = grid_search.best_estimator_
# score = best_model.score(X_test, y_test)
score = best_model.score(X_important_test, y_test)
print(f'Best model accuracy: {score}')
print(best_model)

# Predict and print classification report
# y_pred = best_model.predict(X_test)
y_pred = best_model.predict(X_important_test)
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=[str(speed) for speed in MOTION_SPEED]))

# Save
# pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_SVM.sav'), 'wb'))
# with open(os.path.join(SAVE_PATH, 'pca_model_30_angle.sav'), mode='wb') as f:
#     pickle.dump(pca, f)




