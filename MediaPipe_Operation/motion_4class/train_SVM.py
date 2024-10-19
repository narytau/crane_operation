import os
import cv2
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
data_array = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_original7.txt'))
data_unclass1 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass1.txt'))
data_unclass2 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass2.txt'))
data_unclass3 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass3.txt'))
data_unclass4 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass4.txt'))
data_unclass5 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass5.txt'))
data_unclass6 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass6.txt'))
data_unclass7 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass7.txt'))
data_unclass = np.vstack((data_unclass5.reshape(-1, data_unclass5.shape[1]*frame_num),
                        data_unclass6.reshape(-1, data_unclass6.shape[1]*frame_num),
                        data_unclass7.reshape(-1, data_unclass7.shape[1]*frame_num)))


data_array = create_data_array(data_array, frame_num=frame_num, skip_num=1, is_stop=True)
print(data_unclass.shape, data_array.shape)
# data_array[-data_unclass.shape[0]:, :] = data_unclass

# データの標準化
scaler = StandardScaler()
data_array = scaler.fit_transform(data_array)

label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_array.shape[0] / len(MOTION_SPEED)))])

X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, train_size=0.75, random_state=42)

# param_grid = {'C': [0.01, 0.1, 1], 'gamma': [0.01, 0.1, 1]}
# grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# score = best_model.score(X_test, y_test)

# print(f'Best model accuracy: {score}')
# print(best_model)

model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

model.predict(X_test)

# # reduce 
# pca = PCA(n_components=0.95)
# motion_reduced = pca.fit_transform(motion_scale)
# print("motion_pca")
# print(motion_reduced.shape)

# # Train SVM model with GridSearchCV
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# X_train, X_test, y_train, y_test = train_test_split(motion_scale, label_array, random_state=0)

# forest = RandomForestClassifier(n_estimators=100)
# forest.fit(X_train, y_train)
# selector = SelectFromModel(forest, threshold="median")
# X_important_train = selector.fit_transform(X_train, y_train)
# X_important_test = selector.transform(X_test)

# grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
# grid_search.fit(X_important_train, y_train)
# # grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# score = best_model.score(X_important_test, y_test)
# # score = best_model.score(X_test, y_test)

# print("X important test")
# # print(X_test.shape)
# print(f'Best model accuracy: {score}')
# print(best_model)

# # with open(os.path.join(SAVE_PATH, 'forest_model5.sav'), mode='wb') as f:
# #     pickle.dump(forest, f)

# # with open(os.path.join(SAVE_PATH, 'SVM_model3.sav'), mode='wb') as f:
# #     pickle.dump(best_model, f, protocol=2)


# # Print grid search results
# print("Grid search results:")
# means = grid_search.cv_results_['mean_test_score']
# stds = grid_search.cv_results_['std_test_score']
# params = grid_search.cv_results_['params']
# for mean, std, param in zip(means, stds, params):
#     print(f"Mean: {mean:.3f}, Std: {std:.3f}, Params: {param}")

# best_model = grid_search.best_estimator_
# # score = best_model.score(X_test, y_test)
# score = best_model.score(X_important_test, y_test)
# print(f'Best model accuracy: {score}')
# print(best_model)

# # Predict and print classification report
# # y_pred = best_model.predict(X_test)
# y_pred = best_model.predict(X_important_test)
# print("Classification report:")
# print(classification_report(y_test, y_pred, target_names=[str(speed) for speed in MOTION_SPEED]))

# # Save
# # pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_SVM.sav'), 'wb'))
# # with open(os.path.join(SAVE_PATH, 'pca_model_30_angle.sav'), mode='wb') as f:
# #     pickle.dump(pca, f)




