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
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Constants and Paths
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "motion_model_transformer")
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier.keras")
FINAL_MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier.h5")  # Desired .h5 format
TFLITE_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier.tflite")

MOTION_SPEED = [0, 1, 2]

data = np.loadtxt(os.path.join(SAVE_PATH, 'motion_data_short.txt'))

frame_num = 15
data_shape = data.shape
data_array = data.reshape(-1, data.shape[1] * frame_num)
label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_shape[0] / frame_num / len(MOTION_SPEED)))])
print(data_array.shape, label_array.shape)

# scaler = StandardScaler()
# motion_scale = scaler.fit_transform(data_array)

X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, train_size=0.75, random_state=42)

input_size = 6 * frame_num
num_classes = len(MOTION_SPEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_size, )),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(60, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(30, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(45, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(input_size, )),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(3)  # No activation function for the final layer to get continuous output
# ])

model.summary()

# モデルコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルチェックポイントのコールバック
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, verbose=1, save_weights_only=False)

# 早期打ち切り用コールバック
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

# 推論テスト
print(X_test[0])
print(np.array([X_test[0]]))
predict_result = model.predict(np.array([X_test[0]]))
print(predict_result)
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

model.save(MODEL_SAVE_PATH, include_optimizer=False)



############### tflite #########################

# model = tf.keras.models.load_model(SAVED_MODEL_PATH)

# # # モデルを変換(量子化)
# # print("a")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # print("b")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # print("c")
# tflite_quantized_model = converter.convert()
# print("d")
# open(TFLITE_SAVE_PATH, 'wb').write(tflite_quantized_model)

# # モデルをロード・メモリ確保して推論
# interpreter = tf.lite.Interpreter(model_path=TFLITE_SAVE_PATH)
# interpreter.allocate_tensors()

# # 入出力テンソルを取得
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # 推論実施
# interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
# interpreter.invoke()
# tflite_results = interpreter.get_tensor(output_details[0]['index'])

# print(np.squeeze(tflite_results))
# print(np.argmax(np.squeeze(tflite_results)))


# # Save
# pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_30_angle.sav'), 'wb'))
# with open(os.path.join(SAVE_PATH, 'pca_model_30_angle.sav'), mode='wb') as f:
#     pickle.dump(pca, f)
# with open(os.path.join(SAVE_PATH, 'forest_model_30_angle.sav'), mode='wb') as f:
#     pickle.dump(forest, f)
# with open(os.path.join(SAVE_PATH, 'motion_model_30_angle.sav'), mode='wb') as f:
#     pickle.dump(best_model, f, protocol=2)
