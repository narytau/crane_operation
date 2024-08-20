import os
import numpy as np
import torch
import random
import torch.nn as nn
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Constants and Paths
BASE_PATH = os.path.dirname(__file__)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

# HIGH, MIDDLE, LOW
MOTION_SPEED = [0, 1, 2]

# データを準備
data_high    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high.txt'))
data_high2   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high2.txt'))
data_high3   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high3.txt'))
data_high4   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high4.txt'))
data_high5   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high5.txt'))
data_high6   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high6.txt'))
data_high7   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_high7.txt'))
data_middle  = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle.txt'))
data_middle2 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle2.txt'))
data_middle3 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle3.txt'))
data_middle4 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle4.txt'))
data_middle5 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle5.txt'))
data_middle6 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle6.txt'))
data_middle7 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle7.txt'))
data_middle8 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_middle8.txt'))
data_low     = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low3.txt'))
data_low2    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low4.txt'))
data_low3    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low5.txt'))
data_low4    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low6.txt'))
data_low5    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low7.txt'))
data_low6    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low8.txt'))
data_low7    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low9.txt'))
data_low8    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low10.txt'))
data_low9    = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_low11.txt'))
data_unclass1   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass1.txt'))
data_unclass2   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass2.txt'))
data_unclass3   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass3.txt'))
data_unclass4   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass4.txt'))
data_unclass5   = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass5.txt'))



data = np.concatenate([data_high, 
                       data_high2,
                       data_high3, 
                       data_high4,
                       np.flip(data_high, axis=0),
                       np.flip(data_high2, axis=0),
                       np.flip(data_high3, axis=0),
                       np.flip(data_high4, axis=0),
                       data_middle,
                       data_middle2,
                       data_middle3,
                       data_middle4,
                       np.flip(data_middle, axis=0),
                       np.flip(data_middle2, axis=0),
                       np.flip(data_middle3, axis=0),
                       np.flip(data_middle4, axis=0),
                       data_low,
                       data_low2,
                       data_low3,
                       data_low4,
                       np.flip(data_low, axis=0),
                       np.flip(data_low2, axis=0),
                       np.flip(data_low3, axis=0),
                       np.flip(data_low4, axis=0)], axis=0)

# data = np.concatenate([data_high, 
#                        data_high2,
#                        data_high3, 
#                        np.flip(data_high, axis=0),
#                        np.flip(data_high2, axis=0),
#                        np.flip(data_high3, axis=0),
#                        data_middle,
#                        data_middle2,
#                        data_middle3,
#                        np.flip(data_middle, axis=0),
#                        np.flip(data_middle2, axis=0),
#                        np.flip(data_middle3, axis=0),
#                        data_low,
#                        data_low2,
#                        data_low3,
#                        np.flip(data_low, axis=0),
#                        np.flip(data_low2, axis=0),
#                        np.flip(data_low3, axis=0),
#                        data_unclass1,
#                        data_unclass2,
#                        data_unclass3,
#                        np.flip(data_unclass1, axis=0),
#                        np.flip(data_unclass2, axis=0),
#                        np.flip(data_unclass3, axis=0)], axis=0)

data = np.concatenate([data_high, 
                       data_high2,
                       data_high3, 
                       data_high4,
                       data_high5,
                       np.flip(data_high, axis=0),
                       np.flip(data_high2, axis=0),
                       np.flip(data_high3, axis=0),
                       np.flip(data_high4, axis=0),
                       np.flip(data_high5, axis=0),
                       data_middle,
                       data_middle2,
                       data_middle3,
                       data_middle4,
                       data_middle5,
                       np.flip(data_middle, axis=0),
                       np.flip(data_middle2, axis=0),
                       np.flip(data_middle3, axis=0),
                       np.flip(data_middle4, axis=0),
                       np.flip(data_middle5, axis=0),
                       data_low6,
                       data_low2,
                       data_low3,
                       data_low4,
                       data_low5,
                       np.flip(data_low6, axis=0),
                       np.flip(data_low2, axis=0),
                       np.flip(data_low3, axis=0),
                       np.flip(data_low4, axis=0), 
                       np.flip(data_low5, axis=0)], axis=0)

data = np.concatenate([data_high, 
                       data_high2,
                       data_high3, 
                       data_high4,
                       data_high5,
                       data_high6,
                       data_high7,
                       np.flip(data_high, axis=0),
                       np.flip(data_high2, axis=0),
                       np.flip(data_high3, axis=0),
                       np.flip(data_high4, axis=0),
                       np.flip(data_high5, axis=0),
                       np.flip(data_high6, axis=0),
                       np.flip(data_high7, axis=0),
                       data_middle,
                       data_middle7,
                       data_middle8,
                       data_middle3,
                       data_middle4,
                       data_middle5,
                       data_middle6,
                       np.flip(data_middle, axis=0),
                       np.flip(data_middle7, axis=0),
                       np.flip(data_middle8, axis=0),
                       np.flip(data_middle3, axis=0),
                       np.flip(data_middle4, axis=0),
                       np.flip(data_middle5, axis=0),
                       np.flip(data_middle6, axis=0),
                       data_low7,
                       data_low6,
                       data_low8,
                       data_low3,
                       data_low4,
                       data_low5,
                       data_low9,
                       np.flip(data_low7, axis=0),
                       np.flip(data_low6, axis=0),
                       np.flip(data_low8, axis=0),
                       np.flip(data_low3, axis=0),
                       np.flip(data_low4, axis=0), 
                       np.flip(data_low9, axis=0), 
                       np.flip(data_low5, axis=0)], axis=0)

np.savetxt(os.path.join(DATA_SAVE_PATH, 'motion_data_original7.txt'), data)

def create_data_array(data, frame_num, skip_num, is_stop):
    data_new = data[::skip_num, :]
    data_new = data_new.reshape(-1, data_new.shape[1] * frame_num)

    
    if is_stop:
        random_integers = [random.randint(0, data.shape[0]-1) for _ in range(int(data_new.shape[0] / 3))]
        data_stop = data[random_integers, :]
        data_stop = np.hstack([data_stop] * frame_num)
        data_new = np.vstack((data_new, data_stop))

    gauss_matrix = np.random.randn(data_new.shape[0], data_new.shape[1]) / 1000
    data_new += gauss_matrix

    return data_new    

print(data.shape)

data = create_data_array(data, frame_num=25, skip_num=3, is_stop=True)

# # # # データの標準化
scaler = StandardScaler()
data = scaler.fit_transform(data)

# np.savetxt(os.path.join(DATA_SAVE_PATH, 'motion_data_transformer.txt'), data)

# pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler7.sav'), 'wb'))
# np.savetxt(os.path.join(DATA_SAVE_PATH, 'motion_data7.txt'), data)
print(data.shape)