import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_vectors import calculate_point, calculate_relative_vectors, calculate_answer_vectors

"""This program is for calculating coef
    BUT, it is only for demo, not general parameter

Returns:
    _type_: _description_
"""

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)
DATA_PATH    = os.path.join(BASE_PATH, "calibration_data")

pose_initial_matrix = np.load(os.path.join(DATA_PATH, "pose_matrix.npy"))
pose_initial_matrix = np.load(os.path.join(DATA_PATH, "pose_matrix_iphone.npy"))
print(pose_initial_matrix[0, :])

def predict_missing_point_math(relative_vectors, y):
    p_a = relative_vectors[:, :3]
    p_b = relative_vectors[:, 3:]
    p_c = y
    data_num = p_a.shape[0]    

    A = np.hstack((np.reshape(p_a, (3*data_num, 1)), np.reshape(p_b, (3*data_num, 1))))
    B = np.reshape(p_c, (3*data_num,1))
    X = np.linalg.pinv(A) @ B
    
    return X.flatten()
    
# そもそもデータが間違えているのでは。そこからやり直す
data = np.zeros((18, 6))
for data_i in range(6):
    for tag_i in range(6):
        data[3*data_i:3*(data_i+1), tag_i] = pose_initial_matrix[data_i, 3*tag_i:3*(tag_i+1)]
print(np.round(pose_initial_matrix, 2))

print(np.round(data, 2))
data = data[:, [0, 2, 3, 4, 5]]
data = data - np.tile(data[:, 1].reshape(18,1), (1, 5))
data = data[:, [0,2,3,4]]
print(np.round(data, 2))
X = np.linalg.pinv(data[:, 1:]) @ data[:,0]
print(X)
# np.save(os.path.join(DATA_PATH, "coef_array.npy"), coef_array)
np.save(os.path.join(DATA_PATH, "coef_array_iphone.npy"), X)


# xyz independently calculate
# そもそもデータが間違えているのでは。そこからやり直す
data = np.zeros((18, 6))
for data_i in range(6):
    for tag_i in range(6):
        data[3*data_i:3*(data_i+1), tag_i] = pose_initial_matrix[data_i, 3*tag_i:3*(tag_i+1)]
print(np.round(pose_initial_matrix, 2))

print(np.round(data, 2))
data = data[:, [0, 2, 3, 4, 5]]
data = data - np.tile(data[:, 1].reshape(18,1), (1, 5))
data = data[:, [0,2,3,4]]
print(np.round(data, 2))
X = np.linalg.pinv(data[:, 1:]) @ data[:,0]
print(X)
# np.save(os.path.join(DATA_PATH, "coef_array.npy"), coef_array)
np.save(os.path.join(DATA_PATH, "coef_array_iphone.npy"), X)

"""
[[ 1.00593965 -0.98441679]
 [-1.02763671  1.00607076]
 [-0.97253754  0.97861072]
 [ 0.99378993  1.02143155]]
"""
# plt.show()