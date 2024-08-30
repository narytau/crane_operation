import os
import numpy as np
import matplotlib.pyplot as plt
from create_vectors import calculate_point, calculate_relative_vectors, calculate_answer_vectors

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.join(CURRENT_PATH,"calibration_data")
pose_initial_matrix = np.load(os.path.join(BASE_PATH,"pose_matrix.npy"))
dataset = pose_initial_matrix


def predict_missing_point_math(relative_vectors, y):
    p_a = relative_vectors[:, :3]
    p_b = relative_vectors[:, 3:]
    p_c = y
    data_num = p_a.shape[0]    

    A = np.hstack((np.reshape(p_a, (3*data_num, 1)), np.reshape(p_b, (3*data_num, 1))))
    B = np.reshape(p_c, (3*data_num,1))
    X = np.linalg.pinv(A) @ B
    
    return X.flatten()
    
# 係数を格納するリスト
coefficients = []
points = dataset
N = int(points.shape[1] / 3)

# 各点をターゲットとして除外し、その点を予測
for missing_index in range(N):
    base_point, remaining_point, missing_point = calculate_point(points, missing_index, N)

    relative_vectors = calculate_relative_vectors(base_point, remaining_point, N)
    y = calculate_answer_vectors(base_point, missing_point)
    
    # predicted_point, coef = predict_missing_point(relative_vectors, base_point, y)
    coef = predict_missing_point_math(relative_vectors, y)
    X = np.zeros((6, 3))

    # 対角成分にx1をセット
    np.fill_diagonal(X[:3, :], coef[0])

    # 下半分の対角成分にx2をセット
    np.fill_diagonal(X[3:, :], coef[1])
    predicted_point = relative_vectors @ X + base_point
    
    # 予測結果と実際の値を表示
    print(f"Predicted Point {missing_index}: {predicted_point}")
    print(f"Actual Point {missing_index}: {missing_point}")
    
    # 係数を格納
    coefficients.append(coef)

    # fig = plt.figure()
    # plt.plot(np.arange(1, dataset.shape[0]+1), missing_point[:,2])
    # plt.plot(np.arange(1, dataset.shape[0]+1), predicted_point[:,2])
    
# 最終的な係数を表示
print("Coefficients for each prediction:")

coef_array = np.zeros_like(coef)
for coef in coefficients:
    coef_array = np.vstack((coef_array, coef))
# coef_array = coef_array[3:, :]
coef_array = coef_array[1:, :]
print(coef_array)

np.save(os.path.join(BASE_PATH, "coef_array.npy"), coef_array)

"""
[[ 1.00593965 -0.98441679]
 [-1.02763671  1.00607076]
 [-0.97253754  0.97861072]
 [ 0.99378993  1.02143155]]
"""
# plt.show()