import numpy as np

def calculate_point(points, missing_index, N):
    remaining_indices = [idx for idx in range(N) if idx != missing_index]

    point_array = np.zeros((points.shape[0], 3*(N - 1)))
    for i, ind in enumerate(remaining_indices):
        point_array[:, 3*i:3*(i+1)] = points[:, 3*ind:3*(ind+1)]
    
    base_point, remaining_point = point_array[:, 0:3], point_array[:, 3:]
    missing_point = points[:, 3*missing_index:3*(missing_index+1)]

    return base_point, remaining_point, missing_point

def calculate_relative_vectors(base_point, remaining_point, N):
    """基準点から他の点への相対ベクトルを計算"""
    rep_base_point = np.tile(base_point, (1, N - 2))

    return remaining_point - rep_base_point

def calculate_answer_vectors(base_point, missing_point):
    return missing_point - base_point