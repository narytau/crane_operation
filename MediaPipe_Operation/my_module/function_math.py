import numpy as np
from scipy.ndimage import gaussian_filter1d

def calculate_angle(vec1, vec2):
    x = np.inner(vec1, vec2)
    theta = np.arccos(x / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return theta

def normalize_angle(angle, origin):
    """Function to normalize an angle to the range [- origin, origin + 2pi]"""
    return (angle + origin) % (2 * np.pi) - origin

def calculate_unit_vector(vec):
    return vec / np.linalg.norm(vec)

def calculate_theta(vec):
    """spherical coordinate system"""
    y_axis = np.array([0, -1, 0])
    return calculate_angle(vec, y_axis)

def calculate_phi(vec):
    """spherical coordinate system"""
    x_z_vec = vec[[0, 2]]
    x_z_vec = calculate_unit_vector(x_z_vec)
    return np.arctan2(x_z_vec[1], x_z_vec[0])

def apply_gaussian_smoothing(data, sigma):
    """Apply Gauss kernel to data"""
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=0)
    return smoothed_data

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 数値安定性のために最大値を引きます
    return exp_x / exp_x.sum(axis=0, keepdims=True)
