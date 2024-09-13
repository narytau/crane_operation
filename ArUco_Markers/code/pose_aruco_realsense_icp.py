# The coefficients are calculated mathmatically

import os 
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import sklearn.linear_model
from gram_schmidt import gram_schmidt
from create_vectors import calculate_point, calculate_relative_vectors
from sklearn.linear_model import Ridge, Lasso
from scipy.optimize import minimize
from function_opt import objective_with_penalty, determinant_constraint, arange_position_init

from scipy.optimize import least_squares

def calc_angle(vec1, vec2):
    cos_theta = np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(cos_theta) * 180 / np.pi
    return theta

def arange_position_init(position):
    """coordinate transformation

    Args:
        position (Tags_num x 3):
    """
    point_num = position.shape[0]
    position_diff = position - position[0, :]

    # theta, length
    relative_array = np.zeros((point_num-2, 2))
    for i in range(point_num-2):
        base_length = np.linalg.norm(position_diff[1,:])
        relative_array[i, 0] = calc_angle(position_diff[1,:], position_diff[i+2,:])
        relative_array[i, 1] = np.linalg.norm(position_diff[i+2,:]) / base_length

    # position_init
    position_init = np.zeros((point_num, 3))
    position_init[1, :] = np.array([np.linalg.norm(position_diff[1,:]), 0, 0])
    for i in range(point_num-2):
        theta = relative_array[i, 0] * np.pi / 180
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta),  0],
                                    [0, 0, 1]])
        position_init[i+2, :] =  relative_array[i, 1] * rotation_matrix @ position_init[1, :] 

    # Regularization for inverse matrix computation
    position_init[:, -1] = 1e-6

    return position_init


# Mapping
def map_values(x):
    mapping = {6: 0, 5: 1, 4: 2, 7: 3, 8:4}
    return mapping.get(x, x) 
vectorized_map_values = np.vectorize(map_values)

# Tag label that we use
label_array = np.array([4, 5, 6, 7, 8])
orders = np.zeros(len(label_array)).astype(int)
N = len(label_array)

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.join(CURRENT_PATH,"calibration_data")

# load data
cameraMatrix = np.load(os.path.join(BASE_PATH, "realsense_camera_matrix.npy"))
distCoeffs   = np.load(os.path.join(BASE_PATH, "realsense_distCoeffs.npy"))
relative_matrix = np.load(os.path.join(BASE_PATH, "relative_matrix.npy"))
distance_matrix = np.load(os.path.join(BASE_PATH, "distance_matrix.npy"))
coef_array = np.load(os.path.join(BASE_PATH, "coef_array.npy"))

# Setting for Detector parameters and dictionary
detector_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

aruco_marker_side_length = 0.0575
objPoints = np.array([[-aruco_marker_side_length/2, aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2,  aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2, -aruco_marker_side_length/2, 0],
                    [-aruco_marker_side_length/2, -aruco_marker_side_length/2, 0]])

# Target point on the wall CS
target_point = np.array([3, 3, 0.5])

flag_init = False

# Setting for realsense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

try:
    while True:
            rvec_array = []
            tvec_array = np.zeros((N, 3))
            
            # Wait for frames and get color frame
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            # Convert RGB into numpy array (480x640x3)
            color_image = np.asanyarray(color_frame.get_data())
            image_shape = color_image.shape

            # Detect ArUCo tag
            marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(color_image)
            
            # When Marker ids are in label array  
            if marker_ids is not None and all(elem in label_array for elem in marker_ids.flatten()):
                # Avoid the misdetection
                indices = vectorized_map_values(marker_ids).flatten()
                detected_num = len(indices)
                
                # Draw detected markers
                aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)

                # Calculate R, T
                for i in range(detected_num):
                    retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints, imagePoints=marker_corners[i], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                    # cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec, tvec=tvec, length=0.05)
                    # rvec_array.append(rvec)
                    tvec_array[indices[i], :] = tvec.flatten()
                
                missing_index = [item for item in np.arange(N) if item not in indices]
                remaining_index = [item for item in np.arange(N) if item not in missing_index]
                
                if flag_init == False:
                    key = cv2.waitKey(1) & 0xFF
                    if detected_num == N and key == 9:
                        position_init = arange_position_init(tvec_array)
                        flag_init = True
                        print("Initial position is successfully taken")
                else:
                    if detected_num >= 3:
                        p = tvec_array[remaining_index, :]
                        q = position_init[remaining_index, :]
                        
                        mu_p = np.mean(p, axis=0)
                        mu_q = np.mean(q, axis=0)

                        W = 0
                        for i in range(detected_num-1):
                            W += (p[i, :] - mu_p).reshape(3,1) @ (q[i, :] - mu_q).reshape(1,3)
                        
                        U, S, Vh = np.linalg.svd(W)

                        R = U @ Vh
                        T = mu_p.reshape(3,1) - R @ mu_q.reshape(3,1)
                        
                        cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=R, tvec=T, length=0.1)
                        
                    else:
                        print("The data is not enough")
            # Show image
            cv2.imshow('RGB Image', color_image)
            
            # If the escape button is pressed, exit the loop
            if cv2.waitKey(5) & 0xFF == 27:
                print("Stop")
                break
finally:
    # Stop pipeline
    pipe.stop()
    cv2.destroyAllWindows()

