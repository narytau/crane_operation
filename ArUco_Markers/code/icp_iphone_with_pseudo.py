# The coefficients are calculated mathmatically

import os 
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from create_vectors import calculate_point, calculate_relative_vectors
from scipy.ndimage import gaussian_filter1d


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
    position_init[:, -1] = 1e-30

    return position_init

def ICP(p, q):
    mu_p = np.mean(p, axis=0)
    mu_q = np.mean(q, axis=0)

    W = 0
    for i in range(detected_num-1):
        W += (p[i, :] - mu_p).reshape(3,1) @ (q[i, :] - mu_q).reshape(1,3)
    
    U, S, Vh = np.linalg.svd(W)

    R = U @ Vh
    T = mu_p.reshape(3,1) - R @ mu_q.reshape(3,1)
    
    return R, T


# Mapping
def map_values(x):
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return mapping.get(x, x) 
vectorized_map_values = np.vectorize(map_values)

# Tag label that we use
label_array = np.array([1, 2, 3, 4, 5, 6])
orders = np.zeros(len(label_array)).astype(int)
N = len(label_array)

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.join(CURRENT_PATH, "calibration_data")

# load data
cameraMatrix = np.load(os.path.join(BASE_PATH, "realsense_camera_matrix_iphone.npy"))
distCoeffs   = np.load(os.path.join(BASE_PATH, "realsense_distCoeffs_iphone.npy"))
coef_array = np.load(os.path.join(BASE_PATH, "coef_array_iphone.npy"))

# Setting for Detector parameters and dictionary
detector_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# aruco_marker_side_length = 0.0575
aruco_marker_side_length = 0.098

objPoints = np.array([[-aruco_marker_side_length/2, aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2,  aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2, -aruco_marker_side_length/2, 0],
                    [-aruco_marker_side_length/2, -aruco_marker_side_length/2, 0]])

# Target point on the wall CS
target_point = np.array([3, 3, 0])

flag_init = False

# ガウスカーネルの標準偏差
sigma = 3
window_size = 200  # ウィンドウサイズを調整

data_window = np.zeros((0, 3))

def apply_gaussian_smoothing(data, sigma):
    # ガウスカーネルをデータに適用
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=0)
    return smoothed_data

def process_frame(measurement):
    global data_window

    # ウィンドウに新しい測定値を追加
    if data_window.shape[0] < window_size:
        data_window = np.vstack([data_window, measurement])
    else:
        data_window = np.roll(data_window, -1, axis=0)
        data_window[-1] = measurement

    # ガウス畳み込みを適用して平滑化
    smoothed_data = apply_gaussian_smoothing(data_window, sigma)
    
    # 現在の平滑化された値を取得
    current_smoothed_value = smoothed_data[-1]
    
    return current_smoothed_value

T_smoothed = np.zeros(3)

try:
    print("Initializing...")
    print("Ensure all markers are visible on the screen, then press the Tab key to retrieve the data.")
    print("-------------------------------------------------------------------------------------------")
    
    cap = cv2.VideoCapture(os.path.join(BASE_PATH, "video1.MOV"))
    
    
    while True:
            ret, frame = cap.read()
            rvec_array = []
            tvec_array = np.zeros((N, 3))
            
            # Wait for frames and get color frame
            
            color_image = cv2.resize(frame, (640, 480))
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
                
                print(tvec_array)
                missing_index = [item for item in np.arange(N) if item not in indices]
                remaining_index = [item for item in np.arange(N) if item not in missing_index]
                
                # in this case, we use only 1~4 tags to estimate undetected vectors
                missing_index_for_pseudo = [item for item in np.arange(N-2) if item not in indices]
                
                if flag_init == False:
                    key = cv2.waitKey(1) & 0xFF
                    position_init = np.zeros((N, 3))
                    for i in range(20):
                        if detected_num == N:
                            position_init += arange_position_init(tvec_array)
                    flag_init = True
                    position_init = position_init / 20
                    print("Initial position is successfully taken")
                else:
                    if detected_num >= 3:
                        T_flag = True
                        p = tvec_array[remaining_index, :]
                        q = position_init[remaining_index, :]
                        
                            
                        
                        if 0 in missing_index_for_pseudo:
                            X = np.array([[0.787989627759577,	0.264633943539498	,0.739837702242151],
                                        [0.374846640111150,	-0.401144435567185	,0.446596892409632],
                                        [-1.05770219748462	,-0.778636107141506,	-1.19381446028369]])
                            
                            base_point = tvec_array[2, :]
                            points = tvec_array - np.tile(tvec_array[2, :].reshape(1,3), (N, 1))
                            points = np.delete(points, [0,1,2], axis=0) 
                            
                            print("points", points)
                            # predicted_vector = coef_array[0] * points[0,:] + coef_array[1] * points[1,:] + coef_array[2] * points[2,:] + base_point
                            predicted_vector = np.sum(X * points, axis=0).flatten() + base_point
                            print("com", predicted_vector, base_point)
                            tvec_array[0, :] = predicted_vector
                            T_flag = False
                            
                        T_smoothed = process_frame(tvec_array[0,:])
                        if T_flag:
                            T = tvec_array[0, :]
                        else:
                            T = T_smoothed
                        
                        # ICP step
                        p_new = np.copy(p)
                        p_old = np.copy(p)
                        
                        R_sum = np.eye(3)
                        T_sum = np.zeros((3, 1))
                        
                        mu_p = np.mean(p, axis=0)
                        mu_q = np.mean(q, axis=0)
                        
                        W = 0
                        for i in range(detected_num-1):
                            W += (p[i, :] - T).reshape(3,1) @ (q[i, :]).reshape(1,3)
                        
                        U, S, Vh = np.linalg.svd(W)

                        R = U @ Vh
                        # T = mu_p.reshape(3,1) - R @ mu_q.reshape(3,1)
                        if np.linalg.det(R) > 0.95:
                            R[:,-1] = -R[:,-1]
                        
                        # cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=R, tvec=T, length=0.3)
                        cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=R, tvec=T, length=0.5)
                            
                        
                    else:
                        print("The data is not enough")
                        
            # Show image
            cv2.imshow('RGB Image', color_image)
            
            # If the escape button is pressed, exit the loop
            if cv2.waitKey(5) & 0xFF == 27:
                print("-------------------------------------------------------------------------------------------")
                print("The Escape key has been pressed, and the operation will stop.")
                print("Stop")
                break
finally:
    # Stop pipeline
    cv2.destroyAllWindows()

