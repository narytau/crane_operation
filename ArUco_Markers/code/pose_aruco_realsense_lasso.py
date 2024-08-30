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

# Mapping
def map_values(x):
    mapping = {6: 0, 5: 1, 4: 2, 7: 3}
    return mapping.get(x, x) 
vectorized_map_values = np.vectorize(map_values)

# Tag label that we use
label_array = np.array([4, 5, 6, 7])
orders = np.zeros(len(label_array)).astype(int)
N = len(label_array)

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.join(CURRENT_PATH,"calibration_data")

print(os.path.join(CURRENT_PATH,"calibration_data"))
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

                if detected_num >= 3:
                    if detected_num == 3 and len(missing_index) == 1:
                        points = np.delete(tvec_array, missing_index, axis=0)    
                        base_point, remaining_point = points[0, :], points[1:, :]
                        remaining_point = remaining_point.flatten()
                        
                        relative_vectors = calculate_relative_vectors(base_point, remaining_point, N)

                        coef = coef_array[missing_index[0], :]
                        predicted_vector = coef[0] * relative_vectors[0, :3] + coef[1] * relative_vectors[0, 3:] + base_point
                        
                        tvec_array[missing_index[0], :] = predicted_vector
                

                    # Define the axis
                    x_axis = tvec_array[1, :] - tvec_array[0, :]
                    y_axis = tvec_array[2, :] - tvec_array[0, :]
                    orthogonal_vectors = gram_schmidt([x_axis, y_axis])
                    x_axis, y_axis = orthogonal_vectors[0], orthogonal_vectors[1]
                    z_axis = - np.cross(x_axis, y_axis)
                    rvec_all = np.reshape(np.hstack((x_axis, y_axis, z_axis)), (3,3)).T
                    cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec_all, tvec=tvec_array[0, :], length=0.1)
                    
                    # 
                    origin = tvec_array[0, :]
                    tvec_hole = rvec_all @ np.array([0.1, 0.08, 0.05]) + origin
                    cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec_all, tvec=tvec_hole, length=0.005)
                    # print(tvec_hole)
                    
                if detected_num > 3:
                    
                    X = tvec_array
                    y = np.ones((detected_num, 1))
                    clf = Ridge(alpha=0)
                    
                    alpha = 0.3
                    print(X, y)
                    print("pinv", np.linalg.inv(X.T @ X) @ X.T @ y)
                    print("ridge", np.linalg.inv(X.T @ X + alpha * np.eye(3)) @ X.T @ y)
                
                    # X_new = np.hstack((X, y))
                    # # 方程式を解く（SVDを使う）
                    # U, S, Vt = np.linalg.svd(X_new)
                    # w = Vt[-1, :]  # 最小特異値に対応する特異ベクトル
                    # print("svd", w[:3])

                    
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

