import os 
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Mapping
def map_values(x):
    mapping = {6: 0, 5: 1, 4: 2, 7: 3}
    return mapping.get(x, x) 
vectorized_map_values = np.vectorize(map_values)

# Tag label that we use
label_array = np.array([4, 5, 6, 7])
orders = np.zeros(len(label_array)).astype(int)

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)

cameraMatrix = np.load("code/calibration_data/realsense_camera_matrix.npy")
distCoeffs = np.load("code/calibration_data/realsense_distCoeffs.npy")

# Setting for Detector parameters and dictionary
detector_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

iteration = 0

aruco_marker_side_length = 0.0575
objPoints = np.array([[-aruco_marker_side_length/2, aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2,  aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2, -aruco_marker_side_length/2, 0],
                    [-aruco_marker_side_length/2, -aruco_marker_side_length/2, 0]])

# Setting for realsense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

data_num = 50
label_num = len(label_array)
X = np.zeros((data_num, label_num * 3))

try:
    while True:
            rvec_array = []
            
            # Wait for frames and get color frame
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            # Convert RGB into numpy array (480x640x3)
            color_image = np.asanyarray(color_frame.get_data())
            image_shape = color_image.shape

            # Detect ArUCo tag
            marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(color_image)

            # When Marker ids are in label array  
            if marker_ids is not None and len(marker_ids) == len(label_array):
                indices = vectorized_map_values(marker_ids).flatten()
                detected_num = len(label_array)                
                tvec_array = np.zeros((detected_num, 3))
                
                # Draw detected markers
                aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)

                # Calculate R, T
                for i in range(detected_num):
                    retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints, imagePoints=marker_corners[i], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                    cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec, tvec=tvec, length=0.05)
                    tvec_array[i, :] = tvec.flatten()
                
                # sort 
                tvec_array = tvec_array[np.argsort(indices), :]
                
                key = cv2.waitKey(1) & 0xFF
                if key == 9:
                    X[iteration, :] = np.reshape(tvec_array, label_num * 3)
                    iteration += 1
                    print("iteration: ", iteration)
                    
            # Show image
            cv2.imshow('RGB Image', color_image)
            
            if iteration == data_num:
                print("The necessary data is taken")
                break
            
            # If the escape button is pressed, exit the loop
            if cv2.waitKey(5) & 0xFF == 27:
                print("Stop")
                break
finally:
    # Stop pipeline
    np.save("code/calibration_data/pose_matrix.npy", X)
    pipe.stop()
    cv2.destroyAllWindows()
    
