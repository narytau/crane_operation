import os 
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs


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
BASE_PATH    = os.path.dirname(CURRENT_PATH)
DATA_PATH    = os.path.join(BASE_PATH, "calibration_data")

cameraMatrix = np.load(os.path.join(DATA_PATH, "realsense_camera_matrix_iphone.npy"))
distCoeffs = np.load(os.path.join(DATA_PATH, "realsense_distCoeffs_iphone.npy"))

# Setting for Detector parameters and dictionary
detector_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

iteration = 0

# aruco_marker_side_length = 0.0575
aruco_marker_side_length = 0.098

objPoints = np.array([[-aruco_marker_side_length/2, aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2,  aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2, -aruco_marker_side_length/2, 0],
                    [-aruco_marker_side_length/2, -aruco_marker_side_length/2, 0]])


data_num = 30
label_num = len(label_array)
X = np.zeros((data_num, label_num * 3))

try:
    cap = cv2.VideoCapture(os.path.join(DATA_PATH, "video2.MOV"))

    while True:
            rvec_array = []
            tvec_array = np.zeros((N, 3))
            
            # Wait for frames and get color frame
            ret, frame = cap.read()
            
            # Convert RGB into numpy array (480x640x3)
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
                    tvec_array[indices[i], :] = tvec.flatten()
                
                
                key = cv2.waitKey(1) & 0xFF
                if key == 9 and detected_num == N:
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
    np.save(os.path.join(DATA_PATH, "pose_matrix_iphone.npy"), X)
    np.savetxt(os.path.join(DATA_PATH, "pose_matrix_iphone.txt"), X)
    cv2.destroyAllWindows()
    
