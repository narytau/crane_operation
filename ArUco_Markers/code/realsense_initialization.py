import os 
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

def normalize(q):
    return q / np.linalg.norm(q)

def calc_coef(p, q, r):
    """calculate coefficients

    Args:
        p (ndarray): the vector that we want to calculate
        q (ndarray): the known vector
        r (ndarray): the known vector
    """
    p_q = np.inner(p, q)
    q_r = np.inner(q, r)
    r_p = np.inner(r, q)
    
    A = np.array([[q_r, np.linalg.norm(r)**2], [np.linalg.norm(q)**2, q_r]])
    B = np.array([r_p, p_q])
    X = np.linalg.norm(A, B)
    
    return X

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

# cameraMatrix = np.load("code/calibration_data/realsense_camera_matrix.npy")
cameraMatrix = np.load(os.path.join(CURRENT_PATH, "calibration_data","realsense_camera_matrix.npy"))
distCoeffs = np.load(os.path.join(CURRENT_PATH, "calibration_data","realsense_distCoeffs.npy"))
print(cameraMatrix, distCoeffs)

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


try:
    while True:
            rvec_array = []
            tvec_array = []
            
            # Wait for frames and get color frame
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            # Convert RGB into numpy array (480x640x3)
            color_image = np.asanyarray(color_frame.get_data())
            image_shape = color_image.shape

            # Detect ArUCo tag
            marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(color_image)

            # When Marker ids are in label array  
            if len(marker_ids) == len(label_array):
                detected_num = len(label_array)
                
                # Draw detected markers
                aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)

                # Calculate R, T
                for i in range(detected_num):
                    retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints, imagePoints=marker_corners[i], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                    cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec, tvec=tvec, length=0.05)
                    rvec_array.append(rvec)
                    tvec_array.append(tvec)
                    
                R_0 = cv2.Rodrigues(rvec_array[0])[0]

                relative_matrix = np.zeros((detected_num-1, 3))
                distance_matrix = np.zeros(detected_num-1)
                
                for i in range(detected_num-1):
                    tmp = tvec_array[i+1] - tvec_array[0]
                    relative_matrix[i, :] = tmp.flatten()
                    distance_matrix[i] = np.linalg.norm(tmp.flatten())

                p_r = np.inner(relative_matrix[0,:], relative_matrix[1,:])
                r_q = np.inner(relative_matrix[1,:], relative_matrix[2,:])
                q_p = np.inner(relative_matrix[2,:], relative_matrix[0,:])
                
                A = np.array([[np.linalg.norm(relative_matrix[1,:])**2, r_q],
                            [r_q, np.linalg.norm(relative_matrix[2,:])**2]])
                B = np.array([p_r, q_p])
                X = np.linalg.solve(A, B)
                

                print(X)
                print(distance_matrix)
                
                iteration += 1                
                
            # Show image
            cv2.imshow('RGB Image', color_image)
            
            # If the escape button is pressed, exit the loop
            if cv2.waitKey(5) & 0xFF == 27:
                print("Stop")
                # np.save("code/calibration_data/relative_matrix.npy", relative_matrix)
                # np.save("code/calibration_data/distance_matrix.npy", distance_matrix)
                # np.save("code/calibration_data/vector_coef.npy", X)
                
                break
finally:
    # Stop pipeline
    pipe.stop()
    cv2.destroyAllWindows()