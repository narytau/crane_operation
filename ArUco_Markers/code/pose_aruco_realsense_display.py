import os 
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

def normalize(q):
    return q / np.linalg.norm(q)

def average_quaternion(quaternions, weights=np.ones(4)):
    # 行列Mの初期化
    M = np.zeros((4, 4))
    
    for q, w in zip(quaternions, weights):
        q = normalize(q)  # 正規化
        M += w * np.outer(q, q)  # 外積を加算

    # 固有値問題を解く
    eigenvalues, eigenvectors = np.linalg.eig(M)
    max_index = np.argmax(eigenvalues)  # 最大固有値のインデックス
    
    # 最大固有値に対応する固有ベクトルを取得
    avg_quaternion = eigenvectors[:, max_index]
    return normalize(avg_quaternion) 

def rotvec_to_quaternion(rotvec):
    rotations = R.from_rotvec(rotvec)
    return rotations.as_quat()

def quaternion_to_rotmat(quaternion):
    rotations = R.from_quat(quaternion)
    return rotations.as_matrix()

# Mapping
def map_values(x):
    mapping = {6: 0, 5: 1, 4: 2, 7: 3}
    return mapping.get(x, x) 
vectorized_map_values = np.vectorize(map_values)

label_array = np.array([4, 5, 6, 7])
orders = np.zeros(len(label_array)).astype(int)


CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)

cameraMatrix = np.load("code/calibration_data/realsense_camera_matrix.npy")
distCoeffs = np.load("code/calibration_data/realsense_distCoeffs.npy")

print(cameraMatrix, distCoeffs)
# ArUCoマーカーの検出パラメータと辞書の設定
detector_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

aruco_marker_side_length = 0.0575
objPoints = np.array([[-aruco_marker_side_length/2, aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2,  aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2, -aruco_marker_side_length/2, 0],
                    [-aruco_marker_side_length/2, -aruco_marker_side_length/2, 0]])

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

space_mat = np.array([[0, 0, 0],
                    [-0.1709, 0, 0],
                    [0, -0.109, 0],
                    [-0.1709, -0.109, 0]])


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

            # Flip the image
            # flip_color_image = cv2.flip(color_image, 1)
            
            # Detect ArUCo tag
            marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(color_image)

            if marker_ids is not None:
                # Avoid the misdetection
                indices = vectorized_map_values(marker_ids).flatten()
                detected_num = len(indices)
                quaternion_array = np.zeros((detected_num, 4))
                
                # 検出されたマーカーの描画
                aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)

                for i in range(detected_num):
                    retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints, imagePoints=marker_corners[i], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                    cv2.drawFrameAxes(color_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec, tvec=tvec, length=0.05)
                    quaternion_array[i, :] = rotvec_to_quaternion(rvec[:, 0])
                    rvec_array.append(rvec)
                    tvec_array.append(tvec)
                    
                space_mat_ordered = space_mat[indices, :]
                sum_tvec = np.zeros(3)
                
                print(marker_ids, indices)
                
                for i in range(detected_num):
                    tmp = space_mat_ordered[i, :] + tvec_array[i][:,0]
                    cv2.putText(color_image, str(marker_ids[indices[i]]), (20, 50*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(color_image, str(np.round(tvec_array[i][:,0], 3)), (50, 50*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    sum_tvec = sum_tvec + tmp
                    print(np.round(cv2.Rodrigues(rvec_array[i])[0], 3))
                
                # Average translation
                sum_tvec /= detected_num
                
                
                # cv2.putText(color_image, str(np.round(sum_tvec, 5)), (50, 50*(detected_num+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                
                # cv2.putText(color_image, str(np.round(cv2.Rodrigues(rvec_array[0]), 3)), 
                #             (50, 50*(detected_num+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                
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

