import os 
import cv2
import cv2.aruco as aruco
import numpy as np


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

# 入力画像の読み込み
input_image = cv2.imread(os.path.join(BASE_PATH, "images", "4_tags.jpg"))
input_image = cv2.resize(input_image, (640, 480), interpolation=cv2.INTER_LINEAR)
print(input_image.shape)

# ArUCoマーカーの検出パラメータと辞書の設定
detector_params = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# ArUCoマーカーの検出
marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(input_image)

# 検出されたマーカーの描画
output_image = input_image.copy()
aruco.drawDetectedMarkers(output_image, marker_corners, marker_ids)

aruco_marker_side_length = 0.0575
objPoints = np.array([[-aruco_marker_side_length/2, aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2,  aruco_marker_side_length/2, 0],
                    [ aruco_marker_side_length/2, -aruco_marker_side_length/2, 0],
                    [-aruco_marker_side_length/2, -aruco_marker_side_length/2, 0]])

indices = vectorized_map_values(marker_ids).flatten()
arg_indices = np.argsort(indices)
detected_num = len(indices)
print(marker_ids, indices, arg_indices)

tvec_array = []

for i in range(4):
    retval, rvec, tvec = cv2.solvePnP(objectPoints=objPoints, imagePoints=marker_corners[i], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
    cv2.drawFrameAxes(output_image, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec, tvec=tvec, length=0.05)
    # print("rvec", marker_ids[i], cv2.Rodrigues(rvec)[0])
    print("tvec", marker_ids[i], tvec)
    tvec_array.append(tvec)

    
    
# 結果を表示
cv2.imshow("Detected Markers", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
