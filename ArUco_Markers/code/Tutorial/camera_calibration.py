import numpy as np
import cv2 as cv
import glob
import os


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
print(objp)

CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(os.path.join(os.path.dirname(BASE_PATH), "images", "calib_images", "calib_iphone", "*.jpg"))

# images = glob.glob('.\\images\\calib_images\\calib_iphone\\*.jpg')

print(images)
for fname in images:
    img = cv.imread(fname)
    img = cv.resize(img, (640, 480))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6, 9), None)

    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (6, 9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx, dist, rvecs, tvecs)



np.save(os.path.join(BASE_PATH, "calibration_data", "realsense_camera_matrix_iphone"), mtx)
np.save(os.path.join(BASE_PATH, "calibration_data", "realsense_distCoeffs_iphone"), dist)
np.save(os.path.join(BASE_PATH, "calibration_data", "realsense_rvecs_iphone"), rvecs)
np.save(os.path.join(BASE_PATH, "calibration_data", "realsense_tvecs_iphone"), tvecs)
