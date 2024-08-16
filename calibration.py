import cv2
import numpy as np
import glob
import os
import yaml

# Camera calibration
# set the amount of corner point on width and height.
w_corners = 10               # Set your own data : you can simple calculate the number of lattices on width,  and minus 1.
h_corners = 7              #Set your own data : you can calculate the number of lattices on height,  and minus 1.
images = glob.glob('/home/fate/grand_order/archetype_earth/*.jpg')          #change to your own path

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Capture the coordinate of Calibration Target's corner point
objp = np.zeros((w_corners * h_corners, 3), np.float32)
objp[:, :2] = np.mgrid[0:w_corners, 0:h_corners].T.reshape(-1, 2)
objp = objp * 15.09            # Set your own data : Here is the length of a lattice, with unit 'mm'.

obj_points = []  # save 3 dimension point
img_points = []  # save 2 dimension point

def save_calibration_to_yaml(file_path, cameraMatrix_l, distCoeffs_l):
    data = {
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_l.flatten().tolist()
        },
        'dist_coeff': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_l.flatten().tolist()
        }
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Calibration parameters saved to {file_path}")

for fname in images:
    if not os.path.exists(fname):
        print(f"File not exists : {fname}")
        continue

    try:
        with open(fname, 'rb') as f:
            print(f"Successfully read the file : {fname}")
    except Exception as e:
        print(f"Can not read the file : {fname}, error: {e}")
        continue

    img = cv2.imread(fname)
    if img is None:
        print(f"OpenCV can not read the file : {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (w_corners, h_corners), None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (w_corners, h_corners), corners, ret)
        i += 1

        new_size = (1920, 1080)
        resized_img = cv2.resize(img, new_size)
        cv2.imshow('img', resized_img)
        cv2.waitKey(150)

print(len(img_points))
cv2.destroyAllWindows()

if len(img_points) > 0:
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    save_calibration_to_yaml('calibration.yaml', mtx, dist)         # Change to your own yaml filename and path

    print("ret:", ret)
    print("mtx:\n", mtx)
    print("dist:\n", dist)
    print("rvecs:\n", rvecs)
    print("tvecs:\n", tvecs)

else:
    print("No corner point detected, Camera Calibration can not be processed.")
    
# Example intrinsic matrix, ret and dist:
# mtx:
#  [[1.61002745e+03 0.00000000e+00 6.75628136e+02]
#  [0.00000000e+00 1.61032435e+03 5.15242862e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

# ret: 0.11255757983475725

# dist:
#  [[-1.37463546e-01  4.57019501e-01  1.21836807e-03  2.60201908e-03
#   -1.36875725e+00]]

