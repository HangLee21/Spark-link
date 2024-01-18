#!/usr/bin/env python

import cv2
import numpy as np
import glob
import json
import argparse
import yaml

name_dict = ['front', 'back', 'left', 'right']

def represent_list(dumper, data):
    # 将列表以[]形式输出
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def get_fish_data(index):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (5, 8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(f'../img/{index}_*.png')
    if len(images) == 0:
        print("no image")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # cv2.imshow('img',img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = img.shape[:2]
    calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                             cv2.fisheye.CALIB_CHECK_COND +
                             cv2.fisheye.CALIB_FIX_SKEW)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                (w, h),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dic = {
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": tuple_to_list(rvecs),
        "tvecs": tuple_to_list(tvecs)
    }
    cmx = []
    for i in mtx.tolist():
        for j in i:
            cmx.append(j)
    dmx = []
    for i in dist.tolist():
        for j in i:
            dmx.append(j)
    yaml_dict = {
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": cmx,
        },
        "dist_coeffs": {
            "rows": 4,
            "cols": 1,
            "dt": "d",
            "data": dmx,
        },
        "project_matrix": {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": [-9.3070874510219026e-01, -4.1405550648825917e+00,
                     1.1216126052183415e+03, 1.3669740891976151e-01,
                     -4.6651507085268138e+00, 1.0607568570787648e+03,
                     2.8474752086329793e-04, -6.8625514942363052e-03, 1.],
        },
        "shift_xy": {
            "rows": 2,
            "cols": 1,
            "dt": "f",
            "data": [0, 0],
        },
        "scale_xy": {
            "rows": 2,
            "cols": 1,
            "dt": "f",
            "data": [0, 0],
        },
        "resolution": {
            "rows": 2,
            "cols": 1,
            "dt": "i",
            "data": [1280, 720],
        }
    }
    # 将字典对象转换为 JSON 格式
    json_data = json.dumps(dic)
    yaml.add_representer(list, representer=represent_list)
    yaml_data = yaml.dump(yaml_dict, sort_keys=False)
    # 将 JSON 写入文件
    with open(f"../json/eye_{index}.json", "w") as file:
        file.write(json_data)
    yaml_data_with_header = "%YAML:1.0\n---\n" + yaml_data
    with open(f"../json/eye_{name_dict[index]}.yaml", "w") as f:
        f.write(yaml_data_with_header)

def get_data_json(index):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (5,8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(f'../img/{index}_*.png')
    if len(images) == 0:
        print("no image")
        return
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)

        #cv2.imshow('img',img)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]
    # ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],None,None)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dic = {
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": tuple_to_list(rvecs),
        "tvecs": tuple_to_list(tvecs)
    }
    cmx = []
    for i in mtx.tolist():
        for j in i:
            cmx.append(j)
    dmx = []
    for i in dist.tolist():
        for j in i:
            dmx.append(j)
    yaml_dict = {
        "camera_matrix":  {
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": cmx,
        },
        "dist_coeffs":{
            "rows": 5,
            "cols": 1,
            "dt": "d",
            "data": dmx,
        },
        "project_matrix":{
            "rows": 3,
            "cols": 3,
            "dt": "d",
            "data": [-9.3070874510219026e-01, -4.1405550648825917e+00,
                    1.1216126052183415e+03, 1.3669740891976151e-01,
                    -4.6651507085268138e+00, 1.0607568570787648e+03,
                    2.8474752086329793e-04, -6.8625514942363052e-03, 1. ],
        },
        "shift_xy":{
            "rows": 2,
            "cols": 1,
            "dt": "f",
            "data": [0,0],
        },
        "scale_xy":{
            "rows": 2,
            "cols": 1,
            "dt": "f",
            "data": [0,0],
        },
        "resolution":{
            "rows": 2,
            "cols": 1,
            "dt": "i",
            "data": [1280, 720],
        }
    }
    # 将字典对象转换为 JSON 格式
    json_data = json.dumps(dic)
    yaml.add_representer(list, representer=represent_list)
    yaml_data = yaml.dump(yaml_dict, sort_keys=False)
    # 将 JSON 写入文件
    with open(f"../json/{index}.json", "w") as file:
        file.write(json_data)
    yaml_data_with_header = "%YAML:1.0\n---\n"+yaml_data
    with open(f"../json/{name_dict[index]}.yaml", "w") as f:
        f.write(yaml_data_with_header)

# 递归函数，将元组转换为列表
def tuple_to_list(t):
    result = []
    for item in t:
        if isinstance(item, tuple):
            result.append(tuple_to_list(item))  # 递归调用，处理嵌套的元组
        elif isinstance(item, np.ndarray):
            result.append(item.tolist())  # 将 NumPy 数组转换为列表
        else:
            result.append(item)
    return result

def undistorted(path, index):
    # 从文件中读取 JSON 数据
    with open(f'../json/{index}.json', 'r') as file:
        json_data = json.load(file)
    camera_matrix = np.array(json_data["mtx"])
    dist_coeffs = np.array(json_data["dist"])
    print(camera_matrix)
    print(dist_coeffs)
    # 读取原始图像
    image = cv2.imread(path)
    if image is not None:
        print('success')
    else:
        print('Fail to read the image!')

    # 进行去畸变
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # 显示去畸变后的图像
    cv2.imshow("Origin Image", image)
    cv2.imshow('Undistorted Image', undistorted_image)
    cv2.imwrite(f"undistorted_image_{index}.png", undistorted_image)
    cv2.waitKey(0)

def get_fish_eye(path, index):
    # 从文件中读取 JSON 数据
    with open(f'../json/{index}.json', 'r') as file:
        json_data = json.load(file)
    camera_matrix = np.array(json_data["mtx"])
    dist_coeffs = np.array(json_data["dist"])
    # print(json_data["dist"])
    # dist_cffs = np.array([json_data["dist"][0][0], json_data["dist"][0][1], json_data["dist"][0][2], json_data["dist"][0][3]])
    image_size = (1280, 720)  # 图像尺寸
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, image_size,
                                                     cv2.CV_16SC2)
    image = cv2.imread(path)  # 读取待去畸变的图像

    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)  # 进行去畸变
    cv2.imshow("Origin Image,press ESC to close", image)
    cv2.imshow('Undistorted Image,press ESC to close', undistorted_image)
    
    key=cv2.waitKey(0)
    
    if key==27:
        #close all windows
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='video to rectify')
    parser.add_argument('--json_index', type=int, help='video to rectify')
    args = parser.parse_args()
    get_data_json(args.json_index)
    get_fish_data(args.json_index)
    #undistorted(args.image_path, args.json_index)
    
