#!/usr/bin/env python

import cv2
import numpy as np
import glob
import json

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
    images = glob.glob(f'../img/{index}_*.jpg')
    if len(images) == 0:
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
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

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    # ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],None,None)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dic = {
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": tuple_to_list(rvecs),
        "tvecs": tuple_to_list(tvecs)
    }

    # 将字典对象转换为 JSON 格式
    json_data = json.dumps(dic)

    # 将 JSON 写入文件
    with open(f"../json/{index}.json", "w") as file:
        file.write(json_data)
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

    # 读取原始图像
    image = cv2.imread(path)

    # 进行去畸变
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # 显示去畸变后的图像
    cv2.imshow("Origin Image", image)
    cv2.imshow('Undistorted Image', undistorted_image)
    cv2.waitKey(0)

def get_fish_eye(path, index):
    # 从文件中读取 JSON 数据
    with open(f'../json/{index}.json', 'r') as file:
        json_data = json.load(file)
    camera_matrix = np.array(json_data["mtx"])
    dist_coeffs = np.array(json_data["dist"])
    print(camera_matrix.shape)  # 检查相机矩阵的尺寸
    print(dist_coeffs.shape)  # 检查畸变系数的尺寸
    # print(json_data["dist"])
    # dist_cffs = np.array([json_data["dist"][0][0], json_data["dist"][0][1], json_data["dist"][0][2], json_data["dist"][0][3]])
    image_size = (1280, 720)  # 图像尺寸
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), camera_matrix, image_size,
                                                     cv2.CV_16SC2)
    image = cv2.imread(path)  # 读取待去畸变的图像

    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)  # 进行去畸变
    cv2.imshow("Origin Image", image)
    cv2.imshow('Undistorted Image', undistorted_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    for i in range(4):
        get_data_json(i)
    undistorted("../img/0_210.jpg", 0)
    undistorted("../img/1_210.jpg", 0)
    undistorted("../img/2_210.jpg", 0)
    undistorted("../img/3_210.jpg", 0)
    # get_fish_eye("../img/0_420.jpg", 0)
