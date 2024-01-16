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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
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


if __name__ == '__main__':
    for i in range(3):
        get_data_json(i)
