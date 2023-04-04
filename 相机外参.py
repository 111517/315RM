import cv2
import numpy as np
import math
obj_points=[[-6.8,3,0],[6.8,3,0],[-6.8,-3,0],[6.8,-3,0]]
mtx =[[320.98476465,0,351.02485862],[0,321.81408429,247.80478742],[0,0,1]]
mtx = np.array(mtx)
dist = [[-2.68653784e-02 , 5.68023859e-02 ,-6.36002456e-05 , 1.06945990e-03,-3.27264431e-02]]
dist = np.array(dist)
# 内参数矩阵
Camera_intrinsic = {"mtx": mtx, "dist": dist, }
img_points =[[287,126],[389,126],[287,298],[389,298]]  # 存储2D点
img_points = np.array(img_points,dtype=np.float64)
obj_points=np.array(obj_points,dtype=np.float64)
_, rvec, tvec = cv2.solvePnP(obj_points, img_points, Camera_intrinsic["mtx"], Camera_intrinsic["dist"])  # 解算位姿
distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)  # 计算距离
rvec_matrix = cv2.Rodrigues(rvec)[0]  # 旋转向量->旋转矩阵
print("旋转矩阵")
print(rvec_matrix)
print("tvec")
print(tvec)
