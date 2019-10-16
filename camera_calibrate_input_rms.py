#       made by yeolip.yoon@lge.com
#       camera_calibrate_input_depth ./square151
#       camera_calibrate_input_depth ./cx_cy/ stereo_config_init.json ./input_csv/
#       camera_calibrate_input_depth ./\input_lgit_964/ ./\input_lgit_964/stereo_config_result.json ./\input_lgit_964/
#       camera_calibrate_input_depth ./\input_lgit/ ./\input_lgit/stereo_config_result_lge.json ./\input_lgit/
#       camera_calibrate_input_depth ./\input_lgit/ ./\input_lgit/stereo_config_33_2_1.json ./\input_lgit/
#       camera_calibrate_input_depth ./\input_sm/ ./\input_sm/stereo_config2.json ./\input_sm/
#       camera_calibrate_input_depth ./\input_sm/ ./\input_sm/stereo_config_result_lge.json ./\input_sm/
#       camera_calibrate_input_depth ./dump_pattern ./dump_pattern/stereo_config.json
#       camera_calibrate_input_depth ./dump_pattern ./dump_pattern/stereo_config.json
#       camera_calibrate_input_depth ./\input_sm_idiada/ ./input_sm_idiada/stereo_config2.json ./input_sm_idiada/
#       camera_calibrate_input_depth ./\input_sm_idiada/ ./input_sm_idiada/stereo_config_result_lge.json ./input_sm_idiada/
#       camera_calibrate_input_depth ./\input_sm_lip/ ./input_sm_lip/stereo_config.json ./input_sm_lip/
#./data/rmsecal_input/ ./data/rmsecal_input/stereo_config_result_lge.json ./data/rmsecal_input/




# ../python_rmse/small_cal/3_cover/1/learn/1/ ../python_rmse/small_cal/3_taltail/stereo_config1.json ../python_rmse/small_cal/3_cover/1/learn/1/
# ../python_rmse/small_cal/3_cover/1/learn/3/ ../python_rmse/small_cal/3_taltail/stereo_config1.json ../python_rmse/small_cal/3_cover/1/learn/3/

# ../python_rmse/small_cal/3_cover/1/learn/5/ ../python_rmse/small_cal/3_taltail/stereo_config1.json ../python_rmse/small_cal/3_cover/1/learn/5/
#
# ../python_rmse/small_cal/3_cover/2/learn/1/ ../python_rmse/small_cal/3_taltail/stereo_config2.json ../python_rmse/small_cal/3_cover/2/learn/1/
# ../python_rmse/small_cal/3_cover/2/learn/3/ ../python_rmse/small_cal/3_taltail/stereo_config2.json ../python_rmse/small_cal/3_cover/2/learn/3/
# ../python_rmse/small_cal/3_cover/2/learn/5/ ../python_rmse/small_cal/3_taltail/stereo_config2.json ../python_rmse/small_cal/3_cover/2/learn/5/
#
#
# ../python_rmse/small_cal/3_cover/3/learn/1/ ../python_rmse/small_cal/3_taltail/stereo_config3.json ../python_rmse/small_cal/3_cover/3/learn/1/
# ../python_rmse/small_cal/3_cover/3/learn/3/ ../python_rmse/small_cal/3_taltail/stereo_config3.json ../python_rmse/small_cal/3_cover/3/learn/3/
# ../python_rmse/small_cal/3_cover/3/learn/5/ ../python_rmse/small_cal/3_taltail/stereo_config3.json ../python_rmse/small_cal/3_cover/3/learn/5/
#
#
# ../python_rmse/small_cal/15_cover/1/learn/1/ ../python_rmse/small_cal/15_taltail/stereo_config1.json ../python_rmse/small_cal/15_cover/1/learn/1/
# ../python_rmse/small_cal/15_cover/1/learn/3/ ../python_rmse/small_cal/15_taltail/stereo_config1.json ../python_rmse/small_cal/15_cover/1/learn/3/
# ../python_rmse/small_cal/15_cover/1/learn/5/ ../python_rmse/small_cal/15_taltail/stereo_config1.json ../python_rmse/small_cal/15_cover/1/learn/5/
#
# ../python_rmse/small_cal/15_cover/2/learn/1/ ../python_rmse/small_cal/15_taltail/stereo_config2.json ../python_rmse/small_cal/15_cover/2/learn/1/
# ../python_rmse/small_cal/15_cover/2/learn/3/ ../python_rmse/small_cal/15_taltail/stereo_config2.json ../python_rmse/small_cal/15_cover/2/learn/3/
# ../python_rmse/small_cal/15_cover/2/learn/5/ ../python_rmse/small_cal/15_taltail/stereo_config2.json ../python_rmse/small_cal/15_cover/2/learn/5/
#
# ../python_rmse/small_cal/15_cover/3/learn/1/ ../python_rmse/small_cal/15_taltail/stereo_config3.json ../python_rmse/small_cal/15_cover/3/learn/1/
# ../python_rmse/small_cal/15_cover/3/learn/3/ ../python_rmse/small_cal/15_taltail/stereo_config3.json ../python_rmse/small_cal/15_cover/3/learn/3/
# ../python_rmse/small_cal/15_cover/3/learn/5/ ../python_rmse/small_cal/15_taltail/stereo_config3.json ../python_rmse/small_cal/15_cover/3/learn/5/
#



import numpy as np
import cv2
import glob
#import argparse
import math
import pandas as pd
#lip  #
import matplotlib.pyplot as plt
import json
import os
import sys  #, getopt
import csv

##############
select_detect_pattern   = 0                                #circle: 0, square: 1
enable_debug_detect_pattern_from_image = 0                 #true: 1, false: 0
enable_debug_display_image_point_and_reproject_point = 0   #true: 1, false: 0
enable_debug_pose_estimation_display = 1                   #false: 0, all_enable: 1, left:2, right:3
enable_debug_loop_moving_of_rot_and_trans = 0              #false: 0, left: 1, right:2


enable_debug_dispatiry_estimation_display = 1              #true: 1, false: 0
select_png_or_raw       = 1                                #png: 0, raw: 1

# marker_point_x = 9
# marker_point_y = 9
# marker_length = -30
marker_point_x = 10     #pattern's width point
marker_point_y = 10     #pattern's height point
marker_length = 30      #pattern's gap (unit is cm)
degreeToRadian = math.pi/180
radianToDegree = 180/math.pi
###############

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def check_version_of_opencv():
    (major, minor, mdummy) = cv2.__version__.split(".")
    return (int(major), int(minor), int(mdummy) )

###load point on csv file
def load_point_from_csv(filename):
    ref_point = []
    img_point_l = []
    img_point_r = []

    fp = open(filename, 'r', encoding='utf-8')
    for row in csv.reader(fp):
        # print(row)
        # print(row[0], row[1], row[2])
        if (float(row[3]) < 0 or float(row[4]) < 0 or float(row[5]) < 0 or float(row[6]) < 0):
            print('skip', row, float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        else:
            ref_point.append([float(row[0]), float(row[1]), float(row[2])])
            img_point_l.append([float(row[3]), float(row[4])])
            img_point_r.append([float(row[5]), float(row[6])])
    fp.close()

    print(type(ref_point))
    ret_ref_point = np.array(ref_point, np.float32)
    ret_img_point_l = np.array(img_point_l, np.float32)
    ret_img_point_r = np.array(img_point_r, np.float32)
    print(type(ret_ref_point))

    return ret_ref_point, ret_img_point_l, ret_img_point_r


###load and get about stereo_config.json
def load_value_from_json(filename):
    # fpd = pd.read_json(filename)
    print(filename)
    fp = open(filename)
    print(fp)
    fjs = json.load(fp)
    print(fjs)
    print(type(fjs))

    m_fx, m_fy = fjs["master"]["lens_params"]['focal_len']
    m_cx, m_cy = fjs["master"]["lens_params"]['principal_point']
    m_k1 = fjs["master"]["lens_params"]['k1']
    m_k2 = fjs["master"]["lens_params"]['k2']
    print(m_fx, m_fy, m_cx, m_cy, m_k1, m_k2)
    s_fx, s_fy = fjs["slave"]["lens_params"]['focal_len']
    s_cx, s_cy = fjs["slave"]["lens_params"]['principal_point']
    s_k1 = fjs["slave"]["lens_params"]['k1']
    s_k2 = fjs["slave"]["lens_params"]['k2']
    print(s_fx, s_fy, s_cx, s_cy, s_k1, s_k2)
    print("*" * 50)

    calib_res = fjs["master"]["lens_params"]['calib_res']

    if fjs["slave"].get('w_Rt_c') is None:
        tranx, trany, tranz = fjs["slave"]["camera_pose"]['trans']
        rotx, roty, rotz = fjs["slave"]["camera_pose"]['rot']
        #rot should be set degree.
        print('1[',tranx, trany, tranz, '][',rotx *degreeToRadian , roty *degreeToRadian, rotz *degreeToRadian, ']')
    else:
        tranx, trany, tranz = fjs["slave"]["w_Rt_c"]['w_t_c']
        roty, rotx, rotz = fjs["slave"]["w_Rt_c"]['w_euleryxzdeg_c']
        print('2[',tranx, trany, tranz, '][',roty *degreeToRadian, rotx *degreeToRadian, rotz *degreeToRadian, ']')
    fp.close()

    return m_fx, m_fy, m_cx, m_cy, m_k1, m_k2, s_fx, s_fy, s_cx, s_cy, s_k1, s_k2, tranx, trany, tranz, \
           rotx *degreeToRadian, roty *degreeToRadian, rotz *degreeToRadian, calib_res


###save with result to stereo_config.json
def modify_value_from_json(filename, M1, d1, M2, d2, R, T, imgsize):
    # fpd = pd.read_json(filename)
    print("modify_value_from_json")
    fp = open(filename + '_sample.json')
    fjs = json.load(fp)
    # print(fjs)
    # print(type(fjs))

    # fjs["master"]["lens_params"]['focal_len'] = M1[0][0], M1[1][1]
    fjs["master"]["lens_params"]['focal_len'] = -M1[0][0], -M1[1][1]
    fjs["master"]["lens_params"]['principal_point'] = M1[0][2], M1[1][2]
    fjs["master"]["lens_params"]['k1'] = d1[0][0]
    fjs["master"]["lens_params"]['k2'] = d1[0][1]
    # fjs["slave"]["lens_params"]['focal_len'] = M2[0][0], M2[1][1]
    fjs["slave"]["lens_params"]['focal_len'] = -M2[0][0], -M2[1][1]
    fjs["slave"]["lens_params"]['principal_point'] = M2[0][2], M2[1][2]
    fjs["slave"]["lens_params"]['k1'] = d2[0][0]
    fjs["slave"]["lens_params"]['k2'] = d2[0][1]
    print("*" * 50)

    # fjs["slave"]["camera_pose"]['trans'] = *T[0], *T[1], *T[2]
    fjs["slave"]["camera_pose"]['trans'] = *T[0], *T[1], *T[2]
    euler = rotationMatrixToEulerAngles(R) * radianToDegree
    # print('euler', euler)
    fjs["slave"]["camera_pose"]['rot'] = euler[0], euler[1], euler[2]

    fjs["master"]["lens_params"]['calib_res'] = imgsize
    fjs["slave"]["lens_params"]['calib_res'] = imgsize

    wfp = open(filename + '_result' + '.json', 'w', encoding='utf-8')
    json.dump(fjs, wfp, indent=4)

    fp.close()
    wfp.close()

def modify_value_from_json_from_LGIT_to_SM(filename, M1, d1, M2, d2, R, T, imgsize):
    # fpd = pd.read_json(filename)
    print("modify_value_from_json")
    fp = open(filename + '_sample.json')
    fjs = json.load(fp)
    # print(fjs)
    # print(type(fjs))

    # fjs["master"]["lens_params"]['focal_len'] = M1[0][0], M1[1][1]
    fjs["master"]["lens_params"]['focal_len'] = M1[0][0], M1[1][1]
    fjs["master"]["lens_params"]['principal_point'] = M1[0][2], M1[1][2]
    fjs["master"]["lens_params"]['k1'] = d1[0][0]
    fjs["master"]["lens_params"]['k2'] = d1[0][1]
    # fjs["slave"]["lens_params"]['focal_len'] = M2[0][0], M2[1][1]
    fjs["slave"]["lens_params"]['focal_len'] = M2[0][0], M2[1][1]
    fjs["slave"]["lens_params"]['principal_point'] = M2[0][2], M2[1][2]
    fjs["slave"]["lens_params"]['k1'] = d2[0][0]
    fjs["slave"]["lens_params"]['k2'] = d2[0][1]
    print("*" * 50)

    # fjs["slave"]["camera_pose"]['trans'] = *T[0], *T[1], *T[2]
    fjs["slave"]["camera_pose"]['trans'] = T[0], T[1], T[2]
    euler = rotationMatrixToEulerAngles(R) * radianToDegree
    # print('euler', euler)
    fjs["slave"]["camera_pose"]['rot'] = euler[0], euler[1], euler[2]

    fjs["master"]["lens_params"]['calib_res'] = imgsize
    fjs["slave"]["lens_params"]['calib_res'] = imgsize

    wfp = open(filename + '_result_sm' + '.json', 'w', encoding='utf-8')
    json.dump(fjs, wfp, indent=4)

    fp.close()
    wfp.close()

###extarct and save coordinate point to csv file
def save_coordinate_both_stereo_obj_img(objpoints, imgpoints_l, imgpoints_r, count_ok_dual):
    table = []
    refpointx = []
    refpointy = []
    refpointz = []
    lpointx = []
    lpointy = []
    rpointx = []
    rpointy = []

    print(len(objpoints))
    print(len(imgpoints_l))
    print(len(imgpoints_r))
    print(len(objpoints) / count_ok_dual)
    # print('&&&&&&&&&obj&&&&&&', len(objpoints))
    # print(objpoints)
    # print('&&&&&&&&&img_l&&&&&&', len(imgpoints_l))
    # print(imgpoints_l)
    # print('&&&&&&&&&img_r&&&&&&', len(imgpoints_r))
    # print(imgpoints_r)
    print(type(imgpoints_l))

    for i in objpoints:
        for refpoint in i:
            # print(refpoint[0],refpoint[1],refpoint[2])
            refpointx.append(refpoint[0])
            refpointy.append(refpoint[1])
            refpointz.append(refpoint[2])

    for i in imgpoints_l:
        for j in i:
            for leftpoint in j:
                # print(leftpoint[0], leftpoint[1])
                lpointx.append(leftpoint[0])
                lpointy.append(leftpoint[1])
        # print(i.shape)
    print("&" * 50)
    for i in imgpoints_r:
        for j in i:
            for rightpoint in j:
                # print(rightpoint[0], rightpoint[1])
                rpointx.append(rightpoint[0])
                rpointy.append(rightpoint[1])

    # ret_table = np.zeros((count_ok_dual, len(refpointx)/count_ok_dual), np.float32)
    group_of_value = int(len(refpointx) / count_ok_dual)
    # ret_table = np.zeros((count_ok_dual, group_of_value), np.float32)
    # ret_table = [[0 for cols in range(group_of_value)] for rows in range(count_ok_dual)]
    ret_tables = []
    # print(ret_table)

    for i in range(0, len(refpointx), 1):
        table.append([refpointx[i], refpointy[i], refpointz[i], lpointx[i], lpointy[i], rpointx[i], rpointy[i]])

    col = ['refpX', 'refpY', 'refpZ', 'M_imgX', 'M_imgY', 'S_imgX', 'S_imgY']

    for j in range(0, count_ok_dual, 1):
        temp = []
        for i in range(0, group_of_value, 1):
            temp.append([refpointx[i + group_of_value * j], refpointy[i + group_of_value * j],refpointz[i + group_of_value * j],
                         lpointx[i + group_of_value * j], lpointy[i + group_of_value * j],
                         rpointx[i + group_of_value * j], rpointy[i + group_of_value * j]])
        temp = np.round(temp, 6)
        output2 = pd.DataFrame(temp, columns=col)
        tfilename = "p_from_img%03d.csv"%(j)
        output2.to_csv(tfilename, index=False, header=False)
        # output2.to_csv("calib_point_from_img" + str(j) + ".csv", index=False, header=False)
        # ret_tables.append(temp)

    # for j in range(0, count_ok_dual, 1):
    #     print('7'*50)
    #     print(ret_tables[j])

    table = np.round(table, 6)
    output = pd.DataFrame(table, columns=col)
    output.to_csv("total_p_from_img.csv", index=False, header=False)

def save_coordinate_both_stereo_obj_img_rectify(objpoints, imgpoints_l, imgpoints_r, count_ok_dual):
    table = []
    refpointx = []
    refpointy = []
    refpointz = []
    lpointx = []
    lpointy = []
    rpointx = []
    rpointy = []

    print(len(objpoints))
    print(len(imgpoints_l))
    print(len(imgpoints_r))
    print(len(objpoints) / count_ok_dual)

    for i in objpoints:
        for refpoint in i:
            refpointx.append(refpoint[0])
            refpointy.append(refpoint[1])
            refpointz.append(refpoint[2])

    for i in imgpoints_l:
        for j in i:
            for leftpoint in j:
                lpointx.append(leftpoint[0])
                lpointy.append(leftpoint[1])
        # print(i.shape)
    print("&" * 50)
    for i in imgpoints_r:
        for j in i:
            for rightpoint in j:
                rpointx.append(rightpoint[0])
                rpointy.append(rightpoint[1])

    group_of_value = int(len(refpointx) / count_ok_dual)

    for i in range(0, len(refpointx), 1):
        table.append([refpointx[i], refpointy[i], refpointz[i], lpointx[i], lpointy[i], rpointx[i], rpointy[i]])

    col = ['refpX', 'refpY', 'refpZ', 'M_imgX', 'M_imgY', 'S_imgX', 'S_imgY']

    for j in range(0, count_ok_dual, 1):
        temp = []
        for i in range(0, group_of_value, 1):
            temp.append([refpointx[i + group_of_value * j], refpointy[i + group_of_value * j],refpointz[i + group_of_value * j],
                         lpointx[i + group_of_value * j], lpointy[i + group_of_value * j],
                         rpointx[i + group_of_value * j], rpointy[i + group_of_value * j]])
        temp = np.round(temp, 6)
        output2 = pd.DataFrame(temp, columns=col)
        tfilename = "rectify_from_img%03d.csv"%(j)
        output2.to_csv(tfilename, index=False, header=False)


    table = np.round(table, 6)
    output = pd.DataFrame(table, columns=col)
    output.to_csv("rectifyTotal_p_from_img.csv", index=False, header=False)

#################################################################################################
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    # sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    sy = math.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])
    print(sy)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
        print(x, y, z)
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


#############################################################################################


class StereoCalibration(object):
    def __init__(self, argv):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        #self.objp = np.zeros((9 * 6, 3), np.float32)
        #self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objp = np.zeros((marker_point_x * marker_point_y, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0: marker_point_x, 0: marker_point_y ].T.reshape(-1, 2) * marker_length * 0.001
        # self.objp[:, :2] = self.objp[:, :2] - [((0 + (marker_point_x-1) * marker_length * 0.001) / 2), ((0 + (marker_point_y-1) * marker_length * 0.001) / 2)]
        # self.objp[:, :2] = self.objp[:, :2] - [((0 + (marker_point_x-1) * marker_length * 0.001) ), ((0 + (marker_point_y-1) * marker_length * 0.001) )]

        # print(self.objp)
        # patternsize = (10,10)
        # self.objs_test = np.zeros((np.prod(patternsize),3), np.float32)
        # self.objs_test[:,:2] = np.indices(patternsize).T.reshape(-1,2)
        # self.objs_test *= 30

        print(type(self.objp))
        # print(self.objp)

        # pose estimation
        #self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        self.axis = np.float32([[marker_length * 0.001 *3, 0, 0], [0, marker_length * 0.001 *3, 0], [0, 0, marker_length * 0.001 * -3]]).reshape(-1, 3)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        # self.cal_path = filepath
        if len(argv) >= 2:
            self.cal_path = argv[1]
            print('argv1', argv[1], 'argc', len(argv))

        if len(argv) >= 4:
            self.cal_loadjson = argv[2]
            self.cal_loadpoint = argv[3]
            print('argv2', argv[2], len(argv))
            #self.calc_rms_about_stereo(self.cal_path, self.cal_loadjson, self.cal_loadpoint)
            #self.read_points_with_stereo(self.cal_path, self.cal_loadjson, self.cal_loadpoint)
            self.read_points_with_mono_stereo(self.cal_path, self.cal_loadjson, self.cal_loadpoint)

        elif len(argv) >= 3:
            self.cal_loadjson = argv[2]
            print('argv2', argv[2], len(argv))
            self.read_oneimages_circle_with_stereo(self.cal_path, self.cal_loadjson)
        else:
            if(select_detect_pattern == 0):
                self.read_images_circle(self.cal_path)
            else:
                self.read_images_square(self.cal_path)

        pass

    def nothing(self,x):
        pass

    def loop_moving_of_rot_and_trans(self, cal_path, rot, tran):
        print('loop_moving_of_rot_and_trans')
        # input - all image, output - json, func- mono and stereo calib
        images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
        images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
        images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
        images_left = glob.glob(cal_path + '/LEFT/*.JPG')
        images_left += glob.glob(cal_path + '/LEFT/*.PNG')
        images_left += glob.glob(cal_path + '/LEFT/*.BMP')

        filemax = len(images_right)

        #  cv2.imshow('detected circles',cimg)
        #  cv2.waitKey(0)
        #  cv2.destroyAllWindows()
        tempOffset = 32763
        cv2.namedWindow('RightImage_RT', cv2.WINDOW_AUTOSIZE)  # WINDOW_AUTOSIZE  #WINDOW_NORMAL
        cv2.createTrackbar('fileNum', 'RightImage_RT', 0, filemax - 1, self.nothing)
        cv2.createTrackbar('Tx', 'RightImage_RT', tempOffset, 65535, self.nothing)
        cv2.createTrackbar('Ty', 'RightImage_RT', tempOffset, 65535, self.nothing)
        cv2.createTrackbar('Tz', 'RightImage_RT', tempOffset, 65535, self.nothing)
        cv2.createTrackbar('Rx', 'RightImage_RT', tempOffset, 65535, self.nothing)
        cv2.createTrackbar('Ry', 'RightImage_RT', tempOffset, 65535, self.nothing)
        cv2.createTrackbar('Rz', 'RightImage_RT', tempOffset, 65535, self.nothing)
        while (1):
            # get current positions of four trackbars
            filenum = cv2.getTrackbarPos('fileNum', 'RightImage_RT')
            tx_val = cv2.getTrackbarPos('Tx', 'RightImage_RT')
            ty_val = cv2.getTrackbarPos('Ty', 'RightImage_RT')
            tz_val = cv2.getTrackbarPos('Tz', 'RightImage_RT')
            rx_val = cv2.getTrackbarPos('Rx', 'RightImage_RT')
            ry_val = cv2.getTrackbarPos('Ry', 'RightImage_RT')
            rz_val = cv2.getTrackbarPos('Rz', 'RightImage_RT')

            tx = tx_val - tempOffset
            ty = ty_val - tempOffset
            tz = tz_val - tempOffset
            rx = rx_val - tempOffset
            ry = ry_val - tempOffset
            rz = rz_val - tempOffset

            # init
            img_l = cv2.imread(images_left[filenum])
            img_r = cv2.imread(images_right[filenum])
            print(images_right[filenum])
            #gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

            font = cv2.FONT_HERSHEY_SIMPLEX
            tempText = "tx=%f,ty=%f,tz=%f"%(tx,ty,tz)
            tempText2 = "rx=%f,ry=%f,rz=%f"%(rx,ry,rz)
            cv2.putText(img_r, tempText, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_r, tempText2, (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if(enable_debug_loop_moving_of_rot_and_trans == 1):
                _, rvec_l, tvec_l, _ = cv2.solvePnPRansac(self.objpoints[filenum], self.imgpoints_l[filenum], self.M1,self.d1)
                uR31 = rvec_l
                tran = tvec_l

            uR31 = rot   #cv2.Rodrigues(self.R)

            uR = np.zeros((3), np.float64)
            uR[0] = uR31[0] + (rx /1000 * degreeToRadian)
            uR[1] = uR31[1] + (ry /1000 * degreeToRadian)
            uR[2] = uR31[2] + (rz /1000 * degreeToRadian)
            uR33, _ = cv2.Rodrigues(uR)

            uT = np.zeros((3), np.float64)
            uT[0] = tran[0] + (tx/100000)
            uT[1] = tran[1] + (ty/100000)
            uT[2] = tran[2] + (tz/100000)
            tempText = "tx=%f,ty=%f,tz=%f"%(uT[0],uT[1],uT[2])
            tempText2 = "rx=%f,ry=%f,rz=%f"%(uR[0],uR[1],uR[2])
            cv2.putText(img_r, tempText, (10, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_r, tempText2, (10, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # print('uR', uR, '\nuT', uT, '\nuR33', uR33, '\n')

            if(enable_debug_loop_moving_of_rot_and_trans == 1):
                reprojected_points, _ = cv2.projectPoints(self.objpoints[filenum], uR33, uT, self.M1, self.d1)
                reprojected_points = reprojected_points.reshape(-1, 2)
            else:
                stero_rms, re_left, re_right = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r,
                                                                     self.M1, self.d1, self.M2, self.d2, uR33, uT)
                timgpoint_right = np.array(self.imgpoints_r[filenum])
                treproject_right = np.array(re_right[filenum])

            # print(i, images_left[filenum], images_right[filenum])
            timgpoint_left = np.array(self.imgpoints_l[filenum])

            for j, jname in enumerate(timgpoint_left):
                # print(timgpoint_left[j], '\t', treproject_left[j], '\t', timgpoint_right[j], '\t',
                #       treproject_right[j])
                if (enable_debug_loop_moving_of_rot_and_trans == 1):
                    self.draw_crossline(img_l, timgpoint_left[j][0], timgpoint_left[j][1], (0, 0, 255), 2)
                    self.draw_crossline(img_l, reprojected_points[j][0], reprojected_points[j][1], (0, 255, 0), 2)

                if (enable_debug_loop_moving_of_rot_and_trans == 2):
                    self.draw_crossline(img_r, timgpoint_right[j][0], timgpoint_right[j][1], (0, 0, 255), 2)
                    self.draw_crossline(img_r, treproject_right[j][0], treproject_right[j][1], (0, 255, 0), 2)

            # Draw and display the corners
            if (enable_debug_loop_moving_of_rot_and_trans == 1):
                cv2.imshow("LeftImage_RT", img_l)

            cv2.imshow("RightImage_RT", img_r)

            k = cv2.waitKey(1000) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        pass

    # input - one & all csv point, output - stereo rms calc
    def calc_rms_about_stereo(self, cal_path, cal_loadjson, cal_loadpoint):
        print('/////////calc_rms_about_stereo/////////')
        loadpoint = glob.glob(cal_loadpoint + '/*.CSV')
        print(loadpoint)
        for i, fname in enumerate(loadpoint):
            print(i)
            ref_point, img_point_l, img_point_r = load_point_from_csv(fname)

            self.objpoints.append(ref_point)
            self.imgpoints_l.append(img_point_l)
            self.imgpoints_r.append(img_point_r)

        m_fx, m_fy, m_cx, m_cy, m_k1, m_k2, s_fx, s_fy, s_cx, s_cy, s_k1, s_k2, tranx, trany, tranz, rotx, roty, rotz, calib_res = load_value_from_json(
            cal_loadjson)

        camera_matrix_l = np.zeros((3, 3), np.float64)
        camera_matrix_l[0][0] = -abs(m_fx)
        camera_matrix_l[1][1] = -abs(m_fy)
        camera_matrix_l[0][2] = m_cx
        camera_matrix_l[1][2] = m_cy
        camera_matrix_l[2][2] = 1.0

        dist_coef_l = np.zeros((1, 5), np.float64)
        dist_coef_l[0][0] = m_k1
        dist_coef_l[0][1] = m_k2
        dist_coef_l[0][2] = 0.0
        dist_coef_l[0][3] = 0.0
        dist_coef_l[0][4] = 0.0

        camera_matrix_r = np.zeros((3, 3), np.float64)
        camera_matrix_r[0][0] = -abs(s_fx)
        camera_matrix_r[1][1] = -abs(s_fy)
        camera_matrix_r[0][2] = s_cx
        camera_matrix_r[1][2] = s_cy
        camera_matrix_r[2][2] = 1.0

        dist_coef_r = np.zeros((1, 5), np.float64)
        dist_coef_r[0][0] = s_k1
        dist_coef_r[0][1] = s_k2
        dist_coef_r[0][2] = 0.0
        dist_coef_r[0][3] = 0.0
        dist_coef_r[0][4] = 0.0

        self.M1 = camera_matrix_l
        self.d1 = dist_coef_l
        self.M2 = camera_matrix_r
        self.d2 = dist_coef_r
        img_shape = (calib_res[0], calib_res[1])
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(type(self.M2), type(self.d2))
        # print(type(img_shape))
        # print(type(self.objpoints))
        # print(self.objpoints)
        # print(self.imgpoints_l)

        print('ret', m_fx, m_fy, m_cx, m_cy, m_k1, m_k2)
        print('ret', s_fx, s_fy, s_cx, s_cy, s_k1, s_k2)
        print('ret', tranx, trany, tranz, rotx, roty, rotz)

        # rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, camera_matrix,dist_coef, flags=flags)
        # rt2, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, camera_matrix,dist_coef, flags=flags)

        print("=" * 50)
        print('Input_Intrinsic_mtx_1', *np.round(self.M1, 5), sep='\n')
        print('Input_dist_1', np.round(self.d1, 5))
        print('Input_Intrinsic_mtx_2', *np.round(self.M2, 4), sep='\n')
        print('Input_dist_2', np.round(self.d2, 4))
        print("=" * 50)

        self.stereo_flags = 0
        self.stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        # self.stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # self.stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # self.stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # self.stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        # self.stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_K3
        self.stereo_flags |= cv2.CALIB_FIX_K4
        self.stereo_flags |= cv2.CALIB_FIX_K5

        uR = np.zeros((3), np.float64)
        uR[0] = -rotx
        uR[1] = -roty
        uR[2] = rotz
        uR33, _ = cv2.Rodrigues(uR)

        rotx2 = 1 * rotx
        roty2 = 1 * roty
        rotz2 = 1 * rotz    #radianToDegree

        uR333 = np.zeros((3,3), np.float64)
        uR333[0][0] = math.sin(rotx2) * math.sin(roty2) * math.sin(rotz2) + math.cos(roty2) *math.cos(rotz2)
        uR333[0][1] = math.sin(rotx2) * math.sin(roty2) * math.cos(rotz2) - math.sin(rotz2) * math.cos(roty2)
        uR333[0][2] = math.sin(roty2) * math.cos(rotx2)
        uR333[1][0] = math.sin(rotz2) * math.cos(rotx2)
        uR333[1][1] = math.cos(rotx2) * math.cos(rotz2)
        uR333[1][2] = -math.sin(rotx2)
        uR333[2][0] = math.sin(rotx2) * math.sin(rotz2) * math.cos(roty2) - math.sin(roty2) *math.cos(rotz2)
        uR333[2][1] = math.sin(rotx2) * math.cos(roty2) * math.cos(rotz2) + math.sin(roty2) *math.sin(rotz2)
        uR333[2][2] = math.cos(rotx2) * math.cos(roty2)

        uR333_zyx = np.zeros((3,3), np.float64)
        uR333_zyx[0][0] = math.cos(rotx2) * math.cos(roty2)
        uR333_zyx[0][1] = math.cos(rotx2) * math.sin(roty2) * math.sin(rotz2) - math.sin(rotx2) * math.cos(rotz2)
        uR333_zyx[0][2] = math.cos(rotx2) * math.sin(roty2) * math.cos(rotz2) + math.sin(rotx2) * math.sin(rotz2)
        uR333_zyx[1][0] = math.sin(rotx2) * math.cos(roty2)
        uR333_zyx[1][1] = math.sin(rotx2) * math.sin(roty2) * math.sin(rotz2) + math.cos(rotx2) * math.cos(rotz2)
        uR333_zyx[1][2] = math.sin(rotx2) * math.sin(roty2) * math.cos(rotz2) - math.cos(rotx2) * math.sin(rotz2)
        uR333_zyx[2][0] = -math.sin(roty2)
        uR333_zyx[2][1] = math.cos(roty2) * math.sin(rotz2)
        uR333_zyx[2][2] = math.cos(roty2) * math.cos(rotz2)

        uT = np.zeros((3), np.float64)
        uT[0] = -tranx
        uT[1] = -trany
        uT[2] = tranz
        print('uR', uR, '\nuT', uT, '\nuR33', uR33, '\n', uR333 , '\n', uR333_zyx)

        w_Rt_c = np.eye(4)
        w_Rt_c[0:3, 0:3] = eulerAnglesToRotationMatrix(uR)
        w_Rt_c[0:3, 3] = uT
        print('w_Rt_c',w_Rt_c)
        c_Rt_w = np.linalg.inv(w_Rt_c)
        print('c_Rt_w', c_Rt_w)

        #lip180524 - stero calibration을 지우고, R,T를 json데이터에서 받아 처리해라.

        #self.camera_model = self.stereo_calibrate(img_shape)

        # self.R = np.array([[0.99997492, -0.00157674, -0.00690528],
        #                   [0.0016571, 0.9999308, 0.01164693],
        #                   [0.00688644, -0.01165808, 0.99990833]])
        # self.T = np.array([-0.09235426, -0.00045917, -0.0007854])
        # print("=" * 50)
        # single_rms_l, _, _ = self.reprojection_error2(self.objpoints, self.imgpoints_l,_, _, self.M1, self.d1)
        # print('manual_rms1', single_rms_l)
        # print("=" * 50)
        #
        # # stero_rms, re_left, re_right = self.calc_rms_stereo3(self.objpoints, self.imgpoints_l, self.imgpoints_r,
        # #                                                      self.M1, self.d1, self.M2, self.d2, uR33, uT)
        #stero_rms, re_left, re_right = self.calc_rms_stereo3(self.objpoints, self.imgpoints_l, self.imgpoints_r,
        #                                                      self.M1, self.d1, self.M2, self.d2, w_Rt_c[0:3, 0:3], w_Rt_c[0:3, 3])

        modify_value_from_json_from_LGIT_to_SM("stereo_config",self.M1, self.d1, self.M2, self.d2, c_Rt_w[0:3, 0:3], c_Rt_w[0:3, 3], img_shape)

        if(enable_debug_display_image_point_and_reproject_point == 1):
            self.display_reprojection_point_and_image_point(cal_path, self.imgpoints_l, self.imgpoints_r, re_left, re_right)

        # if(enable_debug_loop_moving_of_rot_and_trans != 0):
        #     self.loop_moving_of_rot_and_trans(cal_path, uR, uT)
        print("END - calc_rms_about_stereo")

    # input - all point, output - json, func- mono and stereo calib
    def read_points_with_mono_stereo(self, cal_path, cal_loadjson, cal_loadpoint):
        loadpoint = glob.glob(cal_loadpoint + '/*.CSV')
        # loadpoint.sort()
        print(loadpoint)

        for i, fname in enumerate(loadpoint):
            print(i)
            ref_point, img_point_l, img_point_r = load_point_from_csv(fname)

            self.objpoints.append(ref_point)
            self.imgpoints_l.append(img_point_l)
            self.imgpoints_r.append(img_point_r)

        print("//////read_points_with_mono_stereo/////////")
        flags = 0
        #flags |= cv2.CALIB_FIX_INTRINSIC
        #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5

        m_fx, m_fy, m_cx, m_cy, m_k1, m_k2, s_fx, s_fy, s_cx, s_cy, s_k1, s_k2, tranx, trany, tranz, rotx, roty, rotz, calib_res = load_value_from_json(
            cal_loadjson)

        m_fx = m_fy = s_fx = s_fy = 1470.0
        m_cx = s_cx = 640
        m_cy = s_cy = 482
        m_k1 = s_k1 = -0.1
        m_k2 = s_k2 = -0.2

        camera_matrix_l = np.zeros((3, 3), np.float64)
        camera_matrix_l[0][0] = abs(m_fx)
        camera_matrix_l[1][1] = abs(m_fy)
        camera_matrix_l[0][2] = m_cx
        camera_matrix_l[1][2] = m_cy
        camera_matrix_l[2][2] = 1.0

        dist_coef_l = np.zeros((1, 5), np.float64)
        dist_coef_l[0][0] = m_k1
        dist_coef_l[0][1] = m_k2
        dist_coef_l[0][2] = 0.0
        dist_coef_l[0][3] = 0.0
        dist_coef_l[0][4] = 0.0

        camera_matrix_r = np.zeros((3, 3), np.float64)
        camera_matrix_r[0][0] = abs(s_fx)
        camera_matrix_r[1][1] = abs(s_fy)
        camera_matrix_r[0][2] = s_cx
        camera_matrix_r[1][2] = s_cy
        camera_matrix_r[2][2] = 1.0

        dist_coef_r = np.zeros((1, 5), np.float32)
        dist_coef_r[0][0] = s_k1
        dist_coef_r[0][1] = s_k2
        dist_coef_r[0][2] = 0.0
        dist_coef_r[0][3] = 0.0
        dist_coef_r[0][4] = 0.0

        # self.M1 = camera_matrix_l
        # self.d1 = dist_coef_l
        # self.M2 = camera_matrix_r
        # self.d2 = dist_coef_r
        img_shape = (calib_res[0], calib_res[1])
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(type(self.M2), type(self.d2))
        # print(type(img_shape))
        # print(type(self.objpoints))
        # print(self.objpoints)
        # print(self.imgpoints_l)

        print('ret', m_fx, m_fy, m_cx, m_cy, m_k1, m_k2)
        print('ret', s_fx, s_fy, s_cx, s_cy, s_k1, s_k2)
        print('ret', tranx, trany, tranz, rotx, roty, rotz)

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape,
                                                                     camera_matrix_l, dist_coef_l, flags=flags)
        rt2, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape,
                                                                      camera_matrix_r, dist_coef_r, flags=flags)

        print("=" * 50)
        print('Input_Intrinsic_mtx_1', *np.round(self.M1, 5), sep='\n')
        print('Input_dist_1', np.round(self.d1, 5))
        print('Input_Intrinsic_mtx_2', *np.round(self.M2, 4), sep='\n')
        print('Input_dist_2', np.round(self.d2, 4))
        print("=" * 50)
        single_rms_l, _, _  = self.reprojection_error(self.objpoints, self.imgpoints_l, self.r1, self.t1, self.M1, self.d1)
        single_rms_r, _, _  = self.reprojection_error(self.objpoints, self.imgpoints_r, self.r2, self.t2, self.M2, self.d2)
        print('manual_rms1', single_rms_l)
        print('manual_rms1', single_rms_r)
        print("=" * 50)

        # print( '\n',self.r2, '\n', self.t2)

        temp_r1 = np.copy(self.r1)
        temp_t1 = np.copy(self.t1)
        temp_M1 = np.copy(self.M1)
        temp_d1 = np.copy(self.d1)
        temp_r2 = np.copy(self.r2)
        temp_t2 = np.copy(self.t2)
        temp_M2 = np.copy(self.M2)
        temp_d2 = np.copy(self.d2)

        self.stereo_flags = 0
        # self.stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        #self.stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        self.stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # self.stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # self.stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        # self.stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_K3
        self.stereo_flags |= cv2.CALIB_FIX_K4
        self.stereo_flags |= cv2.CALIB_FIX_K5

        self.camera_model = self.stereo_calibrate(img_shape)

        print("/" * 50)
        print(self.M1, self.d1, self.M2, self.d2)
        print(type(self.M1), type(self.d1), type(self.M2), type(self.d2))
        print(self.M1.shape, self.d1.shape, self.M2.shape, self.d2.shape)
        print("/" * 50)
        print(temp_M1, temp_d1, temp_M2, temp_d2)
        print(type(temp_M1), type(temp_d1), type(temp_M2), type(temp_d2))
        print(temp_M1.shape, temp_d1.shape, temp_M2.shape, temp_d2.shape)
        print("/" * 50)

        # cv2.useOptimized(0)
        stero_rms, re_left, re_right = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r, self.M1, self.d1, self.M2, self.d2, self.R, self.T)
        print(type(self.imgpoints_l),type(re_left))
        #print((self.imgpoints_l), (re_left))
        if(enable_debug_display_image_point_and_reproject_point == 1):
            self.display_reprojection_point_and_image_point(cal_path, self.imgpoints_l, self.imgpoints_r, re_left, re_right)

        print("END - read_points_with_mono_stereo")
        pass


    # input - all point, output - json, func- stereo calib
    def read_points_with_stereo(self, cal_path, cal_loadjson, cal_loadpoint):
        loadpoint = glob.glob(cal_loadpoint + '/*.CSV')
        # loadpoint.sort()
        # print(loadpoint)

        for i, fname in enumerate(loadpoint):
            print(i)
            ref_point, img_point_l, img_point_r = load_point_from_csv(fname)

            self.objpoints.append(ref_point)
            self.imgpoints_l.append(img_point_l)
            self.imgpoints_r.append(img_point_r)

        # flags = 0
        # #flags |= cv2.CALIB_FIX_INTRINSIC
        # # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # # flags |= cv2.CALIB_RATIONAL_MODEL
        # #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        m_fx, m_fy, m_cx, m_cy, m_k1, m_k2, s_fx, s_fy, s_cx, s_cy, s_k1, s_k2, tranx, trany, tranz, rotx, roty, rotz, calib_res = load_value_from_json(
            cal_loadjson)

        # m_fx = m_fy = s_fx = s_fy = 1470.0
        # m_cx = s_cx = 640
        # m_cy = s_cy = 482
        # m_k1 = s_k1 = -0.1
        # m_k2 = s_k2 = -0.2

        camera_matrix_l = np.zeros((3, 3), np.float32)
        camera_matrix_l[0][0] = abs(m_fx)
        camera_matrix_l[1][1] = abs(m_fy)
        camera_matrix_l[0][2] = m_cx
        camera_matrix_l[1][2] = m_cy
        camera_matrix_l[2][2] = 1.0

        dist_coef_l = np.zeros((1, 5), np.float32)
        dist_coef_l[0][0] = m_k1
        dist_coef_l[0][1] = m_k2
        dist_coef_l[0][2] = 0.0
        dist_coef_l[0][3] = 0.0
        dist_coef_l[0][4] = 0.0

        camera_matrix_r = np.zeros((3, 3), np.float32)
        camera_matrix_r[0][0] = abs(s_fx)
        camera_matrix_r[1][1] = abs(s_fy)
        camera_matrix_r[0][2] = s_cx
        camera_matrix_r[1][2] = s_cy
        camera_matrix_r[2][2] = 1.0

        dist_coef_r = np.zeros((1, 5), np.float32)
        dist_coef_r[0][0] = s_k1
        dist_coef_r[0][1] = s_k2
        dist_coef_r[0][2] = 0.0
        dist_coef_r[0][3] = 0.0
        dist_coef_r[0][4] = 0.0

        self.M1 = camera_matrix_l
        self.d1 = dist_coef_l
        self.M2 = camera_matrix_r
        self.d2 = dist_coef_r
        img_shape = (calib_res[0], calib_res[1])
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(type(self.M2), type(self.d2))
        # print(type(img_shape))
        # print(type(self.objpoints))
        # print(self.objpoints)
        # print(self.imgpoints_l)

        print('ret', m_fx, m_fy, m_cx, m_cy, m_k1, m_k2)
        print('ret', s_fx, s_fy, s_cx, s_cy, s_k1, s_k2)
        print('ret', tranx, trany, tranz, rotx, roty, rotz)

        # rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, camera_matrix,dist_coef, flags=flags)
        # rt2, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, camera_matrix,dist_coef, flags=flags)

        print("=" * 50)
        print('Input_Intrinsic_mtx_1', *np.round(self.M1, 5), sep='\n')
        print('Input_dist_1', np.round(self.d1, 5))
        print('Input_Intrinsic_mtx_2', *np.round(self.M2, 4), sep='\n')
        print('Input_dist_2', np.round(self.d2, 4))
        print("=" * 50)

        self.stereo_flags = 0
        #self.stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        # self.stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        self.stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # self.stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # self.stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        # self.stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_K3
        self.stereo_flags |= cv2.CALIB_FIX_K4
        self.stereo_flags |= cv2.CALIB_FIX_K5


        uR = np.zeros((3), np.float32)
        uR[0] = rotx
        uR[1] = roty
        uR[2] = rotz
        uR33, _ = cv2.Rodrigues(uR)
        self.uR33 = uR33

        uT = np.zeros((3), np.float32)
        uT[0] = tranx
        uT[1] = trany
        uT[2] = tranz
        self.uT31 = uT

        self.camera_model = self.stereo_calibrate(img_shape)

        # #########################################################################
        # uR = np.zeros((3), np.float64)
        # uR[0] = -rotx
        # uR[1] = -roty
        # uR[2] = rotz
        # uR33, _ = cv2.Rodrigues(uR)
        #
        # uT = np.zeros((3), np.float64)
        # uT[0] = -tranx
        # uT[1] = -trany
        # uT[2] = tranz
        # print('uR', uR, '\nuT', uT, '\nuR33', uR33, '\n')
        #
        # w_Rt_c = np.eye(4)
        # w_Rt_c[0:3, 0:3] = eulerAnglesToRotationMatrix(uR)
        # w_Rt_c[0:3, 3] = uT
        # print('w_Rt_c',w_Rt_c)
        # c_Rt_w = np.linalg.inv(w_Rt_c)
        # print('c_Rt_w', c_Rt_w)
        #
        # modify_value_from_json_from_LGIT_to_SM("stereo_config",self.M1, self.d1, self.M2, self.d2, c_Rt_w[0:3, 0:3], c_Rt_w[0:3, 3], img_shape)

        # stero_rms, re_left, re_right = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r,
        #                                                      self.M1, self.d1, self.M2, self.d2, self.R, self.T)
        # if(enable_debug_display_image_point_and_reproject_point == 1):
        #     self.display_reprojection_point_and_image_point(cal_path, self.imgpoints_l, self.imgpoints_r, re_left, re_right)

        print("END - read_points_with_stereo")
        pass

    # input - one image and json, output - json, func- stereo calib
    def read_oneimages_circle_with_stereo(self, cal_path, cal_loadjson):
        print("/////////read_oneimages_circle_with_stereo////////////")
        count_ok_dual = 0

        if (select_png_or_raw == 1):
            print(cal_path + '/RIGHT/*.RAW')
            images_right = glob.glob(cal_path + '/RIGHT/*.RAW')
            images_left = glob.glob(cal_path + '/LEFT/*.RAW')
        else:
            print(cal_path + '/RIGHT/*.JPG')
            images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
            images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
            images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
            images_left = glob.glob(cal_path + '/LEFT/*.JPG')
            images_left += glob.glob(cal_path + '/LEFT/*.PNG')
            images_left += glob.glob(cal_path + '/LEFT/*.BMP')
        # images_left.sort()
        # images_right.sort()
        print(images_left)
        print(images_right)

        for i, fname in enumerate(images_right):
            if (select_png_or_raw == 1):
                fd_l = open(images_left[i], 'rb')
                fd_r = open(images_right[i], 'rb')
                length = len(fd_l.read())
                fd_l.seek(0)
                #print(length)
                rows = 1280
                cols = int(length / rows)
                f_l = np.fromfile(fd_l, dtype=np.uint8, count=rows * cols)
                f_r = np.fromfile(fd_r, dtype=np.uint8, count=rows * cols)
                gray_l = img_l = f_l.reshape((cols, rows))
                gray_r = img_r = f_r.reshape((cols, rows))

                fd_l.close
                fd_r.close
                #cv2.imshow(images_left[i], img_l)
                #cv2.imshow(images_right[i], img_r)
                #cv2.waitKey(20500)

            else:
                img_l = cv2.imread(images_left[i])
                img_r = cv2.imread(images_right[i])

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            #gray_l = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
            #gray_r = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
            # Find the chess board corners
            if(select_detect_pattern == 1):
                ret, gray_l = cv2.threshold(gray_l, 50, 200, cv2.THRESH_BINARY)
                ret, gray_r = cv2.threshold(gray_r, 50, 200, cv2.THRESH_BINARY)
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, (marker_point_x, marker_point_y))
                #ret_l, corners_l = cv2.findChessboardCorners(gray_l, (marker_point_x, marker_point_y), flags = cv2.CALIB_CB_ADAPTIVE_THRESH  )
                #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
                #flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, (marker_point_x, marker_point_y))
                #ret_r, corners_r = cv2.findChessboardCorners(gray_r, (marker_point_x, marker_point_y), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

                #if(enable_debug_detect_pattern_from_image == 1):
                #    cv2.imshow(images_left[i], gray_l)
                #    cv2.waitKey(0)
            else:
                # ret, gray_l = cv2.threshold(gray_l, 40, 255, cv2.THRESH_BINARY)
                # ret, gray_r = cv2.threshold(gray_r, 40, 255, cv2.THRESH_BINARY)
                ret_l, corners_l = cv2.findCirclesGrid(gray_l, (marker_point_x, marker_point_y),
                                                       flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                ret_r, corners_r = cv2.findCirclesGrid(gray_r, (marker_point_x, marker_point_y),
                                                       flags=cv2.CALIB_CB_SYMMETRIC_GRID)

            print((images_left[i], ret_l))
            print((images_right[i], ret_r))

            if ret_l is True and ret_r is True:
                count_ok_dual += 1
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                if (select_detect_pattern == 1):
                    rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),(-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)
                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (marker_point_x, marker_point_y), corners_l, ret_l)

                if(enable_debug_detect_pattern_from_image == 1):
                    cv2.imshow(images_left[i], img_l)
                    cv2.waitKey(500)

                if (select_detect_pattern == 1):
                    rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)
                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (marker_point_x, marker_point_y), corners_r, ret_r)

                if(enable_debug_detect_pattern_from_image == 1):
                    cv2.imshow(images_right[i], img_r)
                    cv2.waitKey(500)

            print(gray_l.shape[::-1], type(gray_l.shape[::-1]))
            img_shape = gray_r.shape[::-1]

        print(len(self.objpoints))
        print(len(self.imgpoints_l))
        print(len(self.imgpoints_r))

        save_coordinate_both_stereo_obj_img(self.objpoints, self.imgpoints_l, self.imgpoints_r, count_ok_dual)

        # flags = 0
        # #flags |= cv2.CALIB_FIX_INTRINSIC
        # # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # # flags |= cv2.CALIB_RATIONAL_MODEL
        # #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        m_fx, m_fy, m_cx, m_cy, m_k1, m_k2, s_fx, s_fy, s_cx, s_cy, s_k1, s_k2, tranx, trany, tranz, rotx, roty, rotz, calib_res = load_value_from_json(
            cal_loadjson)

        camera_matrix_l = np.zeros((3, 3), np.float32)
        camera_matrix_l[0][0] = abs(m_fx)
        camera_matrix_l[1][1] = abs(m_fy)
        camera_matrix_l[0][2] = m_cx
        camera_matrix_l[1][2] = m_cy
        camera_matrix_l[2][2] = 1.0

        dist_coef_l = np.zeros((1, 5), np.float32)
        dist_coef_l[0][0] = m_k1
        dist_coef_l[0][1] = m_k2
        dist_coef_l[0][2] = 0.0
        dist_coef_l[0][3] = 0.0
        dist_coef_l[0][4] = 0.0

        camera_matrix_r = np.zeros((3, 3), np.float32)
        camera_matrix_r[0][0] = abs(s_fx)
        camera_matrix_r[1][1] = abs(s_fy)
        camera_matrix_r[0][2] = s_cx
        camera_matrix_r[1][2] = s_cy
        camera_matrix_r[2][2] = 1.0

        dist_coef_r = np.zeros((1, 5), np.float32)
        dist_coef_r[0][0] = s_k1
        dist_coef_r[0][1] = s_k2
        dist_coef_r[0][2] = 0.0
        dist_coef_r[0][3] = 0.0
        dist_coef_r[0][4] = 0.0

        self.M1 = camera_matrix_l
        self.d1 = dist_coef_l
        self.M2 = camera_matrix_r
        self.d2 = dist_coef_r
        img_shape = (calib_res[0], calib_res[1])
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        print(type(self.M2), type(self.d2))
        print(type(self.objpoints))
        # print(self.objpoints)
        # print(self.imgpoints_l)

        print('ret', m_fx, m_fy, m_cx, m_cy, m_k1, m_k2)
        print('ret', s_fx, s_fy, s_cx, s_cy, s_k1, s_k2)
        print('ret', tranx, trany, tranz, rotx, roty, rotz)

        # rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, camera_matrix_l,dist_coef_l, flags=flags)
        # rt2, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, camera_matrix_r,dist_coef_r, flags=flags)

        print("=" * 50)
        print('Input_Intrinsic_mtx_1', *np.round(self.M1, 5), sep='\n')
        print('Input_dist_1', np.round(self.d1, 5))
        print('Input_Intrinsic_mtx_2', *np.round(self.M2, 4), sep='\n')
        print('Input_dist_2', np.round(self.d2, 4))
        print("=" * 50)

        self.stereo_flags = 0
        #self.stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        # self.stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        self.stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # self.stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # self.stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # self.stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        # self.stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_K3
        self.stereo_flags |= cv2.CALIB_FIX_K4
        self.stereo_flags |= cv2.CALIB_FIX_K5

        self.camera_model = self.stereo_calibrate(img_shape)
        # stero_rms, _, _ = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r, self.M1, self.d1, self.M2, self.d2, self.R, self.T)

        # uR = np.zeros((3), np.float64)
        # uR[0] = rotx #* radianToDegree
        # uR[1] = roty #* radianToDegree
        # uR[2] = rotz #* radianToDegree
        # uR33, _ = cv2.Rodrigues(uR)
        #
        # uT = np.zeros((3), np.float64)
        # uT[0] = tranx
        # uT[1] = trany
        # uT[2] = tranz
        # print('uR', uR, '\nuT', uT, '\nuR33', uR33, '\n')
        # stero_rms, _, _ = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r, self.M1, self.d1,
        #                                         self.M2, self.d2, uR33, uT)


        # uR = np.zeros((3), np.float64)
        # uR[0] = -rotationMatrixToEulerAngles(self.R)[0]
        # uR[1] = -rotationMatrixToEulerAngles(self.R)[1]
        # uR[2] = rotationMatrixToEulerAngles(self.R)[2]
        # uR33, _ = cv2.Rodrigues(uR)
        #
        # uT = np.zeros((3), np.float64)
        # uT[0] = -self.T[0]
        # uT[1] = -self.T[1]
        # uT[2] = self.T[2]
        # # print('uR', uR, '\nuT', uT, '\nuR33', uR33, '\n')
        #
        # w_Rt_c = np.eye(4)
        # w_Rt_c[0:3, 0:3] = eulerAnglesToRotationMatrix(uR)
        # w_Rt_c[0:3, 3] = uT
        # # print('w_Rt_c',w_Rt_c)
        # c_Rt_w = np.linalg.inv(w_Rt_c)
        # # print('c_Rt_w', c_Rt_w)
        #
        #
        # modify_value_from_json_from_LGIT_to_SM("stereo_config",self.Ml, self.dl, self.Mr, self.dr, c_Rt_w[0:3, 0:3], c_Rt_w[0:3, 3], img_shape)


        print("END - read_oneimages_circle_with_stereo")
        pass

    def read_images_square(self, cal_path):
        count_ok_dual = 0

        if (select_png_or_raw == 1):
            print(cal_path + '/RIGHT/*.RAW')
            images_right = glob.glob(cal_path + '/RIGHT/*.RAW')
            images_left = glob.glob(cal_path + '/LEFT/*.RAW')
        else:
            print(cal_path + '/RIGHT/*.JPG')
            images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
            images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
            images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
            images_left = glob.glob(cal_path + '/LEFT/*.JPG')
            images_left += glob.glob(cal_path + '/LEFT/*.PNG')
            images_left += glob.glob(cal_path + '/LEFT/*.BMP')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            if (select_png_or_raw == 1):
                fd_l = open(images_left[i], 'rb')
                fd_r = open(images_right[i], 'rb')
                length = len(fd_l.read())
                fd_l.seek(0)
                #print(length)
                rows = 1280
                cols = int(length / rows)
                f_l = np.fromfile(fd_l, dtype=np.uint8, count=rows * cols)
                f_r = np.fromfile(fd_r, dtype=np.uint8, count=rows * cols)
                gray_l = img_l = f_l.reshape((cols, rows))
                gray_r = img_r = f_r.reshape((cols, rows))

                fd_l.close
                fd_r.close
                #cv2.imshow(images_left[i], img_l)
                #cv2.imshow(images_right[i], img_r)
                #cv2.waitKey(20500)

            else:
                img_l = cv2.imread(images_left[i])
                img_r = cv2.imread(images_right[i])

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            # gray_l = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
            # gray_r = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (marker_point_x, marker_point_y), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (marker_point_x, marker_point_y), None)
            # ret_l, corners_l = cv2.findCirclesGrid(gray_l, (10, 10) , None) #flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
            # ret_r, corners_r = cv2.findCirclesGrid(gray_r, (10, 10) , None)

            print((images_left[i], ret_l))
            print((images_right[i], ret_r))

            if True: #ret_l is True and ret_r is True:
                #print(corners_l)
                count_ok_dual += 1
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

#            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (marker_point_x, marker_point_y),
                                                  corners_l, ret_l)
                if(enable_debug_detect_pattern_from_image == 1):
                    cv2.imshow(images_left[i], img_l)
                    cv2.waitKey(0)

 #           if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (marker_point_x, marker_point_y),
                                                  corners_r, ret_r)
                if(enable_debug_detect_pattern_from_image == 1):
                    cv2.imshow(images_right[i], img_r)
                    cv2.waitKey(0)

            print(gray_l.shape[::-1])
            img_shape = gray_r.shape[::-1]

        save_coordinate_both_stereo_obj_img(self.objpoints, self.imgpoints_l, self.imgpoints_r, count_ok_dual)

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None,
                                                                     None)
        print('rms1', rt)
        rt2, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None,
                                                                      None)

        print("=" * 50)
        single_rms_l, _, _  = self.reprojection_error(self.objpoints, self.imgpoints_l, self.r1, self.t1, self.M1, self.d1)
        single_rms_r, _, _  = self.reprojection_error(self.objpoints, self.imgpoints_r, self.r2, self.t2, self.M2, self.d2)
        print('manual_rms1', single_rms_l)
        print('manual_rms1', single_rms_r)
        print("=" * 50)

        self.stereo_flags = 0
        # self.stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        # self.stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        self.stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # self.stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # self.stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        # self.stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_K3
        self.stereo_flags |= cv2.CALIB_FIX_K4
        self.stereo_flags |= cv2.CALIB_FIX_K5

        self.camera_model = self.stereo_calibrate(img_shape)


        if(enable_debug_dispatiry_estimation_display == 1):
            self.mapL1, self.mapL2, self.mapR1, self.mapR2, _ , _ , _ , _ = self.stereo_rectify(img_shape, self.M1, self.d1, self.M2, self.d2, self.R, self.T, images_left[0], images_right[0])
            self.depth_using_stereo(img_shape, self.M1, self.d1, self.M2, self.d2, self.R, self.T)
        # self.mapL1, self.mapL2, self.mapR1, self.mapR2, _ , _ , _ , _ = self.stereo_rectify(img_shape, self.M1, self.d1, self.M2, self.d2, self.R, self.T)
        # self.depth_using_stereo(img_shape, self.M1, self.d1, self.M2, self.d2, self.R, self.T)

        ##pose_estimation
        self.pose_estimation(cal_path, self.M1, self.d1)
        #
        # stero_rms, _, _ = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r, self.M1, self.d1,
        #                                         self.M2, self.d2, self.R, self.T)
        print("END - read_images_square")

    # input - all image, output - json, func- mono and stereo calib
    def read_images_circle(self, cal_path):
        count_ok_dual = 0

        if (select_png_or_raw == 1):
            print(cal_path + '/RIGHT/*.RAW')
            images_right = glob.glob(cal_path + '/RIGHT/*.RAW')
            images_left = glob.glob(cal_path + '/LEFT/*.RAW')
        else:
            print(cal_path + '/RIGHT/*.JPG')
            images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
            images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
            images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
            images_left = glob.glob(cal_path + '/LEFT/*.JPG')
            images_left += glob.glob(cal_path + '/LEFT/*.PNG')
            images_left += glob.glob(cal_path + '/LEFT/*.BMP')
        # images_left.sort()
        # images_right.sort()
        # print(images_left)

        for i, fname in enumerate(images_right):
            if (select_png_or_raw == 1):
                fd_l = open(images_left[i], 'rb')
                fd_r = open(images_right[i], 'rb')
                length = len(fd_l.read())
                fd_l.seek(0)
                #print(length)
                rows = 1280
                cols = int(length / rows)
                f_l = np.fromfile(fd_l, dtype=np.uint8, count=rows * cols)
                f_r = np.fromfile(fd_r, dtype=np.uint8, count=rows * cols)
                gray_l = img_l = f_l.reshape((cols, rows))
                gray_r = img_r = f_r.reshape((cols, rows))

                fd_l.close
                fd_r.close
                #cv2.imshow(images_left[i], img_l)
                #cv2.imshow(images_right[i], img_r)
                #cv2.waitKey(20500)

            else:
                img_l = cv2.imread(images_left[i])
                img_r = cv2.imread(images_right[i])

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            # gray_l = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
            # gray_r = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
            #ret, gray_l = cv2.threshold(gray_l, 25, 200, cv2.THRESH_BINARY)
            #ret, gray_r = cv2.threshold(gray_r, 25, 200, cv2.THRESH_BINARY)
            # Find the chess board corners
            # ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            # ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)
            ret_l, corners_l = cv2.findCirclesGrid(gray_l, (marker_point_x, marker_point_y),
                                                   flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            # flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
            ret_r, corners_r = cv2.findCirclesGrid(gray_r, (marker_point_x, marker_point_y),
                                                   flags=cv2.CALIB_CB_SYMMETRIC_GRID)

            print((images_left[i], ret_l))
            print((images_right[i], ret_r))
            #print("L",corners_l)
            #print("R",corners_r)

            if ret_l is True and ret_r is True:
                count_ok_dual += 1
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                # rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                #                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (marker_point_x, marker_point_y),
                                                  corners_l, ret_l)
                if(enable_debug_detect_pattern_from_image == 1):
                    cv2.imshow(images_left[i], img_l)
                    cv2.waitKey(500)

                # rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                #                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (marker_point_x, marker_point_y),
                                                  corners_r, ret_r)

                if(enable_debug_detect_pattern_from_image == 1):
                    cv2.imshow(images_right[i], img_r)
                    cv2.waitKey(500)

            print(gray_l.shape[::-1])
            img_shape = gray_r.shape[::-1]

        print(len(self.objpoints))
        print(len(self.imgpoints_l))
        print(len(self.imgpoints_r))

        save_coordinate_both_stereo_obj_img(self.objpoints, self.imgpoints_l, self.imgpoints_r, count_ok_dual)

        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5

        camera_matrix = np.zeros((3, 3), np.float32)
        camera_matrix[0][0] = 1470.0
        camera_matrix[1][1] = 1470.0
        camera_matrix[0][2] = 640.0
        camera_matrix[1][2] = 482.0
        camera_matrix[2][2] = 1.0

        dist_coef = np.zeros((1, 5), np.float32)
        dist_coef[0][0] = -0.1
        dist_coef[0][1] = -0.2
        dist_coef[0][2] = 0.0
        dist_coef[0][3] = 0.0
        dist_coef[0][4] = 0.0

        # a,camMatrix, c, rvec, tvec = cv2.calibrateCamera([obj_points], [img_points], size, camera_matrix, dist_coefs, flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT)
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape,
                                                                     camera_matrix,
                                                                     dist_coef, flags=flags)
        rt2, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape,
                                                                      camera_matrix,
                                                                      dist_coef, flags=flags)

        print("=" * 50)
        print('Intrinsic_mtx_1', *np.round(self.M1, 5), sep='\n')
        print('dist_1', np.round(self.d1, 5))
        print('Intrinsic_mtx_2', *np.round(self.M2, 4), sep='\n')
        print('dist_2', np.round(self.d2, 4))
        print('R1', *np.round(self.r1, 4), sep='\n')
        print('T1', *np.round(self.t1, 4), sep='\n')
        print('R2', *np.round(self.r2, 4), sep='\n')
        print('T2', *np.round(self.t2, 4), sep='\n')
        print("=" * 50)
        print('rms1', rt)
        print('rms2', rt2)
        print("=" * 50)
        single_rms_l, _, _ = self.reprojection_error(self.objpoints, self.imgpoints_l, self.r1, self.t1, self.M1, self.d1)
        single_rms_r, _, _ = self.reprojection_error(self.objpoints, self.imgpoints_r, self.r2, self.t2, self.M2, self.d2)
        print('manual_rms1', single_rms_l)
        print('manual_rms1', single_rms_r)
        print("=" * 50)

        temp_r1 = np.copy(self.r1)
        temp_t1 = np.copy(self.t1)
        temp_M1 = np.copy(self.M1)
        temp_d1 = np.copy(self.d1)
        temp_r2 = np.copy(self.r2)
        temp_t2 = np.copy(self.t2)
        temp_M2 = np.copy(self.M2)
        temp_d2 = np.copy(self.d2)

        self.stereo_flags = 0
        # self.stereo_flags |= cv2.CALIB_FIX_INTRINSIC
        # self.stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        self.stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # self.stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
        self.stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # self.stereo_flags |= cv2.CALIB_RATIONAL_MODEL
        # self.stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        self.stereo_flags |= cv2.CALIB_FIX_K3
        self.stereo_flags |= cv2.CALIB_FIX_K4
        self.stereo_flags |= cv2.CALIB_FIX_K5

        self.camera_model = self.stereo_calibrate(img_shape)

        # print(self.camera_model)
        # stereodata = json.dumps(self.camera_model)
        #
        # print(str(stereodata['M1']))

        self.Ml[0][0] = - np.float64(self.Ml[0][0])
        self.Ml[1][1] = - np.float64(self.Ml[1][1])
        self.Mr[0][0] = - np.float64(self.Mr[0][0])
        self.Mr[1][1] = - np.float64(self.Mr[1][1])

        if(enable_debug_dispatiry_estimation_display == 1):
            self.mapL1, self.mapL2, self.mapR1, self.mapR2, _ , _ , _ , _ = self.stereo_rectify(img_shape, self.Ml, self.dl, self.Mr, self.dr, self.R, self.T, images_left[0], images_right[0])
            self.depth_using_stereo(img_shape, self.Ml, self.dl, self.Mr, self.dr, self.R, self.T)

        # print(self.camera_model.values(M1))
        # self.mapL1, self.mapL2, self.mapR1, self.mapR2, _ , _ , _ , _  = self.stereo_rectify(img_shape, self.M1, self.d1, self.M2,
        #                                                                      self.d2, self.R, self.T)
        #
        # self.depth_using_stereo(img_shape, self.M1, self.d1, self.M2, self.d2, self.R, self.T)

        ##pose_estimation
        #self.pose_estimation_circle(cal_path, self.M1, self.d1)
        #self.pose_estimation_square(cal_path, self.M1, self.d1)

        # # rms
        # print(self.M1, self.d1)
        # print(self.M2, self.d2)
        # single_rms_l, a1, a2 = self.reprojection_error(self.objpoints, self.imgpoints_l, self.r1, self.t1, self.M1, self.d1)
        # single_rms_r, b1, b2 = self.reprojection_error(self.objpoints, self.imgpoints_r, self.r2, self.t2, self.M2, self.d2)
        #
        # c1 = np.sqrt((a1 + b1) / (a2+b2))
        # print('rms3_c1', c1)

        # print("?" * 50)
        # print(self.M1, self.d1, self.M2, self.d2)
        # print(type(self.M1), type(self.d1), type(self.M2), type(self.d2))
        # print(self.M1.shape, self.d1.shape, self.M2.shape, self.d2.shape)
        # print("?" * 50)
        # print(temp_M1, temp_d1, temp_M2, temp_d2)
        # print(type(temp_M1), type(temp_d1), type(temp_M2), type(temp_d2))
        # print(temp_M1.shape, temp_d1.shape, temp_M2.shape, temp_d2.shape)
        # print("?" * 50)



        stero_rms, _, _ = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r, self.Ml, self.dl, self.Mr, self.dr, self.R, self.T)
        # stero_rms = self.calc_rms_stereo2(self.objpoints, self.imgpoints_l, self.imgpoints_r, temp_M1, temp_d1, temp_M2, temp_d2, self.R, self.T)
        print(stero_rms)
        self.pose_estimation(cal_path, self.Ml, self.dl)

        pass

    def stereo_calibrate(self, dims):
        # flags = 0
        # #flags |= cv2.CALIB_FIX_INTRINSIC
        # # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # #flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # # flags |= cv2.CALIB_RATIONAL_MODEL
        # #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5


        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        lgit_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        lip_criteria = (cv2.TERM_CRITERIA_MAX_ITER    , 1, 0.5)
        # stereo calibration test code for varify

        # ret, Ml, dl, Mr, dr, R, T, E, F = cv2.stereoCalibrate(
        #     self.objpoints, self.imgpoints_l,
        #     self.imgpoints_r, self.M1, self.d1, self.M2, self.d2, dims,
        #     criteria=lgit_criteria, flags=self.stereo_flags)
        # R = np.array([[0.99997219, -0.00162369, -0.00727876],
        #                   [0.00169173, 0.99995485, 0.00935097],
        #                   [0.00726325, -0.00936303, 0.99992979]])
        # T = np.array([-0.09232336, -0.1000415491683, -0.00071886])
        #
        # Ml = np.array([[1496.71883,    0.,       641.60057],
        #             [0.,      1496.71883,  490.26952],
        #             [0., 0., 1.]])
        # dl = np.array([-0.11003, -0.29897,    0.,    0.,     0.])
        # Mr = np.array([[1494.86203,   0.,       652.97403],
        #             [0.   ,   1494.86203,  472.24791],
        #             [0., 0., 1.]])
        # dr = np.array([-0.11309, -0.25242, 0, 0, 0])
        #
        # print('beforeRT', R,'\n', T[0], T[1],T[2])
        #
        # self.stereo_flags = cv2.CALIB_USE_EXTRINSIC_GUESS
        # ret2, Ml2, dl2, Mr2, dr2, R2, T2, E2, F2, perViewErrors = cv2.stereoCalibrateExtended(
        #     self.objpoints, self.imgpoints_l,
        #     self.imgpoints_r, Ml, dl, Mr, dr, dims, R, T,
        #     criteria=lgit_criteria, flags=self.stereo_flags)
        # print('perViewErrors', perViewErrors)
        # print('userRT', R,'\n', T[0], T[1],T[2])
        # print('Ml', Ml2, dl2, 'M2' , Mr2, dr2, '\nret_RT', R2,'\n', T2[0], T2[1],T2[2] )
        # print('*'*30, 'rms3_2', ret2 )


        (major, minor, mdumuy) = check_version_of_opencv()
        print('opencv', major, minor, mdumuy)
        print(major == 3 , minor >= 4 , mdumuy >= 1)
        if(major == 3 and minor >= 4 and mdumuy >= 1):
            #self.stereo_flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
            print(self.stereo_flags & cv2.CALIB_USE_EXTRINSIC_GUESS)
            if((self.stereo_flags & cv2.CALIB_USE_EXTRINSIC_GUESS) > 0):
                print("OK")
                userR = np.array([[0.99999774, -0.000082858249, -0.0021352633],
                                  [0.000088781271, 0.99999613, 0.0027739638],
                                  [0.0021350251, -0.0027741471, 0.99992979]])
                # userR = np.array([[0.99999774, 0.000088781271, -0.0021350251],
                #                   [-0.000082858249, 0.99999613, 0.0027741471],
                #                   [0.0021350251, -0.0027739638, 0.99999386]])

                userT = np.array([-0.091707, -0.000120558, 0.00195439])
                print(self.uT31)
                #print(type(userR), userR.shape, type(userT), userT.shape)
                print()
                #print(type(self.uR33), type(userR))
                #print(self.uR33.shape, userR.shape)
                #print(userT.shape)
                print(self.uR33)
                print(self.uR33.shape, self.uT31.shape)
                print(self.stereo_flags)
                ret, Ml, dl, Mr, dr, R, T, E, F, perViewErrors = cv2.stereoCalibrateExtended(self.objpoints, self.imgpoints_l,self.imgpoints_r,
                            self.M1, self.d1, self.M2, self.d2, dims,
                            userR, userT, criteria=lgit_criteria, flags=self.stereo_flags)
                print('perViewErrors', perViewErrors)
                #print('userRT', userR, userT)

            else:
                print("NG")
                ret, Ml, dl, Mr, dr, R, T, E, F = cv2.stereoCalibrate(
                    self.objpoints, self.imgpoints_l,
                    self.imgpoints_r, self.M1, self.d1, self.M2,
                    self.d2, dims,
                    criteria=lgit_criteria, flags=self.stereo_flags)



        else:
            ret, Ml, dl, Mr, dr, R, T, E, F = cv2.stereoCalibrate(
                self.objpoints, self.imgpoints_l,
                self.imgpoints_r, self.M1, self.d1, self.M2,
                self.d2, dims,
                criteria=lgit_criteria, flags=self.stereo_flags)

        # print(type(T),T.shape)
        T =  T.reshape(3,1)

        print("*" * 50)
        print('Intrinsic_mtx_1', *np.round(Ml, 5), sep='\n')
        print('dist_1', *np.round(dl, 5))
        print('Intrinsic_mtx_2', *np.round(Mr, 5), sep='\n')
        print('dist_2', *np.round(dr, 5))
        print("")
        # Roll/Pitch/Yaw -> Pitch/Yaw/Roll 로 변경
        print('R_3x3', np.round(R, 8))  # 3x3
        print('R', rotationMatrixToEulerAngles(R)[0], rotationMatrixToEulerAngles(R)[1], rotationMatrixToEulerAngles(R)[2])  # 1x3
        # print('T', np.round(T,3)*0.01)
        # print('T', np.round(T[0],3)*0.01, np.round(T[1],3)*0.01, np.round(T[2],3)*0.01)
        # print('T', T[0]*0.01, T[1]*0.01, T[2]*0.01)
        print('degreeR', rotationMatrixToEulerAngles(R)[0]*radianToDegree, rotationMatrixToEulerAngles(R)[1]*radianToDegree,
              rotationMatrixToEulerAngles(R)[2]*radianToDegree)  # 1x3

        print('T', T[0], T[1], T[2])
        print("=" * 50)
        print('rms3', ret)
        print("=" * 50)

        print('E', *np.round(E, 5), sep='\n')
        print('F', *np.round(F, 5), sep='\n')

        modify_value_from_json("stereo_config", Ml, dl, Mr, dr, R, T, dims)


        uR = np.zeros((3), np.float64)
        uR[0] = -rotationMatrixToEulerAngles(R)[0]
        uR[1] = -rotationMatrixToEulerAngles(R)[1]
        uR[2] = rotationMatrixToEulerAngles(R)[2]

        uT = np.zeros((3), np.float64)
        uT[0] = -T[0]
        uT[1] = -T[1]
        uT[2] = T[2]
        print('uR', uR, '\nuT', uT)

        w_Rt_c = np.eye(4)
        w_Rt_c[0:3, 0:3] = eulerAnglesToRotationMatrix(uR)
        w_Rt_c[0:3, 3] = uT
        print('w_Rt_c',w_Rt_c)
        c_Rt_w = np.linalg.inv(w_Rt_c)
        print('c_Rt_w', c_Rt_w)

        Ml[0][0] = - np.float64(Ml[0][0])
        Ml[1][1] = - np.float64(Ml[1][1])
        Mr[0][0] = - np.float64(Mr[0][0])
        Mr[1][1] = - np.float64(Mr[1][1])

        modify_value_from_json_from_LGIT_to_SM("stereo_config", Ml, dl, Mr, dr, c_Rt_w[0:3, 0:3], c_Rt_w[0:3, 3], dims)

        # print(np.round(num, 2))

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        # camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
        #                     ('dist2', d2), ('rvecs1', self.r1),
        #                     ('rvecs2', self.r2), ('R', R), ('T', T),
        #                     ('E', E), ('F', F)])
        camera_model = dict([('M1', Ml), ('M2', Mr), ('dist1', dl),
                             ('dist2', dr), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        self.Ml = Ml
        self.dl = dl
        self.Mr = Mr
        self.dr = dr
        self.R = R
        self.T = T

        cv2.destroyAllWindows()
        return camera_model


    def extract_point_from_chart(self, img_l, img_r):
        if img_l is False or img_r is False:
            print('NG - please check rectified file.')
            return 0

        count_ok_dual = 0
        local_objpoints = []  # 3d point in real world space
        local_imgpoints_l = []  # 2d points in image plane.
        local_imgpoints_r = []  # 2d points in image plane.

        #gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        #gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        gray_l = np.copy(img_l)
        gray_r = np.copy(img_r)

        # ret, gray_l = cv2.threshold(gray_l, 50, 200, cv2.THRESH_BINARY)
        # ret, gray_r = cv2.threshold(gray_r, 50, 200, cv2.THRESH_BINARY)
        # gray_l = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
        # gray_r = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
        # Find the chess board corners
        if (select_detect_pattern == 1):
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (marker_point_x, marker_point_y), flags = cv2.CALIB_CB_ADAPTIVE_THRESH  )
            # flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (marker_point_x, marker_point_y), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

            # if(enable_debug_detect_pattern_from_image == 1):
            #    cv2.imshow(images_left[i], gray_l)
            #    cv2.waitKey(0)
        else:
            ret_l, corners_l = cv2.findCirclesGrid(gray_l, (marker_point_x, marker_point_y),  flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            ret_r, corners_r = cv2.findCirclesGrid(gray_r, (marker_point_x, marker_point_y),  flags=cv2.CALIB_CB_SYMMETRIC_GRID)

        #print((images_left[i], ret_l))
        #print((images_right[i], ret_r))

        if ret_l is True and ret_r is True:
            count_ok_dual += 1
            # If found, add object points, image points (after refining them)
            local_objpoints.append(self.objp)

            if (select_detect_pattern == 1):
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
            local_imgpoints_l.append(corners_l)
            # Draw and display the corners
            ret_l = cv2.drawChessboardCorners(gray_l, (marker_point_x, marker_point_y), corners_l, ret_l)

            if (enable_debug_detect_pattern_from_image == 1):
                cv2.imshow("Left find", gray_l)
                cv2.waitKey(500)

            if (select_detect_pattern == 1):
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
            local_imgpoints_r.append(corners_r)
            # Draw and display the corners
            ret_r = cv2.drawChessboardCorners(gray_r, (marker_point_x, marker_point_y), corners_r, ret_r)

            if (enable_debug_detect_pattern_from_image == 1):
                cv2.imshow("Right find", gray_r)
                cv2.waitKey(500)

        #print(gray_l.shape[::-1], type(gray_l.shape[::-1]))
        #img_shape = gray_r.shape[::-1]

        print(len(local_objpoints))
        print(len(local_imgpoints_l))
        print(len(local_imgpoints_r))

        save_coordinate_both_stereo_obj_img_rectify(local_objpoints, local_imgpoints_l, local_imgpoints_r, count_ok_dual)

    def stereo_rectify(self, img_shape, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, filename_l, filename_r):
        print("-> process of sreteo rectify")
        # img_l = cv2.imread('./image/LEFT/LEFT13.jpg')
        # img_r = cv2.imread('./image/RIGHT/RIGHT13.jpg')
        # img_l = cv2.imread('./image33/LEFT/LEFT_Step_112.png')
        # img_r = cv2.imread('./image33/RIGHT/RIGHT_Step_112.png')
        #filename_l = 'D:/data/Calibration/Right_Zig/ALL/LEFT/CaptureImage_Left_005_01.raw'
        #filename_r = 'D:/data/Calibration/Right_Zig/ALL/RIGHT/CaptureImage_Right_005_01.raw'
        print(filename_l, filename_r)
        if (select_png_or_raw == 1):
            fd_l = open(filename_l, 'rb')
            fd_r = open(filename_r, 'rb')
            length = len(fd_l.read())
            fd_l.seek(0)
            # print(length)
            rows = 1280
            cols = int(length / rows)
            f_l = np.fromfile(fd_l, dtype=np.uint8, count=rows * cols)
            f_r = np.fromfile(fd_r, dtype=np.uint8, count=rows * cols)
            gray_l = img_l = f_l.reshape((cols, rows))
            gray_r = img_r = f_r.reshape((cols, rows))

            fd_l.close
            fd_r.close
            # cv2.imshow(images_left[i], img_l)
            # cv2.imshow(images_right[i], img_r)
            # cv2.waitKey(20500)

        else:
            img_l = cv2.imread(filename_l)
            img_r = cv2.imread(filename_r)
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        frameL = img_l
        frameR = img_r
        # frameL = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        # frameR = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        # frameL = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
        # frameL = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)

        # STAGE 4: rectification of images (make scan lines align left <-> right
        # N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
        # only valid pixels are visible (no black areas after rectification). alpha=1 means
        # that the rectified image is decimated and shifted so that all the pixels from the original images
        # from the cameras are retained in the rectified images (no source image pixels are lost)." - ?
        RL, RR, PL, PR, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_matrix_l, dist_coeffs_l,
                                                                          camera_matrix_r, dist_coeffs_r,
                                                                          img_shape, R, T, alpha=-1, flags=cv2.CALIB_ZERO_DISPARITY);

        # compute the pixel mappings to the rectified versions of the images
        print(camera_matrix_l, '-> \n', PL)
        print(camera_matrix_r, '-> \n', PR)
        mapL1, mapL2 = cv2.initUndistortRectifyMap(camera_matrix_l, dist_coeffs_l, RL, PL, img_shape, cv2.CV_32FC1);
        mapR1, mapR2 = cv2.initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r, RR, PR, img_shape, cv2.CV_32FC1);

        print("-> performing rectification")
        print(type(mapL1), type(mapL2))
        print(mapL1.shape, mapL2.size)
        # undistort and rectify based on the mappings (could improve interpolation and image border settings here)
        undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR);
        undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR);

        # display data
        print('\nLeft  3x3 rectification(rot) \n', *np.round(RL[0], 8), '\n', *np.round(RL[1], 8), '\n',
              *np.round(RL[2], 8))
        # print('Left  3x3 rectification(rot) \n', *RL, sep='\n')
        print('\nRight 3x3 rectification(rot) ', *RR, sep='\n')
        print('\nLeft  3x4 proj in new (rectified) coordinate ', *PL, sep='\n')
        print('\nRight 3x4 proj in new (rectified) coordinate ', *PR, sep='\n')
        print('\nQ', *Q, sep='\n')
        print('validPixROI1', *validPixROI1)
        print('validPixROI2', *validPixROI2)

        self.RL = RL
        self.RR = RR
        self.PL = PL
        self.PR = PR
        self.Q = Q

        # display image
        # cv2.imshow("LEFT Camera rectification Input", undistorted_rectifiedL);
        # cv2.imshow("RIGHT Camera rectification Input", undistorted_rectifiedR);
        self.extract_point_from_chart(undistorted_rectifiedL,undistorted_rectifiedR)
        self.depth_using_stereo_param2(undistorted_rectifiedL,undistorted_rectifiedR)
        # cv2.waitKey(0)
        return mapL1, mapL2, mapR1, mapR2, RL, PL, RR, PR


    def depth_using_stereo_param(self, left, right):
        fx = 1482.88842705  # lense focal length
        baseline = 93.40152  # distance in mm between the two cameras
        disparities = 16  # num of disparities to consider = 16xn
        block = 31  # block size to match
        units = 0.512  # depth units, adjusted for the output to fit in one byte

        sbm = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)

        # calculate disparities
        disparity = sbm.compute(left, right)
        valid_pixels = disparity > 0

        # calculate depth data
        depth = np.zeros(shape=left.shape).astype("uint8")
        depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

        # visualize depth data
        # plt.imshow(depth)
        depth = cv2.equalizeHist(depth)
        colorized_depth = np.zeros((left.shape[0], left.shape[1], 3), dtype="uint8")
        temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        colorized_depth[valid_pixels] = temp[valid_pixels]
        plt.imshow(colorized_depth)
        plt.show()

    def depth_using_stereo_param2(self, left, right):
        #fx = 1482.88842705  # lense focal length
        #baseline = 93.40152  # distance in mm between the two cameras
        block = 25  # block size to match
        #units = 0.512  # depth units, adjusted for the output to fit in one byte

        disparities = 256  # num of disparities to consider = 16xn
        blockSizes = [5, 15, 25, 35, 45, 55]

        # for b_idx in range(len(blockSizes)):
        #     sbm = cv2.StereoBM_create(numDisparities=disparities, blockSize=blockSizes[b_idx])
        #
        #     # calculate disparities
        #     disparity = sbm.compute(left, right)
        #     cv2.filterSpeckles(disparity, 0, 200, 128);
        #     disparity = disparity / 16.
        #
        #     plt.imshow(disparity)  # , 'gray')
        #     plt.title('Block Matching Disparity Map, blocksize: ' + str(blockSizes[b_idx]))
        #     plt.colorbar()
        #     plt.show()
        #     # fname = 'disparity_BM_blocksize_' + str(blockSizes[b_idx])
        #     # np.save(fname, disparity)
        #     # cv2.imshow('Disparity Map', disparity)
        #     # cv2.waitKey(0)

        # # Semi Global Block Matching
        # disparities = 128  # num of disparities to consider = 16xn
        # blockSizes = [5, 11, 15, 23, 35]
        # for b_idx in range(len(blockSizes)):
        #     sbm = cv2.StereoSGBM_create(numDisparities=disparities, blockSize=blockSizes[b_idx])
        #     right_matcher = cv2.ximgproc.createRightMatcher(sbm)
        #
        #     # FILTER Parameters
        #     lmbda = 80000
        #     sigma = 1.2
        #     visual_multiplier = 1.0
        #
        #     wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sbm)
        #     wls_filter.setLambda(lmbda)
        #     wls_filter.setSigmaColor(sigma)
        #
        #     print('computing disparity...')
        #     displ = sbm.compute(left, right)  # .astype(np.float32)/16
        #     dispr = right_matcher.compute(right, left)  # .astype(np.float32)/16
        #     displ = np.int16(displ)
        #     dispr = np.int16(dispr)
        #     filteredImg = wls_filter.filter(displ, left, None, dispr)
        #     filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        #     # filteredImg = filteredImg / 16.
        #     filteredImg = np.uint8(filteredImg)
        #     plt.imshow(filteredImg)
        #
        #     plt.title('Semi-Global Block Matching Disparity Map, blocksize: ' + str(blockSizes[b_idx]))
        #     plt.colorbar()
        #     plt.show()
        #     fname = 'disparity_SGBM_blocksize_filt_' + str(blockSizes[b_idx])
        #     np.save(fname, filteredImg)

        # SGBM Parameters -----------------
        window_size = 7  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        windowSize = [3 , 5, 7]
        blockSizes = [5, 15, 25, 35, 45, 55]
        for w_idx in range(len(windowSize)):
            for b_idx in range(len(blockSizes)):
                left_matcher = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=512,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                    blockSize=blockSizes[b_idx],
                    P1=8 * 3 * windowSize[w_idx] ** 2,
                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                    P2=32 * 3 * windowSize[w_idx] ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=15,
                    speckleWindowSize=0,
                    speckleRange=2,
                    preFilterCap=63,
                    mode= cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )

                right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

                # FILTER Parameters
                lmbda = 80000
                sigma = 1.2
                visual_multiplier = 1.0

                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
                wls_filter.setLambda(lmbda)
                wls_filter.setSigmaColor(sigma)

                print('computing disparity...')
                displ = left_matcher.compute(left, right)  # .astype(np.float32)/16
                dispr = right_matcher.compute(right, left)  # .astype(np.float32)/16
                displ = np.int16(displ)
                dispr = np.int16(dispr)
                filteredImg = wls_filter.filter(displ, left, None, dispr)  # important to put "imgL" here!!!

                filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
                filteredImg = np.uint8(filteredImg)

                plt.imshow(filteredImg)
                plt.title('StereoSGBM_create Disparity Map: windowSize='+ str(windowSize[w_idx]) + str(',blockSize=') + str(blockSizes[b_idx]))
                plt.colorbar()
                plt.show()

        'generating 3d point cloud...'
        min_disp = 16
        num_disp = 112 - min_disp

        h, w = left.shape[:2]
        print(h,w)

        f = self.PL[0][0]   #0.8 * w  # guess for focal length
        print(f)
        # Q = np.float32([[1, 0, 0, -0.5 * w],
        #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
        #                 [0, 0, 0, -f],  # so that y-axis looks up
        #                 [0, 0, 1, 0]])
        Q = self.Q
        print(Q)
        points = cv2.reprojectImageTo3D(filteredImg, Q)
        colors = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
        mask = filteredImg > filteredImg.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'out.ply'
        write_ply('out.ply', out_points, out_colors)
        print('out.ply saved')

        cv2.imshow('left', left)
        cv2.imshow('disparity', (filteredImg - min_disp) / num_disp)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def depth_using_stereo(self, img_shape, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T):
        # img_l = cv2.imread('./image/LEFT/LEFT13.jpg')
        # img_r = cv2.imread('./image/RIGHT/RIGHT13.jpg')
        # img_l = cv2.imread('./image33/LEFT/LEFT_Step_112.png')
        # img_r = cv2.imread('./image33/RIGHT/RIGHT_Step_112.png')
        #img_l = cv2.imread('./data/LGIT_data/LEFT/LEFT_Step_11.png')
        #img_r = cv2.imread('./data/LGIT_data/RIGHT/RIGHT_Step_11.png')
        # img_l = cv2.imread('./dump_pattern/LEFT/Cal0_00001.png')
        # img_r = cv2.imread('./dump_pattern/RIGHT/Cal1_00001.png')
        # filename_l = 'D:/data/Calibration/Left_Zig/ALL/LEFT/CaptureImage_Left_005_01.raw'
        # filename_r = 'D:/data/Calibration/Left_Zig/ALL/RIGHT/CaptureImage_Right_005_01.raw'
        filename_l = 'E:\C\headeye\stereo_rectification_disparity\DSM\Left_Zig\ALL\LEFT\CaptureImage_Left_005_01.raw'
        filename_r = 'E:\C\headeye\stereo_rectification_disparity\DSM\Left_Zig\ALL\RIGHT\CaptureImage_Right_005_01.raw'



        print(filename_l, filename_r)
        if (select_png_or_raw == 1):
            fd_l = open(filename_l, 'rb')
            fd_r = open(filename_r, 'rb')
            length = len(fd_l.read())
            fd_l.seek(0)
            # print(length)
            rows = 1280
            cols = int(length / rows)
            f_l = np.fromfile(fd_l, dtype=np.uint8, count=rows * cols)
            f_r = np.fromfile(fd_r, dtype=np.uint8, count=rows * cols)
            gray_l = img_l = f_l.reshape((cols, rows))
            gray_r = img_r = f_r.reshape((cols, rows))

            fd_l.close
            fd_r.close

        else:
            img_l = cv2.imread(filename_l)
            img_r = cv2.imread(filename_r)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # frameL = img_l
        # frameR = img_r
        # frameL = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        # frameR = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        # STAGE 5: calculate stereo depth information
        # uses a modified H. Hirschmuller algorithm [HH08] that differs (see opencv manual)
        # parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]
        # FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);
        # From help(cv2): StereoBM_create(...)
        #        StereoBM_create([, numDisparities[, blockSize]]) -> retval
        #
        #    StereoSGBM_create(...)
        #        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
        # disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

        print("-> calc. disparity image")

        max_disparity = 128;
        # stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21, mode=cv2.STEREO_SGBM_MODE_HH);
        stereoProcessor = cv2.StereoSGBM_create(0, 128, 5, 100, 1000, 32, 0, 15, 50, 2, mode=cv2.STEREO_SGBM_MODE_HH);


        # remember to convert to grayscale (as the disparity matching works on grayscale)

        # undistort and rectify based on the mappings (could improve interpolation and image border settings here)
        # N.B. mapping works independant of number of image channels

        undistorted_rectifiedL = cv2.remap(gray_l, self.mapL1, self.mapL2, cv2.INTER_LINEAR);
        undistorted_rectifiedR = cv2.remap(gray_r, self.mapR1, self.mapR2, cv2.INTER_LINEAR);

        # compute disparity image from undistorted and rectified versions
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)
        # np.set_printoptions(edgeitems=720)

        disparity = stereoProcessor.compute(undistorted_rectifiedL, undistorted_rectifiedR);
        cv2.normalize(disparity,disparity,beta=0,alpha=255,norm_type=cv2.NORM_MINMAX)
        print('disparity', disparity)
        # print(type(disparity))
        # cv2.filterSpeckles(disparity, 0, 4000, 128);

        # scale the disparity to 8-bit for viewing

        disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())
        print('disparity_scaled', disparity_scaled)

        # display image
        cv2.imshow("LEFT Camera Input", undistorted_rectifiedL);
        cv2.imshow("RIGHT Camera Input", undistorted_rectifiedR);

        # display disparity
        cv2.imshow("SGBM Stereo Disparity - Output", disparity);
        cv2.imshow("SGBM Stereo Disparity - Output2", disparity_scaled);
        # im_color = cv2.applyColorMap(disparity_scaled, cv2.COLORMAP_JET)
#lip        # plt.imshow(im_color, 'gray')
#lip        # plt.imshow(disparity_scaled, 'gray')
#lip        # plt.show()
        cv2.waitKey(0)

    def draw_xyz_axis(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        print('x',corner, tuple(imgpts[0].ravel()))
        print('y',corner, tuple(imgpts[1].ravel()))
        print('z',corner, tuple(imgpts[2].ravel()))

        return img

    def pose_estimation(self, cal_path, mtx, dist):
        if(enable_debug_pose_estimation_display == 0):
            print('skip pose estimation')
            return
        print('pose estimation')
        # input - all image, output - json, func- mono and stereo calib
        print(cal_path)
        if (select_png_or_raw == 1):
            print(cal_path + '/RIGHT/*.RAW')
            images_right = glob.glob(cal_path + '/RIGHT/*.RAW')
            images_left = glob.glob(cal_path + '/LEFT/*.RAW')
        else:
            print(cal_path + '/RIGHT/*.JPG')
            images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
            images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
            images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
            images_left = glob.glob(cal_path + '/LEFT/*.JPG')
            images_left += glob.glob(cal_path + '/LEFT/*.PNG')
            images_left += glob.glob(cal_path + '/LEFT/*.BMP')

        for i, fname in enumerate(images_right):
            if (select_png_or_raw == 1):
                fd_l = open(images_left[i], 'rb')
                fd_r = open(images_right[i], 'rb')
                length = len(fd_l.read())
                fd_l.seek(0)
                #print(length)
                rows = 1280
                cols = int(length / rows)
                f_l = np.fromfile(fd_l, dtype=np.uint8, count=rows * cols)
                f_r = np.fromfile(fd_r, dtype=np.uint8, count=rows * cols)
                gray_l = img_l = f_l.reshape((cols, rows))
                gray_r = img_r = f_r.reshape((cols, rows))

                fd_l.close
                fd_r.close
                #cv2.imshow(images_left[i], img_l)
                #cv2.imshow(images_right[i], img_r)
                #cv2.waitKey(20500)

            else:
                img_l = cv2.imread(images_left[i])
                img_r = cv2.imread(images_right[i])

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        for i, fname in enumerate(images_right):
            if (enable_debug_pose_estimation_display == 1 or enable_debug_pose_estimation_display == 2 ):
                # img_l = cv2.imread(images_left[i])
                # gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                # gray_l = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
                if(select_detect_pattern == 1):
                    ret, corners_l = cv2.findChessboardCorners(gray_l, (marker_point_x, marker_point_y), None)
                else:
                    ret, corners_l = cv2.findCirclesGrid(gray_l, (marker_point_x, marker_point_y), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                if ret == True:
                    if (select_detect_pattern == 1):
                        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                        ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners2_l, mtx, dist)
                        imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
                        img = self.draw_xyz_axis(img_l, corners2_l, imgpts)
                    else:
                        ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners_l, mtx, dist)
                        imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)


                        img = self.draw_xyz_axis(img_l, corners_l, imgpts)

                    print('rvecs',rvecs,'tvecs',tvecs)
                    print(self.axis, imgpts)
                    cv2.imshow('poseEstimate_Left', img)
                    k =  cv2.waitKey(0)

            if (enable_debug_pose_estimation_display == 1 or enable_debug_pose_estimation_display == 3 ):
                # img_r = cv2.imread(images_right[i])
                # gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
                # gray_r = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
                if (select_detect_pattern == 1):
                    ret, corners_r = cv2.findChessboardCorners(gray_r, (marker_point_x, marker_point_y), None)
                else:
                    ret, corners_r = cv2.findCirclesGrid(gray_r, (marker_point_x, marker_point_y), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                if ret == True:
                    if (select_detect_pattern == 1):
                        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                        ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners2_r, mtx, dist)
                        imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
                        img = self.draw_xyz_axis(img_r, corners2_r, imgpts)
                    else:
                        ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners_r, mtx, dist)
                        imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
                        img = self.draw_xyz_axis(img_r, corners_r, imgpts)

                    cv2.imshow('poseEstimate_Right', img)
                    k = cv2.waitKey(0)




    #
    # def pose_estimation_circle(self, cal_path, mtx, dist):
    #     print('pose estimation circle')
    #     # input - all image, output - json, func- mono and stereo calib
    #     # images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
    #     # images_right += glob.glob(cal_path + 'RIGHT/*.BMP')
    #     # images_right += glob.glob(cal_path + 'RIGHT/*.PNG')
    #     images_left = glob.glob(cal_path + '/LEFT/*.JPG')
    #     images_left += glob.glob(cal_path + '/LEFT/*.PNG')
    #     images_left += glob.glob(cal_path + '/LEFT/*.BMP')
    #
    #     for i, fname in enumerate(images_left):
    #         img_l = cv2.imread(images_left[i])
    #         gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    #         # gray_l = cv2.adaptiveThreshold(gray_l, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
    #         ret_l, corners_l = cv2.findCirclesGrid(gray_l, (10, 10), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    #
    #         print(ret_l, corners_l)
    #         if ret_l == True:
    #             #corners2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
    #             # Find the rotation and translation vectors.
    #             ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners_l, mtx, dist)
    #             # project 3D points to image plane
    #             imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
    #             img = self.draw(img_l, corners_l, imgpts)
    #             cv2.imshow('img', img)
    #             k = cv2.waitKey(0)
    #
    # def pose_estimation_squre(self, cal_path, mtx, dist):
    #     print('pose estimation sqaure')
    #     # input - all image, output - json, func- mono and stereo calib
    #     images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
    #     images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
    #     images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
    #     images_left = glob.glob(cal_path + '/LEFT/*.JPG')
    #     images_left += glob.glob(cal_path + '/LEFT/*.PNG')
    #     images_left += glob.glob(cal_path + '/LEFT/*.BMP')
    #
    #     for i, fname in enumerate(images_right):
    #         img_l = cv2.imread(images_left[i])
    #         img_r = cv2.imread(images_right[i])
    #
    #         gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    #         gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    #         ret, corners = cv2.findChessboardCorners(gray_l, (9, 6), None)
    #         if ret == True:
    #             corners2 = cv2.cornerSubPix(gray_l, corners, (11, 11), (-1, -1), self.criteria)
    #             # Find the rotation and translation vectors.
    #             ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners2, mtx, dist)
    #             # project 3D points to image plane
    #             imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
    #             img = self.draw(img_l, corners2, imgpts)
    #             cv2.imshow('img', img)
    #             k = cv2.waitKey(0)

    def reprojection_error(self, obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
        min_error = 1
        max_error = 0

        # print('obj_points', obj_points[0])
        # print('img_points', img_points[0])
        img_points_reshape = np.array(img_points)
        #print(img_points_reshape[0].reshape(-1,2))
        #print(img_points_reshape[1].reshape(-1, 2))

        # this is 3d reference point and 2d image point to extract R,T
        # print(len(obj_points), len(obj_points[0]))
        # print(len(img_points), len(img_points[0]))
        # print(len(rvecs), len(rvecs[0]))
        # print(len(tvecs), len(tvecs[0]))
        #print(len(camera_matrix), len(camera_matrix[0]))
        #print(len(dist_coeffs), len(dist_coeffs[0]))
        # print(camera_matrix, dist_coeffs)
        # ret, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[0], img_points[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE  )
        # print('temp r t', temp_rvecs, temp_tvecs)
        # ret, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[0], img_points[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP  )
        # print('temp r t', temp_rvecs, temp_tvecs)
        # ret, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[0], img_points[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP  )
        # print('temp r t', temp_rvecs, temp_tvecs)
        # print('cal r t', rvecs[0], tvecs[0])

        # for i in range(len(obj_points)):
        #     print('cal all', rvecs[i], tvecs[i])
        # print('\n')

        tot_error = 0
        total_points = 0
        ret_error = 0
        for i in range(len(obj_points)):
            reprojected_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            reprojected_points = reprojected_points.reshape(-1, 2)
            temp_points = img_points_reshape[i].reshape(-1,2)
            #print(reprojected_points.shape, temp_points.shape)
            #print('start', reprojected_points , "check\n", temp_points)
            tot_error += np.sum(np.abs(temp_points - reprojected_points) ** 2)
            total_points += len(obj_points[i])

        # print('tot', tot_error, total_points)
        ret_error = np.sqrt(tot_error / total_points)
        # print("Mean reprojection error: ", ret_error)

        return ret_error, tot_error, total_points

    def reprojection_error2(self, obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
        min_error = 1
        max_error = 0

        img_points_reshape = np.array(img_points)

        # print('temp r t', temp_rvecs, temp_tvecs)
        # print('cal r t', rvecs[0], tvecs[0])

        tot_error = 0
        total_points = 0
        for i in range(len(obj_points)):
            ret, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[i], img_points[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE  )
            temp_rvecs[0] = temp_rvecs[0]
            temp_rvecs[1] = temp_rvecs[1]
            temp_rvecs[2] = temp_rvecs[2]
            temp_tvecs[0] = temp_tvecs[0]
            temp_tvecs[1] = temp_tvecs[1]
            temp_tvecs[2] = temp_tvecs[2]
            reprojected_points, _ = cv2.projectPoints(obj_points[i], temp_rvecs, temp_tvecs, camera_matrix, dist_coeffs)
            reprojected_points = reprojected_points.reshape(-1, 2)
            temp_points = img_points_reshape[i].reshape(-1, 2)

            tot_error += np.sum(np.abs(temp_points - reprojected_points) ** 2)
            total_points += len(obj_points[i])

        # print('tot', tot_error, total_points)
        ret_error = np.sqrt(tot_error / total_points)
        # print("Mean reprojection error: ", ret_error)

        return ret_error, tot_error, total_points

    # def solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec=None, tvec=None, useExtrinsicGuess=None,
    #              flags=None):  # real signature unknown; restored from __doc__
    #     """ solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) -> retval, rvec, tvec """
    #     pass
    #
    # def solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec=None, tvec=None,
    #                    useExtrinsicGuess=None, iterationsCount=None, reprojectionError=None, confidence=None,
    #                    inliers=None, flags=None):  # real signature unknown; restored from __doc__
    #     """ solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError[, confidence[, inliers[, flags]]]]]]]]) -> retval, rvec, tvec, inliers """
        pass

    def calc_rms_stereo(self, obj_points, imgpoints_l, imgpoints_r, A1, D1, A2, D2, R, T):
        tot_error = 0
        total_points = 0

        # print(type(obj_points[0]),type(imgpoints_l[0]),type(imgpoints_r[0]))
        # print((obj_points[0].shape), (imgpoints_l[0].shape), (imgpoints_r[0].shape))
        # print((obj_points[0]), (imgpoints_l[0]), (imgpoints_r[0]))

        for i in range(len(obj_points)):
            # calculate world <-> cam1 transformation

            #cv2.FindExtrinsicCameraParams2(obj_points[i], imgpoints_l[i], A1, D1, temp_rvecs, temp_tvecs)
            _, rvec_l, tvec_l, _ = cv2.solvePnPRansac(obj_points[i], imgpoints_l[i], A1, D1 )
            #_, t_rvec_l, t_tvec_l = cv2.solvePnP(obj_points[i], imgpoints_l[i], A1, D1, useExtrinsicGuess = True, flags=cv2.SOLVEPNP_ITERATIVE )

            #_, rvec_l, tvec_l = cv2.solvePnP(obj_points[i], t_imgpoints_l, A1, D1, flags=cv2.SOLVEPNP_ITERATIVE     )
            #print('rvec_l', 'tvec_l',rvec_l, tvec_l)

            # _, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[i], imgpoints_l[i], A1, D1, flags=cv2.SOLVEPNP_ITERATIVE  )
            #print('temp_rvecs', 'temp_tvecs', temp_rvecs, temp_tvecs)
            # compute reprojection error for cam1
            rp_l, _ = cv2.projectPoints(obj_points[i], rvec_l, tvec_l, A1, D1)
            #print( 'rp_l' , rp_l )
            #tot_error += np.sum(np.square(np.float64(imgpoints_l[i] - rp_l)))
            tot_error += np.sum(np.square(np.float64(rp_l - imgpoints_l[i])))
            total_points += len(obj_points[i])

            # calculate world <-> cam2 transformation
            rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]
            #print('rvec_r', 'tvec_r', rvec_r, tvec_r)

            # compute reprojection error for cam2
            rp_r, _ = cv2.projectPoints(obj_points[i], rvec_r, tvec_r, A2, D2)  #A1, D1
            #print( 'rp_r' , rp_r )
            # tot_error += np.square(imgpoints_r[i] - rp_r).sum()
            tot_error += np.sum(np.square(np.float64(rp_r - imgpoints_r[i])))
            total_points += len(obj_points[i])

        mean_error = np.sqrt(tot_error / total_points)

        # print("=" * 50)
        # print('stero_rms_manual: ', mean_error)
        # print("=" * 50)
        # for i in range(len(obj_points)):
        #     # calculate world <-> cam1 transformation
        #     _, rvec_l, tvec_l, _ = cv2.solvePnPRansac(obj_points[i], imgpoints_l[i], A1, D1)
        #     # compute reprojection error for cam1
        #     rp_l, _ = cv2.projectPoints(obj_points[i], rvec_l, tvec_l, A1, D1)
        #     tot_error += np.sum(np.square(np.float64(imgpoints_l[i] - rp_l)))
        #     total_points += len(obj_points[i])
        #
        #     # calculate world <-> cam2 transformation
        #     rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]
        #
        #     # compute reprojection error for cam2
        #     rp_r, _ = cv2.projectPoints(obj_points[i], rvec_r, tvec_r, A2, D2)
        #     tot_error += np.square(imgpoints_r[i] - rp_r).sum()
        #     total_points += len(obj_points[i])
        #
        # mean_error = np.sqrt(tot_error / total_points)
        return mean_error

    def calc_rms_stereo2(self, obj_points, imgpoints_l, imgpoints_r, A1, D1, A2, D2, R, T):
        tot_error = 0
        total_points = 0
        p_reproj_left = []
        p_reproj_right = []

        # print(type(obj_points[0]),type(imgpoints_l[0]),type(imgpoints_r[0]))
        # print((obj_points[0]), (imgpoints_l[0]), (imgpoints_r[0]))
        # print((obj_points[0].shape), (imgpoints_l[0].shape), (imgpoints_r[0].shape))

        A1[0][0] = -A1[0][0]
        A1[1][1] = -A1[1][1]
        print(A1, D1)

        for i in range(len(obj_points)):
            i = 21  #특정장수 체크
            t_imgpoints_l = imgpoints_l[i].reshape(-1, 1, 2)
            t_imgpoints_r = imgpoints_r[i].reshape(-1, 1, 2)
            t_undist_imgpoints_l = cv2.undistortPoints(t_imgpoints_l,  A1, np.zeros((1,5)))
            t_undist_imgpoints_l =  t_undist_imgpoints_l * [A1[0][0], A1[1][1]]+[A1[0][2],A1[1][2]]
            print('t_imgpoints_l',np.squeeze(t_imgpoints_l), 't_undist_imgpoints_l',np.squeeze(t_undist_imgpoints_l))
            # print('test', t_obj_points, t_imgpoints_l, t_imgpoints_r)
            # calculate world <-> cam1 transformation
            # t_undist_imgpoints_r = cv2.undistortPoints(t_imgpoints_r,  A2, D2)
            # t_undist_imgpoints_r =  t_undist_imgpoints_r * [A2[0][0], A2[1][1]]+[A2[0][2],A2[1][2]]
            # print('t_imgpoints_r', np.squeeze(t_imgpoints_r), 't_undist_imgpoints_r', np.squeeze(t_undist_imgpoints_r))
            # check = np.ones((100,3))
            # check[:,0:2] = t_undist_imgpoints_r.reshape(100,2)
            # check = check.reshape(100, 3, 1)
            #
            # RT = np.eye(4,4)
            # RT[0:3,0:3] = np.matrix(R)
            # RT[0:3, 3]  = np.matrix(T).T
            # RT_inv = np.linalg.inv(RT)
            # print('\nRT',RT, '\nRT_inv',RT_inv)
            # t_undist_imgpoints_r_trans_A2_to_A1 = []
            # for k in t_undist_imgpoints_r:
            #     # print([[k[0][0]], [k[0][1]], [1.]])
            #     check5 = np.ones((4,1))
            #     check4 =  np.linalg.inv(A2) * np.matrix([[k[0][0]], [k[0][1]], [1.]])
            #     # print(check4[:,0:2])
            #     check5[0:3,:] = check4
            #     # print(check5)
            #     check6 = RT_inv * np.matrix(check5)
            #     print('check6',check6)
            #     # check6_1 = check6[0:2, :] / check6[2, :].reshape((-1, 1))
            #     check6_1 = check6[0:3, :] / check6[2, :]
            #     print('check6_1',check6_1)
            #     check7 = A1 * check6_1
            #     # check7 = check6_1[0:2, :] *[A1[0][0], A1[1][1]] + [A1[0][2], A1[1][2]]
            #     # print('input\n',RT_inv, check5, 'ret\n',check6)
            #     # print(A1)
            #     print('check7',check7)
            #     t_undist_imgpoints_r_trans_A2_to_A1.append(check7[0:2,:])
            #
            # # np.stack(t_undist_imgpoints_r_trans_A2_to_A1)
            # t_undist_imgpoints_r_trans_A2_to_A1 = np.asarray(t_undist_imgpoints_r_trans_A2_to_A1).reshape(-1, 1, 2)
            #
            # # print(t_undist_imgpoints_r_trans_A2_to_A1)
            # print(t_undist_imgpoints_l.shape, t_undist_imgpoints_r_trans_A2_to_A1.shape)
            # both_imgpoints = t_undist_imgpoints_l
            # both_imgpoints = np.append(both_imgpoints, t_undist_imgpoints_r_trans_A2_to_A1, axis=0)
            # both_obj_points= obj_points[i]
            # both_obj_points = np.append(both_obj_points, obj_points[i], axis=0)
            # print(both_imgpoints.shape, both_obj_points.shape)

            #cv2.FindExtrinsicCameraParams2(obj_points[i], imgpoints_l[i], A1, D1, temp_rvecs, temp_tvecs)

            #_, rvec_l, tvec_l, _ = cv2.solvePnPRansac(obj_points[i], t_imgpoints_l, A1, D1 )
            #_, t_rvec_l, t_tvec_l = cv2.solvePnP(obj_points[i], imgpoints_l[i], A1, D1, useExtrinsicGuess = True, flags=cv2.SOLVEPNP_ITERATIVE )
            _, rvec_l, tvec_l = cv2.solvePnP(obj_points[i], t_imgpoints_l, A1, D1, flags=cv2.SOLVEPNP_ITERATIVE     )
            # _, rvec_l, tvec_l = cv2.solvePnP(obj_points[i], t_undist_imgpoints_l, A1, np.zeros((1,5)), flags=cv2.SOLVEPNP_ITERATIVE     )
            # _, rvec_l, tvec_l = cv2.solvePnP(both_obj_points, both_imgpoints, A1, np.zeros((1,5)), flags=cv2.SOLVEPNP_ITERATIVE     )
            # print('rvec_l', 'tvec_l',rvec_l, tvec_l)

            # _, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[i], imgpoints_l[i], A1, D1, flags=cv2.SOLVEPNP_ITERATIVE  )
            #print('temp_rvecs', 'temp_tvecs', temp_rvecs, temp_tvecs)
            # compute reprojection error for cam1
            rp_l, _ = cv2.projectPoints(obj_points[i], rvec_l, tvec_l, A1, D1)
            #print( 'rp_l' , rp_l )
            #tot_error += np.sum(np.square(np.float64(imgpoints_l[i] - rp_l)))
            tot_error += np.sum(np.square(np.float64(rp_l - t_imgpoints_l)))
            total_points += len(obj_points[i])

            # calculate world <-> cam2 transformation
            rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]
            #print('rvec_r', 'tvec_r', rvec_r, tvec_r)

            # compute reprojection error for cam2
            rp_r, _ = cv2.projectPoints(obj_points[i], rvec_r, tvec_r, A2, D2)
            #print( 'rp_r' , rp_r )
            # tot_error += np.square(imgpoints_r[i] - rp_r).sum()
            tot_error += np.square(rp_r - t_imgpoints_r).sum()
            total_points += len(obj_points[i])

            #temp_error = np.square(rp_r - t_imgpoints_r).sum() + np.square(rp_l - t_imgpoints_l).sum()
            temp_error = np.sum(np.square(np.float64(rp_l - t_imgpoints_l))) + np.sum(np.square(np.float64(rp_r - t_imgpoints_r)))
            temp_points = 2 * len(obj_points[i])
            temp_mean_error = np.sqrt(temp_error / temp_points)
            print(temp_mean_error)

            rp_l = rp_l.reshape(-1,2)
            rp_r = rp_r.reshape(-1,2)
            p_reproj_left.append(rp_l)
            p_reproj_right.append(rp_r)
            if (i == 21):
                break       # 21번째 장만 체크

        mean_error = np.sqrt(tot_error / total_points)

        print("=" * 50)
        print('stero_rms_manual2: ', mean_error)
        print("=" * 50)

        return mean_error, p_reproj_left, p_reproj_right

    def calc_rms_stereo3(self, obj_points, imgpoints_l, imgpoints_r, A1, D1, A2, D2, R, T):
        tot_error = 0
        total_points = 0
        p_reproj_left = []
        p_reproj_right = []

        for i in range(len(obj_points)):
            t_imgpoints_l = imgpoints_l[i].reshape(-1, 1, 2)
            t_imgpoints_r = imgpoints_r[i].reshape(-1, 1, 2)

            # print('test', t_obj_points, t_imgpoints_l, t_imgpoints_r)
            # calculate world <-> cam1 transformation
            print(A1)
            # cv2.FindExtrinsicCameraParams2(obj_points[i], imgpoints_l[i], A1, D1, temp_rvecs, temp_tvecs)

            # _, rvec_l, tvec_l, _ = cv2.solvePnPRansac(obj_points[i], t_imgpoints_l, A1, D1 )
            # _, t_rvec_l, t_tvec_l = cv2.solvePnP(obj_points[i], imgpoints_l[i], A1, D1, useExtrinsicGuess = True, flags=cv2.SOLVEPNP_ITERATIVE )

            _, rvec_l, tvec_l = cv2.solvePnP(obj_points[i], t_imgpoints_l, A1, D1, flags=cv2.SOLVEPNP_ITERATIVE)
            # rvec_test = rvec_l.copy()
            # rvec_test[0] = -rvec_test[0]
            # rvec_test[1] = -rvec_test[1]
            # rvec_test[2] = -rvec_test[2]
            print('rvec_l', 'tvec_l',rvec_l, tvec_l,'r33\n',cv2.Rodrigues(rvec_l)[0])
            # print(rvec_l.shape, tvec_l.shape)
            tvec_l[0] = tvec_l[0]
            tvec_l[1] = tvec_l[1]
            tvec_l[2] = tvec_l[2]
            rvec_l[0] = rvec_l[0]
            rvec_l[1] = rvec_l[1]
            rvec_l[2] = rvec_l[2]
            #print('rvec_l', 'tvec_l',rvec_l, tvec_l)
            # print('*'*50)
            # _, temp_rvecs, temp_tvecs = cv2.solvePnP(obj_points[i], imgpoints_l[i], A1, D1, flags=cv2.SOLVEPNP_ITERATIVE  )
            # print('temp_rvecs', 'temp_tvecs', temp_rvecs, temp_tvecs)
            # compute reprojection error for cam1
            rp_l, _ = cv2.projectPoints(obj_points[i], rvec_l, tvec_l, A1, D1)
            # print( 'rp_l' , rp_l )
            # tot_error += np.sum(np.square(np.float64(imgpoints_l[i] - rp_l)))
            tot_error += np.sum(np.square(np.float64(rp_l - t_imgpoints_l)))
            total_points += len(obj_points[i])

            # calculate world <-> cam2 transformation
            rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]
            #print('rvec_r', 'tvec_r', rvec_r, tvec_r)
            #, rvec_r_mono, tvec_r_mono = cv2.solvePnP(obj_points[i], t_imgpoints_r, A2, D2, flags=cv2.SOLVEPNP_ITERATIVE)
            #print('rvec_r_mono', 'tvec_r_mono', rvec_r_mono, tvec_r_mono)

            # compute reprojection error for cam2
            rp_r, _ = cv2.projectPoints(obj_points[i], rvec_r, tvec_r, A2, D2)
            # print( 'rp_r' , rp_r )
            # tot_error += np.square(rp_r - t_imgpoints_r).sum()
            tot_error += np.sum(np.square(np.float64(rp_r - t_imgpoints_r)))
            total_points += len(obj_points[i])

            #temp_error = np.square(rp_r - t_imgpoints_r).sum() + np.square(rp_l - t_imgpoints_l).sum()
            temp_error = np.sum(np.square(np.float64(rp_l - t_imgpoints_l))) + np.sum(np.square(np.float64(rp_r - t_imgpoints_r)))
            temp_points = 2 * len(obj_points[i])
            temp_mean_error = np.sqrt(temp_error / temp_points)
            print(temp_mean_error)


            rp_l = rp_l.reshape(-1, 2)
            rp_r = rp_r.reshape(-1, 2)
            p_reproj_left.append(rp_l)
            p_reproj_right.append(rp_r)

        mean_error = np.sqrt(tot_error / total_points)

        print("=" * 50)
        print('stero_rms_manual2_3: ', mean_error)
        print("=" * 50)

        return mean_error, p_reproj_left, p_reproj_right

    def draw_crossline(self, img, x, y, color, thinkness):
        width = 3
        if(x > width and y > width):
            cv2.line(img, (int(x), int(y-width)), (int(x), int(y+width)), color, thinkness)
            cv2.line(img, (int(x-width), int(y)), (int(x+width), int(y)), color, thinkness)
        else:
            print("error - invalid point")
        pass

    def display_reprojection_point_and_image_point(self, cal_path, imgpoint_left, imgpoint_right, reproject_left, reproject_right):
        print(cal_path + '/RIGHT/*.JPG')
        images_right = glob.glob(cal_path + '/RIGHT/*.JPG')
        images_right += glob.glob(cal_path + '/RIGHT/*.BMP')
        images_right += glob.glob(cal_path + '/RIGHT/*.PNG')
        images_left = glob.glob(cal_path + '/LEFT/*.JPG')
        images_left += glob.glob(cal_path + '/LEFT/*.PNG')
        images_left += glob.glob(cal_path + '/LEFT/*.BMP')

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            print(i, images_left[i],images_right[i] )
            timgpoint_left = np.array(imgpoint_left[i])
            timgpoint_right = np.array(imgpoint_right[i])
            treproject_left = np.array(reproject_left[i])
            treproject_right = np.array(reproject_right[i])
            #print(timgpoint_left.shape, timgpoint_right.shape, treproject_left.shape, treproject_right.shape)
            for j, jname in enumerate(timgpoint_left):
                print(timgpoint_left[j],'\t', treproject_left[j],'\t',timgpoint_right[j],'\t',treproject_right[j])

                #print(timgpoint_left[j])
                #cv2.circle(img_l, (timgpoint_left[j][0], timgpoint_left[j][1]), 2, (0, 0, 255), 2)
                self.draw_crossline(img_l, timgpoint_left[j][0], timgpoint_left[j][1], (0, 0, 255), 2)
                # print(treproject_left[j])
                #cv2.circle(img_l, (treproject_left[j][0], treproject_left[j][1]), 2, (0, 255, 0), 2)
                self.draw_crossline(img_l, treproject_left[j][0], treproject_left[j][1], (0, 255, 0), 2)
                # print(timgpoint_right[j])
                #cv2.circle(img_r, (timgpoint_right[j][0], timgpoint_right[j][1]), 2, (0, 0, 255), 2)
                self.draw_crossline(img_r, timgpoint_right[j][0], timgpoint_right[j][1], (0, 0, 255), 2)
                # print(treproject_right[j])
                #cv2.circle(img_r, (treproject_right[j][0], treproject_right[j][1]), 2, (0, 255, 0), 2)
                self.draw_crossline(img_r, treproject_right[j][0], treproject_right[j][1], (0, 255, 0), 2)

            # Draw and display the corners
            cv2.imshow("RMS_LeftImage", img_l)
            cv2.imshow("RMS_RightImage", img_r)
            cv2.waitKey(0)

        pass

def display_guide():
    print("="*80)
    print("Please follow below - made by yeolip.yoon@lge.com")
    print("this tool is support stereo calibration using both image or camera param.")
    print("if you want to use images,\n please make folder and make subfolder name about LEFT and RIGHT. and copy left,right image to each folder.")
    print("go to #1")
    print("if you want to use camear data,\n please set up json(camera intrinsic, extrinsic param) and path of points(pattern 3d coordinate and L/R image coordinate)")
    print("go to #2")
    print("if you want to test images based on designed camear data,\n please make folder and make subfolder name about LEFT and RIGHT. and copy left,right image to each folder")
    print("go to #3")

    print("="*80)
    print("#1   {} [path_of_image]".format(os.path.basename(sys.argv[0])))
    print("ex1) {} ./image33/\n".format(os.path.basename(sys.argv[0])))
    print("#2   {} [path_of_image] [json file] [path of csv]".format(os.path.basename(sys.argv[0])))
    print("ex2) {} ./input_sm/ ./input_sm/stereo_config2.json ./input_sm/\n".format(os.path.basename(sys.argv[0])))
    print("ex3) {} ./input_lgit/ ./input_lgit/stereo_config_33_2_1.json ./input_lgit/".format(os.path.basename(sys.argv[0])))
    print("#3   {} [path_of_image] [json file] ".format(os.path.basename(sys.argv[0])))
    print("ex2) {} ./image33/ ./input_sm/stereo_config2.json \n".format(os.path.basename(sys.argv[0])))
    print("="*80)

    pass

if __name__ == '__main__':
    # main(sys.argv[1:])
    # load_value_from_json('stereo_config_init.json')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filepath', help='String Filepath')
    # # #parser.add_argument('intrinsic_flag', help='Use intrinsic')
    # # #parser.add_argument('intrinsic_path', help='Path intrinsic')
    # args = parser.parse_args()
    # print(args)
    # cal_data = StereoCalibration(args.filepath)
    display_guide()
    if(len(sys.argv) == 1):
        print("Your command is wrong. \nplease see guide and follow it")
        exit(0)
    print(sys.argv[1:], sys.argv[2:])
    cal_data = StereoCalibration(sys.argv)