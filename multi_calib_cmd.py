#import numpy as np
#import cv2
#import glob
import argparse
#import math
#import pandas as pd
#lip  #
#import matplotlib.pyplot as plt
#import json
import os
import sys  #, getopt
import csv
import datetime as dt
#import scipy.optimize
import camera_calibrate_input_rms as stereoCalib

class SearchManager(object):
    def __init__(self, argv):
        print(argv)
        # self.cal_path = filepath
        # if len(argv) >= 2:
        #     self.cal_path = argv[1]
        #     print('argv[1]= ', argv[1], ', argc=', len(argv), '\n\n')
        #
        # if len(argv) >= 4:
        #     self.cal_loadjson = argv[2]
        #     self.cal_loadpoint = argv[3]
        #     print('argv[2]= ', argv[2], ', len= ', len(argv), '\n\n')
        #     # self.calc_rms_about_stereo(self.cal_path, self.cal_loadjson, self.cal_loadpoint)
        #     # self.read_points_with_stereo(self.cal_path, self.cal_loadjson, self.cal_loadpoint)
        #     # self.repeat_calibration(3, 3, self.cal_path, self.cal_loadjson, self.cal_loadpoint)
        #     # self.read_points_with_mono_stereo(self.cal_path, self.cal_loadjson, self.cal_loadpoint)
        #
        # elif len(argv) >= 3:
        #     self.cal_loadjson = argv[2]
        #     print('argv[2]=', argv[2], ', len= ', len(argv), '\n\n')
        #     # self.read_param_and_images_with_stereo(self.cal_path, self.cal_loadjson)
        # else:
        #     # self.repeat_calibration(1,1,self.cal_path, 0, 0)
        #     # self.read_images_with_mono_stereo(self.cal_path)
        pass


    def repeat_calibration(self, action, idx, cal_path, cal_loadjson, cal_loadpoint):
        CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

        if(action==1 and idx==1):
            try:
                os.chdir(cal_path)
                print("change path", cal_path)
                files_to_replace = []
                for dirpath, dirnames, filenames in os.walk("."):
                    if(os.path.exists(dirpath+'/LEFT') and os.path.exists(dirpath+'/RIGHT')):
                        print(dirpath)
                        files_to_replace.append(os.path.join(dirpath)+'\\')
                    elif(os.path.exists(dirpath + '/L') and os.path.exists(dirpath + '/R')):
                        print(dirpath)
                        files_to_replace.append(os.path.join(dirpath)+'\\')
                        # files_to_replace.append(os.path.join(dirpath))
                        # print("ok")
                    # else:
                    #     print('ng')
                    # for dirnames in [f for f in dirnames if f.endswith("LEFT") ]:
                    #     for dirnames in [f for f in dirnames if f.endswith("RIGHT")]:

                        # files_to_replace.append(os.path.join(dirpath))
                        # if("distance_from_img" in filename):
                        #     # print('skip file: ',filename)
                        #     continue
                        # if("rectify_from_img" in filename):
                        #     # print('skip file: ', filename)
                        #     continue
                        # files_to_replace.append(os.path.abspath(dirpath))
                        # print(dirnames)
                        # break
                        # print("ok")
                    # for filename in [f for f in filenames if f.endswith(".json")]:
                    # print(os.path.join(dirpath))
                # os.chdir(CURRENT_DIR)
            except OSError:
                print("Can't change the Current Working Directory")

            print(files_to_replace)
            for tpath in files_to_replace:
                self.objpoints = []         # 3d point in real world space
                self.objpoints_center = []  # 3d point in real world space for center of chart
                self.imgpoints_l = []       # 2d points in image plane.
                self.imgpoints_r = []       # 2d points in image plane.
                print('tpath', tpath)
                self.cal_path = tpath
                self.cal_loadpoint = tpath
                self.cal_loadjson = cal_loadjson
                # self.read_points_with_stereo(tpath, cal_loadjson, tpath)
                # self.read_points_with_mono_stereo(tpath, cal_loadjson, tpath)
                # self.read_images_with_mono_stereo(tpath)
        else:
            try:
                os.chdir(cal_path)
                print("change path", cal_path)
                files_to_replace = []
                for dirpath, dirnames, filenames in os.walk("."):
                    for filename in [f for f in filenames if f.endswith(".csv")]:
                        # files_to_replace.append(os.path.join(dirpath))
                        if("distance_from_img" in filename):
                            # print('skip file: ',filename)
                            continue
                        if("rectify_from_img" in filename):
                            # print('skip file: ', filename)
                            continue
                        files_to_replace.append(os.path.abspath(dirpath))
                        # print(filename)
                        break
                        # print("ok")
                    # for filename in [f for f in filenames if f.endswith(".json")]:
                    # print(os.path.join(dirpath))
                os.chdir(CURRENT_DIR)
            except OSError:
                print("Can't change the Current Working Directory")

            print(files_to_replace)
            for tpath in files_to_replace:
                self.objpoints = []         # 3d point in real world space
                self.objpoints_center = []  # 3d point in real world space for center of chart
                self.imgpoints_l = []       # 2d points in image plane.
                self.imgpoints_r = []       # 2d points in image plane.
                print('tpath', tpath)
                self.cal_path = tpath
                self.cal_loadpoint = tpath
                self.cal_loadjson = cal_loadjson
                # self.read_points_with_stereo(tpath, cal_loadjson, tpath)
                # self.read_points_with_mono_stereo(tpath, cal_loadjson, tpath)
                # self.read_images_with_mono_stereo(tpath)

        # try:
        #     os.chdir(cal_path)
        #     print("change path", cal_path)
        #     files_to_replace = []
        #     for dirpath, dirnames, filenames in os.walk("."):
        #         if(os.path.exists(dirpath+'/LEFT') and os.path.exists(dirpath+'/RIGHT')):
        #             files_to_replace.append(os.path.join(dirpath))
        #             print("ok")
        #         else:
        #             print('ng')
        #         # for filename in [f for f in filenames if f.endswith(".json")]:
        #         print(os.path.join(dirpath))
        # except OSError:
        #     print("Can't change the Current Working Directory")
        #
        # print(files_to_replace)
        # self.read_images_with_mono_stereo(self.cal_path)
        return

if __name__ == '__main__':
    #action
    #1) calibration
    #2) recalibration
    #3) calculation Reprojection error
    # def read_images_with_mono_stereo(self, cal_path):
    # def read_param_and_images_with_stereo(self, cal_path, cal_loadjson):
    # def read_points_with_stereo(self, cal_path, cal_loadjson, cal_loadpoint):
    # def read_points_with_mono_stereo(self, cal_path, cal_loadjson, cal_loadpoint):
    # def calc_rms_about_stereo(self, cal_path, cal_loadjson, cal_loadpoint):

    parser = argparse.ArgumentParser(description='Stereo camera calibration investigation')
    parser.add_argument('--action', type=int, required=True, help='action')
    parser.add_argument('--recursive', action='store_true', required=False, help='find recursive subdirectory')
    parser.add_argument('--path_point', type=str, required=False ,help='points path containing 3d object and 2d position')
    parser.add_argument('--path_img', type=str, required=False, help='images path')
    parser.add_argument('--path_json', required=False, help='json path containing camera calib param')
    args = parser.parse_args()
    print(args)
    if(args.action == '1'):
        print("ok")
    print("recursive %d"%(args.recursive))
    if(args.path_img != None):
        print(args.path_img)
    if(args.path_point != None):
        print(args.path_point)
    if(args.path_json != None):
        print(args.path_json)

    # --action 1 --path_img   ./image
    # --action 1 --path_point ./point
    # --action 1 --path_img   ./image --recursive
    # --action 1 --path_point ./point --recursive
    # --action 2 --path_img   ./image --path_json ./calib.json
    # --action 2 --path_point ./point --path_json ./calib.json
    # --action 2 --path_img   ./image --path_json ./calib.json --recursive
    # --action 2 --path_point ./point --path_json ./calib.json --recursive
    # --action 2 --path_img   ./image --recursive
    # --action 2 --path_point ./point --recursive
    # --action 3 --path_img   ./image --path_json ./calib.json
    # --action 3 --path_point ./point --path_json ./calib.json
    # --action 3 --path_img   ./image --path_json ./calib.json --recursive
    # --action 3 --path_point ./point --path_json ./calib.json --recursive
    # --action 3 --path_img   ./image --recursive
    # --action 3 --path_point ./point --recursive

    # main(sys.argv[1:])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filepath', help='String Filepath')
    # args = parser.parse_args()
    # print(args)
    # cal_data = StereoCalibration(args.filepath)
    # display_guide()

    # if(len(sys.argv) == 1):
    #     print("Your command is wrong. \nplease see guide and follow it")
    #     exit(0)
    # print(sys.argv[1:], sys.argv[2:])
    # cal_data = objCal.StereoCalibration(sys.argv)
    # del objCal
    # objCal
    objCal = stereoCalib.StereoCalibration("None")
    # objCal.repeat_calibration(1, 1, args.path_img, 0, 0)
    # self.repeat_calibration(1, 1, self.cal_path, 0, 0)
    objCal.cal_path = "D:\Project\HET\calib\jawha/2D-CAL\master_#01/09_51_25/"
    objCal.read_images_with_mono_stereo("D:\Project\HET\calib\jawha/2D-CAL\master_#01/09_51_25/")
    # del objCal
    objCal = stereoCalib.StereoCalibration("None")
    objCal.cal_path = "D:\Project\HET\calib\jawha/2D-CAL\master_#02/09_53_37/"
    objCal.read_images_with_mono_stereo("D:\Project\HET\calib\jawha/2D-CAL\master_#02/09_53_37/")


