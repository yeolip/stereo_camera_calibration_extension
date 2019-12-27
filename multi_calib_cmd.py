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
    def __del__(self):
        print("*************delete SearchManager class***********\n")
    def __init__(self, args):
        # --action 1 --path_img   ./image
        # --action 1 --path_point ./point
        # --action 2 --path_img   ./image --path_json ./calib.json
        # --action 2 --path_point ./point --path_json ./calib.json
        # --action 3 --path_img   ./image --path_json ./calib.json
        # --action 3 --path_point ./point --path_json ./calib.json

        ######### --recursive  check image
        # --action 1 --path_img   ./image --recursive
        # --action 2 --path_img   ./image --path_json ./calib.json --recursive
        # --action 3 --path_img   ./image --path_json ./calib.json --recursive

        ######### --recursive  check point
        # --action 1 --path_point ./point --recursive
        # --action 2 --path_point ./point --path_json ./calib.json --recursive
        # --action 3 --path_point ./point --path_json ./calib.json --recursive

        # ***######## --recursive  check image & json
        # --action 2 --path_img   ./image --recursive
        # --action 3 --path_img   ./image --recursive

        # ***######## --recursive  check point & json
        # --action 2 --path_point ./point --recursive
        # --action 3 --path_point ./point --recursive

        CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        print("CURRENT RUN DIR ",CURRENT_DIR)
        if (self.recursive_process(args)== False):
            self.one_process(args)
        pass

    def one_process(self, args):
        # 1) calibration
        # 2) recalibration
        # 3) calculation Reprojection error
        if(args.action == 1):
            print("1) calibration (action is %d)\n"%(args.action))
            objCal = stereoCalib.StereoCalibration("Manual")
            if (args.path_img != None):
                print("\nIMAGE ", args.path_img)
                objCal.initialize(args.path_img)
                objCal.read_images_with_mono_stereo(args.path_img)
            elif (args.path_point != None):
                print("\nPOINT ", args.path_point)
                objCal.initialize(args.path_point)
                objCal.read_points_with_mono_stereo(args.path_point, None , None)
                # objCal.read_points_with_stereo(args.path_point, None, None)

        elif(args.action == 2):
            print("2) recalibration (action is %d)\n"%(args.action))
            objCal = stereoCalib.StereoCalibration("Manual")
            if (args.path_json != None ):
                if(args.path_img != None):
                    objCal.initialize(args.path_img)
                    objCal.read_param_and_images_with_stereo(args.path_img, args.path_json)
                elif(args.path_point != None):
                    objCal.initialize(args.path_point)
                    # objCal.read_points_with_mono_stereo(args.path_point, args.path_json, None)
                    objCal.read_points_with_stereo(args.path_point, args.path_json, None)
                    #please check intrinsic flag (GUESS or FIX)

        elif(args.action == 3):
            print("3) calculation Reprojection error (action is %d)\n"%(args.action))
            objCal = stereoCalib.StereoCalibration("Manual")
            if (args.path_point != None and args.path_json != None ):
                objCal.initialize(args.path_point)
                objCal.calc_rms_about_stereo(args.path_point, args.path_json, None)
            # currently, not support for calculating Rp from image
            # elif (args.path_img != None):
            #     objCal.initialize(args.path_img)
            #     self.calc_rms_about_stereo(args.path_img, args.path_json, None)
        else:
            print("action is wrong. value = ", args.action)
        return

    def recursive_process(self, args):
        # tlist_of_images = []
        # tlist_of_points = []
        ######### --recursive  check image
        # --action 1 --path_img   ./image --recursive
        # --action 2 --path_img   ./image --path_json ./calib.json --recursive
        # --action 3 --path_img   ./image --path_json ./calib.json --recursive

        ######### --recursive  check point
        # --action 1 --path_point ./point --recursive
        # --action 2 --path_point ./point --path_json ./calib.json --recursive
        # --action 3 --path_point ./point --path_json ./calib.json --recursive
        if(args.recursive == 1):
            tlist_of_images, tlist_of_points = self.extract_available_folder(args.path_img, args.path_point)
            objCal = stereoCalib.StereoCalibration("Manual")
            if (args.action == 1):
                print("1) calibration (action is %d)\n" % (args.action))
                if (args.path_img != None and len(tlist_of_images) >= 1):
                    for tpath in tlist_of_images:
                        print("\nIMAGE ", tpath)
                        objCal.initialize(tpath)
                        objCal.read_images_with_mono_stereo(tpath)
                if (args.path_point != None and len(tlist_of_points) >= 1):
                    for tpath in tlist_of_points:
                        print("\nPOINT ", tpath)
                        objCal.initialize(tpath)
                        objCal.read_points_with_mono_stereo(tpath, None , None)

            elif(args.action == 2):
                print("2) recalibration (action is %d)\n"%(args.action))
                if (args.path_json != None ):
                    if (args.path_img != None and len(tlist_of_images) >= 1):
                        for tpath in tlist_of_images:
                            print("\nIMAGE ", tpath)
                            objCal.initialize(tpath)
                            objCal.read_param_and_images_with_stereo(tpath, args.path_json)
                    if (args.path_point != None and len(tlist_of_points) >= 1):
                        for tpath in tlist_of_points:
                            print("\nPOINT ", tpath)
                            objCal.initialize(tpath)
                            # objCal.read_points_with_mono_stereo(tpath, args.path_json, None)
                            objCal.read_points_with_stereo(tpath, args.path_json, None)
                            #please check intrinsic flag (GUESS or FIX)
                else:
                    print("special 처리 필요 - auto load json\n")
            elif(args.action == 3):
                print("3) calculation Reprojection error (action is %d)\n"%(args.action))
                if (args.path_json != None ):
                    if (args.path_point != None and len(tlist_of_points) >= 1):
                        for tpath in tlist_of_points:
                            print("\nPOINT ", tpath)
                            objCal.initialize(tpath)
                            objCal.calc_rms_about_stereo(tpath, args.path_json, None)
                    # currently, not support for calculating Rp from image
                    # elif (args.path_img != None):
                    #     objCal.initialize(args.path_img)
                    #     self.calc_rms_about_stereo(args.path_img, args.path_json, None)
                else:
                    print("special 처리 필요 - auto load json\n")
        else:
            return False

        return True

    def extract_available_calib_data(self, tlist_of_images, tlist_of_points):
        # CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # print("CURRENT RUN DIR ",CURRENT_DIR)

        if(len(tlist_of_images) >= 1):
            try:
                os.chdir(path_img)
                print("Search image path", path_img)
                for dirpath, dirnames, filenames in os.walk("."):
                    if(os.path.exists(dirpath+'/LEFT') and os.path.exists(dirpath+'/RIGHT')):
                        # print(dirpath)
                        list_of_images.append(os.path.join(dirpath)+'\\')
                    elif(os.path.exists(dirpath + '/L') and os.path.exists(dirpath + '/R')):
                        # print(dirpath)
                        list_of_images.append(os.path.join(dirpath)+'\\')
            except OSError:
                print("Can't change the Current Working Directory!")

        if(len(tlist_of_points) >= 1):
            try:
                os.chdir(path_point)
                print("Search point path", path_point)
                for dirpath, dirnames, filenames in os.walk("."):
                    for filename in [f for f in filenames if f.endswith(".json")]:
                        # list_of_points.append(os.path.join(dirpath))
                        if("distance_from_img" in filename):
                            continue
                        if("rectify_from_img" in filename):
                            continue
                        if("temperature" in filename):
                            continue
                        list_of_points.append(os.path.abspath(dirpath))
                        # print(filename)
                        break
                # os.chdir(CURRENT_DIR)
            except OSError:
                print("Can't change the Current Working Directory!!")

        print('\nlist_of_images len=', len(list_of_images))
        for tpath in list_of_images:
            print('list_of_images ', tpath)
            # self.read_points_with_stereo(tpath, cal_loadjson, tpath)
            # self.read_points_with_mono_stereo(tpath, cal_loadjson, tpath)
            # self.read_images_with_mono_stereo(tpath)

        print('\nlist_of_points len=', len(list_of_points))
        for tpath in list_of_points:
            print('list_of_points ', tpath)
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
        return list_of_images, list_of_points

    def extract_available_folder(self, path_img, path_point):
        # CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # print("CURRENT RUN DIR ",CURRENT_DIR)
        list_of_images = []
        list_of_points = []

        if(path_img != None):
            try:
                os.chdir(path_img)
                print("Search image path", path_img)
                for dirpath, dirnames, filenames in os.walk("."):
                    if(os.path.exists(dirpath+'/LEFT') and os.path.exists(dirpath+'/RIGHT')):
                        # print(dirpath)
                        list_of_images.append(os.path.join(dirpath)+'\\')
                    elif(os.path.exists(dirpath + '/L') and os.path.exists(dirpath + '/R')):
                        # print(dirpath)
                        list_of_images.append(os.path.join(dirpath)+'\\')
            except OSError:
                print("Can't change the Current Working Directory!")

        if (path_point != None):
            try:
                os.chdir(path_point)
                print("Search point path", path_point)
                for dirpath, dirnames, filenames in os.walk("."):
                    for filename in [f for f in filenames if f.endswith(".csv")]:
                        # list_of_points.append(os.path.join(dirpath))
                        if("distance_from_img" in filename):
                            continue
                        if("rectify_from_img" in filename):
                            continue
                        if("temperature" in filename):
                            continue
                        list_of_points.append(os.path.abspath(dirpath))
                        # print(filename)
                        break
                # os.chdir(CURRENT_DIR)
            except OSError:
                print("Can't change the Current Working Directory!!")

        print('\nlist_of_images len=', len(list_of_images))
        for tpath in list_of_images:
            print('list_of_images ', tpath)
            # self.read_points_with_stereo(tpath, cal_loadjson, tpath)
            # self.read_points_with_mono_stereo(tpath, cal_loadjson, tpath)
            # self.read_images_with_mono_stereo(tpath)

        print('\nlist_of_points len=', len(list_of_points))
        for tpath in list_of_points:
            print('list_of_points ', tpath)
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
        return list_of_images, list_of_points

if __name__ == '__main__':
    #action
    #1) calibration
    #2) recalibration
    #3) calculation Reprojection error
    parser = argparse.ArgumentParser(description='Stereo camera calibration investigation')
    parser.add_argument('--action', type=int, required=True, help='action')
    parser.add_argument('--recursive', action='store_true', required=False, help='find recursive subdirectory')
    parser.add_argument('--path_point', type=str, required=False ,help='points path containing 3d object and 2d position')
    parser.add_argument('--path_img', type=str, required=False, help='images path')
    parser.add_argument('--path_json', required=False, help='json path containing camera calib param')
    parser.add_argument('--log', action='store_true', required=False, help='save from stdout to DebugLog.txt')
    args = parser.parse_args()
    print(args)
    # if(args.action == '1'):
    #     print("ok")
    # print("recursive %d"%(args.recursive))
    # if(args.path_img != None):
    #     print(args.path_img)
    # if(args.path_point != None):
    #     print(args.path_point)
    # if(args.path_json != None):
    #     print(args.path_json)
    if(args.log == True):
        sys.stdout = open('DebugLog.txt', 'w')
    SearchManager(args)
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

    # objCal = stereoCalib.StereoCalibration("Manual")
    # tpath = "D:\Project\HET\calib\jawha/2D-CAL/test/master_#01/09_51_25/"
    # objCal.initialize(tpath)
    # objCal.read_images_with_mono_stereo(tpath)
    # # # del objCal
    # objCal = stereoCalib.StereoCalibration("Manual")
    # tpath = "D:\Project\HET\calib\jawha/2D-CAL/test/master_#01/09_51_25/"
    # objCal.initialize(tpath)
    # objCal.read_images_with_mono_stereo(tpath)


