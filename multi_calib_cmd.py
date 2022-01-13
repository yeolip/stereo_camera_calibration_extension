#import numpy as np
#import cv2
import glob
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
#import camera_calibrate_input_rms as stereoCalib
import camera_calibrate_input_rms_ext as stereoCalib
# import camera_calibrate_input_rms_fisheye as stereoCalib    #opt1 & opt2 flag have hard coding

# CALIB_CHECK_COND = 4
# CALIB_FIX_INTRINSIC = 256
# CALIB_FIX_K1 = 16
# CALIB_FIX_K2 = 32
# CALIB_FIX_K3 = 64
# CALIB_FIX_K4 = 128
# CALIB_FIX_PRINCIPAL_POINT = 512
# CALIB_FIX_SKEW = 8
# CALIB_RECOMPUTE_EXTRINSIC = 2
# CALIB_USE_INTRINSIC_GUESS = 1

C_FIX_INTRINSIC = 256
C_FIX_PRINCIPAL_POINT = 4
C_USE_INTRINSIC_GUESS = 1
C_FIX_FOCAL_LENGTH = 16
C_FIX_ASPECT_RATIO = 2
C_ZERO_TANGENT_DIST = 8
C_RATIONAL_MODEL = 16384
C_SAME_FOCAL_LENGTH = 512
C_FIX_K1 = 32
C_FIX_K2 = 64
C_FIX_K3 = 128
C_FIX_K4 = 2048
C_FIX_K5 = 4096
C_FIX_K6 = 8192
C_FIX_TAUX_TAUY = 524288
C_FIX_S1_S2_S3_S4 = 65536
C_USE_EXTRINSIC_GUESS = 4194304
C_RATIONAL_MODEL = 16384
C_THIN_PRISM_MODEL = 32768


#fisheye flag
C_FISHEYE_CHECK_COND = 4
C_FISHEYE_FIX_INTRINSIC = 256
C_FISHEYE_FIX_K1 = 16
C_FISHEYE_FIX_K2 = 32
C_FISHEYE_FIX_K3 = 64
C_FISHEYE_FIX_K4 = 128
C_FISHEYE_FIX_PRINCIPAL_POINT = 512
C_FISHEYE_FIX_SKEW = 8
C_FISHEYE_RECOMPUTE_EXTRINSIC = 2
C_FISHEYE_USE_INTRINSIC_GUESS = 1


CALIB_CHECK_COND = 4
CALIB_FIX_INTRINSIC = 256
CALIB_FIX_K1 = 16
CALIB_FIX_K2 = 32
CALIB_FIX_K3 = 64
CALIB_FIX_K4 = 128
CALIB_FIX_PRINCIPAL_POINT = 512
CALIB_FIX_SKEW = 8
CALIB_RECOMPUTE_EXTRINSIC = 2
CALIB_USE_INTRINSIC_GUESS = 1

def user_calib_option(ttype, targetName):
    tflag = 0
    if(targetName == 'fisheye'):
        if(ttype == 'BAGIC'):
            tflag = CALIB_USE_INTRINSIC_GUESS|CALIB_CHECK_COND|CALIB_RECOMPUTE_EXTRINSIC|CALIB_FIX_SKEW
        elif(ttype == 'GUESS'): #current is not support skew. (we don't have skew interface)if you need, let me know.
            tflag = CALIB_USE_INTRINSIC_GUESS|CALIB_CHECK_COND|CALIB_RECOMPUTE_EXTRINSIC|CALIB_FIX_SKEW
        elif (ttype == 'FIX'):
            tflag = CALIB_FIX_INTRINSIC
        elif (ttype == 'USER1'):
            tflag = CALIB_USE_INTRINSIC_GUESS|CALIB_CHECK_COND|CALIB_RECOMPUTE_EXTRINSIC
    else:
        if(ttype == 'BAGIC'):
            tflag = C_USE_INTRINSIC_GUESS|C_FIX_ASPECT_RATIO|C_ZERO_TANGENT_DIST|C_FIX_K3|C_FIX_K4|C_FIX_K5
        elif(ttype == 'GUESS'):
            tflag = C_USE_INTRINSIC_GUESS
        elif (ttype == 'FIX'):
            tflag = C_FIX_INTRINSIC
        elif (ttype == 'USER1'):
            tflag = C_USE_INTRINSIC_GUESS|C_ZERO_TANGENT_DIST
        elif (ttype == 'USER2'):
            tflag = C_USE_INTRINSIC_GUESS|C_SAME_FOCAL_LENGTH|C_FIX_FOCAL_LENGTH|C_FIX_ASPECT_RATIO|C_ZERO_TANGENT_DIST|C_FIX_K3|C_FIX_K4|C_FIX_K5
        elif (ttype == 'USER3'):
            tflag = C_USE_INTRINSIC_GUESS|C_FIX_ASPECT_RATIO|C_FIX_K3|C_FIX_K4|C_FIX_K5
        elif (ttype == 'USER4'):
            tflag = C_USE_INTRINSIC_GUESS|C_RATIONAL_MODEL|C_ZERO_TANGENT_DIST|C_FIX_K5|C_FIX_K6

        if (targetName == 'extend'):
            tflag |= C_RATIONAL_MODEL

    return tflag


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
                print(objCal.getName())
                objCal.read_images_with_mono_stereo(args.path_img, opt1=user_calib_option('USER1',objCal.getName()), opt2=user_calib_option('USER1',objCal.getName()))
            elif (args.path_point != None):
                print("\nPOINT ", args.path_point)
                objCal.initialize(args.path_point)
                objCal.read_points_with_mono_stereo(args.path_point, None, None, opt1=user_calib_option('BAGIC',objCal.getName()), opt2=user_calib_option('BAGIC',objCal.getName()))
                # objCal.read_points_with_mono_stereo(args.path_point, None , None)
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
                    # objCal.read_points_with_stereo(args.path_point, args.path_json, None)
                    objCal.read_points_with_stereo(args.path_point, args.path_json, None, opt2=user_calib_option('FIX',objCal.getName()))
                    #please check intrinsic flag (GUESS or FIX)

        elif(args.action == 3):
            print("3) calculation Reprojection error (action is %d)\n"%(args.action))
            objCal = stereoCalib.StereoCalibration("Manual")
            if (args.path_point != None and args.path_json != None ):
                objCal.initialize(args.path_point)
                objCal.calc_rms_about_stereo(None, args.path_json, args.path_point)
            elif (args.path_img != None):
                objCal.initialize(args.path_img)
                objCal.calc_rms_about_stereo(None, args.path_json, None, args.path_img)
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
                    for tnum, tpath in enumerate(tlist_of_images):
                        print('\n###### ', tnum + 1, 'st#################')
                        print("\nIMAGE ", tpath)
                        objCal.initialize(tpath)
                        objCal.read_images_with_mono_stereo(tpath)
                if (args.path_point != None and len(tlist_of_points) >= 1):
                    for tnum, tpath in enumerate(tlist_of_points):
                        print('\n###### ', tnum + 1, 'st#################')
                        print("\nPOINT ", tpath)
                        objCal.initialize(tpath)
                        # print(user_calib_option('GUESS'))
                        objCal.read_points_with_mono_stereo(tpath, None, None)
                        # objCal.read_points_with_mono_stereo(tpath, None, None, opt1=user_calib_option('GUESS',objCal.getName()), opt2=user_calib_option('GUESS',objCal.getName()))


            elif(args.action == 2):
                print("2) recalibration (action is %d)\n"%(args.action))
                if (args.path_json != None ):
                    if (args.path_img != None and len(tlist_of_images) >= 1):
                        for tnum, tpath in enumerate(tlist_of_images):
                            print('\n###### ', tnum + 1, 'st#################')
                            print("\nIMAGE ", tpath)
                            objCal.initialize(tpath)
                            objCal.read_param_and_images_with_stereo(tpath, args.path_json)
                            # objCal.read_param_and_images_with_stereo(tpath, args.path_json, opt2=user_calib_option('FIX',objCal.getName()))
                    if (args.path_point != None and len(tlist_of_points) >= 1):
                        for tnum, tpath in enumerate(tlist_of_points):
                            print('\n###### ', tnum + 1, 'st#################')
                            print("\nPOINT ", tpath)
                            objCal.initialize(tpath)
                            # objCal.read_points_with_mono_stereo(tpath, args.path_json, None)
                            objCal.read_points_with_stereo(tpath, args.path_json, None)
                            #please check intrinsic flag (GUESS or FIX)
                else:
                    tlist_images_with_json , tlist_points_with_json = self.extract_available_calib_data(tlist_of_images, tlist_of_points)
                    for tnum, tlist_data in enumerate(tlist_images_with_json):
                        print('\n###### ', tnum+1, 'st#################')
                        print("\nIMAGE ", tlist_data[0], '\n,JSON ',  tlist_data[1])
                        objCal.initialize(tlist_data[0])
                        # objCal.read_param_and_images_with_stereo(tlist_data[0], tlist_data[1])
                        objCal.read_param_and_images_with_stereo(tlist_data[0], tlist_data[1], opt2=user_calib_option('BAGIC',objCal.getName()))
                    for tnum, tlist_data in enumerate(tlist_points_with_json):
                        print('\n###### ', tnum+1, 'st#################')
                        print("\nPOINT ", tlist_data[0], '\n,JSON ',  tlist_data[1])
                        objCal.initialize(tlist_data[0])
                        # objCal.read_points_with_mono_stereo(tlist_data[0], tlist_data[1], None)
                        # objCal.read_points_with_stereo(tlist_data[0], tlist_data[1], None)
                        objCal.read_points_with_stereo(tlist_data[0], tlist_data[1], None, opt2=user_calib_option('BAGIC',objCal.getName()))

            elif(args.action == 3):
                print("3) calculation Reprojection error (action is %d)\n"%(args.action))
                if (args.path_json != None ):
                    if (args.path_point != None and len(tlist_of_points) >= 1):
                        for tnum, tpath in enumerate(tlist_of_points):
                            print('\n###### ', tnum + 1, 'st#################')
                            print("\nPOINT ", tpath)
                            objCal.initialize(tpath)
                            objCal.calc_rms_about_stereo(None, args.path_json, tpath)
                    elif (args.path_img != None and len(tlist_of_images) >= 1):
                        for tnum, tpath in enumerate(tlist_of_images):
                            print('\n###### ', tnum + 1, 'st#################')
                            print("\nIMAGE ", tpath)
                            objCal.initialize(tpath)
                            objCal.calc_rms_about_stereo(None, args.path_json, None, tpath)
                else:
                    tlist_images_with_json , tlist_points_with_json = self.extract_available_calib_data(tlist_of_images, tlist_of_points)
                    for tnum, tlist_data in enumerate(tlist_images_with_json):
                        print('\n###### ', tnum+1, 'st#################')
                        print("\nIMAGE ", tlist_data[0], '\n,JSON ',  tlist_data[1])
                        objCal.initialize(tlist_data[0])
                        objCal.calc_rms_about_stereo(None, tlist_data[1], None, tlist_data[0])

                    for tnum, tlist_data in enumerate(tlist_points_with_json):
                        print('\n###### ', tnum+1, 'st#################')
                        print("\nPOINT ", tlist_data[0], '\n,JSON ',  tlist_data[1])
                        objCal.initialize(tlist_data[0])
                        objCal.calc_rms_about_stereo(None, tlist_data[1], tlist_data[0])


        else:
            return False

        return True

    def extract_available_calib_data(self, tlist_of_images, tlist_of_points):
        print("\t>>>>>%s>>>>>START>>>>>"%(sys._getframe(0).f_code.co_name))
        tlist_of_images_with_json = []
        tlist_of_points_with_json = []

        if(len(tlist_of_images) >= 1):
            try:
                for tpath in tlist_of_images:
                    # print("Search image path", tpath)
                    list_of_json = glob.glob(tpath + '\\' + '*.json')
                    for tjson in list_of_json:
                        tlist_of_images_with_json.append([tpath, tjson])
            except OSError:
                print("Can't change the Current Working Directory!")

        if(len(tlist_of_points) >= 1):
            try:
                for tpath in tlist_of_points:
                    # print("Search point path", tpath)
                    list_of_json = glob.glob(tpath + '\\' + '*.json')
                    for tjson in list_of_json:
                        tlist_of_points_with_json.append([tpath, tjson])
                # for tpath in tlist_of_points:
                #     os.chdir(tpath)
                #     print("Search point path", tpath)
                #     for dirpath, dirnames, filenames in os.walk("."):
                #         for filename in [f for f in filenames if f.endswith(".json")]:
                #             # print(os.path.abspath(dirpath)+'\\'+filename)
                #             tlist_of_points_with_json.append([tpath , os.path.abspath(dirpath)+'\\'+filename])
            except OSError:
                print("Can't change the Current Working Directory!!")

        print('\nlist_of_images len=', len(tlist_of_images_with_json))
        for timage_addr, tjson_addr in tlist_of_images_with_json:
            print('timage_addr=', timage_addr, ',tjson_addr=', tjson_addr)

        print('\ntlist_of_points_with_json len=', len(tlist_of_points_with_json))
        for tpoint_addr, tjson_addr in tlist_of_points_with_json:
            print('tpoint_addr=', tpoint_addr, ',tjson_addr=', tjson_addr)

        print("\t<<<<<%s<<<<<END<<<<<" % (sys._getframe(0).f_code.co_name))
        return tlist_of_images_with_json, tlist_of_points_with_json

    def extract_available_folder(self, path_img, path_point):
        print("\t>>>>>%s>>>>>START>>>>>"%(sys._getframe(0).f_code.co_name))
        list_of_images = []
        list_of_points = []

        if(path_img != None):
            try:
                os.chdir(path_img)
                print("\tSearch image path", path_img)
                for dirpath, dirnames, filenames in os.walk("."):
                    if(os.path.exists(dirpath+'/LEFT') and os.path.exists(dirpath+'/RIGHT')):
                        # print(dirpath)
                        list_of_images.append(os.path.join(dirpath)+'\\')
                    elif(os.path.exists(dirpath + '/L') and os.path.exists(dirpath + '/R')):
                        # print(dirpath)
                        list_of_images.append(os.path.join(dirpath)+'\\')
                    # for filename in [f for f in filenames if f.endswith(".raw")]:
                    #     list_of_images.append(os.path.abspath(dirpath))
                        # print(filename)
                        # break
            except OSError:
                print("Can't change the Current Working Directory!")

        if (path_point != None):
            try:
                os.chdir(path_point)
                print("\tSearch point path", path_point)
                for dirpath, dirnames, filenames in os.walk("."):
                    for filename in [f for f in filenames if f.endswith(".csv")]:
                        if("distance_from_img" in filename):
                            continue
                        if("rectify_from_img" in filename):
                            continue
                        if("temperature" in filename):
                            continue
                        list_of_points.append(os.path.abspath(dirpath))
                        # print(filename)
                        break
            except OSError:
                print("Can't change the Current Working Directory!!")

        print('\n\tlist_of_images len=', len(list_of_images))
        for tpath in list_of_images:
            print('\tlist_of_images ', tpath)

        print('\n\tlist_of_points len=', len(list_of_points))
        for tpath in list_of_points:
            print('\tlist_of_points ', tpath)

        print("\t<<<<<%s<<<<<END<<<<<" % (sys._getframe(0).f_code.co_name))
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
    # tpath = "D:\Project\HET\calibration\python_stereo\input_sm"
    # tjson = "D:/Project/HET/calibration/python_stereo/input_sm/stereo_config.json"
    # objCal.initialize(tpath)
    # objCal.calc_rms_about_stereo(None, tjson, tpath)
    # objCal.read_points_with_mono_stereo(tpath, None, None)


