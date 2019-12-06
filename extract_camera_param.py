#camaera intrinsic and extrinsic extract and check
#                                   by yeolip.yoon

import os
import re
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, getopt
import math

CURRENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def parsing_lens_params(data, inputList):
    data = data.sort_index(axis=1, ascending= True)
    # print('data\n',data)
    temp = ['k1', 'k2', 'k3', 'k4', 'k5', 'skew']
    for i in data.columns:
        for j in data.index:
            # print('data_ij ',data[i][j])
            for k in enumerate(temp):
                # print('k',k[1])
                if (data[i].name == k[1]):
                    if (data[i].name == 'k1' or data[i].name == 'k2' or data[i].name == 'k3' or data[i].name == 'k4' or data[i].name == 'k5' or data[i].name == 'skew'):
                        inputList.append(data[i][j])
                        # print("@@",data[i][j])
                    break
            # print("result",data[i].name , k[1])
            if (data[i].name == k[1]):
                break

            inputList.append(data[i][j])
            # print("##", data[i][j])
    return inputList

def parsing_camera_pose(data, inputList):
    for i in data.columns:
        for j in data.index:
            # print(data[i][j])

            inputList.append(data[i][j])
    return inputList

flag_MASTER = 1                 # exist: 1
flag_SLAVE  = 10                # exist: 10
flag_plusFOCAL = 100            # plus: 100
flag_minusFOCAL = 1000          # minus = 1000
flag_LEFTtoRIGHT = 10000        # LEFTtoRIGHT: 10000
flag_RIGHTtoLEFT = 100000       # RIGHTtoLEFT: 100000

def load_json_to_table(filename, filenum):
  data = json.load(open(filename))
  # dictdata = data['master']
  #
  # # d = data[camera_name]
  # dictdata = data['master']
  # for tkey in dictdata.keys():
  #     print('tkey', tkey)
  #
  #
  # dictdata = data['reprojection_error']
  # print('dictdata',dictdata)
  #
  # dictdata = data['slave']['camera_pose']['trans']
  # print(dictdata)
  #
  # dictdata = data['slave'].get('camera_pose')
  # print(dictdata)
  # dictdata = data['master'].get('camera_pose')
  # print(dictdata)

  tflag = 0

  if (data['master'].get('camera_pose') is not None):
      # tflag |= (flag_MASTER)
      # print("tflag_m", tflag)
      ttrans = data['master']['camera_pose']['trans']
      trot = data['master']['camera_pose']['rot']
      if(ttrans[0]==0 and ttrans[1]==0 and ttrans[2]==0 and trot[0]==0 and trot[1]==0 and trot[2]==0):
          print("master trans(0,0,0), rot(0,0,0)")
      else:
          tflag += (flag_MASTER | flag_LEFTtoRIGHT)

  if (data['slave'].get('camera_pose') is not None):
      # tflag |= (flag_SLAVE)
      # print("tflag_s", tflag)
      ttrans = data['slave']['camera_pose']['trans']
      trot = data['slave']['camera_pose']['rot']
      if(ttrans[0]==0 and ttrans[1]==0 and ttrans[2]==0 and trot[0]==0 and trot[1]==0 and trot[2]==0):
          print("slave trans(0,0,0), rot(0,0,0)")
      else:
          tflag += (flag_SLAVE | flag_RIGHTtoLEFT)

  tfocal = data['master']['lens_params'].get('focal_len')
  if (tfocal is not None):
      print(tfocal[0], tfocal[1])
      if(tfocal[0] >= 0 ):
          tflag += flag_plusFOCAL
      else:
          print('test2', tflag, flag_minusFOCAL)
          tflag += (flag_minusFOCAL)
  print("tflag", tflag)
  df = pd.DataFrame(data['master']['lens_params'])
  df2 = pd.DataFrame(data['slave']['lens_params'])
  if(tflag & flag_MASTER == 1):
      print("master")
      df3 = pd.DataFrame(data['master']['camera_pose'])
  else:
      print("slave")
      df3 = pd.DataFrame(data['slave']['camera_pose'])

  # t2trans = fpd['master'].get['camera_pose']
  # t2rot = fpd.master.camera_pose.get('rot')
  #
  # ttrans = fpd.slave.camera_pose.get('trans')
  # trot = fpd.slave.camera_pose.get('rot')
  # print(ttrans[0], ttrans[1], ttrans[2])
  # print(trot[0], trot[1], trot[2])
  # print(fpd['master']['camera_pose'])
  # print(fpd['master'].get('camera_pose'))
  # print('---------')
  # if (fpd['master'].get('camera_pose') == float('nan')):
  #     print('aaaa')
  #     if (fpd['slave']['camera_pose'].get('trans')):
  #         print('aaaa2')
  #         df3 = pd.DataFrame(fpd.slave.camera_pose)
  #         if (df3['trans'][0] == 0 and df3['trans'][1] == 0 and df3['trans'][2] == 0
  #                 and df3['rot'][0] == 0 and df3['rot'][1] == 0 and df3['rot'][2] == 0):
  #             # go to slave
  #             print('aaaa4')
  #             df3 = pd.DataFrame(fpd.master.camera_pose)
  #         # print(df3['trans'][0])
  #         # print(df3['rot'][0])
  # else:
  #     print('bbbb')
  #     if (fpd['master']['camera_pose'].get('trans')):
  #         print('bbbb2')
  #         df3 = pd.DataFrame(fpd.master.camera_pose)
  #         if(df3['trans'][0] == 0 and df3['trans'][1] == 0 and df3['trans'][2] == 0
  #                 and df3['rot'][0] == 0 and df3['rot'][1] == 0 and df3['rot'][2] == 0):
  #             #go to slave
  #             print('bbbb4')
  #             df3 = pd.DataFrame(fpd.slave.camera_pose)
  #         # print(df3['trans'][0])
  #         # print(df3['rot'][0])
  #     print('test')


      # if (fpd['master']['camera_pose'].get('trans') is None):

  # if(fpd['master']['camera_pose'].get('trans') is None):
  #     df3 = pd.DataFrame(fpd.slave.camera_pose)
  # else:
  #     df3 = pd.DataFrame(fpd.master.camera_pose)

  if data.get('reprojection_error') is None:
      print("reprojection none")
  else:
      # print(data['reprojection_error'].keys())
      # print(data['reprojection_error'].values())
      # df4 = pd.DataFrame(list(data['reprojection_error']))
      # df4 = pd.DataFrame(list(data['reprojection_error'].key()),list(data['reprojection_error'].value()))
      dictdata = data['reprojection_error']
      # print(data.get('reprojection_error'))
      # print(data.values())
      # df4 = pd.DataFrame({'reprojection_error': data['reprojection_error']} )
      # df4 = pd.Series(data['reprojection_error'])
      # print(df4)
  indata = pd.get_dummies(df)
  # print(indata)
  indata2 = pd.get_dummies(df2)
  # print(indata2)
  indata3 = pd.get_dummies(df3)
  # print(indata3)
  if data.get('reprojection_error') is None:
      print("reprojection none")


  wantedList = [filename]
  wantedList = parsing_lens_params(indata, wantedList)
  wantedList = parsing_lens_params(indata2, wantedList)
  wantedList = parsing_camera_pose(indata3, wantedList)
  print(wantedList)
  if data.get('reprojection_error') is None:
      wantedList.append(0)
  else:
      wantedList.append(data['reprojection_error'])
      print('Rp',data['reprojection_error'])
      print(wantedList)

  # col = ['Path' , 'M_imageX','M_imageY','M_focalX','M_focalY','M_k1','M_k2','M_principalX','M_principalY',
  # 'S_imageX','S_imageY','S_focalX','S_focalY','S_k1','S_k2','S_principalX','S_principalY',
  # 'Ext_RotX','Ext_RotY','Ext_RotZ','Ext_TransX','Ext_TransY','Ext_TransZ']
  # col = ['Path' , 'M_imageX','M_imageY','M_focalX','M_focalY','M_k1','M_k2','M_principalX','M_principalY',
  # 'S_imageX','S_imageY','S_focalX','S_focalY','S_k1','S_k2','S_principalX','S_principalY',
  # 'Ext_RotX','Ext_RotY','Ext_RotZ','Ext_TransX','Ext_TransY','Ext_TransZ','RMS_Stereo']
  col = ['Path' , 'M_imageX','M_imageY','M_focalX','M_focalY','M_k1','M_k2','M_p1','M_p2','M_k3','M_principalX','M_principalY','M_skew',
  'S_imageX','S_imageY','S_focalX','S_focalY','S_k1','S_k2','M_p1','M_p2','M_k3','S_principalX','S_principalY','S_skew',
  'Ext_RotX','Ext_RotY','Ext_RotZ','Ext_TransX','Ext_TransY','Ext_TransZ','RMS_Stereo']

  output = pd.DataFrame(wantedList, index=col, columns=['param'+str(filenum)])
  output =output.T
  print(output)
  # output.to_excel('output.xls')

  return output

def add_list(argument, text):
    argument.append(text)

def add_column_on_table(dataframe, inlist, colname):
    for i in dataframe.index:
        dataframe[colname] = inlist

def export_excel(dataframe, finename):
    dataframe.to_excel(finename)

def draw_graph_about_camera_param(table):
    # ax = plt.subplots(facecolor='w')
    ax = table.plot(kind='scatter' ,x = ['M_principalX'], y = ['M_principalY'], color='Red', label='Principle[Master]', alpha = 0.5)
    ax2 = table.plot(kind='scatter', x = ['S_principalX'], y = ['S_principalY'], color='Blue', label='Principle[Slave]', alpha = 0.5 ,ax = ax)
    for i, txt in enumerate(table.index):
        ax.annotate(txt, (table.M_principalX.iat[i], table.M_principalY.iat[i]), color='DarkRed', alpha = 0.5)
        ax2.annotate(txt, (table.S_principalX.iat[i], table.S_principalY.iat[i]), color='DarkBlue', alpha = 0.5)

    plt.text(s='CenterPoint', x=640, y=482, color='DarkGreen', fontsize = 16)
    plt.scatter(640, 482,  c='Green', label='Center')
    plt.grid(True)
    plt.title('Stereo camera principle point')

    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    plt.title('Translate')
    ax3.scatter(
        np.asarray(table.Ext_TransX.values, dtype="float"),np.asarray(table.Ext_TransY.values, dtype="float"),np.asarray(table.Ext_TransZ.values, dtype="float")
        # table.Ext_TransX.values, table.Ext_TransY.values, table.Ext_TransZ.values
        , c='r', marker='o')
    ax3.scatter(
        np.asarray(-0.092, dtype="float"), np.asarray(0.0, dtype="float"), np.asarray(0.0, dtype="float")
        , c='Green', label='Center')
    ax3.set_xlabel('Trans-X', fontsize=16)
    ax3.set_ylabel('Trans-Y', fontsize=16)
    ax3.set_zlabel('Trans-Z', fontsize=16)


    ax4 = table.plot(kind='scatter' ,x = ['M_focalX'], y = ['M_focalY'], color='Red', label='FocalLength[Master]', alpha = 0.5)
    ax5 = table.plot(kind='scatter', x = ['S_focalX'], y = ['S_focalY'], color='Blue', label='FocalLength[Slave]', alpha = 0.5 ,ax = ax4)
    for i, txt in enumerate(table.index):
        ax4.annotate(txt, ((table.M_focalX.iat[i]), (table.M_focalY.iat[i])), color='DarkRed', alpha = 0.5)
        ax5.annotate(txt, ((table.S_focalX.iat[i]), (table.S_focalY.iat[i])), color='DarkBlue', alpha = 0.5)

    plt.text(s='CenterPoint', x=-1470, y=-1470, color='DarkGreen', fontsize = 16)
    plt.scatter(-1470, -1470,  c='Green', label='Center')
    plt.grid(True)
    plt.title('Focal length')

    plt.show()


def calc_camera_param(table, transOffset, principleOffset, focalOffset):
  resultTx = []
  resultTy = []
  resultTz = []
  resultCx1 = []
  resultCy1 = []
  resultCx2 = []
  resultCy2 = []
  resultC2_C1 = []
  resultFx1 = []
  resultFy1 = []
  resultFx2 = []
  resultFy2 = []
  resultF2_F1 = []

  transpos = table.groupby(['Ext_TransX', 'Ext_TransY','Ext_TransZ',table.index], sort=False)
  # print(transpos.size())
  for trans in transpos:
      xCood = trans[0][0]
      yCood = trans[0][1]
      zCood = trans[0][2]
      # print(xCood, yCood, zCood)
      print(xCood)
      if  0.000-(transOffset) >= float(zCood) and float(zCood) <= 0.000+(transOffset):
          # print('NG-tz')
          add_list(resultTz, 'NG-tz')
      else:
          # print('OK_tz')
          add_list(resultTz, 'OK_tz')
      if -0.092-(transOffset) >= float(xCood) and float(xCood) <= -0.092+(transOffset):
          # print('NG_tx')
          add_list(resultTx, 'NG_tx')
      else:
          # print('OK_tx')
          add_list(resultTx, 'OK_tx')
      if 0.000-(transOffset) >= float(yCood) and float(yCood) <= 0.000+(transOffset):
          # print('NG_ty')
          add_list(resultTy, 'NG_ty')
      else:
          # print('OK_ty')
          add_list(resultTy, 'OK_ty')
  print("*"*50)

  cxy = table.groupby(['M_imageX', 'M_imageY', 'M_principalX', 'M_principalY','S_imageX', 'S_imageY', 'S_principalX', 'S_principalY',table.index], sort=False)
  # print(cxy.size())
  for i in cxy:
      preCx = 0
      preCy = 0
      for j in range(2):
          step = 4
          imageCenterX = float(i[0][0+(j*step)]) /2
          imageCenterY = float(i[0][1+(j*step)]) /2
          cx = float(i[0][2+(j*step)])
          cy = float(i[0][3+(j*step)])
          if imageCenterX - principleOffset <= float(cx) and imageCenterX + principleOffset >= float(cx):
            # print("OK_cx",j+1)
            add_list(resultCx1 if j == 0 else resultCx2, "OK_cx"+str(j+1))
          else:
            # print("NG_cx",j+1)
            add_list(resultCx1 if j == 0 else resultCx2, "NG_cx"+str(j+1))
          if imageCenterY - principleOffset <= float(cy) and imageCenterY + principleOffset >= float(cy):
            # print("OK_cy",j+1)
            add_list(resultCy1 if j == 0 else resultCy2, "OK_cy"+str(j+1))
          else:
            # print("NG_cy",j+1)
            add_list(resultCy1 if j == 0 else resultCy2, "NG_cy" + str(j + 1))
          if preCx != 0 and preCy != 0:
              if abs(float(preCx) - float(cx)) <=  principleOffset and abs(float(preCy) - float(cy)) <=  principleOffset:
                  # print("OK_c2-c1")
                  add_list(resultC2_C1, "OK_c2-c1")
              else:
                  # print("NG_c2-c1")
                  add_list(resultC2_C1, "NG_c2-c1")
          preCx = cx
          preCy = cy
  print("*"*50)

  focaldata = table.groupby(['M_focalX', 'M_focalY', 'S_focalX', 'S_focalY',table.index], sort=False)

  tFocalLen = 1470
  for i in focaldata:
      preFx = 0
      preFy = 0
      for j in range(2):
          step = 2
          fx = float(i[0][0+(j*step)])
          fy = float(i[0][1+(j*step)])
          # print(fx,fy)
          if abs(tFocalLen) - focalOffset <= abs(float(fx)) and  abs(tFocalLen) + focalOffset >= abs(float(fx)):
              # print("OK_fx",j+1)
              add_list(resultFx1 if j == 0 else resultFx2, "OK_fx" + str(j + 1))
          else:
              # print("NG_fx",j+1)
              add_list(resultFx1 if j == 0 else resultFx2, "NG_fx" + str(j + 1))
          if abs(tFocalLen) - focalOffset <= abs(float(fy)) and  abs(tFocalLen) + focalOffset >= abs(float(fy)):
              # print("OK_fy",j+1)
              add_list(resultFy1 if j == 0 else resultFy2, "OK_fy" + str(j + 1))
          else:
              # print("NG_fy",j+1)
              add_list(resultFy1 if j == 0 else resultFy2, "NG_fy" + str(j + 1))
          if preFx != 0 and preFy != 0:
              if abs(float(preFx) - float(fx)) <=  focalOffset and abs(float(preFy) - float(fy)) <=  focalOffset:
                  # print("OK_f2-f1")
                  add_list(resultF2_F1, "OK_f2-f1")
              else:
                  # print("NG_f2-f1")
                  add_list(resultF2_F1, "NG_f2-f1")
          preFx = fx
          preFy = fy

  print(resultTx)
  print(resultTy)
  print(resultTz)
  print(resultCx1)
  print(resultCy1)
  print(resultCx2)
  print(resultCy2)
  print(resultC2_C1)
  print(resultFx1)
  print(resultFy1)
  print(resultFx2)
  print(resultFy2)
  print(resultF2_F1)
  add_column_on_table(table, resultTx, 'resultTx')
  add_column_on_table(table, resultTy, 'resultTy')
  add_column_on_table(table, resultTz, 'resultTz')
  add_column_on_table(table, resultCx1, 'resultCx1')
  add_column_on_table(table, resultCy1, 'resultCy1')
  add_column_on_table(table, resultCx2, 'resultCx2')
  add_column_on_table(table, resultCy2, 'resultCy2')
  add_column_on_table(table, resultC2_C1, 'resultC2_C1')
  add_column_on_table(table, resultFx1, 'resultFx1')
  add_column_on_table(table, resultFy1, 'resultFy1')
  add_column_on_table(table, resultFx2, 'resultFx2')
  add_column_on_table(table, resultFy2, 'resultFy2')
  add_column_on_table(table, resultF2_F1, 'resultF2_F1')
  # print(table)

def main(argv):
  if len(argv) == 0:
    print('How to use : $> python chkCamaraParam.py [FILE_NAME]')
    print('\n option \`--all` : process all files recursively')
    return

  files_to_replace = []
  print(argv[0])
  if (argv[0] == '--all'):
    for dirpath, dirnames, filenames in os.walk("."):
      for filename in [f for f in filenames if f.endswith(".json")]:
          files_to_replace.append(os.path.join(dirpath, filename))
          # print(os.path.join(dirpath, filename))
  else:
    print("ok")
    files_to_replace.append(CURRENT_DIR + '/' + argv[0])

  print(CURRENT_DIR)
  print(files_to_replace)
  print("*"*50)

  table = []
  total_number = 0
  for single_file in files_to_replace:
    table.append(load_json_to_table(single_file, total_number+1))
    total_number+=1
  result = pd.concat([i for i in table], axis=0)

  print(len(result))
  print("&&" * 50)

  # calc_camera_param(result, 0.001, 10, 10)

  print('result', result)

  for i in range(0,1000,1):
      filename = 'List_CameraParam%03d.xls'%(i)
      if not os.path.isfile(filename):
        export_excel(result, filename)
        print('save', filename)
        break

  draw_graph_about_camera_param(result)

if __name__ == '__main__':
  main(sys.argv[1:])
  #main(['--all'])
