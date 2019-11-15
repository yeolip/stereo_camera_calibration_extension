# stereo_camera_calibration

Stereo camera calibration

this section is doing stereo camera calibration using opencv python.
this tool have some functions below about stereo camera

![enter image description here](./desc/StereoCalibrate_phase_one.png)
![enter image description here](./desc/StereoCalibrate_phase_two_three.png)

 1. **stereo camera calibration using stereo images**
    - **support circle grid marker**
    - **support square grid marker**
2. **stereo camera calibration using stereo points**
    - **support marker's point calibration** 
    - **(can be compare performance between calibration algorithms)**
3. **optimize calibration using camera-calibration data and additional images**
4. **optimize calibration using camera-calibration data and additional points**
5. **can calculate reprojection error for stereo** 
    - **it is sightly bigger than values of reprojection error**
6.  **display detected point  of target for debugging**
7.  **display pose estimation for debugging about marker chart**
8.  **display detection point on image and reprojection point from marker using calibration data**
9. **can transform from plus to minus or revert about focal length**
10. **can transform from left to right image or revert about RT**

Please follow below - made by  [magicst3@gmail.com](mailto:magicst3@gmail.com)  
this tool is support stereo calibration using both image or camera param. if you want to use images, please make folder and make subfolder name about LEFT and RIGHT. and copy left,right image to each folder go to #1 if you want to use camear data, please set up json(camera intrinsic, extrinsic param) and path of points(pattern 3d coordinate and L/R image coordinate)

go to #2 if you want to test images based on designed camear data, please make folder and make subfolder name about LEFT and RIGHT. and copy left,right image to each folder

go to #3

#1 camera_calibrate_input_rms.py [path_of_image] ex1) camera_calibrate_input_rms.py ./image33/

#2 camera_calibrate_input_rms.py [path_of_image] [json file] [path of csv] ex2) camera_calibrate_input_rms.py ./input_sm/ ./input_sm/stereo_config2.json ./input_sm/

ex3) camera_calibrate_input_rms.py ./input_lgit/ ./input_lgit/stereo_config_33_2_1.json ./input_lgit/ #3 camera_calibrate_input_rms.py [path_of_image] [json file] ex2) camera_calibrate_input_rms.py ./image33/ ./input_sm/stereo_config2.json

================================================================================

#example #D:\HET\calib\data\example\image\cal\circle\raw #change option

[](https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_png_or_raw--------1--------------------------------png-0-raw-1)select_png_or_raw = 1 #png: 0, raw: 1 #D:\HET\calib\data\example\image\cal\circle\png #change option # [](https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_png_or_raw--------0--------------------------------png-0-raw-1)select_png_or_raw = 0 #png: 0, raw: 1 #D:\HET\calib\data\example\image\cal\square\png_8_5 # [](https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#marker_point_x--8-----patterns-width-point)marker_point_x = 8 #pattern's width point # [](https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#marker_point_y--5-----patterns-height-point)marker_point_y = 5 #pattern's height point # [](https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#marker_length--60------patterns-gap-unit-is-mm)marker_length = 60 #pattern's gap (unit is mm)

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_detect_pattern----1--------------------------------circle-0-square-1)select_detect_pattern = 1 #circle: 0, square: 1

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_png_or_raw--------0--------------------------------png-0-raw-1-1)select_png_or_raw = 0 #png: 0, raw: 1

#D:\HET\calib\data\example\image\cal\square\raw_6_4

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#marker_point_x--6-----patterns-width-point)marker_point_x = 6 #pattern's width point

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#marker_point_y--4-----patterns-height-point)marker_point_y = 4 #pattern's height point

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#marker_length--60------patterns-gap-unit-is-mm-1)marker_length = 60 #pattern's gap (unit is mm)

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_detect_pattern----1--------------------------------circle-0-square-1-1)select_detect_pattern = 1 #circle: 0, square: 1

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_png_or_raw--------1--------------------------------png-0-raw-1-1)select_png_or_raw = 1 #png: 0, raw: 1

#D:\HET\calib\data\example\image\cal\circle\raw D:\HET\calib\data\example\image\cal\circle\raw\stereo_config_result_r_to_l.json

(https://github.com/yeolip/stereo_camera_calibration/tree/readme_edit#select_png_or_raw--------1--------------------------------png-0-raw-1-2)select_png_or_raw = 1 #png: 0, raw: 1

# reference

[https://github.com/bvnayak/stereo_calibration](https://github.com/bvnayak/stereo_calibration)


![enter image description here](./desc/detected_point.png)
![enter image description here](./desc/detected_point_square.png)
![enter image description here](./desc/distance.png)
![enter image description here](./desc/pose_estimate.png)
![enter image description here](./desc/reprojection_and_image_point.png)
![enter image description here](./desc/RT_XYZaxis.png)
![enter image description here](./desc/correcspond_epilines.png)
