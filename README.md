# stereo_camera_calibration
Stereo camera calibration

this section is doing stereo camera calibration using opencv python.

this tool have some functions below about stereo camera 
1. support circle grid marker calibration
2. support square grid marker calibration
3. support marker's point calibration (can be compare performance between calibration algorithms)
4. optimize calibration using before-calibration data and additional images
5. can calculate reprojection error for stereo (but it is sightly bigger than calibration reprojection error)
6. can change marker point (ex) 10x10 9x9 8x5 and so on.
6. display detection of target for debugging
7. display pose estimation for debugging
8. display detection point on image and reprojection point from marker using calibration data


Please follow below - made by magicst3@gmail.com
this tool is support stereo calibration using both image or camera param.
if you want to use images,
please make folder and make subfolder name about LEFT and RIGHT. and copy left,right image to each folder
go to #1
if you want to use camear data,
please set up json(camera intrinsic, extrinsic param) and path of points(pattern 3d coordinate and L/R image coordinate)

go to #2
if you want to test images based on designed camear data,
please make folder and make subfolder name about LEFT and RIGHT. and copy left,right image to each folder

go to #3")
