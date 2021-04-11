# Stereo camera calibration extension

2018년도에 Stereo camera calibration의 생산 라인 카메라 셋업을 담당하게 되어, Stereo camera calibration의 성능 보장을 위한 다양한 검증을 위해, opencv 기반의 검사 툴을 만들어 사용하였다.

## Stereo camera calibration의 정의

동일한 차트로 다양한 각도의 영상을 촬영하여, 두 카메라의 특성의 내부 파라메터와 외부 파라메터를 계산하기 위한 과정을 말하며, 최적의 파라메터를 추출하는 과정을 말한다.
>**Intrinsic parameter(내부파라메터)** - Focal length(초점거리), Principal point(이미지중심), Distortion(K1, K2, K3, P1, P2-왜곡지수)
>
>**Extrinsic parameter(외부파라메터)** - Translation, Rotation - 두 카메라간의 위치, 각도 관계

## 지원기능
이 툴은 간단하게 아래와 같은 기능을 가지고 있다.
1. 원형 그리드 마커 / 사각 그리드 마커 사진 입력 지원 ( M x N 확장 가능 )
2. 마커 대체 좌표 입력 지원 (M x N 확장 가능) - **타사 calibration algorithm 성능 비교 검증 가능**
3. 학습된 켈 결과 데이터를 입력으로  re-calibration 추가 최적화  지원( 사진/ 좌표 입력 지원)
4. 학습된 켈 결과와 입력 데이터 간의 성능검증  stereo re-projection error 계산 지원(Stereo RMS)
5.  차트로 부터 Stereo camera를 이용한 rectify, depth 계산 지원(사진/ 좌표 입력 지원)


## 확장기능

1. 모든 하위폴더 Stereo camera calibration 지원(사진/좌표 입력) 
	- **빠른 대응 - 3000대 샘플 생산 데이터 검증 및 켈 옵션 변경에 따른 정밀도 차이 검증**
2. 모든 하위폴더 Stereo camera re-calibration 지원(추가 사진/좌표 입력)
3. 모든 하위폴더 Stereo RMS 계산 지원(사진/좌표 입력) 
	- **생산에서 생산된 데이터를 입력으로 검증 가능**
4. \+ focal length와 \- focal length 둘다 변환 지원 ( Default : minus focal length)
5. 외부파라메터 두 카메라간의 RT정보 변환 지원 Left->Right/Right->Left (Default: Right->Left)

<img  src = "./desc/StereoCalibrate_phase_one.png"  width="800px" >  

 - > **Phase One** - Basic stereo camera calibration flow

<img  src = "./desc/StereoCalibrate_phase_two_three.png"  width="800px">   
 
 - > **Phase Two** - Stereo camera re-calibration flow using image and point

 - > **Phase Three** - Verify calibration parameter and image among different calibration algorithms 




## 실행방법

All your files and folders are presented as a tree in the file explorer. You can switch from one to another by clicking a file in the tree.

## 실행결과

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## 참고문헌
1. [https://github.com/bvnayak/stereo_calibration](https://github.com/bvnayak/stereo_calibration)
2. https://sourishghosh.com/2016/stereo-calibration-cpp-opencv/
3. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
Please follow below - made by  [magicst3@gmail.com](mailto:magicst3@gmail.com)  



![enter image description here](./desc/detected_point.png)
![enter image description here](./desc/detected_point_square.png)
![enter image description here](./desc/distance.png)
![enter image description here](./desc/pose_estimate.png)
![enter image description here](./desc/reprojection_and_image_point.png)
![enter image description here](./desc/RT_XYZaxis.png)
![enter image description here](./desc/correcspond_epilines.png)
