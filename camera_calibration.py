import cv2
import numpy as np


row=0
col=0
distorted=[]
for i in range(20):
    image= cv2.imread("./calibration/calibration"+str(i+1)+".jpg")
    distorted.append(image)
    col+=1
    if(col==5):
        col=0
        row+=1


row=0
nx=9
ny=6
objpoints=[]
imgpoints=[]
objp=np.zeros((nx*ny,3),np.float32)
objp[:,:2]= np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for image in distorted:

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if (ret):
        objpoints.append(objp)
        imgpoints.append(corners)
        if row > 3:
            continue
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        row += 1

image_test= cv2.cvtColor(cv2.imread("./calibration/calibration1.jpg"),cv2.COLOR_BGR2RGB)
y=image_test.shape[0]
x=image_test.shape[1]
_,mtx,dist,_,_= cv2.calibrateCamera(objpoints, imgpoints,(y,x),None,None)
undistorted_image= cv2.undistort(image_test,mtx,dist, None, mtx)


with open('mtx.npy', 'wb') as f:
    np.save(f, mtx)

with open('dist.npy', 'wb') as f:
    np.save(f, dist)
