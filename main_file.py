import cv2
import numpy as np
import matplotlib.pyplot as plt

with open('dist.npy', 'rb') as f:
    dist = np.load(f)

with open('mtx.npy', 'rb') as f:
    mtx = np.load(f)

image_test1= cv2.imread("./test/test1.jpg")
image_test2= cv2.imread("./test/test4.jpg")
image_test3= cv2.imread("./test/test6.jpg")
image_test4= cv2.imread("./test/test2.jpg")
image_test5= cv2.imread("./test/test3.jpg")
image_test6= cv2.imread("./test/test5.jpg")

my_full = [image_test1,image_test2,image_test3,image_test4,image_test5,image_test6]



def undistort(img):
    return cv2.undistort(img,mtx,dist, None, mtx)

for v , image in enumerate(my_full):
    image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    y=image.shape[0]
    x=image.shape[1]
    undistorted_image= undistort(image)
    file_loc = "undistort/"+str(v) + ".jpg"
    rgb = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_loc,rgb)



