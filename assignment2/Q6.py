import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def octave(img,S=6,kernel_size=(5, 5)):
    x, y = np.shape(img)
    blur_list=np.zeros((x,y,S))
    blur=img
    diff_blur=np.zeros((x,y,S-1))
    blur_list[:, :, 0] = blur
    for i in range(1,S):
        blur = cv.GaussianBlur(blur, kernel_size, 0)
        blur_list[:,:,i]=blur
        diff_blur[:,:,i-1]=blur_list[:,:,i]=blur_list[:,:,i-1]
    return diff_blur
def is_maxima(patch1,patch2,patch3):
    max_patch1=np.max(patch1)
    min_patch1=np.min(patch1)
    max_patch2 = np.max(patch2)
    min_patch2 = np.min(patch2)
    max_patch3 = np.max(patch3)
    min_patch3 = np.min(patch3)
    result=0
    if patch2[1,1]>=np.max([max_patch1,max_patch2,max_patch3]) or patch2[1,1]<=np.min([min_patch1,min_patch2,min_patch3]):
        result=patch2[1,1]
    return result
def find_exterema(diff_blur):
    x, y ,z = np.shape(diff_blur)
    maxima_minima = np.zeros((x, y, z-2))
    for n in range(1,z-1):
        for i in range(1,x-1):
            for j in range(1,y-1):
                maxima_minima[i,j,n]=is_maxima(diff_blur[x-1:x+2,y-1:y+2,n-1],diff_blur[x-1:x+2,y-1:y+2,n],diff_blur[x-1:x+2,y-1:y+2,n+1])
    return maxima_minima




def sift(img,num_octave=4,S=6,kernel_size=(5, 5)):
    x, y = np.shape(img)
    img = cv.resize(img, None, x * 2, y * 2, interpolation=cv.INTER_AREA)

    for i in range(0,num_octave):
        diff_blur=octave(img,S,kernel_size)
        maxima_minima=find_exterema(diff_blur)
        img = cv.resize(img, None, x/2, y/2, interpolation=cv.INTER_AREA)
        x, y = np.shape(img)

img1= cv.imread("hough1.png",0)
img2= cv.imread("hough2.png",0)
num_octave=4
S=6
kernel_size=(5, 5)
Sift_img1=sift(img1,num_octave,S,kernel_size)
Sift_img2=sift(img2,num_octave,S,kernel_size)



blur = cv.GaussianBlur(img,(5,5),0)
