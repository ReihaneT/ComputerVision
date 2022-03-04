import cv2 as cv
import numpy as np

def sigma_Haris(Ix2,Iy2,Ixy,k,kernel):
    x, y = np.shape(Ix2)
    padding=kernel-1
    halh_padd=int(padding/2)
    padding_Ix2=np.zeros((x+padding,y+padding))
    padding_Ix2[halh_padd:x + halh_padd, halh_padd:y + halh_padd] = Ix2

    padding_Iy2 = np.zeros((x + padding, y + padding))
    padding_Iy2[halh_padd:x + halh_padd, halh_padd:y + halh_padd] = Iy2

    padding_Ixy = np.zeros((x + padding, y + padding))
    padding_Ixy[halh_padd:x + halh_padd, halh_padd:y + halh_padd] = Ixy

    R = np.zeros((x+padding, y+padding))
    for i in range(halh_padd, x+halh_padd ):
        for j in range(halh_padd, y+halh_padd ):
            Sx2 = np.sum(padding_Ix2[i - halh_padd:i + halh_padd+1, j - halh_padd:j + halh_padd+1] )
            Sy2 = np.sum(padding_Iy2[i - halh_padd:i + halh_padd+1, j - halh_padd:j + halh_padd+1] )
            Sxy = np.sum(padding_Ixy[i - halh_padd:i + halh_padd+1, j - halh_padd:j + halh_padd+1] )
            H = np.array([[Sx2, Sxy],
                          [Sxy, Sy2]])
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R[i,j] = det - k * (tr ** 2)
    return R

def drow_circle(originalimg,img,R,thershold,kernel):
    keypoints = np.argwhere(R> thershold )
    keypoints2=[]
    for b in keypoints:
        keypoints2.append(cv.KeyPoint(int(b[1]), int(b[0]), 1))

    cv.drawKeypoints(originalimg, keypoints2,  originalimg, color=(255,0,0))
    cv.imshow('key_points ', originalimg)

def thresh_R(R,thershold):
    x,y = np.shape(R)
    new_img=np.zeros(np.shape(R))
    for i in range(0,x):
        for j in range(0,y):
            val = R[i,j]
            if val > thershold:
                new_img[i,j]=R[i,j]
    return new_img
def non_maximum_suppression(R,kernel=3):
    x, y = np.shape(R)
    new_R=np.zeros((x,y))
    for i in range(1, x -1):
        for j in range(1, y-1):
            if R[i,j]==np.max(R[i - 1:i + 2, j - 1:j + 2]):
                new_R[i,j]=R[i,j]

    return new_R


originalImage = cv.imread("hough1.png")
#originalImage = cv.imread("contrast1.jpg")
img = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
cv.imshow('sobelx ',sobelx)
cv.imshow('sobely ',sobely)
Ix2=sobelx*sobelx
Iy2=sobely*sobely
Ixy=sobelx*sobely
cv.imshow('sobelxy ',Ixy)
k=0.04
kernel=3
R=sigma_Haris(Ix2,Iy2,Ixy,k,kernel)
R=((R-np.min(R))/(np.max(R)-np.min(R)))
thershold=0.41
#thershold=0.39
cv.imshow('R ',R)
R_threshed=thresh_R(R,thershold)
cv.imshow('R_threshed ',R_threshed)
non_max=non_maximum_suppression(R_threshed,3)
cv.imshow('non_maximum_suppression ',non_max)
drow_circle(originalImage,img,non_max,thershold,kernel)

dst = cv.cornerHarris(img,2,5,0.008)
cv.imshow('cornerHarris ',dst)
drow_circle(originalImage,img,non_max,thershold,kernel)
cv.waitKey()