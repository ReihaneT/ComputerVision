# Q1 assignment1
import numpy as np
import cv2 as cv
# ***************** part A **************

def func(img,factor):
    x,y,z= np.shape(img)
    dim_x=int(x / factor)
    dim_y = int(y / factor)
    new_img=np.zeros((dim_x,dim_y,z),np.uint8)
    new_x=-1
    for i in range(0,x):
        if i % factor == 0:
            new_x = new_x + 1
            new_y = -1
            for j in range(0,y):
                if  j%factor==0:
                    new_y = new_y + 1
                    if new_y!=dim_y and new_x!=dim_x:
                        new_img[new_x,new_y,:]=img[i,j,:]

    return new_img
img=cv.imread('santorini.jpeg')
downsample2=func(img,2)
downsample4=func(img,4)
downsample8=func(img,8)
downsample16=func(img,16)
# ***************** part B **************
cv.imshow('main',img)
cv.imshow('down sample 2',downsample2)
cv.imshow('down sample 4',downsample4)
cv.imshow('down sample 8',downsample8)
cv.imshow('down sample 16',downsample16)
cv.waitKey(0)
# ***************** part C **************
# (I) nearest neighbour, (II) bilinear interpolation, (III) bicubic interpolation.
x,y,z= np.shape(downsample16)
dim = (x*10, y*10)
resized_NB = cv.resize(downsample16, dim, interpolation =cv.INTER_NEAREST)
resized_bilinear = cv.resize(downsample16, dim, interpolation = cv.INTER_LINEAR)
resized_bicubic = cv.resize(downsample16, dim, interpolation = cv.INTER_CUBIC)
cv.imshow('nearest neighbour',resized_NB)
cv.imshow('bilinear interpolation',resized_bilinear)
cv.imshow('bicubic interpolation',resized_bicubic)
cv.waitKey(0)