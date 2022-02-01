import numpy as np
import cv2 as cv
def sobel_y(img):
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],dtype=np.float)
    x, y, z = np.shape(img)
    padding_image=np.zeros((x+2,y+2,3),dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1, :] = img
    new_img = np.zeros((x, y, z))
    for i in range(0, x ):
        for j in range(0, y ):
            new_img[i, j, 0] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 0] * kernel)
            new_img[i, j, 1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 1] * kernel)
            new_img[i, j, 2] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 2] * kernel)
    return new_img
def sobel_x(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],dtype=np.float)
    x, y, z = np.shape(img)
    padding_image=np.zeros((x+2,y+2,3),dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1, :] = img
    new_img = np.zeros((x, y, z))
    for i in range(0, x ):
        for j in range(0, y ):
            new_img[i, j, 0] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 0] * kernel)
            new_img[i, j, 1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 1] * kernel)
            new_img[i, j, 2] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 2] * kernel)
    return new_img

def sobel_y_gray(img):
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8#,dtype=np.float
    x, y = np.shape(img)
    padding_image = np.zeros((x + 2, y + 2), dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1] = img
    new_img = np.zeros((x, y))
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            new_img[i - 1, j - 1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel)
    img_max = np.max(new_img)
    img_min = np.min(new_img)
    new_img = (new_img - img_min) / (img_max - img_min)
    new_img = new_img * 255
    new_img = new_img.astype(np.uint8)
    return new_img
def sobel_x_gray(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])*1/8
    x, y = np.shape(img)
    padding_image=np.zeros((x+2,y+2),dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1] = img
    new_img = np.zeros((x, y))
    for i in range(1, x+1 ):
        for j in range(1, y+1 ):
            new_img[i-1, j-1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel)
    img_max=np.max(new_img)
    img_min = np.min(new_img)
    new_img=(new_img-img_min)/(img_max-img_min)
    new_img=new_img*255
    new_img=new_img.astype(np.uint8)
    return new_img
def orientation(img):
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])*1/8
    x, y = np.shape(img)
    padding_image = np.zeros((x + 2, y + 2), dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1] = img
    new_img = np.zeros((x, y,3))
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            g_y = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_y)
            g_x = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_x)
            magnitute = np.sqrt((g_y ** 2) + (g_x ** 2))
            red_channel=0
            green_channel=0
            blue_channel = 0
            theta = np.arctan2(g_x, g_y)
            #print(magnitute,'   ', theta)
            if magnitute>0:
                red_channel=np.abs(np.cos(theta)*255)
                green_channel = np.abs(np.sin(theta) * 255)
            new_img[i-1,j-1,:]=[red_channel,green_channel,blue_channel]
    return new_img

def magnitute(img):
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])*1/8
    x, y = np.shape(img)
    padding_image = np.zeros((x + 2, y + 2), dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1] = img
    new_img = np.zeros((x, y))
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            g_y = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_y)
            g_x = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_x)
            magnitute = np.sqrt((g_y ** 2) + (g_x ** 2))
            new_img[i-1,j-1]=magnitute
    img_max = np.max(new_img)
    img_min = np.min(new_img)
    new_img = (new_img - img_min) / (img_max - img_min)
    new_img = new_img * 255
    new_img = new_img.astype(np.uint8)
    return new_img



#**************** part A******************
img=cv.imread('santorini.jpeg',0)
#img=cv.imread('img2.png',0)
cv.imshow('main img ',img)
img_x= sobel_x_gray(img)
cv.imshow('sobel x ',img_x)
img_y= sobel_y_gray(img)
cv.imshow('sobel y ',img_y)

#**************** part B******************
img_orientation=orientation(img)
cv.imshow('orientation ',img_orientation)
#**************** part C******************
img_magnitute=magnitute(img)
cv.imshow('magnitute ',img_magnitute)
#**************** part D******************
edges = cv.Canny(img,100,200)
cv.imshow('canny edges ',edges)
cv.waitKey(0)