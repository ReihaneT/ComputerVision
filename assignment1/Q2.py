# Q2 assignment1
import numpy as np
import cv2 as cv


def shifting_right(img):
    kernel = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    x, y ,z= np.shape(img)
    padding_image = np.zeros((x + 2, y + 2,3), dtype=np.uint8)
    padding_image[1:x + 1, 1:y + 1,:] = img
    cv.imshow('padd', padding_image)
    new_img = np.zeros((x, y,z),dtype=np.uint8)

    for i in range(1, x + 1):
        for j in range(1, y + 1):
            new_img[i - 1, j - 1,0] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2,0] * kernel)
            new_img[i - 1, j - 1, 1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 1] * kernel)
            new_img[i - 1, j - 1, 2] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2, 2] * kernel)


    return new_img


def gkern(l, sig=1):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    # size = int(size) // 2
    # x, y = np.mgrid[-size:size + 1, -size:size + 1]
    # normal = 1 / (2.0 * np.pi * sigma ** 2)
    # g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = (1 / np.sqrt(2 * 3.14 * sig)) * np.exp(-0.5 * np.square(ax) / np.square(sig))  #
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)
    return kernel


def filter_func(img,kernel,N):
    padd = int(N / 2)
    x, y, z = np.shape(img)
    padding_image = np.zeros((x + 2*padd, y + 2*padd, 3), dtype=np.uint8)
    padding_image[padd:x + padd, padd:y + padd, :] = img
    #cv.imshow('padd', padding_image)
    new_img = np.zeros((x, y, z), dtype=np.uint8)
    for i in range(padd, x + padd):
        for j in range(padd, y + padd):
            new_img[i - padd, j - padd,0] = np.sum(padding_image[i - padd:i + padd+1, j - padd:j + padd+1,0] * kernel)
            new_img[i - padd, j - padd,1] = np.sum(padding_image[i - padd:i + padd+1, j - padd:j + padd+1,1] * kernel)
            new_img[i - padd, j - padd,2] = np.sum(padding_image[i - padd:i + padd+1, j - padd:j + padd+1,2] * kernel)

    return new_img

def func_Gaussian_a_b(alpha,beta,img,N):
    kernel_alpha = gkern(N, alpha)
    kernel_beta = gkern(N, beta)
    new_img_alpha=filter_func(img, kernel_alpha, N)
    new_img_beta = filter_func(img, kernel_beta, N)
    cv.imshow('new_img_alpha',new_img_alpha)
    cv.imshow('new_img_beta', new_img_beta)
    x, y, z = np.shape(img)
    # new_img = np.zeros((x, y, z), dtype=np.uint8)
    new_img = new_img_beta - new_img_alpha
    # new_img[:, :, 0] = new_img_beta[:, :, 0] - new_img_alpha[:, :, 0]
    # new_img[:, :, 1] = new_img_beta[:, :, 1] - new_img_alpha[:, :, 1]
    # new_img[:, :, 2] = new_img_beta[:, :, 2] - new_img_alpha[:, :, 2]
    return new_img
# ***************** part A **************
img=cv.imread('santorini.jpeg')
#img=cv.imread('img2.png')
cv.imshow('main image',img)
shifted_img=shifting_right(img)
cv.imshow('shifted imgage',shifted_img)

# ***************** part B **************
N=5
sigma=0.8
kernel=gkern(N,sigma)
new_img=filter_func(img,kernel,N)
cv.imshow('Gaussian',new_img)
# ***************** part C **************
alpa=np.sqrt(2)
beta=3
N=5
new_img=func_Gaussian_a_b(alpa,beta,img,N)
cv.imshow('Gaussian_alpha_beta',new_img)
cv.waitKey(0)