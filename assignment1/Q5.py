import numpy as np
import cv2 as cv


def gkern(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    # gauss = (1 / np.sqrt(2 * 3.14 * sig)) * np.exp(-0.5 * np.square(ax) / np.square(sig))  #
    # kernel = np.outer(gauss, gauss)
    # kernel = kernel / np.sum(kernel)
    return g



def filter_func(img,kernel,N):
    padd = int(N / 2)
    x, y = np.shape(img)
    padding_image = np.zeros((x + 2*padd, y + 2*padd), dtype=np.uint8)
    padding_image[padd:x + padd, padd:y + padd] = img
    new_img = np.zeros((x, y), dtype=np.uint8)
    for i in range(padd, x + padd):
        for j in range(padd, y + padd):
            new_img[i - padd, j - padd] = np.sum(padding_image[i - padd:i + padd+1, j - padd:j + padd+1] * kernel)


    return new_img
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = filter_func(img, Kx,3)
    Iy = filter_func(img, Ky,3)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.2):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

# def sobel_y_gray(img):
#     kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8#,dtype=np.float
#     x, y = np.shape(img)
#     padding_image = np.zeros((x + 2, y + 2), dtype=np.uint8)
#     padding_image[1:x + 1, 1:y + 1] = img
#     new_img = np.zeros((x, y))
#     for i in range(1, x + 1):
#         for j in range(1, y + 1):
#             new_img[i - 1, j - 1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel)
#     img_max = np.max(new_img)
#     img_min = np.min(new_img)
#     new_img = (new_img - img_min) / (img_max - img_min)
#     new_img = new_img * 255
#     new_img = new_img.astype(np.uint8)
#     return new_img
# def sobel_x_gray(img):
#     kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])*1/8
#     x, y = np.shape(img)
#     padding_image=np.zeros((x+2,y+2),dtype=np.uint8)
#     padding_image[1:x + 1, 1:y + 1] = img
#     new_img = np.zeros((x, y))
#     for i in range(1, x+1 ):
#         for j in range(1, y+1 ):
#             new_img[i-1, j-1] = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel)
#     img_max=np.max(new_img)
#     img_min = np.min(new_img)
#     new_img=(new_img-img_min)/(img_max-img_min)
#     new_img=new_img*255
#     new_img=new_img.astype(np.uint8)
#     return new_img
# def orientation(img,theresh=0):
#     kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8
#     kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])*1/8
#     x, y = np.shape(img)
#     padding_image = np.zeros((x + 2, y + 2), dtype=np.uint8)
#     padding_image[1:x + 1, 1:y + 1] = img
#     new_img = np.zeros((x, y))
#     for i in range(1, x + 1):
#         for j in range(1, y + 1):
#             g_y = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_y)
#             g_x = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_x)
#             theta = np.arctan2(g_x, g_y)* 180 / np.pi
#             if theta<0:
#                 theta=theta+360
#             new_img[i-1,j-1]=theta
#
#     cv.imshow('theta',new_img)
#     return new_img

# def magnitute(img):
#     kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8
#     kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])*1/8
#     x, y = np.shape(img)
#     padding_image = np.zeros((x + 2, y + 2), dtype=np.uint8)
#     padding_image[1:x + 1, 1:y + 1] = img
#     new_img = np.zeros((x, y))
#     for i in range(1, x + 1):
#         for j in range(1, y + 1):
#             g_y = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_y)
#             g_x = np.sum(padding_image[i - 1:i + 2, j - 1:j + 2] * kernel_x)
#             magnitute = np.sqrt((g_y ** 2) + (g_x ** 2))
#             new_img[i-1,j-1]=magnitute
#     img_max = np.max(new_img)
#     img_min = np.min(new_img)
#     new_img = (new_img - img_min) / (img_max - img_min)
#     new_img = new_img * 255
#     new_img = new_img.astype(np.uint8)
#     return new_img

# def non_maximum_suppression(orientation_img,magnitute_img):
#     N=3
#     padd = int(N / 2)
#     x, y = np.shape(magnitute_img)
#     padding_image = np.zeros((x + 2 * padd, y + 2 * padd), dtype=np.uint8)
#     padding_image[padd:x + padd, padd:y + padd] = magnitute_img
#     new_img = np.zeros((x, y), dtype=np.uint8)
#     for i in range(padd, x + padd):
#         for j in range(padd, y + padd):
#             theta= orientation_img[i-padd,j-padd]
#             if ((theta<=22.5 and theta>=0) or (theta<=360 and theta>=337.5)) or (theta<=202.5 and theta>=157.5) :
#                 if padding_image[i,j]>padding_image[i][j+1] and padding_image[i,j]>padding_image[i][j-1]:
#                     new_img[i-padd ,j-padd ]=padding_image[i,j]
#                 else:
#                     new_img[i-padd,j-padd]=0
#             elif (theta <=67.5  and theta >= 22.5) or (theta <= 247.5 and theta >= 202.5):
#                 if padding_image[i, j] > padding_image[i-1][j + 1] and padding_image[i, j] > padding_image[i+1][j - 1]:
#                     new_img[i-padd , j-padd ] = padding_image[i, j]
#                 else:
#                     new_img[i-padd, j-padd] = 0
#             elif (theta <=112.5  and theta >= 67.5) or (theta <= 292.5 and theta >= 247.5):
#                 if padding_image[i, j] > padding_image[i-1][j ] and padding_image[i, j] > padding_image[i+1][j ]:
#                     new_img[i-padd , j-padd ] = padding_image[i, j]
#                 else:
#                     new_img[i-padd , j-padd] = 0
#             elif (theta <=157.5 and theta >= 112.5 ) or (theta <= 337.5 and theta >= 292.5):
#                 if padding_image[i, j] > padding_image[i-1][j-1 ] and padding_image[i, j] > padding_image[i+1][j+1 ]:
#                     new_img[i-padd, j-padd] = padding_image[i, j]
#                 else:
#                     new_img[i-padd, j-padd] = 0
#             else:
#                 print ('in wrong position',i,j,magnitute_img[i, j],theta)
#
#
#
#     return new_img

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



img=cv.imread('santorini.jpeg',0)
#img=cv.imread('img2.png',0)


#### gausssian
N=11
sigma=2
kernel=gkern(N,sigma)
new_img=filter_func(img,kernel,N)
cv.imshow('gaussian  ',new_img)

#### sobel
G, theta=sobel_filters(new_img)
G = magnitute(img)
tmp=np.uint8(G)
cv.imshow('gradient  ', tmp)
#### non maximum supression
cany_output=non_max_suppression(G, theta)
cv.imshow('canny ',np.uint8(cany_output))
threshhold_img,weak, strong=threshold(cany_output)
cv.imshow('threshhold_img ',np.uint8(threshhold_img))
hysteresis_img=hysteresis(threshhold_img, weak, strong)
cv.imshow('hysteresis_img ',np.uint8(hysteresis_img))
edges = cv.Canny(img,100,200)
cv.imshow('cv_canny ',edges)
cv.waitKey()
