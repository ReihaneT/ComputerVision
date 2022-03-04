import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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

    R = np.zeros((x, y))
    for i in range(halh_padd, x+halh_padd ):
        for j in range(halh_padd, y+halh_padd ):
            Sx2 = np.sum(padding_Ix2[i - halh_padd:i + halh_padd+1, j - halh_padd:j + halh_padd+1] )
            Sy2 = np.sum(padding_Iy2[i - halh_padd:i + halh_padd+1, j - halh_padd:j + halh_padd+1] )
            Sxy = np.sum(padding_Ixy[i - halh_padd:i + halh_padd+1, j - halh_padd:j + halh_padd+1] )
            H = np.array([[Sx2, Sxy],
                          [Sxy, Sy2]])
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R[i-halh_padd,j-halh_padd] = det - k * (tr ** 2)
    return R

def drow_circle(originalimg,img,R,thershold,kernel,name='key_points '):
    keypoints = np.argwhere(R> thershold )
    keypoints2=[]
    for b in keypoints:
        keypoints2.append(cv.KeyPoint(int(b[1]), int(b[0]), 1))
    points_name=name
    cv.drawKeypoints(originalimg, keypoints2,  originalimg, color=(255,0,0))
    cv.imshow(points_name, originalimg)


def thresh_R(img,R,thershold,kernel):
    x, y = np.shape(R)
    new_img = np.zeros(np.shape(R))
    for i in range(0, x):
        for j in range(0, y):
            val = R[i, j]
            if val > thershold:
                new_img[i, j] = R[i, j]
    return new_img
def non_maximum_suppression(R,kernel=3):
    x, y = np.shape(R)

    new_R = np.zeros((x, y))
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            if R[i, j] == np.max(R[i - 1:i + 2, j - 1:j + 2]):
                new_R[i, j] = R[i, j]

    return new_R


def N_max_elements(R, N):
    matrix=R.copy()
    x, y = np.shape(matrix)
    result_list = np.zeros((x,y))

    for i in range (0,N):
        max_element=np.max(matrix)
        m,n=np.argwhere(matrix == max_element)[0]
        result_list[m,n]=matrix[m,n]
        matrix[m,n]=0
    return result_list
def distance(result_matrix,num_keypoint):
    distance_matrix=np.zeros((num_keypoint,num_keypoint))
    point_list= np.argwhere(result_matrix>0)

    for i in range(0,num_keypoint):
        for j in range(0,num_keypoint):
            distance_matrix[i,j]=np.sqrt((point_list[i][0]-point_list[j][0])**2+(point_list[i][1]-point_list[j][1])**2)
    return distance_matrix,point_list
def condition(point1,point2,k=0.9):
    ret=0
    if point1<(point2*k):
        ret= 1
    return ret
def find_radius(R,distance_matrix,point_list):
    x,y =np.shape(distance_matrix)
    new_matrix=np.zeros((x,y))
    for i in range (0,x):
        min = distance_matrix[i, 0]
        min_j=0
        for j in range(0,y):
            poit1 = R.item(point_list[i][0],point_list[i][1])
            poit2 = R.item(point_list[j][0],point_list[j][1])
            if condition(poit1,poit2):
                if distance_matrix[i, j]<=min:
                    min = distance_matrix[i, j]
                    min_j=j
        new_matrix[i,min_j]=distance_matrix[i, min_j]

    return new_matrix
def find_point_on_image(R,max_radius,point_list):
    kernels=np.zeros(np.shape(R))
    x,y=np.shape(max_radius)
    for i in range(0,x):
        for j in range(0,y):
            if (max_radius[i,j]==1):
                kernels[point_list[i][0],point_list[i][1]]=1
                # kernels[point_list[j][0],point_list[j][1]] = 1


    return kernels

def find_maxima(R,radius,point_list,num_out_keypoints):
    martix_radius=radius.copy()
    x,y = np.shape(martix_radius)
    max_radious=np.zeros((x,y))

    for m in range(0,num_out_keypoints):
        max_R = 0
        max_i = -1
        max_j = -1
        for i in range(0,x):
            for j in range(0, y):
                if max_R<martix_radius[i,j]:
                    max_i = i
                    max_j = j
                    max_R=martix_radius[i,j]
        max_radious[max_i,max_j]=1
        martix_radius[max_i, max_j]=0
    img= find_point_on_image(R,max_radious,point_list)
    return img

def adaptive_non_maximum_suppression(R,num_keypoint=400,num_out_keypoints=100):
    result_matrix=N_max_elements(R, num_keypoint)
    distance_matrix,point_list=distance(result_matrix,num_keypoint)
    martix_radius=find_radius(R,distance_matrix,point_list)
    maxima_matrix=find_maxima(R,martix_radius,point_list,num_out_keypoints)
    return maxima_matrix

def harris(originalImage,k = 0.4,kernel = 5,thershold = 0.7,num_keypoint=400):
    img = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

    Ix2 = sobelx * sobelx
    Iy2 = sobely * sobely
    Ixy = sobelx * sobely

    k = 0.04
    kernel = 3
    R = sigma_Haris(Ix2, Iy2, Ixy, k, kernel)
    R = ((R - np.min(R)) / (np.max(R) - np.min(R)))
    thershold = 0.4
    # cv.imshow('R ', R)
    R_threshed = thresh_R(img, R, thershold, kernel=5)
    # cv.imshow('R_threshed ', R_threshed)
    non_max_adaptive = adaptive_non_maximum_suppression(R_threshed,num_keypoint)
    # cv.imshow('non_maximum_adaptive ', non_max_adaptive)
    drow_circle(originalImage, img, non_max_adaptive, thershold, kernel,'non_maximum_adaptive')

    non_max = non_maximum_suppression(R_threshed, num_keypoint)
    # cv.imshow('non_maximum_suppresion ', non_max)
    drow_circle(originalImage, img, non_max, thershold, kernel,'non_maximum_suppresion ')

    return non_max_adaptive
def extract_16_window(originalImage,non_max):
    m, n = np.shape(originalImage)
    features_places = np.argwhere(non_max != 0)
    num_features, _ = np.shape(features_places)
    window_16 = []
    keypoints2 = []
    for i in range(0, num_features):
        x, y = features_places[i]
        if ~(x - 7 < 0 or m < x + 9 or y - 7 < 0 or y + 9 > n):
            keypoints2.append(cv.KeyPoint(int(y), int(x), 1))
            a = originalImage[x - 7:x + 9, y - 7:y + 9]
            window_16.append(originalImage[x - 7:x + 9, y - 7:y + 9])
    window_16 = np.array(window_16)
    b, n, m = np.shape(window_16)
    new_window16 = np.zeros((n, m, b))
    for i in range(0, b):
        new_window16[:, :, i] = window_16[i, :, :]
    return new_window16,keypoints2
def extract_4_cell(window_16):
    cell_4 = np.zeros((4, 4,  16))
    num=0
    for i in range(0,13,4):
        for j in range(0,13,4):
            cell_4[:,:,num]=window_16[i:i+4,j:j+4]
            num=num+1
    return cell_4
def extract_orientaion(img):
    # img = cv.cvtColor(cell_4, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    theta = np.arctan2(sobely, sobelx)

    angle = theta * 180. / np.pi

    angle[angle < 0] += 360
    dir=np.zeros((8,1))

    for i in range(0, 4):
        for j in range(0, 4):

            if (0 <= angle[i, j] < 22.5 or 337.5 <= angle[i, j] ):
                dir[0,0]=dir[0,0]+1
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5 ):
                dir[1,0]=dir[1,0]+1
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                dir[2,0]=dir[2,0]+1
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                dir[3,0]=dir[3,0]+1
            elif (157.5 <= angle[i, j] < 202.5):
                dir[4,0]=dir[4,0]+1
            elif (202.5 <= angle[i, j] < 247.5):
                dir[5, 0] = dir[5, 0] + 1
            elif (247.5 <= angle[i, j] < 292.5):
                dir[6, 0] = dir[6, 0] + 1
            elif (292.5 <= angle[i, j] < 337.5):
                dir[7, 0] = dir[7, 0] + 1

    return dir
def normalazer(dir_vec):
    normalized_vec=dir_vec/np.sum(dir_vec)
    return normalized_vec

def SIFT_descriptor(originalImage1,img):
    non_max1 = harris(originalImage1)

    extracted_windows,keypoints = extract_16_window(img, non_max1)
    _, _,  num_vec = np.shape(extracted_windows)
    total_descriptor=[]
    for i in range(0, num_vec):
        cells4_4 = extract_4_cell(extracted_windows[:, :,  i])
        descriptor=[]
        for j in range (0,16):
            dir=extract_orientaion(cells4_4[:,:,j])
            descriptor.append(dir)
        descriptor=np.reshape(descriptor,(128,1))
        descriptor=normalazer(descriptor)
        total_descriptor.append(descriptor)

    return total_descriptor,keypoints
def SSD(descriptor1,descriptor2):
    s = np.sum((descriptor1 - descriptor2) ** 2)
    return s
def SSD_matrix(total_descriptor1,total_descriptor2):
    x1, _, _ = np.shape(total_descriptor1)
    x2, _, _ = np.shape(total_descriptor2)
    matches = np.zeros((x1, x2))
    for i in range(0, x1):
        for j in range(0, x2):
            matches[i, j] = SSD(total_descriptor1[i], total_descriptor2[j])
    return matches
def find_matches(matrix):
    tmp_matrix = matrix.copy()
    x, y = np.shape(tmp_matrix)
    place_min1 = []
    place_min2 = []
    for i in range(0, x):
        min1 = np.min(tmp_matrix[i, :])
        max = np.max(tmp_matrix[i, :])
        b = np.argmax(tmp_matrix[i, :] == min1)
        tmp_matrix[i, b] = max
        place_min1.append([i, b])
        min1 = np.min(tmp_matrix[i, :])
        max = np.max(tmp_matrix[i, :])
        b = np.argmax(tmp_matrix[i, :] == min1)
        tmp_matrix[i, b] = max
        place_min2.append([i, b])
    return place_min1,place_min2

def drow_matches(img1,img2,keypoints1,keypoints2):
    out_img = np.array([])
    matches = [cv.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) for idx in range(len(keypoints1))]
    img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    print('hkjh')
def ratio(matches1,matches2,matrix,thresh=0.9):
    x, y = np.shape(matrix)
    best_points = []
    for i in range(0, x):
        if matrix[matches1[i][0], matches1[i][1]] / matrix[matches2[i][0], matches2[i][1]] < thresh:
            best_points.append(i)
    return  best_points
def ratio_test(originalImage1,img1,total_descriptor1,originalImage2,img2,total_descriptor2,keypoints1,keypoints2):
    matrix = SSD_matrix(total_descriptor1, total_descriptor2)
    a = matrix.copy()
    matches1, matches2 = find_matches(a)
    best_points = ratio(matches1, matches2, matrix)
    new_key1 = []
    new_key2 = []
    for i in range(0, len(best_points)):
        point = matches1[best_points[i]]
        new_key1.append(keypoints1[point[0]])
        new_key2.append(keypoints2[point[1]])

    drow_matches(img1, img2, new_key1, new_key2)


originalImage1 = cv.imread("contrast1.jpg")
img = cv.cvtColor(originalImage1, cv.COLOR_BGR2GRAY)
total_descriptor,keypoints=SIFT_descriptor(originalImage1,img)

# originalImage2 = cv.imread("contrast5.jpg")
# width, height,ch=np.shape(originalImage1)
# dim = (height,width )
# originalImage2 = cv.resize(originalImage2, dim, interpolation = cv.INTER_AREA)
# img2 = cv.cvtColor(originalImage2, cv.COLOR_BGR2GRAY)
# total_descriptor2,keypoints2=SIFT_descriptor(originalImage2,img2)
#
# ratio_test(originalImage1,img,total_descriptor,originalImage2,img2,total_descriptor2,keypoints,keypoints2)


cv.waitKey()







