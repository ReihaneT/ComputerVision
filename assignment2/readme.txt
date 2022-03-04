#Q1:
Hough Transform
First GaussianBlur is used to blur the image and then cv2.Canny is used to detect edges. we can use every kind of edge detection algorithms.
I used  line_detection_hough to detect edges.
First the diameter is calculated and create a matrix in shape of 2d * 180(comes from different degree in a circle)
Then for each pixel that is edge in the image we have a loop for all the degree in all around the point.
Then rho(the distance between each point and the center) for the pixel with specific location is calculated.
With rho quantization we can get rho_ind that is index of rho in the matrix.
Then the specific location in the matrix with this theta and rho_ind increase by one.
In the next loop, if numbers is higher than the threshold it means that there is a line with the theta and rho in the matrix.
#Q2
Harris corner detection
first we calculate sobelx, sobely,Ix2=sobelx*sobelx, Iy2=sobely*sobely, Ixy=sobelx*sobely.
Then in sigma_Haris, images are padded first and then with help of kernel 3*3, sum every 3*3 windows in the images and try to calculate matrix H = np.array([[Sx2, Sxy],[Sxy, Sy2]]).
For calculating R we need trace and determinant H.
In thresh_R, we keep R values that are grater than threshold.
In non_maximum_suppression, the center pixel is saved if it is maxima in the 3*3 kernel. and then draw circles on the main image with drow_circle function.
#Q3
Feature descriptors/matching
SIFT_descriptor: in this function we have harris(originalImage1) and harris_opencv(img) methods for kornel detection. they have different result in different images and it depends on the parameters in these function like thresholds, kernel size and K.
extract_16_window: After that in this function all 16*16 windows around interest points extracted.
extract_4_cell: separate every 16*16 windows in to 4*4 windows
extract_orientation:get orientation for every 4*4 cells and concat to gather to have 16*8=128 descriptor for every window.
normalizer: in this function I normalized every descriptor. If normalizer is used the SIFT become contrast invariant. It is used for Q5.A
ratio_test: first form SSD matrix that contains sum of square error between descriptors in image 1 and 2
find_matches: find two minimum SSD in each row
ratio: we chose the first point in each row with the lowest SSD if the ration between 2 minimum is less than threshold
in the next for all the points that have ratio near to 1 are omitted and draw the rest of the points with drow_matches
#Q4
sobel in direct direction
sobel()
Gx = torch.tensor([[-1.0, 0, 1.0], [-2.0, 0, 2.0], [-1.0, 0, 1.0]])*1/8
Gy = torch.tensor([[-1.0, -2.0, -1.0], [0, 0, 0], [1.0, 2.0, 1.0]])*1/8
sobel in opposite direction
Sobel(Sx=np.array([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0, -1.0]])*1/8,
Sy=np.array([[1.0, 2.0, 1.0], [0, 0, 0], [-1.0, -2.0, -1.0]])*1/8)
#Q5
A: Make your feature descriptor contrast invariant
go to Q3 and uncomment #descriptor=normalazer(descriptor) in line 173
B:adaptive_non_maximum_suppression
ri = minj |xi − xj |, s.t. f (xi) < Crobust f(xj), for all xj in image
First find N(=nem_keypoints) biggest element in R with N_max_elements function.
Then find the distance between every two pairs of N maximum elements in the distance function.
find_radius: traverse all the distance matrix. in each row try to find a pixel in this row with minimum distance and satisfy the condition.
condition: f (xi) < 0.9 f(xj)
After that we find the num_out_keypoints numbers that have maximum distance.


