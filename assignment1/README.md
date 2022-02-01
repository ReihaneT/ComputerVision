# computer_vision_assignment1
#Q1 
 func(img,factor)
in  func(img,factor), we get image and the factor of resizing to the function and the out put of this function is resized image by the input factor 
#Q2 
shifting_right(img) function the input image is shifted to the right side with the kernel ([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
gkern(size, sigma=1) function the inputs are size of kernel abd sigma and the output is gaussian kernel with the input size and input sigma
filter_func(img,kernel,N) function the inputs are image and kernel that in this question we get from the gkern (a gaussian kernel) and N that is the size of kernel. In this function first the image is padded with zero inorder to have the same size images for input and output
func_Gaussian_a_b(alpha,beta,img,N) in this function alpha is sigma for the first gaussian and beta is the sigma for the second gaussian. img is our input image and N is the size of our kernel . like before first we get tow different  gaussian functions and apply them on the image and then get their comparison
#Q3
sobel_y_gray(img): apply ([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])*1/8 kernel on the image and get the edges in the y direction
sobel_x_gray(img): apply [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]*1/8 kernel on the image and get the edges in the y direction
orientation(img) : after applying sobel in x and y direction  and getting the magnitude by magnitute = np.sqrt((g_y ** 2) + (g_x ** 2)). after that get the orientation of image by red =np.abs(np.cos(theta)*255) and green = np.abs(np.sin(theta) * 255) and blue =0
magnitude(img) : getting the magnitude of image and normalize it to show          
#Q4
ds = CIFAR10('~/.torch/data/', train=True, download=True): downloading the cfar10 images
myDict is a dictionary of classes that contains index of all the images belong to the certain class 
#Q5
sobel_filters : calculate the sobel gradient in x and y direction and then calculate the magnitude with  np.hypot(Ix, Iy) and  theta with np.arctan2(Iy, Ix)
non_max_suppression(img, D): calculate the non maximum suppression of the input magnitude image with the orientation of D 
first convert the theta (in radian )to the degree with  D * 180. / np.pi and then in the for loop we search for the direction that the pixel point on 
in if (img[i, j] >= q) and (img[i, j] >= r) we consider to keep the pixel magnitude if the pixel value is higher than the pointed pixels otherwise set it to zero
threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.2): we set pixels lower than  highThreshold * lowThresholdRatio to zero and higher than img.max() * highThresholdRatio to 255 
pixels are between these two are set to 25
hysteresis(img, weak, strong=255): in this function pixels that have value equals to 25 and are neighbour of strong pixels with the value of 255 are set to 255 and otherwise are set to zero