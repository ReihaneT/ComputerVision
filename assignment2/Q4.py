#import all the necessary libraries
import torch
import torchvision
import numpy as np
from torch import nn
#random
import random
from torchvision import datasets, transforms
#plotting
import plotly.express as px

# load and show an image with Pillow
from PIL import Image
import cv2
import random
# define the Sobel operator
class Sobel(torch.nn.Module):
    # constructor
    def __init__(self,Sx=np.array([[-1.0, 0, 1.0], [-2.0, 0, 2.0], [-1.0, 0, 1.0]])*1/8,Sy=np.array([[-1.0, -2.0, -1.0], [0, 0, 0], [1.0, 2.0, 1.0]])*1/8):
        super(Sobel, self).__init__()

        # TODO: Define the sobel kernels Sx and Sy; Use numpy and define the 3x3 Sobel filters
        Sx = Sx

        # reshape
        Sx = np.reshape(Sx, (1, 1, 3, 3))

        # TODO: Define the sobel kernels Sx and Sy; Use numpy and define the 3x3 Sobel filters
        Sy = Sy

        # reshape
        Sy = np.reshape(Sy, (1, 1, 3, 3))

        # TODO: use torch.nn.Conv2D to create a convolutional layer for the Sx; set the bias=False, kernel_size=3

        self.Sx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        # Overwrite the kernel values
        self.Sx.weight = torch.nn.Parameter(torch.from_numpy(Sx).float())

        # TODO: use torch.nn.Conv2D to create a convolutional layer for the Sx; set the bias=False, kernel_size=3
        self.Sy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        # Overwrite the kernel values
        self.Sy.weight = torch.nn.Parameter(torch.from_numpy(Sy).float())

        return

    def forward(self, x):
        # x is the input image;

        # reshape it to 1x1x28x28
        x = torch.reshape(x, (1, 1, 28, 28))

        # apply the kernels Sx and Sy
        gx = self.Sx(x)
        gy = self.Sy(x)

        # reshape it back to 1x28x28
        gx = gx.squeeze(0)
        gy = gy.squeeze(0)
        return gx, gy

def np_img_to_tensor(img):
    x, y = np.shape(img)
    img_edged = np.zeros([1, 1, x, y])
    img_edged[0, 0, :, :] = img
    img_edged=torch.from_numpy(img_edged).float()
    return img_edged
#Load the example image to be used for debugging, convert to grayscale, and then convert to numpy array

image = np.asarray(Image.open('example.png').convert('L'))

#Show the image
px.imshow(image)

#Convert to tensor and reshape
image = torch.from_numpy(image).float()
image = torch.reshape(image, (1,28,28))

#TODO: Create an instance of the class Sobel
#model = Sobel()
model = Sobel(Sx=np.array([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0, -1.0]])*1/8,Sy=np.array([[1.0, 2.0, 1.0], [0, 0, 0], [-1.0, -2.0, -1.0]])*1/8)
#apply the kernel to the image by calling the forward function
gx,gy = model.forward(image)

#convert to numpy; size is now 28x28
gx = gx.permute(1,2,0).detach().numpy().squeeze(2)
gy = gy.permute(1,2,0).detach().numpy().squeeze(2)

#show the gradient x
cv2.imshow('gx',gx)

#show the gradient y
cv2.imshow('gy',gy)

#TODO: calculate and show the gradient magnitude
gradient_magnitude = np.sqrt(gx*gx+gy*gy)
cv2.imshow('gradient_magnitude',gradient_magnitude)
#TODO: Calculate the gradient orientation and threshold anything less than e.g. 100
gradient_orientation =  np.arctan2(gy, gx)
gradient_orientation=((gradient_orientation-np.min(gradient_orientation))/(np.max(gradient_orientation)-np.min(gradient_orientation)))*255
threshold=np.zeros((np.shape(gradient_orientation)))
places=np.argwhere(gradient_orientation<100)
for i in range(0,len(places)):
    threshold[places[i][0],places[i][1]]=gradient_orientation[places[i][0],places[i][1]]

cv2.imshow('gradient_orientation',gradient_orientation)
cv2.imshow('gradient_orientation_threshold',threshold)



px.imshow(gradient_orientation)
#TODO: Calculate the *edge* direction
edge_orientation =  np.arctan2(gy, gx)*180/np.pi
minus=np.argwhere(edge_orientation<0)
for i in range(0,len(minus)):
    edge_orientation[minus[i][0],minus[i][1]]=edge_orientation[minus[i][0],minus[i][1]]+360
edge_orientation=edge_orientation-90
cv2.imshow('edge_orientation',edge_orientation)



DATASET_PATH = "data"

#TODO: Download the MNIST dataset
dataset1 = datasets.MNIST('../data', train=True, download=True)
dataset2 = datasets.MNIST('../data', train=False)
training_dataset = torch.utils.data.DataLoader(dataset1)
testing_dataset = torch.utils.data.DataLoader(dataset2)

#TODO: Get a random image from the training dataset and show it
image = random.randint(0, len(testing_dataset)) #The images in training_dataset are of type tensor

train_image_, train_target_ = dataset1[image]
train_image_.show()
input=np_img_to_tensor(train_image_)
result=model(input)
img_x = result[0]
img_y = result[1]
img_edged_x = img_x[0, :, :]
img_edged_y = img_y[0, :, :]
edged_x=img_edged_x.detach().numpy()
cv2.imshow('img_edged_x',edged_x)
edged_y=img_edged_y.detach().numpy()
cv2.imshow('img_edged_y',edged_y)

#Make sure there are 60K training examples, and 10K testing examples
print(training_dataset, testing_dataset)
cv2.waitKey()
#TODO: Repeat the same steps as before
#1. Apply the Sobel kernels and show the gx and gy
#2. Calculate gradient magnitude and show it
#3. Calculate gradient orientation and show it
#4. Calculate the *edge* orientation and show it
#5. Change the signs of the Sobel filter and see how that affects the edge orientation