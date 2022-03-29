import random

import torch
import torchvision
from torchvision.datasets import ImageFolder,KMNIST
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
import numpy as np

# load the alexnet model using pytorch hub from:
# https://github.com/pytorch/vision/blob/winbuild/v0.6.0/torchvision/models/alexnet.py
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# switch the model to "eval" mode
model.eval()
# load dataset from path
# set path to images location on your local machine or google drive
#path = 'drive/MyDrive/images/Images'  # Google drive
path = './images/Images'               #local machine
dataset = ImageFolder(path)
print(f'number of images: {len(dataset)}')
print(f'number of classes: {len(dataset.classes)}')

# Create train and test splits of original dataset
test_pct = 0.3
test_size = int(len(dataset)*test_pct)
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])


# In order to apply transformations, we use a custom dataset
# see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
class DogBreedDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label
batch_size =64

#train set transforms
train_transform = transforms.Compose([
   transforms.Resize((240, 240)),
    transforms.ToTensor()
])

# test set transforms
test_transform = transforms.Compose([
    transforms.Resize((240,240)),
    transforms.ToTensor()
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize train and test sets
train_dataset = DogBreedDataset(train_ds, train_transform)
test_dataset = DogBreedDataset(test_ds, test_transform)

# Create DataLoaders
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)

data,_ = next(iter(train_dl))
data=data.to(device)
sample_image = data[0]

# the mean and standard deviations of ImageNet dataset
# that were used for preprocessing AlexNet training data
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transform_norm = transforms.Compose([
    transforms.Normalize(mean, std)
])

img_normalized = transform_norm(data)
### TODO: visualize the sample image
rand=np.random.randint(batch_size,size=1)
img=img_normalized[rand,:,:,:]
cpu_img=img.to('cpu')
plt.imshow(cpu_img[0,-1])
plt.show()

# first layer of Alexnet
model.to(device)
first_layer = model.features[0]

### TODO: get weights of first layer kernels of the model
first_layer_weights=first_layer.weight
### TODO: pass sample image to the first layer
first_layer_output=first_layer(img)

### print summery
print(model)
### get all the layers

visualisation = {}
def hook_fn(m, i, o):
    visualisation[m] = o

def get_all_layers(net):
  for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer)
    else:
      # it's a non sequential. Register a hook
      layer.register_forward_hook(hook_fn)

get_all_layers(model)
prediction= model(img)
# model.
### TODO: randomly select 20 filters out of 64
### TODO: show selected kernel and their convolution output on sample image.

for i in range(0,20):
    figure = plt.figure(figsize=(8, 8))
    random_num = random.randint(0, 63)
    kernel=first_layer_weights[random_num,:,:,:]
    out_kernel=first_layer_output[0,random_num,:,:]
    kernel_cpu=kernel.to('cpu')
    kernel_cpu=kernel_cpu.detach().numpy()

    out_kernel_cpu = out_kernel.to('cpu')
    out_kernel_cpu = out_kernel_cpu.detach().numpy()

    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.imshow(out_kernel_cpu)
    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.imshow(kernel_cpu[-1,:])
    plt.show()

