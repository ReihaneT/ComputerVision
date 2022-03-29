import torch
import torchvision
from torchvision.datasets import ImageFolder,KMNIST
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
import numpy as np
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

batch_size =128

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

# Initialize train and test sets
train_dataset = DogBreedDataset(train_ds, train_transform)
test_dataset = DogBreedDataset(test_ds, test_transform)

# Create DataLoaders
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)

def show_batch(dl):
    for img, lb in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(img.cpu(), nrow=16).permute(1,2,0))
        plt.show()
        break
show_batch(train_dl)


class conv_netpytorch(nn.Module):
    def __init__(self):
        """ Initialize conv_net
        Args:
            None
        Returns:
            None
        """
        super(conv_netpytorch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=11, stride=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=512, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        ### TODO: calculate number of in_features for this layer
        self.linear = nn.Linear(in_features=256*2*2, out_features=120)
        self.init_weights()

    def init_weights(self):
        ### EXTRA CREDIT FOR ALL: initialize network weights based on Xavier initialization version 1
        self.conv1.weight = torch.nn.init.normal_(self.conv1.weight, mean=0.0, std= 1/ np.sqrt(self.conv1.weight.shape[1]))
        self.conv2.weight = torch.nn.init.normal_(self.conv2.weight, mean=0.0, std= 1/ np.sqrt( self.conv2.weight.shape[1]))
        self.conv3.weight = torch.nn.init.normal_(self.conv3.weight, mean=0.0, std= 1/ np.sqrt(self.conv3.weight.shape[1]))
        self.conv4.weight = torch.nn.init.normal_(self.conv4.weight, mean=0.0, std= 1/ np.sqrt(self.conv4.weight.shape[1]))
        self.conv5.weight = torch.nn.init.normal_(self.conv5.weight, mean=0.0, std= 1/ np.sqrt(self.conv5.weight.shape[1]))

        ### EXTRA CREDIT FOR ALL: initialize network weights based on Xavier initialization version 2
        # self.conv1.weight = torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=6 / np.sqrt(
        #     self.conv1.weight.shape[0] + self.conv1.weight.shape[1]))
        # self.conv2.weight = torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=6 / np.sqrt(
        #     self.conv2.weight.shape[0] + self.conv2.weight.shape[1]))
        # self.conv3.weight = torch.nn.init.normal_(self.conv3.weight, mean=0.0, std=6 / np.sqrt(
        #     self.conv3.weight.shape[0] + self.conv3.weight.shape[1]))
        # self.conv4.weight = torch.nn.init.normal_(self.conv4.weight, mean=0.0, std=6 / np.sqrt(
        #     self.conv4.weight.shape[0] + self.conv4.weight.shape[1]))
        # self.conv5.weight = torch.nn.init.normal_(self.conv5.weight, mean=0.0, std=6 / np.sqrt(
        #     self.conv5.weight.shape[0] + self.conv5.weight.shape[1]))

        ### EXTRA CREDIT FOR ALL: initialize network weights based on Xavier initialization version pytorch
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.xavier_uniform_(self.conv3.weight)
        # nn.init.xavier_uniform_(self.conv4.weight)
        # nn.init.xavier_uniform_(self.conv5.weight)
        # nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        """ Perform forward pass
        Args:
            x (torch.tensor): tensor of images of shape  (B, C, H, W)
        Returns:
            torch.tensor: tesnor of output of shape (B, N_classes)
        """
        ### TODO: fill out this function
        x = self.conv1(x)
        x=F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x= self.max_pool3(x)
        x = nn.Flatten()(x)
        x = self.linear(x)
        return x

def Train(model, optimizer, dataloader, device):
    """ performs training on train set
    Args:
        model (my_nn instance): model to be trained
        optimizer (torch.optim instance)
        dataloader (torch.utils.data.DataLoader instance): dataloader for train set
        device (str): computation device ['cpu','cuda',...]
    Returns:
        list of floats: mini_batch loss sampled every 20 steps for visualization purposes
        list of floats: mini_batch accuracy sampled every 20 steps for visualization purposes
    """
    loss_tracker = []
    accuracy_tracker = []
    for i, (data, label) in enumerate(dataloader):

        ### TODO: Put data and label on target device
        inputs = data.to(device=device)
        labels = label.to(device=device)

        ### TODO: Set gradients to zero
        optimizer.zero_grad()

        ### TODO: Pass data to the model
        predictions = model.forward(inputs)

        ### TODO: Calculate the loss of predicted labels vs ground truth labels
        one_hot_label = nn.functional.one_hot(labels, num_classes=120)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(predictions, labels)



        ### TODO: Calculate gradients and update weights and biases
        loss.backward()
        # a=torch.norm(model.linear.weight.data)
        optimizer.step()
        # b = torch.norm(model.linear.weight.data)
        # print(a,b)

        if i % 20:
            accuracy = 0
            with torch.no_grad():
                loss_tracker.append(loss.item())
                ### TODO: calculate accuracy of mini_batch
                for i in range(0, np.shape(predictions)[0]):
                    if torch.argmax(predictions[i,:])==torch.argmax(one_hot_label[i,:]):
                        accuracy =accuracy + 1
            # print(accuracy,np.shape(predictions)[0])

            accuracy_tracker.append(100*accuracy / np.shape(predictions)[0])

    return loss_tracker, accuracy_tracker


def Test(model, dataloader, device):
    """ performs training on train set
    Args:
        model (my_nn instance): model to be trained
        dataloader (torch.utils.data.DataLoader instance)
        device (str): computation device ['cpu','cuda',...]
    Returns:
        floats: test set loss for visualization purposes
        floats: test set accuracy for visualization purposes
    """
    loss_tracker = []
    accuracy_tracker = []
    for i, (data, label) in enumerate(dataloader):
        if np.shape(data)[0]!=batch_size:
            break
        ### TODO: Put data and label on target device
        inputs = data.to(device=device)
        labels = label.to(device=device)


        with torch.no_grad():
            ### TODO: Pass data to the model
            predictions = model.forward(inputs)

            ### TODO: Calculate the loss of predicted labels vs ground truth labels
            one_hot_label = nn.functional.one_hot(labels, num_classes=120)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(predictions, labels)

            ### TODO: calculate accuracy of mini_batch        ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            accuracy = 0

            for j in range(0, np.shape(predictions)[0]):
                if torch.argmax(predictions[j, :]) == torch.argmax(one_hot_label[j, :]):
                    accuracy = accuracy + 1

            # accuracy_tracker.append(accuracy / np.shape(predictions)[0])
            ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        loss_tracker.append(loss.item())
        accuracy_tracker.append(100*accuracy / data.size(0))

    return sum(loss_tracker) / len(loss_tracker), sum(accuracy_tracker) / len(accuracy_tracker)
# Training hyper parameters
epochs = 5
learning_rate = 0.001

### TODO: Set target device for computations
device =  'cuda' if torch.cuda.is_available() else 'cpu' #cpu'
print(f'device: {device}')

model = conv_netpytorch()

### TODO: Put model parameters on target device
model=model.cuda()
# model=model.to(device=device)
### TODO: Initialize Adam optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

train_loss_tracker = []
train_accuracy_tracker = []

test_loss_tracker = []
test_accuracy_tracker = []

for epoch in range(epochs):
    train_loss,train_accuracy = Train(model,optimizer,train_dl,device)
    test_loss , test_accuracy = Test(model,test_dl,device)
    train_loss_tracker.extend(train_loss)
    train_accuracy_tracker.extend(train_accuracy)
    test_loss_tracker.append(test_loss)
    test_accuracy_tracker.append(test_accuracy)
    print(f'epoch: {epoch}')
    print('\t training loss/accuracy: {0:.2f}/{1:.2f}'.format(sum(train_loss)/len(train_loss), sum(train_accuracy)/len((train_accuracy))))
    print('\t testing loss/accuracy: {0:.2f}/{1:.2f}'.format(test_loss, test_accuracy))

### TODO: visualize train_loss and train_accuracy
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].plot(train_loss_tracker)
axs[0].set_title('train loss')
axs[0].set_xlabel('iteration')
axs[0].set_ylabel('loss')
axs[1].plot(train_accuracy_tracker)
axs[1].set_title('train accuracy')
axs[1].set_xlabel('iteration')
axs[1].set_ylabel('accuracy')
### TODO: visualize test_loss and test_accuracy
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].plot(test_loss_tracker)
axs[0].set_title('test loss')
axs[0].set_xlabel('iteration')
axs[0].set_ylabel('loss')
axs[1].plot(test_accuracy_tracker)
axs[1].set_title('test accuracy')
axs[1].set_xlabel('iteration')
axs[1].set_ylabel('accuracy')
plt.show()