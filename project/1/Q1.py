# import required libraries, DO NOT MODIFY!
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

### TODO: set random seed to your Student ID
random_seed =40193325
torch.manual_seed(random_seed);
# datasets hyper parameters
batch_size = 32
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize kmnist train and test datasets
# These two lines will download the datasets in a folder called KMNIST.
# The folder will be written in the same directory as this script.
# The download will occur once. Subsequent executions will not re-download the datasets if they exist.
kmnist_train_set = KMNIST(root='.',
                         train=True,
                         download=True,
                         transform=train_transform)
kmnist_test_set = KMNIST(root='.',
                         train=False,
                         download=True,
                         transform=test_transform)

# Initialize kmnist train and test data loaders.
kmnist_train_loader = torch.utils.data.DataLoader(kmnist_train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)
kmnist_test_loader = torch.utils.data.DataLoader(kmnist_test_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)
### TODO: visualize a sample image and corresponding label from KMNIST
for images in kmnist_train_loader:
    data = images[0]
    lable = images[1]
    for i in range(0,1):
        a=data[i,0,:,:]
        b=lable[i]
        c=np.array(b)
        plt.imshow(data[i,0,:,:])
        plt.title(np.array(b))
        plt.show()
def Sigmoid(x):
    """ Identity activation function
    Args:
        x (torch.tensor)
    Return:
        torch.tensor: a tensor of shape of x
    """
    ### TODO: Fill out this function
    return 1 / (1 + torch.exp(x))
def ReLU(x):
    """ ReLU activation function
    Args:
        x (torch.tensor)
    Return:
        torch.tensor: a tensor of shape of x
    """
    ### TODO: Fill out this function
    out = torch.max(torch.tensor(0), x)
    # batch, kernel, height, width= np.shape(x)
    # out=torch.maximum(torch.zeros((batch, kernel, height, width)).to(device), x)
    return out
def Identity(x):
    """ Identity activation function
    Args:
        x (torch.tensor)
    Return:
        torch.tensor: a tensor of shape of x
    """
    ### TODO: Fill out this function

    return x
def Softmax(x):
    """ Softmax function
    Args:torch.log(
        x (torch.tensor): inputs tensor of size (B,F)
        dim (int): A dimension along which Softmax will be computed
    Return:
        torch.tensor: a tensor of shape of x
    """
    ### TODO: Fill out this function
    batch, classes=np.shape(x)
    for i in range(0,batch):
        x[i]=torch.tensor(x[i]-torch.max(x[i])) / torch.sum(torch.tensor(x[i]-torch.max(x[i])))#x[i]=torch.exp(x[i]-torch.max(x[i])) / torch.sum(torch.exp(x[i]-torch.max(x[i])))
    return x
def CE_loss(predictions,labels):
    """ Cross entropy loss
    Args:
        predictions (torch.tensor): tensor of shape of (B,C)
        labels (torch.tensor): tensor of shape of (B,1)
    Returns:
        torch.tensor: a tensor of shape of (1,)
    """
    ### TODO: Fill out this function
    # eps=torch.tensor(torch.finfo(torch.float32).eps)
    np.shape(predictions)[0]
    one_hot_label = nn.functional.one_hot(labels, num_classes=10)
    logsum=torch.logsumexp(predictions, 1)
    logsum=torch.reshape(logsum,(np.shape(predictions)[0],1))
    log_loss= torch.sub(predictions,logsum)
    out=torch.mul(log_loss,one_hot_label)
    minus_one=torch.tensor(-1)
    out=torch.mul(minus_one,out)
    batch_loss=torch.sum(out)/np.shape(predictions)[0]




    # batch_loss=torch.tensor(0)
    # batch_loss=batch_loss.to(device)
    # x,y=np.shape(predictions)
    #
    # for batch in range(0,x):
    #     # print('predictions',predictions[batch, labels[batch]])
    #     batch_log=torch.log2(predictions[batch, labels[batch]] + eps)
    #     batch_log=-batch_log
    #     batch_loss=batch_loss+batch_log#torch.sum(batch_loss,batch_log )
    #     # print(batch_loss,"**", labels[batch])
    # # print('&&',batch_loss)


    return  batch_loss

params = {}


class my_nn:
    def __init__(self, layers_dim, layers_activation='Sigmoid', device='cpu'):
        """ Initialize network
        Args:
            layers_dims (List of ints): list of Size of each layer of the network
                                        [inputs,layer1,...,outputs]
            layers_activation (List of strings): list of activation function for each hidden layer
                                        of the network[layer1,...,outputs]
            device (str): a device that will be used for computation
                Default: 'cpu'

        """
        self.layers_activation = layers_activation
        self.params = {}
        self.num_layers = len(layers_dim) - 1
        self.layers_dim = layers_dim
        self.device = device
        self.init_weights()

    def init_weights(self):
        """ Initialize weights and biases of network based on layers dimension.
            Store weights and biases in self.params.
            weights and biases key should be of format "W#" and "b#" where # is the layer number.
            Example: for layer 1, weight and bias key is "W1" and "b1"
        Args:
            None

        Returns:
            None
        """
        ### TODO: Initialize weights and bias of network
        ### TODO: Store weights and biases in self.params
        for i in range(0, self.num_layers):
            weight_name = 'W' + str(i)
            bias_name = 'b' + str(i)
            self.params[weight_name] = torch.normal(mean=0,std=1,size=(self.layers_dim[i], self.layers_dim[i + 1]),dtype=torch.float64)
            self.params[bias_name] = torch.zeros(self.layers_dim[i + 1],dtype=torch.float64)#torch.normal(mean=0,std=1,size=(self.layers_dim[i + 1],1))

        ### HINT: Remember to set require_grad to True
        ### HINT: Remember to put tensors of target device
        for name, param in self.params.items():
            self.params[name] = param.to(device)
            self.params[name].requires_grad = True
            param.requires_grad = True
        return self.params

    def forward(self, x):
        """ Perform forward pass
        Args:
            x (torch.tensor): tensor of shape of (B, C, H, W)

        Return:
            torch.tensor: tensor of shape of (B, N_classes)
        """
        ### TODO: Fill out this function
        B, C, H, W = np.shape(x)
        data = x.view(-1, H * W * C)
        data=data.to(torch.float64)
        for i in range(0, self.num_layers):
            weight_name = 'W' + str(i)
            bias_name = 'b' + str(i)
            if i == self.num_layers-1:
                data = data @ self.params[weight_name] + self.params[bias_name]
            elif self.layers_activation == 'Sigmoid':
                data = Sigmoid(data @ self.params[weight_name] + self.params[bias_name])# ) self.layers_activation(
            elif self.layers_activation == 'ReLU':
                data = ReLU(data @ self.params[weight_name] + self.params[bias_name])
            else:
                print('define activation')


        return data

    def grad_true(self):
        for name, param in self.params.items():
            self.params[name] = param.to(device)
            self.params[name].requires_grad = True
            param.requires_grad = True


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
        one_hot_label = nn.functional.one_hot(labels, num_classes=10)
        loss = CE_loss(predictions, labels)
        # loss_func = nn.CrossEntropyLoss()
        # loss = loss_func(predictions, labels)
        # print(loss,'***')

        ### TODO: Calculate gradients and update weights and biases
        loss.backward()
        optimizer.step()

        if i % 20:
            accuracy = 0
            with torch.no_grad():
                loss_tracker.append(loss.item())
                ### TODO: calculate accuracy of mini_batch
                for j in range(0, np.shape(predictions)[0]):
                    # print(torch.argmax(predictions[i,:]),torch.argmax(one_hot_label[i,:]))
                    if torch.argmax(predictions[j,:])==torch.argmax(one_hot_label[j,:]):
                        accuracy =accuracy + 1

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
        ### TODO: Put data and label on target device
        inputs = data.to(device=device)
        labels = label.to(device=device)

        with torch.no_grad():
            ### TODO: Pass data to the model
            predictions = model.forward(inputs)

            ### TODO: Calculate the loss of predicted labels vs ground truth labels
            one_hot_label = nn.functional.one_hot(labels, num_classes=10)
            # loss = CE_loss(predictions, labels)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(predictions, labels)

            ### TODO: calculate accuracy of mini_batch
            accuracy = 0

            for j in range(0, np.shape(predictions)[0]):
                if torch.argmax(predictions[j, :]) == torch.argmax(one_hot_label[j, :]):
                    accuracy = accuracy + 1


        loss_tracker.append(loss.item())
        accuracy_tracker.append(100*accuracy / data.size(0))

    return sum(loss_tracker) / len(loss_tracker), sum(accuracy_tracker) / len(accuracy_tracker)


# Training hyper parameters
epochs =10
learning_rate = 0.001
layers_dim = [28*28,512,512,10]

### TODO: Set target device for computations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')


### TODO: Initialize model using layers_dim
model = my_nn(layers_dim, device=device)
### TODO: Initialize Adam optimizer
optimizer = Adam(model.params.values(), lr=learning_rate)

train_loss_tracker = []
train_accuracy_tracker = []

test_loss_tracker = []
test_accuracy_tracker = []

for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train_loss,train_accuracy = Train(model,optimizer,kmnist_train_loader,device)
    test_loss , test_accuracy = Test(model,kmnist_test_loader,device)
    train_loss_tracker.extend(train_loss)
    train_accuracy_tracker.extend(train_accuracy)
    test_loss_tracker.append(test_loss)
    test_accuracy_tracker.append(test_accuracy)
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

