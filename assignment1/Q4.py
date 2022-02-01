
from torchvision.datasets import CIFAR10
from random import randint
import matplotlib.pyplot as plt

ds = CIFAR10('~/.torch/data/', train=True, download=True)

myDict = {}
classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range (0,10):
  myDict[classes[i]]=[]

dog_idx, deer_idx = ds.class_to_idx['dog'], ds.class_to_idx['deer']
plane_idx, car_idx = ds.class_to_idx['airplane'], ds.class_to_idx['automobile']
bird_idx, cat_idx = ds.class_to_idx['bird'], ds.class_to_idx['cat']
frog_idx, horse_idx = ds.class_to_idx['frog'], ds.class_to_idx['horse']
ship_idx, truck_idx = ds.class_to_idx['ship'], ds.class_to_idx['truck']

for i in range(len(ds)):
  current_class = ds[i][1]
  if current_class == dog_idx:
    myDict['dog'].append(i)
  elif current_class == deer_idx:
    myDict['deer'].append(i)
  elif current_class == plane_idx:
    myDict['airplane'].append(i)
  elif current_class == car_idx:
    myDict['car'].append(i)
  elif current_class == bird_idx:
    myDict['bird'].append(i)
  elif current_class == cat_idx:
    myDict['cat'].append(i)
  elif current_class == frog_idx:
    myDict['frog'].append(i)
  elif current_class == horse_idx:
    myDict['horse'].append(i)
  elif current_class == ship_idx:
    myDict['ship'].append(i)
  elif current_class == truck_idx:
    myDict['truck'].append(i)


for i in range(0,10):
  value = randint(0, len(myDict[classes[i]]))
  pic=myDict[classes[i]][value]
  print(ds[pic][0])
  plt.imshow(ds[pic][0])
  plt.show()






