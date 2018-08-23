# -*- coding: utf-8 -*-
"""
    Training a classifier
    
    We will do the following steps in order:
    
    1. Load and normalizing the CIFAR10 training and test datasets using
    ``torchvision``
    2. Define a Convolution Neural Network
    3. Define a loss function
    4. Train the network on the training data
    5. Test the network on the test data
    
    1. Loading and normalizing CIFAR10
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    Using ``torchvision``, itâ€™s extremely easy to load CIFAR10.
    """
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].


transform = transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

#import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # convolution layer
        self.conv1 = nn.Conv2d(3, 96, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, 10, 1, stride=1, padding=0)
        
        # pooling layer
        self.pool = nn.AvgPool2d((6,6), stride=1, padding=0)
        # dropout
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
    
    
    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.drop2(x)
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.drop3(x)
        x = F.relu(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)
        x = x.view(-1,10)
        return x


net = Net()
net.to(device)


########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(100):  # loop over the dataset multiple times
    
    running_loss = 0.0
    count = 0;
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        count = count + 1;


print(running_loss / count)

print('Finished Training')


########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

# print images
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
    images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
                                                                   100 * correct / total))

########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
    images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
                                        classes[i], 100 * class_correct[i] / class_total[i]))
