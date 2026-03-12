import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES

class CNN(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class LeNet5(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(LeNet5, self).__init__()

        fc1_in_size = 16*14*14 if in_channels == 3 else 16*5*5
        # First convolutional layer: 1 or 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        # First pooling layer
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Second pooling layer
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_in_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        # Input: 32x32
        x = self.pool1(F.relu(self.conv1(x)))  # Conv -> ReLU -> Pooling
        x = self.pool2(F.relu(self.conv2(x)))  # Conv -> ReLU -> Pooling
        x = x.view(x.size(0), -1)   # Flatten
        x = F.relu(self.fc1(x))     # First fully connected layer
        x = F.relu(self.fc2(x))     # Second fully connected layer
        x = self.fc3(x)             # Output layer

        return x
