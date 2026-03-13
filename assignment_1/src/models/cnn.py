import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES

class LeNet5(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(LeNet5, self).__init__()

        fc1_in_size = 16*14*14 if in_channels == 3 else 16 * 5 * 5
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

class CNN(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Dropout(dropout),
            nn.Linear(84, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MediumCNN(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super(MediumCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super(DeepCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), # handles any input size
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super(MultiScaleCNN, self).__init__()

        # Block 1
        self.b1_conv1 = nn.Conv2d(in_channels, 16, 3, padding=1) # Branch 1
        self.b1_conv2 = nn.Conv2d(in_channels, 16, 5, padding=2) # Branch 2
        self.b1_batchnorm = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        # Block 2
        self.b2_conv1 = nn.Conv2d(32, 32, 3, padding=1) # Branch 1
        self.b2_conv2 = nn.Conv2d(32, 32, 5, padding=2) # Branch 2
        self.b2_batchnorm = nn.BatchNorm2d(64)

        # Block 3
        self.b3_conv1 = nn.Conv2d(64, 64, 3, padding=1) # Branch 1
        self.b3_conv2 = nn.Conv2d(64, 64, 5, padding=2) # Branch 2
        self.b3_batchnorm = nn.BatchNorm2d(128)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 2 * 2, NUM_CLASSES)
        )

    def forward(self, x):
        # Block 1
        x1 = self.b1_conv1(x)
        x2 = self.b1_conv2(x)
        x = torch.cat([x1, x2], dim=1) # Merge both branches into one
        x = F.relu(self.b1_batchnorm(x))
        x = self.pool(x)

        # Block 2
        x1 = self.b2_conv1(x)
        x2 = self.b2_conv2(x)
        x = torch.cat([x1, x2], dim=1) # Merge both branches into one
        x = F.relu(self.b2_batchnorm(x))
        x = self.pool(x)

        # Block 3
        x1 = self.b3_conv1(x)
        x2 = self.b3_conv2(x)
        x = torch.cat([x1, x2], dim=1) # Merge both branches into one
        x = F.relu(self.b3_batchnorm(x))
        x = self.pool(x)

        x = self.classifier(x)
        return x