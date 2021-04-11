# -*- coding utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

# A simple AlexNet architecture as baseline model
class BasicCNN(nn.Module):
    def __init__(self, n_classes):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, 4, padding=2)
        self.conv2 = nn.Conv2d(48, 128, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(128, 192, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(192, 128, 3, 1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(6 * 6 * 128, 2048, bias=True)
        self.fc2 = nn.Linear(2048, 2048, bias=True)
        self.fc3 = nn.Linear(2048, n_classes, bias=True)
        
        self.dropout = nn.Dropout(0.5)
        
    # forward behavior
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 6 * 6 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = sigmoid(x)
        
        return x


# Dipper AlexNet architecture
class BasicCNN_v1(nn.Module):
    def __init__(self, n_classes):
        super(BasicCNN_v1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, 4, padding=2)
        self.conv2 = nn.Conv2d(64, 156, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(156, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(256, 192, 3, 1, padding=1)

        self.pool = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Linear(6 * 6 * 192, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 2048, bias=True)
        self.fc3 = nn.Linear(2048, n_classes, bias=True)

        self.dropout = nn.Dropout(0.5)

    # forward behavior
    def forward(self, x):
        # print(f"x: {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"x after conv1: {x.shape}")
        x = self.pool(x)
        # print(f"x after max pooling: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"x after conv2: {x.shape}")
        x = self.pool(x)
        # print(f"x after max pooling: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"x after conv3: {x.shape}")
        x = F.relu(self.conv4(x))
        # print(f"x after conv4: {x.shape}")
        x = F.relu(self.conv5(x))
        # print(f"x after conv5: {x.shape}")
        x = self.pool(x)
        # print(f"x after max pool: {x.shape}")
        x = x.view(-1, 6 * 6 * 192)
        # print(f"x after resize: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"x after fc1: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # print(f"x after fc2: {x.shape}")
        x = self.dropout(x)
        x = self.fc3(x)
        # print(f"x after fc3: {x.shape}")
        x = sigmoid(x)

        return x


# Dipper AlexNet architecture, such as in torchvision's implementation
class BasicCNN_v2(nn.Module):
    def __init__(self, n_classes):
        super(BasicCNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, 4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.pool = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Linear(6 * 6 * 256, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, n_classes, bias=True)

        self.dropout = nn.Dropout(0.5)

    # forward behavior
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x