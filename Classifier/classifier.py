import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedEncoder(nn.Module):
    def __init__(self, input_dim=3):
        super(ModifiedEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 256, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  
        self.bn4 = nn.BatchNorm2d(32)

        self.skip1 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.skip2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.skip3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))
        x1_pool = F.max_pool2d(x1, kernel_size=2)

        x2 = F.leaky_relu(self.bn2(self.conv2(x1_pool)))
        x2_pool = F.max_pool2d(x2, kernel_size=2)

        x3 = F.leaky_relu(self.bn3(self.conv3(x2_pool))) 
        x3_pool = x3

        x4 = F.leaky_relu(self.bn4(self.conv4(x3_pool)))  

        skip3 = F.interpolate(self.skip3(x3_pool), size=x4.shape[2:], mode="bilinear", align_corners=False)
        skip2 = F.interpolate(self.skip2(x2_pool), size=x4.shape[2:], mode="bilinear", align_corners=False)
        skip1 = F.interpolate(self.skip1(x1_pool), size=x4.shape[2:], mode="bilinear", align_corners=False)

        out = x4 + skip3 + skip2 + skip1
        return out


class classifier(nn.Module):
    def __init__(self, input_dim = 3, num_classes = 43):
        super(classifier, self).__init__()

        self.encoder = ModifiedEncoder(input_dim=input_dim)

        self.fc1 = None
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self,x):

        features = self.encoder(x)

        if self.fc1 is None:
            flatten_dim = features.shape[1] * features.shape[2] * features.shape[3]
            self.fc1 = nn.Linear(flatten_dim, 128).to(features.device)

        flatten = torch.flatten(features, start_dim=1)

        x = F.relu(self.fc1(flatten))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x