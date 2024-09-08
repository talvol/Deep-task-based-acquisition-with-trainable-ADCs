import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, filters, kernel_size, first_layer=False):
        super(ResBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.first_layer = first_layer
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding='same')
        if first_layer:
            # Adding a 1x1 convolution for matching dimensions if it is the first layer of the network
            self.if_first = nn.Conv2d(in_channels=int(filters/2), out_channels=filters, kernel_size=1, padding='same')

    def forward(self, x):
        if self.first_layer:
            x = self.if_first(x)
        fx = self.conv1(x)
        fx = F.relu(fx, inplace=True)
        fx = self.conv2(fx)
        out = x + fx
        out = F.relu(out, inplace=True)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, height, width, input_channels=1, num_classes=512):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock(32, 3)
        self.resblock2 = ResBlock(32, 3)
        self.resblock3 = ResBlock(64, 3, first_layer=True)
        self.resblock4 = ResBlock(64, 3)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        output_height = ((height - 7 + 2 * 3) // 2 + 1) // 2  # Conv1 and pooling
        output_width = ((width - 7 + 2 * 3) // 2 + 1) // 2   # Conv1 and pooling
        flat_features = output_height * output_width * 64
        self.fc = nn.Linear(flat_features, num_classes)
        self.dropout1 = nn.Dropout(0.5)  # Dropout before the final fully connected layers
        self.fc2 = nn.Linear(num_classes, 18)

    def forward(self, x):
        # Datasize - Batchsize x Channel x Height x Width
        # Datasize - 32x1x102x62
        x = self.relu(self.conv1(x))
        # Datasize - 32x32x51x31
        x = self.resblock1(x)
        # Datasize - 32x32x51x31
        x = self.resblock2(x)
        # Datasize - 32x32x51x31
        x = self.resblock3(x)
        # Datasize - 32x64x51x31
        x = self.resblock4(x)
        # Datasize - 32x64x51x31
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = F.normalize(x, p=2, dim=1) #Tal Removed
        # Tal added
        x = self.dropout1(x)  # Apply Dropout before the last fully connected layer
        x = self.relu(x)
        x = self.fc2(x)
        return x
