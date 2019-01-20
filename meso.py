import torch
import torch.nn as nn


class MesoNet(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        conv_kern = 3
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.maxpool2 = nn.MaxPool2d(2,stride=1,padding=1)
        self.maxpool4 = nn.MaxPool2d(4,stride=2,padding=2)

        self.conv1 = nn.Conv2d(3, 8, 3,stride=2,padding=0)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8 ,5,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8,16,5,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16,16,5,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(1024,16)

        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(16,1)

    def forward(self, input):
        """
        Perform forward pass of Mesonet.

        """

        x = self.conv1(input)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool2(x)
        # print(x.shape)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        # print(x.shape)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.maxpool2(x)
        # print(x.shape)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.maxpool4(x)
        # print(x.shape)

        x = x.reshape(self.batch_size,-1)
        # print(x.shape)

        x = self.dropout1(x)
        x = self.fc1(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        x = self.sigmoid(x)
        print(x)

        return x



input = torch.rand(1,3,256,256)
net = MesoNet(1)
output = net.forward(input)