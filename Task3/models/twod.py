import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self, n_classes, **kwargs):
        """
        Define the layers of the model

        Args:
            n_classes (int): Number of classes in our classification problem
        """
        super(CNN, self).__init__()
        nb_filters = 16
        self.n_classes = n_classes
        self.conv2d_1 = nn.Conv2d(36,nb_filters,3) #36 input channels
        #nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2d_2 = nn.Conv2d(nb_filters, nb_filters*2, 3, padding=1)
        self.conv2d_3 = nn.Conv2d(nb_filters*2, nb_filters*4, 3, padding=1)

        self.linear_1 = nn.Linear(246016, 2048)
        self.linear_2 = nn.Linear(2048, 1024)
        self.linear_3 = nn.Linear(1024, n_classes)
        #nn.MaxPool2d(kernel_size)
        self.maxpool2d = nn.MaxPool2d(4, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, X, **kwargs):
        """
        Forward Propagation

        Args:
            X: batch of training examples with dimension (batch_size, 36, 1000, 1000) 
        """
        x1 = self.relu(self.conv2d_1(X))
        maxpool1 =  self.maxpool2d(x1)
        x2 = self.relu(self.conv2d_2(maxpool1))
        x3 = self.relu(self.conv2d_3(x2))
        maxpool2 = self.maxpool2d(x3)
        maxpool2 = self.dropout(maxpool2)
        maxpool2 = maxpool2.reshape(maxpool2.shape[0],-1) #flatten (batch_size,)
        x4 = self.dropout(self.relu(self.linear_1(maxpool2)))
        x5 = self.relu(self.linear_2(x4))
        x6 = self.linear_3(x5)
        return x6

# Defined a CNN based on the AlexNet model,
# based on: https://medium.com/analytics-vidhya/alexnet-a-simple-implementation-using-pytorch-30c14e8b6db2 (visited on April 27, 2022)
class AlexNet(nn.Module):
    def __init__(self, n_classes, **kwargs):
        """
        Define the layers of the model
        Args:
            n_classes (int): Number of classes in our classification problem
        """
        super(AlexNet, self).__init__()
        nb_filters = 8 #number of filters in the first layer
        self.n_classes = n_classes
        self.conv2d_1 = nn.Conv2d(36,nb_filters,11,stride=4) #36 input channels
        #nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2d_2 = nn.Conv2d(nb_filters, nb_filters*2, 5, padding=2)
        self.conv2d_3 = nn.Conv2d(nb_filters*2, nb_filters*4, 3, padding=1)
        self.conv2d_4 = nn.Conv2d(nb_filters*4, nb_filters*8, 3, padding=1)
        self.conv2d_5 = nn.Conv2d(nb_filters*8, 256, 3, padding=1)
        self.linear_1 = nn.Linear(9216, 4096)
        self.linear_2 = nn.Linear(4096, 2048)
        self.linear_3 = nn.Linear(2048, n_classes)
        #nn.MaxPool2d(kernel_size)
        self.maxpool2d = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)

    def forward(self, X, **kwargs):
        """
        Forward Propagation
        Args:
            X: batch of training examples with dimension (batch_size, 36, 256, 256) 
        """
        x1 = self.relu(self.conv2d_1(X))
        maxpool1 =  self.maxpool2d(x1)
        maxpool1 = self.dropout(maxpool1)
        x2 = self.relu(self.conv2d_2(maxpool1))
        maxpool2 = self.maxpool2d(x2)
        maxpool2 = self.dropout(maxpool2)
        x3 = self.relu(self.conv2d_3(maxpool2))
        x4 = self.relu(self.conv2d_4(x3))
        x5 = self.relu(self.conv2d_5(x4))
        x6 = self.maxpool2d(x5)
        x6 = self.dropout(x6)
        x6 = x6.reshape(x6.shape[0],-1) #flatten (batch_size,)
        x7 = self.relu(self.linear_1(x6))
        x8 = self.relu(self.linear_2(x7))
        x9 = self.linear_3(x8)
        return x9

#based on https://medium.com/@tioluwaniaremu/vgg-16-a-simple-implementation-using-pytorch-7850be4d14a1 (visited on May 22, 2022)
class VGG16(nn.Module):
    def __init__(self, n_classes, **kwargs):
        super(VGG16, self).__init__()
        self.n_classes = n_classes
        n_filters = 16
        self.conv1_1 = nn.Conv2d(36, n_filters, 3, padding=1)
        self.conv1_2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.conv2_1 = nn.Conv2d(n_filters, n_filters*2, 3, padding=1)
        self.conv2_2 = nn.Conv2d(n_filters*2, n_filters*2, 3, padding=1)
        self.conv3_1 = nn.Conv2d(n_filters*2, n_filters*4, 3, padding=1)
        self.conv3_2 = nn.Conv2d(n_filters*4, n_filters*4, 3, padding=1)
        self.conv3_3 = nn.Conv2d(n_filters*4, n_filters*4,3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(65536, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.dropout(F.relu(self.conv2_2(x)),0.3)
        x = self.maxpool(x)
        x = F.dropout(x,0.3)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.dropout(F.relu(self.conv3_3(x)),0.3)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

#code for ResNet adapted from:
#https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        return x


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()
      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))
      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=36):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=16)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=32, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=64, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128*ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=36):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
