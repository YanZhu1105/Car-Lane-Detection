#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[8]:


def conv3x3(in_planes, out_planes, stride=1, padding=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False)


# In[9]:


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)  # ？ why no bias: 如果卷积层之后是BN层，那么可以不用骗纸参数，可以节省内存


# In[10]:


class BasicBlock(nn.Module):
    expansion = 1  # 经过block之后channel的变化量

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果bn层没有自定义，就是用标准的bn层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.relu(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identtity = self.downsample(x)  # downsample调整x的维度，F(X)+X一致才能相加
        out += identtity
        out = self.relu(out)  # 先相加再激活

        return out


# In[12]:


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BottleBlock, self).__init__()
        assert inplanes == planes*self.expansion
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, padding=1, groups=32)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)  # 输出nnel数： planes*self.explansion
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# In[13]:


class ResNeXt(nn.Module):
    def __init__(self, name, block, layers, num_class=1000, norm_layer=None):
        super(ResNeXt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, 128, layers[0])
        self.inplanes = 256
        self.layer2 = self._make_layer(block, self.inplanes, 256, layers[1])
        self.inplanes = 512
        self.layer3 = self._make_layer(block, self.inplanes, 512, layers[2])
        self.inplanes = 1024
        self.layer4 = self._make_layer(block, self.inplanes, 1024, layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (1,1) 等于GAP
        self.fc = nn.Linear(512 * block.expansion, num_class)
        self.name = name

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        # 生成不同的stage/layer
        # block: block type(basic block/bottle block)
        # blocks: blocks的数量
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),  # 同时调整spatial和channel两个方向
                norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=norm_layer))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 使用sequential层组合blocks，行程stage。

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# In[ ]:


def resnext50():
    return ResNeXt('resnext50', BottleBlock, [3, 4, 6, 3])

# In[ ]:


# In[ ]:




