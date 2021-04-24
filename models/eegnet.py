# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from functools import reduce
from operator import __add__


class SamePadding(nn.Module):

    def __init__(self, kernel_size):
        super(SamePadding, self).__init__()

        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)

    def forward(self, inp):
        return self.pad(inp)


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EEGNet(nn.Module):

    def __init__(self, F1=8, F2=16, D=2, C=56, dropout_rate=.25):
        super(EEGNet, self).__init__()

        self.block1 = nn.Sequential(OrderedDict({
            "samePadding": SamePadding(kernel_size=(1, 64)),
            "conv1": nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 64), bias=False),
            "BN1": nn.BatchNorm2d(F1),
            "depthConv1": nn.Conv2d(in_channels=F1, out_channels=D * F1, kernel_size=(C, 1), groups=F1, bias=False),
            "BN2": nn.BatchNorm2d(D * F1),
            "elu": nn.ELU(),
            "AvgPool": nn.AvgPool2d(kernel_size=(1, 4)),
            "dropout": nn.Dropout(dropout_rate)
        }))

        self.block2 = nn.Sequential(OrderedDict({
            "samePadding": SamePadding(kernel_size=(1, 16)),
            "separableConv1": depthwise_separable_conv(in_channels=D * F1, out_channels=F2, kernel_size=(1, 16)),
            "BN3": nn.BatchNorm2d(F2),
            "elu": nn.ELU(),
            "AvgPool": nn.AvgPool2d(kernel_size=(1, 8)),
            "dropout": nn.Dropout(dropout_rate),
            "flatten": nn.Flatten()
        }))

        self.classifier = nn.Linear(8 * F2, 1)

    def forward(self, inp):

        h1 = self.block1(inp.unsqueeze(1))
        h2 = self.block2(h1)
        return torch.sigmoid(self.classifier(h2))

