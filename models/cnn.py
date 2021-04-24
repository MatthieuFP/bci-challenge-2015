# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""

import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, F1=8, F2=16, C=56, dropout_rate=.25):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 128))
        self.BN1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(C, 1), bias=False)
        self.BN2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(512, 1)

    def forward(self, inp):

        h1 = self.BN1(self.conv1(inp.unsqueeze(1)))
        h2 = self.BN2(self.conv2(h1))
        h3 = self.avgpool(self.elu(h2))
        h4 = self.dropout(h3)

        return torch.sigmoid(self.classifier(self.flatten(h4)))

