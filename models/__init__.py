# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""


from .erpenet import ERPENet
from .eegnet import EEGNet, SamePadding
from .cnn import ConvNet


__all__ = ["ERPENet",
           "EEGNet",
           "SamePadding",
           "ConvNet"
           ]

