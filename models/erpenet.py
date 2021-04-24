# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from .eegnet import SamePadding


class EncoderBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, last: bool, dropout: float):
        super(EncoderBlock, self).__init__()

        self.last = last
        self.block = nn.Sequential(OrderedDict({
            "padding": SamePadding(kernel_size=(3, 3)),
            "conv1": nn.Conv2d(in_channels=in_channel, out_channels=out_channel * 2, kernel_size=(3, 3), stride=(2, 2)),
            "BN1": nn.BatchNorm2d(out_channel * 2),
            "leaky_relu1": nn.LeakyReLU(0.1),
            "dropout1": nn.Dropout(0.2),
            "padding2": SamePadding(kernel_size=(3, 3)),
            "conv2": nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=(3, 3)),
            "BN2": nn.BatchNorm2d(out_channel),
            "leaky_relu2": nn.LeakyReLU(0.1),
            "dropout2": nn.Dropout(0.2),
            "padding3": SamePadding(kernel_size=(3, 3)),
            "conv3": nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3)),
            "BN3": nn.BatchNorm2d(out_channel),
            "leaky_relu3": nn.LeakyReLU(0.1),
            "dropout3": nn.Dropout(0.2),
        }))

        if self.last:
            self.latent = nn.LSTM(input_size=96, hidden_size=512, batch_first=True, dropout=dropout, num_layers=1)

    def forward(self, inp, bs, t):
        if not self.last:
            inp = inp.view(bs * t, inp.shape[2], inp.shape[-2], inp.shape[-1])
            out = self.block(inp)
            return out
        else:
            out = self.block(inp)
            out = out.view(bs, t, -1)
            latent, _ = self.latent(out)
            return latent[:, -1]



class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        return torch.sigmoid(self.classifier(x))


class EncoderNetwork(nn.Module):

    def __init__(self, channel1: int, channel2: int, dropout: float):
        super(EncoderNetwork, self).__init__()

        self.block1 = EncoderBlock(in_channel=1,
                                   out_channel=channel1,
                                   last=False,
                                   dropout=dropout)

        self.block2 = EncoderBlock(in_channel=channel1,
                                   out_channel=channel2,
                                   last=True,
                                   dropout=dropout)

    def forward(self, inp, bs, t):

        h_1 = self.block1(inp, bs, t)
        h_2 = self.block2(h_1, bs, t)
        return h_2


class DecoderBlock(nn.Module):

    def __init__(self, first: bool, dropout: float, out_channel: int):
        super(DecoderBlock, self).__init__()

        self.first = first
        if self.first:
            self.lstm = nn.LSTM(input_size=512, hidden_size=96, batch_first=True, dropout=dropout, num_layers=1)

        self.block = nn.Sequential(OrderedDict({
            "upsample": nn.Upsample(scale_factor=2, mode='bilinear'),
            "zero_pad": nn.ZeroPad2d((0, 1, 0, 1)),
            "conv1": nn.Conv2d(in_channels=16, out_channels=out_channel * 2, kernel_size=(3, 3)),
            "BN1": nn.BatchNorm2d(num_features=out_channel * 2),
            "leaky1": nn.LeakyReLU(.1),
            "dropout1": nn.Dropout(.2),
            "same_pad": SamePadding((3, 3)),
            "conv2": nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=(3, 3)),
            "BN2": nn.BatchNorm2d(num_features=out_channel),
            "leaky2": nn.LeakyReLU(.1),
            "dropout2": nn.Dropout(.2),
            "same_pad2": SamePadding((3, 3)),
            "conv3": nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3)),
            "BN3": nn.BatchNorm2d(num_features=out_channel),
            "leaky3": nn.LeakyReLU(.1),
            "dropout3": nn.Dropout(.2)
        }))

        if not self.first:
            self.same_pad = SamePadding((1, 1))
            self.reconstruct = nn.Conv2d(in_channels=out_channel, out_channels=1, kernel_size=(1, 1))

    def forward(self, inp):

        if self.first:
            inp = inp.repeat(1, 256).reshape(-1, 256, inp.shape[-1])
            h1, _ = self.lstm(inp)
            h2 = self.block(h1.reshape(h1.size(0) * h1.size(1), 16, 2, 3))
            return h2
        else:
            h1 = self.block(inp)
            h1 = self.same_pad(h1)
            out = self.reconstruct(h1)
            return out


class DecoderNetwork(nn.Module):
    def __init__(self, dropout, chan1, chan2):
        super(DecoderNetwork, self).__init__()

        self.decoder1 = DecoderBlock(first=True, dropout=dropout, out_channel=chan1)
        self.decoder2 = DecoderBlock(first=False, dropout=dropout, out_channel=chan2)

    def forward(self, inp):

        h1 = self.decoder1(inp)
        out = self.decoder2(h1)
        return out


class ERPENet(nn.Module):
    def __init__(self, dropout, encoder_chan1=8, encoder_chan2=16, decoder_chan1=16, decoder_chan2=8):
        super(ERPENet, self).__init__()

        self.encoder = EncoderNetwork(channel1=encoder_chan1, channel2=encoder_chan2, dropout=dropout)
        self.decoder = DecoderNetwork(chan1=decoder_chan1, chan2=decoder_chan2, dropout=dropout)
        self.classifer = Classifier()

    def forward(self, inp, bs, t):

        h1 = self.encoder(inp, bs, t)
        reconstruct = self.decoder(h1).reshape(*inp.shape)
        out = self.classifer(h1)

        return out, reconstruct




