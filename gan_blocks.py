import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pickle
import itertools
import math

class LinearSFT(nn.Module):
    def __init__(self, input_chn, out_chn, dropOut=False, with_condition=False) -> None:
        super().__init__()
        self.dropOut = dropOut
        self.with_condition = with_condition


        self.linear = torch.nn.Linear(input_chn, out_chn)
        self.lrelu = torch.nn.LeakyReLU()

        if self.dropOut:
            self.dropoutModule = torch.nn.Dropout(0.3)
        
        label_condition_emb = [torch.nn.Linear(input_chn, out_chn),
                              torch.nn.LeakyReLU()]
        self.label_condition_emb = torch.nn.Sequential(*label_condition_emb)


    def forward(self, x):
        if self.with_condition:
            x, label_fea = x[0], x[1]
            label_fea = self.label_condition_emb(label_fea)        
            img_fea = self.linear(x) * label_fea
            img_fea = self.lrelu(img_fea)
            if self.dropOut:
                img_fea = self.dropoutModule(img_fea)
            return [img_fea, label_fea]
        else:
            x = self.lrelu(self.linear(x))
            if self.dropOut:
                x = self.dropoutModule(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self, init_chn_num=784, channels = [512, 256, 128], with_condition=False):
        super().__init__()
        self.with_condition = with_condition
        self.model = None

        init_chn_num = 784

        label_linear = [torch.nn.Linear(10, 784),
                          torch.nn.LeakyReLU()]


        linear_list = [LinearSFT(init_chn_num, channels[0], dropOut=True, with_condition=self.with_condition)]
        
        for i in range(len(channels)-1):
            in_ch_num, out_ch_num = channels[i], channels[i+1]
            linear_list.append(LinearSFT(in_ch_num, out_ch_num, dropOut=True, with_condition=self.with_condition))
        

        linear_logits = [torch.nn.Linear(channels[-1], 1),
                              torch.nn.Sigmoid()]

        self.label_linear = torch.nn.Sequential(*label_linear)
        self.linearSFT_blocks = torch.nn.Sequential(*linear_list)
        self.linear_logits = torch.nn.Sequential(*linear_logits)

            
    def forward(self, x, label=None):
        x = x.view(x.size(0), 784)
        
        if self.with_condition:
            assert label is not None
            # label = torch.nn.functional.one_hot(label, num_classes=10)
            # label = label.float()
            label_fea = self.label_linear(label)
            x, label_fea = self.linearSFT_blocks([x, label_fea])

        else:
            label_fea = label
            x = self.linearSFT_blocks(x)
        out = self.linear_logits(x)
        return out

class Generator(nn.Module):
    def __init__(self, dim_z=100, channels = [128, 256, 512], with_condition=False):
        super().__init__()
        self.dim_z = dim_z
        self.with_condition = with_condition

        init_chn_num = self.dim_z

        label_linear = [torch.nn.Linear(10, self.dim_z),
                        torch.nn.LeakyReLU()]


        linear_list = [LinearSFT(init_chn_num, channels[0], dropOut=False, with_condition=self.with_condition)]
        
        for i in range(len(channels)-1):
            in_ch_num, out_ch_num = channels[i], channels[i+1]
            linear_list.append(LinearSFT(in_ch_num, out_ch_num, dropOut=False, with_condition=self.with_condition))
        

        linear_out = [torch.nn.Linear(channels[-1], 28*28)]

        self.label_linear = torch.nn.Sequential(*label_linear)
        self.linearSFT_blocks = torch.nn.Sequential(*linear_list)
        self.linear_out = torch.nn.Sequential(*linear_out)


    def forward(self, x, label=None):
        x = x.view(x.size(0), self.dim_z)
        
        if self.with_condition:
            assert label is not None
            # label = torch.nn.functional.one_hot(label, num_classes=10)
            # label = label.float()
            label_fea = self.label_linear(label)
            x, label_fea = self.linearSFT_blocks([x, label_fea])
        else:
            label_fea = label
            x = self.linearSFT_blocks(x)
        out = self.linear_out(x)
        return out