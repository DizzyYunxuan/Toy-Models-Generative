import torch
import torch.nn as nn
import numpy as np
import yaml
import torch.nn.functional as F
from matplotlib import pyplot as plt


class MaskedConv2d(nn.Conv2d):
    def __init__(self, input_num_c, output_num_c, kernel_size=5, padding=2, type='A', device='cuda'):
        super().__init__(input_num_c, output_num_c, [kernel_size, kernel_size], stride=1, padding=padding)
        h_masked_index = kernel_size // 2
        w_masked_index = kernel_size // 2
        self.mask = torch.ones_like(self.weight)
        self.mask[:,:, h_masked_index+1:, :] = 0.0
        if type == 'A':
          self.mask[:,:, h_masked_index , w_masked_index:] = 0.0
        elif type == 'B':
          self.mask[:,:, h_masked_index , w_masked_index+1:] = 0.0
        self.mask = self.mask.to(device)


    def forward(self, x):
        with torch.no_grad():
          self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class residualMaskedConv(nn.Module):
    def __init__(self, input_num_dim, output_num_dim):
        super().__init__()

        net = []
        net.append(torch.nn.Conv2d(input_num_dim, input_num_dim // 2, 1))
        net.append(torch.nn.ReLU())
        net.append(MaskedConv2d(input_num_dim // 2, input_num_dim // 2, 3, padding=1, type='B'))
        net.append(torch.nn.ReLU())
        net.append(torch.nn.Conv2d(input_num_dim // 2, output_num_dim, 1))
        net.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*net)
    
    def forward(self, x):
        x = x + self.net(x)
        return x


class PixelCNN(nn.Module):
    ##################
    ### Problem 2(b): Implement PixelCNN
    def __init__(self, num_input_c=1, num_inner_c=64, num_output_c=1, num_masked_convs=4):
        super(PixelCNN, self).__init__()
        
        shallow_fea = []
        shallow_fea.append(MaskedConv2d(num_input_c, num_inner_c, 7, 3, 'A'))
        shallow_fea.append(torch.nn.LeakyReLU(0.1))


        net = []
        for i in range(num_masked_convs):
          net.append(residualMaskedConv(num_inner_c, num_inner_c))


        final_fea = []
        final_fea.append(MaskedConv2d(num_inner_c, num_inner_c, 1, 0, 'B'))
        final_fea.append(torch.nn.LeakyReLU(0.1))
        final_fea.append(MaskedConv2d(num_inner_c, num_output_c, 1, 0, 'B'))
        final_fea.append(torch.nn.Sigmoid())

        self.sf = torch.nn.Sequential(*shallow_fea)
        self.net = torch.nn.Sequential(*net)
        self.ff = torch.nn.Sequential(*final_fea)

    def forward(self, x):

        x = self.sf(x)
        x = x + self.net(x)
        x = self.ff(x)
        return x


class ConditionalMaskedConv2d(MaskedConv2d):
    ##################
    ### Problem 3(b): Implement ConditionalMaskedConv2d
    def __init__(self, input_num_c, output_num_c, kernel_size=5, padding=2, type='A', device='cuda'):
        super().__init__(input_num_c, output_num_c, kernel_size=kernel_size, padding=padding, type=type)


        self.h_func = torch.nn.Linear(10, 28 * 28).to(device)
        self.to(device)

    def forward(self, x, class_condition):
        vh = self.h_func(class_condition).view(-1, 1, 28, 28)
        return super(ConditionalMaskedConv2d, self).forward(x) + vh
    ##################


class residualConditionalMaskedConv(nn.Module):
    def __init__(self, input_num_dim, output_num_dim):
        super().__init__()

        net0 = []
        net0.append(torch.nn.Conv2d(input_num_dim, input_num_dim // 2, 1))
        net0.append(torch.nn.ReLU())
        
        self.conv0 = ConditionalMaskedConv2d(input_num_dim // 2, input_num_dim // 2, 3, padding=1, type='B')
        self.relu0 = torch.nn.ReLU()

        net2 = []
        net2.append(torch.nn.Conv2d(input_num_dim // 2, output_num_dim, 1))
        net2.append(torch.nn.ReLU())

        self.net0 = torch.nn.Sequential(*net0)
        # self.net1 = torch.nn.Sequential(*net1)
        self.net2 = torch.nn.Sequential(*net2)
    
    def forward(self, x, class_condition):
        x = x + self.net2(self.relu0(self.conv0(self.net0(x), class_condition)))
        return x
    


class ConditionalPixelCNN(nn.Module):
    ##################
    ### Problem 3(b): Implement ConditionalPixelCNN
    def __init__(self, num_input_c=3, num_inner_c=64, num_output_c=3, num_masked_convs=4):
        super(ConditionalPixelCNN, self).__init__()

        self.num_masked_convs = num_masked_convs
        
        self.shallow_fea = torch.nn.ModuleList([])
        self.shallow_fea.append(ConditionalMaskedConv2d(num_input_c, num_inner_c, 7, 3, 'A'))
        self.shallow_fea.append(torch.nn.LeakyReLU(0.1))


        self.resCMConvNet = torch.nn.ModuleList([])
        for i in range(num_masked_convs):
          self.resCMConvNet.append(residualConditionalMaskedConv(num_inner_c, num_inner_c))


        self.final_fea_conv0 = torch.nn.ModuleList([])
        self.final_fea_conv0.append(ConditionalMaskedConv2d(num_inner_c, num_inner_c, 1, 0, 'B'))
        self.final_fea_conv0.append(torch.nn.LeakyReLU(0.1))


        self.final_fea_conv1 = torch.nn.ModuleList([])
        self.final_fea_conv1.append(ConditionalMaskedConv2d(num_inner_c, 1, 1, 0, 'B'))
        self.final_fea_conv1.append(torch.nn.Sigmoid())
        

    def forward(self, x, class_condition):

        x = self.shallow_fea[0](x, class_condition)
        x = self.shallow_fea[1](x)
        x_sf = torch.clone(x)
        for i in range(self.num_masked_convs):
            x_sf = self.resCMConvNet[i](x_sf, class_condition)
        x = x + x_sf

        x = self.final_fea_conv0[0](x, class_condition)
        x = self.final_fea_conv0[1](x)
        x = self.final_fea_conv1[0](x, class_condition)
        x = self.final_fea_conv1[1](x)

        return x