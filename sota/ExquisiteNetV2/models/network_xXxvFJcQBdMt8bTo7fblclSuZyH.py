# --PROMPT LOG--

import collections
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import *
from torch.optim import SGD
# Adding some modules that llama3 seems to favor
import random
import math

# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
#from ranger import RangerQH  
#from ranger import RangerVA  
#from ranger import Ranger
#from ranger21 import Ranger21

# --OPTION--
def get_optimizer(model, lr, weight_decay=0, nesterov=True):  
    if weight_decay != 0:
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)
        
        opt = SGD(g0, lr, 0.9, nesterov=nesterov)
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        opt.add_param_group({'params': g2})  # add g2 (biases)
    else:
        opt = SGD(model.parameters(), lr, 0.9, nesterov=nesterov)
    return opt


# --OPTION--
class SE(nn.Module):
    def __init__(self, cin, ratio):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(cin, int(cin/ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, cin, bias=False)
        self.act2 = nn.Sigmoid()
        
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1,x.size()[1])
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = x.view(-1,x.size()[1],1,1)
        return x*y

# --OPTION--
class SE_LN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))  
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x*y

# --OPTION--

# -- NOTE --
# Note: The classes SE_LN and SE used in this architecture are pre-existing and fully implemented elsewhere. 
# It is not necessary to create new implementations or modify these classes for this architecture. They should be used as-is. 
# -- NOTE --

def pad_num_x(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side
    
class DFSEBV2(nn.Module):
    def __init__(self, cin, dw_s, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.SiLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin,3)
            
        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x += y
        
        x = self.pw2(x)       
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x += y
        return x
        
# --OPTION--
class MinPool2d_y(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)

class FCT(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 4, 2, 1, groups=cin, bias=False)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d_y(2, ceil_mode=True)
        self.pw = nn.Conv2d(3*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x),
            self.dw(x),
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x

# --OPTION--
class MinPool2d_x(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)

class EVE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d_x(2, ceil_mode=True)
        self.pw = nn.Conv2d(2*cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x

# --OPTION--
class ME(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

# --OPTION--
import torch
import torch.nn as nn

def pad_num_y(k_s):
    pad_per_side = int((k_s-1)*0.5)
    return pad_per_side

class DW(nn.Module):
    def __init__(self, cin, dw_s):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_s, 1, pad_num_y(dw_s), groups=cin)
        self.act = nn.Hardswish()
        
    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=pad_num_y(3))
        self.bn1 = nn.BatchNorm2d(16)
        self.hswish1 = nn.Hardswish()
        
        self.dw1 = DW(16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16 // 16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.hswish2 = nn.Hardswish()
        
        self.dw2 = DW(16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.hswish3 = nn.Hardswish()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=pad_num_y(3))
        self.bn4 = nn.BatchNorm2d(32)
        self.hswish4 = nn.Hardswish()
        
        self.dw3 = DW(32, 3)
        self.bn5 = nn.BatchNorm2d(32)
        self.se2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 // 16, 32, kernel_size=1),
            nn.Sigmoid()
        )
        self.hswish5 = nn.Hardswish()
        
        self.dw4 = DW(32, 3)
        self.bn6 = nn.BatchNorm2d(32)
        self.hswish6 = nn.Hardswish()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=pad_num_y(3))
        self.bn7 = nn.BatchNorm2d(64)
        self.hswish7 = nn.Hardswish()
        
        self.dw5 = DW(64, 3)
        self.bn8 = nn.BatchNorm2d(64)
        self.se3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64 // 16, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.hswish8 = nn.Hardswish()
        
        self.dw6 = DW(64, 3)
        self.bn9 = nn.BatchNorm2d(64)
        self.hswish9 = nn.Hardswish()
        
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hswish1(out)
        
        out = self.dw1(out)
        out = self.bn2(out)
        se_weight = self.se(out)
        out = out * se_weight
        out = self.hswish2(out)
        
        out = self.dw2(out)
        out = self.bn3(out)
        out = self.hswish3(out)
        
        out = self.conv2(out)
        out = self.bn4(out)
        out = self.hswish4(out)
        
        out = self.dw3(out)
        out = self.bn5(out)
        se_weight2 = self.se2(out)
        out = out * se_weight2
        out = self.hswish5(out)
        
        out = self.dw4(out)
        out = self.bn6(out)
        out = self.hswish6(out)
        
        out = self.conv3(out)
        out = self.bn7(out)
        out = self.hswish7(out)
        
        out = self.dw5(out)
        out = self.bn8(out)
        se_weight3 = self.se3(out)
        out = out * se_weight3
        out = self.hswish8(out)
        
        out = self.dw6(out)
        out = self.bn9(out)
        out = self.hswish9(out)
        
        out = self.avgpool(out)
        out = out.view(-1, 64 * 8 * 8)
        out = self.fc(out)
        return out
# --OPTION--

# -- NOTE --
# Note: The classes FCT, EVE, ME, and DFSEBV2 used in this architecture are pre-existing and fully implemented elsewhere. 
# It is not necessary to create new implementations or modify these classes for this architecture. They should be used as-is. 
# -- NOTE --

class ExquisiteNetV2(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.FCT = FCT(img_channels, 12)
        self.DFSEB1 = DFSEBV2(12, 3, True) #
        self.EVE = EVE(12, 48)  
        self.DFSEB2 = DFSEBV2(48, 3, True) #
        self.ME3 = ME(48, 96)  
        self.DFSEB3 = DFSEBV2(96, 3, True) #
        self.ME4 = ME(96, 192)  
        self.DFSEB4 = DFSEBV2(192, 3, True) #
        self.ME5 = ME(192, 384)  
        self.DFSEB5 = DFSEBV2(384, 3, True) #
        self.DW = DW(384, 3)                #
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(384, class_num)
                        
    def forward(self, x):
        x = self.FCT(x)
        x = self.DFSEB1(x) #
        x = self.EVE(x)  
        x = self.DFSEB2(x) #
        x = self.ME3(x)  
        x = self.DFSEB3(x) #
        x = self.ME4(x)  
        x = self.DFSEB4(x) #
        x = self.ME5(x)  
        x = self.DFSEB5(x) #
        x = self.DW(x)     #
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x