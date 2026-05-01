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
import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, cin, ratio=16):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(cin, int(cin/ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(int(cin/ratio), cin, bias=False)
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
        return x * y

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.se1 = SE(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.se2 = SE(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.se3 = SE(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.ReLU()
        self.se4 = SE(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        self.se5 = SE(256)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()
        self.se6 = SE(256)
        
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.se3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.se4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.se5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)
        x = self.se6(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# --OPTION--
import torch
import torch.nn as nn

class SE_LN(nn.Module):
    def __init__(self, cin, reduction_ratio=16):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))  
        self.use_ln = True
        if self.use_ln:
            self.ln = nn.LayerNorm(cin)
        self.fc1 = nn.Linear(cin, cin//reduction_ratio, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(cin//reduction_ratio, cin, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = x
        x = self.gavg(x).view(-1, x.size(1))
        if self.use_ln:
            x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x * y

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SE_LN(64),  
        )
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.GELU()
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            SE_LN(128),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.conv_block1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.conv_block2(x)
        x = self.avgpool(x).view(-1, 128)
        x = self.fc(x)
        return x
# --OPTION--
# -- NOTE --
# Note: The classes SE_LN and SE used in this architecture are pre-existing and fully implemented elsewhere. 
# It is not necessary to create new implementations or modify these classes for this architecture. They should be used as-is.# -- NOTE --
import torch
import torch.nn as nn

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
            self.seln = SE(cin, 3)
            
        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)
        
        self.rev_f = nn.Hardswish()
        self.rev_g = nn.Hardswish()

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        
        # Reversible block insertion
        c = x.shape[1]
        x1, x2 = torch.split(x, c // 2, dim=1)
        x1 = x1 + self.rev_f(x2)
        x2 = x2 + self.rev_g(x1)
        x = torch.cat([x1, x2], dim=1)
        
        x += y  # First residual connection
        
        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x += y  # Second residual connection
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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.pw = nn.Conv2d(2*cin, cout, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SELU()

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.SELU()
        
        self.eve1 = EVE(64, 64)
        self.eve2 = EVE(64, 64)
        self.eve3 = EVE(64, 128)
        self.eve4 = EVE(128, 128)
        self.eve5 = EVE(128, 256)
        self.eve6 = EVE(256, 256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.eve1(x)
        x = self.eve2(x)
        x = self.eve3(x)
        x = self.eve4(x)
        x = self.eve5(x)
        x = self.eve6(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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

# --OPTION--
# -- NOTE --
# Note: The classes FCT, EVE, ME, and DFSEBV2 used in this architecture are pre-existing and fully implemented elsewhere. 
# It is not necessary to create new implementations or modify these classes for this architecture. They should be used as-is.# -- NOTE --
import torch
import torch.nn as nn

class ExquisiteNetV2(nn.Module):
    def __init__(self, class_num=10, img_channels=3):
        super().__init__()
        self.FCT = FCT(img_channels, 24) # Increased channels for better feature extraction
        self.DFSEB1 = DFSEBV2(24, 3, True)
        self.EVE = EVE(24, 64)       # Adjusted dimensions for balance
        self.DFSEB2 = DFSEBV2(64, 3, True)
        self.ME3 = ME(64, 128)       # Progressive channel increase
        self.DFSEB3 = DFSEBV2(128, 3, True)
        self.ME4 = ME(128, 256)
        self.DFSEB4 = DFSEBV2(256, 3, True)
        self.ME5 = ME(256, 512)      # Final expansion before dense layers
        self.DFSEB5 = DFSEBV2(512, 3, True)
        self.DW = DW(512, 3)
        self.gavg = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(512, class_num) # Output adjusted for new final features
        
    def forward(self, x):
        x = self.FCT(x)
        x = self.DFSEB1(x)
        x = self.EVE(x)
        x = self.DFSEB2(x)
        x = self.ME3(x)
        x = self.DFSEB3(x)
        x = self.ME4(x)
        x = self.DFSEB4(x)
        x = self.ME5(x)
        x = self.DFSEB5(x)
        x = self.DW(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x
