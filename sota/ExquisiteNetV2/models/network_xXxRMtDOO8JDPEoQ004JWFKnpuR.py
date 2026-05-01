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
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ChainedScheduler

def get_optimizer(model, lr, weight_decay=0, nesterov=True, warmup_epochs=5, total_epochs=100):
    if weight_decay != 0:
        g0, g1, g2 = [], [], []  
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  
                g2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):  
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  
                g1.append(v.weight)
        
        opt = optim.SGD(g0, lr, momentum=0.9, nesterov=nesterov)
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})  
        opt.add_param_group({'params': g2})  
    else:
        opt = optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=nesterov)
    
    scheduler = LambdaLR(opt, lambda epoch: min(epoch / warmup_epochs, 1))
    full_scheduler = ChainedScheduler([
        ('warmup', scheduler),
        ('cosine', CosineAnnealingLR(opt, T_max=total_epochs - warmup_epochs))
    ])
    return opt, full_scheduler
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
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Helper Factories & Modules ===

class SE_LN(nn.Module):
    def __init__(self, cin, use_gate=False):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))  
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Parameter(torch.tensor(1.0))  # Learnable gate
    
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = x.view(-1, x.size(1), 1, 1)
        if self.use_gate:
            x = self.gate * x  # Apply learnable gate
        return x * y

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ActivationStack(nn.Module):
    def __init__(self, acts=['relu', 'swish']):
        super().__init__()
        self.acts = nn.ModuleList([getattr(F, act) for act in acts])
    
    def forward(self, x):
        for act in self.acts:
            x = act(x)
        return x

class AdaptivePooling(nn.Module):
    def __init__(self, mode='avg'):
        super().__init__()
        self.mode = mode
    
    def forward(self, x):
        if self.mode == 'avg':
            return F.avg_pool2d(x, x.size()[2:])
        elif self.mode == 'max':
            return F.max_pool2d(x, x.size()[2:])
        else:
            raise ValueError("Invalid pooling mode")

# === Modular Stage Builder ===

class ModularStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, 
                 use_se=True, use_activation_stack=False, pooling_mode='avg'):
        super().__init__()
        self.blocks = nn.ModuleList([DepthwiseSeparableBlock(in_channels if i==0 else out_channels, out_channels) for i in range(num_blocks)])
        self.se = SE_LN(out_channels, use_gate=True) if use_se else None
        self.activation_stack = ActivationStack() if use_activation_stack else None
        self.pooling = AdaptivePooling(pooling_mode)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            if self.se:
                x = self.se(x)
            if self.activation_stack:
                x = self.activation_stack(x)
        x = self.pooling(x)
        return x

# === Full Model ===

class EfficientCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ModularStage(3, 16, 2, use_se=True, use_activation_stack=True, pooling_mode='avg')
        self.stage2 = ModularStage(16, 32, 3, use_se=True, pooling_mode='max')
        self.stage3 = ModularStage(32, 64, 4, use_activation_stack=True)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
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