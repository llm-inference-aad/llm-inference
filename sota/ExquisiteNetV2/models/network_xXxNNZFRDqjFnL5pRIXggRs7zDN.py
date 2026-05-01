
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:
# ========== End:
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
import torch
import torch.nn as nn

class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = torch.max_pool2d(out, 2, 2)
        out = torch.relu(self.bn3(self.conv3(out)))
        out = torch.relu(self.bn4(self.conv4(out)))
        out = torch.max_pool2d(out, 2, 2)
        out = torch.relu(self.bn5(self.conv5(out)))
        out = torch.relu(self.bn6(self.conv6(out)))
        out = torch.max_pool2d(out, 2, 2)
        out = out.view(-1, 256 * 4 * 4)
        out = torch.relu(self.bn7(self.fc1(out)))
        out = self.fc2(out)
        return out

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
        
        opt = torch.optim.SGD(g0, lr, momentum=0.9, nesterov=nesterov)
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        opt.add_param_group({'params': g2})  # add g2 (biases)
    else:
        opt = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=nesterov)
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
import torch
import torch.nn as nn

def pad_num_x(k_s):
    """Calculates padding size for a given kernel size."""
    pad_per_side = int((k_s - 1) * 0.5)
    return pad_per_side

class SE_LN(nn.Module):
    """Squeeze-and-Excitation Layer with Layer Normalization."""
    def __init__(self, num_features):
        super(SE_LN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_features, bias=True)
        
    def forward(self, x):
        """Forward pass through SE_LN."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid(self.fc(y))
        y = y.view(b, c, 1, 1)
        return x * y

class SE(nn.Module):
    """Generic Squeeze-and-Excitation Layer."""
    def __init__(self, num_features, reduction):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=True),
            nn.GELU(),  # Updated activation
            nn.Linear(num_features // reduction, num_features, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass through generic SE layer."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DFSEBV2_Updated(nn.Module):
    """Updated Depthwise Separable Convolution Block with SE and modern activations."""
    def __init__(self, cin, dw_s, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.SiLU()  # Consistent modern activation
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)
        # Choose between SE_LN and generic SE based on is_LN flag
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin, 3)
            
        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()  # Retained for diversity
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num_x(dw_s), groups=cin)
        self.act_final = nn.GELU()  # Additional non-linearity at the end

    def forward(self, x):
        """Forward pass through updated DFSEBV2 block."""
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x += y  # Residual connection
        
        x = self.pw2(x)       
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x += y  # Additional residual connection
        x = self.act_final(x)  # Final activation for enhanced non-linearity
        return x 

class CIFAR10Model_Updated(nn.Module):
    """Updated Example Model for CIFAR10 Dataset with deeper feature extraction."""
    def __init__(self):
        super(CIFAR10Model_Updated, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.SiLU()
        
        # Deeper stack with the updated block
        self.blocks = nn.ModuleList([DFSEBV2_Updated(16, 3, True if i%2 ==0 else False) for i in range(6)])
        
        self.transition = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_transition = nn.BatchNorm2d(32)
        self.act_transition = nn.Hardswish()
        
        self.blocks_32 = nn.ModuleList([DFSEBV2_Updated(32, 3, True if i%2 ==0 else False) for i in range(6)])
        
        self.transition_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_transition_2 = nn.BatchNorm2d(64)
        self.act_transition_2 = nn.SiLU()
        
        self.blocks_64 = nn.ModuleList([DFSEBV2_Updated(64, 3, True if i%2 ==0 else False) for i in range(6)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 10, bias=True)

    def forward(self, x):
        """Forward pass through the updated CIFAR10Model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.transition(x)
        x = self.bn_transition(x)
        x = self.act_transition(x)
        
        for block in self.blocks_32:
            x = block(x)
        
        x = self.transition_2(x)
        x = self.bn_transition_2(x)
        x = self.act_transition_2(x)
        
        for block in self.blocks_64:
            x = block(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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