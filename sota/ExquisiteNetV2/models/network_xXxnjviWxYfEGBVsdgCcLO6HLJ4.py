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
import torch
import torch.nn as nn
import torch.nn.functional as F

class SA_DynCS(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))  
        self.conv = nn.Conv2d(cin, cin, kernel_size=1, bias=True) # Lightweight Conv
        self.bn = nn.BatchNorm2d(cin)
        self.sig_temp = nn.Parameter(torch.tensor(1.0)) # Learnable Temperature
        self.act = lambda x: torch.sigmoid(x / self.sig_temp)
        
    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = self.conv(x) # Reintroduce spatial awareness
        x = self.bn(x)
        x = self.act(x)
        return x * y

class EConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw_conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1, groups=cin, stride=stride)
        self.pw_conv = nn.Conv2d(cout, cout, kernel_size=1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = Swish()
        self.sa_dyn_cs = SA_DynCS(cout) if cout > 1 else nn.Identity()
        self.spatial_scale = nn.Conv2d(cout, cout, kernel_size=1, bias=False) if stride > 1 else nn.Identity()
        
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.sa_dyn_cs(x)
        x = self.spatial_scale(x) # Adaptive spatial scaling
        return x

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, c1, c2, c3, cout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c1 + c2 + c3, cout),
            nn.ReLU(),
            nn.Linear(cout, 3), # Output weights for fusion
            nn.Softmax(dim=1)
        )
        
    def forward(self, f1, f2, f3):
        concat = torch.cat([f1.mean((2,3)), f2.mean((2,3)), f3.mean((2,3))], dim=1)
        weights = self.mlp(concat)
        return weights[0]*f1 + weights[1]*f2 + weights[2]*f3

class ECAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            Swish()
        )
        self.blocks = nn.Sequential(
            EConvBlock(16, 24, stride=2), # 16->24 C, 32->16 H/W
            EConvBlock(24, 24),
            EConvBlock(24, 40, stride=2), # 24->40 C, 16->8 H/W
            EConvBlock(40, 40),
            EConvBlock(40, 80, stride=2), # 40->80 C, 8->4 H/W
            EConvBlock(80, 80)
        )
        self.hff = HierarchicalFeatureFusion(24, 40, 80, 128) # Example dims, adjust based on actual block outs
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10) # Adjust input based on HFF output
        )
        
    def forward(self, x):
        x = self.stem(x)
        f1 = self.blocks[0](x) # First block's output
        f2 = self.blocks[2](self.blocks[1](f1)) # Third block's output (assuming sequence)
        f3 = self.blocks[-1](self.blocks[-2](self.blocks[-3](f2))) # Last block's output
        fused = self.hff(f1, f2, f3)
        return self.head(fused)

# Example Usage (Training Loop Snippet)
if __name__ == "__main__":
    model = ECAN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy Data
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")
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

class SE_LN(nn.Module):
    def __init__(self, num_features):
        super(SE_LN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_features, bias=True)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid(self.fc(y))
        y = y.view(b, c, 1, 1)
        return x * y

class SE(nn.Module):
    def __init__(self, num_features, reduction):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.SiLU()
        
        self.block1 = DFSEBV2(16, 3, True)
        self.block2 = DFSEBV2(16, 3, False)
        self.block3 = DFSEBV2(16, 3, True)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.Hardswish()
        
        self.block4 = DFSEBV2(32, 3, False)
        self.block5 = DFSEBV2(32, 3, True)
        self.block6 = DFSEBV2(32, 3, False)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.SiLU()
        
        self.block7 = DFSEBV2(64, 3, True)
        self.block8 = DFSEBV2(64, 3, False)
        self.block9 = DFSEBV2(64, 3, True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(64, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            DW(16, 3),
            nn.MaxPool2d(2, 2)
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 2, 1),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
            DW(24, 3),
            nn.MaxPool2d(2, 2)
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(24, 40, 3, 2, 1),
            nn.BatchNorm2d(40),
            nn.Hardswish(),
            DW(40, 3),
            nn.AvgPool2d(2, 2)
        )
        
        self.classifier = nn.Linear(40*4*4, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(-1, 40*4*4)
        x = self.classifier(x)
        return x

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