
# ========== Start: GeneCrossed

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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)  # Adjusted momentum
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256, momentum=0.1)
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.dropout = nn.Dropout(p=0.2)  # Added Dropout
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)
        return x

def get_optimizer(model, lr, weight_decay=0.01):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def train(model, device, loader, optimizer, epoch, scheduler=None):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Example with CosineAnnealingLR
    optimizer = get_optimizer(model, lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):  # Train for 10 epochs as example
        train(model, device, torch.utils.data.DataLoader(torch.randn(100, 3, 32, 32), batch_size=32), optimizer, epoch, scheduler)
    
    # Alternative with OneCycleLR (uncomment to use)
    # optimizer = get_optimizer(model, lr=1.0)  # Higher initial LR for OneCycleLR
    # scheduler = OneCycleLR(optimizer, max_lr=1.0, total_steps=10*len(torch.utils.data.DataLoader(torch.randn(100, 3, 32, 32), batch_size=32)))
    # for epoch in range(10):
    #     train(model, device, torch.utils.data.DataLayer(torch.randn(100, 3, 32, 32), batch_size=32), optimizer, epoch, scheduler)

if __name__ == "__main__":
    main()
# --OPTION--
import torch
import torch.nn as nn

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
        return x * y 

class ResNetBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = nn.ReLU()
        self.se_ln = SE_LN(cout)
        if cin != cout or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride),
                nn.BatchNorm2d(cout)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se_ln(out)  
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.act2(out)
        return out

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.block1 = ResNetBlock(64, 64, stride=1)
        self.block2 = ResNetBlock(64, 64, stride=1)
        self.block3 = ResNetBlock(64, 128, stride=2)
        self.block4 = ResNetBlock(128, 128, stride=1)
        self.block5 = ResNetBlock(128, 256, stride=2)
        self.block6 = ResNetBlock(256, 256, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# --OPTION--
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ActivationStack(nn.Module):
    def __init__(self, acts=['relu', 'swish']):
        super().__init__()
        self.acts = nn.ModuleList([getattr(F, act) for act in acts])
    
    def forward(self, x):
        for act in self.acts:
            x = act(x)
        return x

class ModularResNetBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, use_se=True, use_activation_stack=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se_ln = SE_LN(cout, use_gate=True) if use_se else None
        self.activation_stack = ActivationStack() if use_activation_stack else None
        if cin != cout or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride),
                nn.BatchNorm2d(cout)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se_ln:
            out = self.se_ln(out)
        if self.activation_stack:
            out = self.activation_stack(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        return out  # Removed final ReLU for direct comparison; consider adding based on model needs

class EnhancedCIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # Simplified to directly compare; in practice, configure based on performance needs
        self.block1 = ModularResNetBlock(64, 64, stride=1, use_se=True, use_activation_stack=True)
        self.block2 = ModularResNetBlock(64, 64, stride=1)
        self.block3 = ModularResNetBlock(64, 128, stride=2, use_se=True)
        self.block4 = ModularResNetBlock(128, 128, stride=1, use_activation_stack=True)
        self.block5 = ModularResNetBlock(128, 256, stride=2)
        self.block6 = ModularResNetBlock(256, 256, stride=1, use_se=True, use_activation_stack=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
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
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class EfficientCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = FCT(3, 16)
        self.stage2 = FCT(16, 32)
        self.stage3 = FCT(32, 64)
        self.stage4 = FCT(64, 128)
        self.fc = nn.Linear(128 * 2 * 2, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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