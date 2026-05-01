
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed

# ========== End:

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

class SE_LN(nn.Module):
    def __init__(self, cin):
        super(SE_LN, self).__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1,1))  
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()
        self.dwsc = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, groups=cin, padding=1, bias=False),  
            nn.Conv2d(cin, cin, kernel_size=1, bias=False)  
        )
        self.bn = nn.BatchNorm2d(cin)      
        
    def forward(self, x):
        x_dwsc = self.dwsc(x)
        x_dwsc = self.bn(x_dwsc)  
        x_dwsc = nn.ReLU()(x_dwsc)  
        x_avg = self.gavg(x)
        x_avg = x_avg.view(-1, x_avg.size(1))
        x_avg = self.ln(x_avg)
        x_avg = x_avg.view(-1, x_avg.size(1), 1, 1)
        return x_dwsc * self.act(x_avg)
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
    def __init__(self, cin, cout, dw_kernel_size=4, dw_stride=2, dw_padding=1, 
                 pool_ks=2, pw_kernel_size=1, use_bn=True, activation='relu'):
        """
        Flexible Channel Transformer with configurable depthwise convolution,
        pooling, and pointwise convolution parameters.
        
        :param cin: Input channels
        :param cout: Output channels
        :param dw_kernel_size: Depthwise convolution kernel size
        :param dw_stride: Depthwise convolution stride
        :param dw_padding: Depthwise convolution padding
        :param pool_ks: Pooling kernel size
        :param pw_kernel_size: Pointwise convolution kernel size
        :param use_bn: Whether to use batch normalization
        :param activation: Activation function ('relu', 'swish')
        """
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_kernel_size, dw_stride, dw_padding, groups=cin, bias=False)
        self.maxpool = nn.MaxPool2d(pool_ks, ceil_mode=True)
        self.minpool = MinPool2d_y(pool_ks, ceil_mode=True)
        self.pw = nn.Conv2d(3*cin, cout, pw_kernel_size, 1, bias=False)
        
        if use_bn:
            self.bn = nn.BatchNorm2d(cout)
        else:
            self.bn = None
            
        self.activation = activation.lower()
        if self.activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(x)
        elif self.activation == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        else:
            raise ValueError("Unsupported activation. Choose 'relu' or 'swish'.")
            
    def forward(self, x):
        # Enhanced feature extraction pathway
        x_max = self.maxpool(x)
        x_min = self.minpool(x)
        x_dw = self.dw(x)
        
        # Channel concatenation for diverse feature representation
        x_concat = torch.cat((x_max, x_min, x_dw), 1)
        
        # Pointwise convolution for channel reduction/transformation
        x_pw = self.pw(x_concat)
        
        # Optional batch normalization and activation
        if self.bn:
            x_pw = self.bn(x_pw)
        x_pw = self.act_fn(x_pw)
        
        return x_pw

# Example usage with a simple classifier on top for CIFAR-10
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fct_block = FCT(cin=3, cout=64, dw_kernel_size=3, dw_stride=1, dw_padding=1, 
                              pool_ks=2, use_bn=True, activation='swish')  # Adjusted for input layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # Assuming input size 32x32 reduces to 16x16 after FCT
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.fct_block(x)
        x = self.classifier(x)
        return x

# Initialization for demonstration
if __name__ == "__main__":
    model = CIFAR10Classifier()
    print(model)
    # Dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
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
        
        # Depthwise Separable Convolution
        self.dw = nn.Conv2d(2*cin, 2*cin, kernel_size=3, groups=2*cin, padding=1, bias=False)
        self.pw = nn.Conv2d(2*cin, cout, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()  

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.dw(x)  
        x = self.pw(x)   
        x = self.bn(x)
        x = self.act(x)  
        return x
# --OPTION--
import torch
import torch.nn as nn

class ME(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.AvgPool2d(2, ceil_mode=True) # Changed to AvgPool for smoother downsampling
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() # Added SiLU activation for better feature representation

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x) # Applied activation after BN for enhanced non-linearity
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()
        
        self.me1 = ME(64, 64) # Reduced channel growth for parameter efficiency
        self.me2 = ME(64, 128)
        self.me3 = ME(128, 256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.me1(x)
        x = self.me2(x)
        x = self.me3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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