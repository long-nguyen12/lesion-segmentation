import torch
import torch.nn as nn
import torch.nn.functional as F
from .axial_atten import AA_kernel
from .cbam import CBAM
from .conv_layer import Conv

class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        
        self.aa_1 = AA_kernel(in_channels, out_channels)
        self.aa_2 = AA_kernel(out_channels, out_channels)
        self.cbam = CBAM(out_channels)
        self.conv_1 = Conv(2 * out_channels, out_channels, 3, 1, 1)
        self.conv_2 = Conv(2 * out_channels, out_channels, 5, 1, 2)
    
    def forward(self, i_high, i_low):
        i_high = F.interpolate(
            i_high, size=i_low.shape[2:], mode="bilinear", align_corners=False
        )
        
        i_high = self.aa_1(i_high)
        i_low = self.aa_2(i_low)
        
        i_cat = torch.cat([i_low, i_high], dim=1)
        i_3 = self.conv_1(i_cat)
        i_5 = self.conv_2(i_cat)
        
        i_total = i_high + i_low + i_3 + i_5
        
        return i_total
        
        
