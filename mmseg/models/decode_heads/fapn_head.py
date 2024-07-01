import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from mmcv.cnn import ConvModule
from .psp_head import PPM


from ..builder import HEADS
from .decode_head import BaseDecodeHead

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2,  g* 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)
    
    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bicubic', align_corners=False)
        
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


@HEADS.register_module()
class FaPNHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(FaPNHead, self).__init__(input_transform='multiple_select', **kwargs)
        
        in_channels = self.in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], self.channels, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, self.channels))
            self.output_convs.append(ConvModule(self.channels, self.channels, 3, 1, 1))

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def psp_forward(self, x):
        """Forward function of PSP module."""
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])
        
        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        out = self.conv_seg(self.dropout(out))
        return out