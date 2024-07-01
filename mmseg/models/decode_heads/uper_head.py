# import torch
# import torch.nn as nn
# from mmcv.cnn import ConvModule

# from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
# from .psp_head import PPM
# import torch.nn.functional as F


# @HEADS.register_module()
# class UPerHead(BaseDecodeHead):
#     """Unified Perceptual Parsing for Scene Understanding.

#     This head is the implementation of `UPerNet
#     <https://arxiv.org/abs/1807.10221>`_.

#     Args:
#         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
#             Module applied on the last feature. Default: (1, 2, 3, 6).
#     """

#     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
#         super(UPerHead, self).__init__(
#             input_transform='multiple_select', **kwargs)
#         # PSP Module
#         self.psp_modules = PPM(
#             pool_scales,
#             self.in_channels[-1],
#             self.channels,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg,
#             align_corners=self.align_corners)
#         self.bottleneck = ConvModule(
#             self.in_channels[-1] + len(pool_scales) * self.channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         # FPN Module
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
#         for in_channels in self.in_channels[:-1]:  # skip the top layer
#             l_conv = ConvModule(
#                 in_channels,
#                 self.channels,
#                 1,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg,
#                 inplace=False)
#             fpn_conv = ConvModule(
#                 self.channels,
#                 self.channels,
#                 3,
#                 padding=1,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg,
#                 inplace=False)
#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)

#         self.fpn_bottleneck = ConvModule(
#             len(self.in_channels) * self.channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)

#     def psp_forward(self, inputs):
#         """Forward function of PSP module."""
#         x = inputs[-1]
#         psp_outs = [x]
#         psp_outs.extend(self.psp_modules(x))
#         psp_outs = torch.cat(psp_outs, dim=1)
#         output = self.bottleneck(psp_outs)

#         return output

#     def forward(self, inputs):
#         """Forward function."""

#         inputs = self._transform_inputs(inputs)

#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         laterals.append(self.psp_forward(inputs))

#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             prev_shape = laterals[i - 1].shape[2:]
#             laterals[i - 1] += resize(
#                 laterals[i],
#                 size=prev_shape,
#                 mode='bicubic',
#                 align_corners=self.align_corners)

#         # build outputs
#         fpn_outs = [
#             self.fpn_convs[i](laterals[i])
#             for i in range(used_backbone_levels - 1)
#         ]
#         # append psp feature
#         fpn_outs.append(laterals[-1])

#         for i in range(used_backbone_levels - 1, 0, -1):
#             fpn_outs[i] = resize(
#                 fpn_outs[i],
#                 size=fpn_outs[0].shape[2:],
#                 mode='bicubic',
#                 align_corners=self.align_corners)
#         fpn_outs = torch.cat(fpn_outs, dim=1)
#         output = self.fpn_bottleneck(fpn_outs)
#         output = self.cls_seg(output)
#         return output

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bicubic', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out
    
@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    scales: Pooling scales used in PPM module applied on the last feature
    """
    def __init__(self, scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PPM Module
        self.ppm = PPM(self.in_channels[-1], self.channels, scales)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for in_ch in self.in_channels[:-1]: # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, self.channels, 1))
            self.fpn_out.append(ConvModule(self.channels, self.channels, 3, 1, 1))

        self.bottleneck = ConvModule(len(self.in_channels)*self.channels, self.channels, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, 1)


    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features)-1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(f, size=feature.shape[-2:], mode='bicubic', align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bicubic', align_corners=False)
 
        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output
