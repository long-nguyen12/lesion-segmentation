import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
from torch import Tensor
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..segmentors.lib.cbam import CBAM


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        add_relu: bool = False,
        interpolate: bool = False,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=True)
        return x


class FPABlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPABlock, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # midddle branch
        # self.conv0 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)
        self.conv0 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)

        self.conv13_0 = nn.Conv2d(
            out_channels, out_channels, (1, 3), padding=(0, 1), groups=out_channels
        )
        self.conv13_1 = nn.Conv2d(
            out_channels, out_channels, (3, 1), padding=(1, 0), groups=out_channels
        )

        self.conv15_0 = nn.Conv2d(
            out_channels, out_channels, (1, 5), padding=(0, 2), groups=out_channels
        )
        self.conv15_1 = nn.Conv2d(
            out_channels, out_channels, (5, 1), padding=(2, 0), groups=out_channels
        )

        self.conv17_0 = nn.Conv2d(
            out_channels, out_channels, (1, 7), padding=(0, 3), groups=out_channels
        )
        self.conv17_1 = nn.Conv2d(
            out_channels, out_channels, (7, 1), padding=(3, 0), groups=out_channels
        )

        self.mixer = ConvBnRelu(out_channels, out_channels, 1)

    def forward(self, x):
        b1 = self.branch1(x)

        mid = self.conv0(x)

        c13 = self.conv13_0(mid)
        c13 = self.conv13_1(c13)

        c15 = self.conv15_0(mid)
        c15 = self.conv15_1(c15)

        c17 = self.conv17_0(mid)
        c17 = self.conv17_1(c17)

        att = c13 + c15 + c17
        att = self.mixer(att)

        x = torch.mul(mid, att)

        x = x + b1

        return x


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
        )


@HEADS.register_module()
class PANUPerHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(PANUPerHead, self).__init__(input_transform="multiple_select", **kwargs)
        # PPM Module
        self.fpa = FPABlock(
            in_channels=self.in_channels[-1], out_channels=self.channels
        )

        # FPN Module
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()

        for in_ch in self.in_channels[:-1]:  # skip the top layer
            self.fpn_in.append(ConvModule(in_ch, self.channels, 1))
            self.fpn_out.append(ConvModule(self.channels, self.channels, 3, 1, 1))

        self.bottleneck = ConvModule(
            len(self.in_channels) * self.channels, self.channels, 3, 1, 1
        )
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, 1)

    def forward(self, features) -> Tensor:
        f = self.fpa(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + F.interpolate(
                f, size=feature.shape[-2:], mode="bicubic", align_corners=False
            )
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(features)):
            fpn_features[i] = F.interpolate(
                fpn_features[i],
                size=fpn_features[0].shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output
