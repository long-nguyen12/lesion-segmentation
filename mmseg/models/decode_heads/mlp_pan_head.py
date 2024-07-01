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

        x = mid * att

        x = x + b1

        return x


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class MLPPanHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)

        self.fpa = FPABlock(
            in_channels=sum(self.in_channels), out_channels=self.channels
        )
        for i, dim in enumerate(self.in_channels):
            self.add_module(f"cbam_c{i+1}", CBAM(dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, features):
        B, _, H, W = features[0].shape
        outs = []

        for i, cf in enumerate(features):
            cf = eval(f"self.cbam_c{i+1}")(cf)
            outs.append(
                F.interpolate(cf, size=(H, W), mode="bicubic", align_corners=True)
            )

        seg = self.fpa(torch.cat(outs, dim=1))
        output = self.cls_seg(seg)
        return output
