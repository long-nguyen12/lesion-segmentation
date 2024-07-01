import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_

from ..builder import HEADS
from ...utils import Upsample
from .decode_head import BaseDecodeHead



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
        add_relu: bool = True,
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
    def __init__(self, in_channels, out_channels, upscale_mode="bicubic"):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == "bicubic":
            self.align_corners = True
        else:
            self.align_corners = False

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
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

        self.conv0 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)

        self.att_branch1 = nn.Sequential(
            ConvBnRelu(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            ConvBnRelu(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            ConvBnRelu(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.att_branch2 = nn.Sequential(
            ConvBnRelu(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            ConvBnRelu(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            ConvBnRelu(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.att_branch3 = nn.Sequential(
            ConvBnRelu(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(out_channels, out_channels, 3, padding=7, dilation=7)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(mode=self.upscale_mode, align_corners=self.align_corners)
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)

        # mid = self.conv0(x)

        # att_branch1 = self.att_branch1(mid)
        # att_branch2 = self.att_branch2(mid)
        # att_branch3 = self.att_branch3(mid)

        # att = att_branch1 + att_branch2 + att_branch3
        # x = torch.mul(mid, att)
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)

        x = torch.mul(x, mid)
        x = x + b1
        # return x
        return x


class GAUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_mode: str = "bicubic"):
        super(GAUBlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = True if upscale_mode == "bicubic" else None

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                add_relu=False,
            ),
            nn.Sigmoid(),
        )
        self.conv2 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


@HEADS.register_module()
class PANDecoder(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.fpa = FPABlock(in_channels=self.in_channels[-1], out_channels=self.channels)
        self.gau3 = GAUBlock(
            in_channels=self.in_channels[-2],
            out_channels=self.channels,
            upscale_mode="bicubic",
        )
        self.gau2 = GAUBlock(
            in_channels=self.in_channels[-3],
            out_channels=self.channels,
            upscale_mode="bicubic",
        )
        self.gau1 = GAUBlock(
            in_channels=self.in_channels[-4],
            out_channels=self.channels,
            upscale_mode="bicubic",
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        bottleneck = features[-1]
        x5 = self.fpa(bottleneck)  # 1/32
        x4 = self.gau3(features[-2], x5)  # 1/16
        x3 = self.gau2(features[-3], x4)  # 1/8
        x2 = self.gau1(features[-4], x3)  # 1/4
        output = self.cls_seg(x2)
        return output