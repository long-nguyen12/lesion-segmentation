import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.segmentors.lib.conv_layer import Conv
from mmseg.models.segmentors.lib.self_attention import self_attn
import math


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


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode="h")
        self.Wattn = self_attn(out_channel, mode="w")

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


class AA_kernel_1(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, upscale_mode: str = "bicubic"
    ):
        super(AA_kernel_1, self).__init__()

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
        self.conv2 = ConvBnRelu(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(
            y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners
        )
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class PSA(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * channel_out
        return spatial_out
