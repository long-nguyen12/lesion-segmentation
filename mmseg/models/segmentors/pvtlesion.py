import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..builder import SEGMENTORS

from .lib.conv_layer import Conv, BNPReLU
from .lib.attentions import Modified_ECA
from .lib.context_module import CFPModule
from .lib.cbam import CBAM


@SEGMENTORS.register_module()
class LesionSegmentation(nn.Module):
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(LesionSegmentation, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        self.in_channels = decode_head["in_channels"]

        self.cbam_0 = CBAM(self.in_channels[0])
        self.cbam_1 = CBAM(self.in_channels[1])
        self.cbam_2 = CBAM(self.in_channels[2])
        self.cbam_3 = CBAM(self.in_channels[3])

    def forward(self, x):
        segout = self.backbone(x)

        x1 = segout[0]  #  64x88x88
        x2 = segout[1]  # 128x44x44
        x3 = segout[2]  # 320x22x22
        x4 = segout[3]  # 512x11x11
        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        x1 = self.cbam_0(x1)
        x2 = self.cbam_0(x2)
        x3 = self.cbam_0(x3)
        x4 = self.cbam_0(x4)


        decoder_1 = self.decode_head.forward(segout)  # 88x88
        lateral_map_1 = F.interpolate(decoder_1, size=x.shape[2:], mode="bicubic")

        return lateral_map_1
