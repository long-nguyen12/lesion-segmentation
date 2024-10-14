import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..builder import SEGMENTORS

from .lib.conv_layer import Conv, BNPReLU
from .lib.attentions import Modified_ECA
from .lib.context_module import CFPModule
from .lib.cbam import CBAM
from .lib.axial_atten import AA_kernel
from .lib.fusion import FusionModule

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

        # self.cbam_0 = CBAM(self.in_channels[0])
        # self.cbam_1 = CBAM(self.in_channels[1])
        # self.cbam_2 = CBAM(self.in_channels[2])
        # self.cbam_3 = CBAM(self.in_channels[3])

        # self.CFP_0 = CFPModule(64, d=8)
        # self.CFP_1 = CFPModule(128, d=8)
        # self.CFP_2 = CFPModule(320, d=8)
        # self.CFP_3 = CFPModule(512, d=8)
        ###### dilation rate 4, 62.8

        self.ra0_conv1 = Conv(64, 32, 3, 1, padding=1, bn_acti=True)
        self.ra0_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra0_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra1_conv1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(320, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        # self.ra3_conv1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        # self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        # self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        # self.aa_kernel_0 = AA_kernel(64, 64)
        # self.aa_kernel_1 = AA_kernel(128, 128)
        # self.aa_kernel_2 = AA_kernel(320, 320)
        # self.aa_kernel_3 = AA_kernel(512, 512)
        
        self.fusion_1 = FusionModule(512, 320)
        self.fusion_2 = FusionModule(320, 128)
        self.fusion_3 = FusionModule(128, 64)

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

        # xc_1 = self.cbam_0(x1)
        # xc_2 = self.cbam_1(x2)
        # xc_3 = self.cbam_2(x3)
        # xc_4 = self.cbam_3(x4)

        decoder_3 = self.decode_head.forward([x1, x2, x3, x4])  # 88x88
        lateral_map_1 = F.interpolate(
            decoder_3, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        
        fuse_1 = self.fusion_1(x4, x3)
        fuse_2 = self.fusion_2(x3, x2)
        fuse_3 = self.fusion_3(x2, x1)

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(
            decoder_3, size=fuse_1.size()[2:], mode="bilinear", align_corners=False
        )
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_2_o = decoder_2_ra.expand(-1, 320, -1, -1).mul(fuse_1)

        ra_2 = self.ra2_conv1(aa_atten_2_o)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)

        x_2 = ra_2 + decoder_2
        lateral_map_2 = F.interpolate(
            x_2, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        # ------------------- atten-three -----------------------
        decoder_1 = F.interpolate(
            x_2, size=x2_size, mode="bilinear", align_corners=False
        )
        # cfp_out_3 = self.CFP_1(x2)
        decoder_1_ra = -1 * (torch.sigmoid(decoder_1)) + 1
        # aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        # aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_1_ra.expand(-1, 128, -1, -1).mul(fuse_2)

        ra_1 = self.ra1_conv1(aa_atten_1_o)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)

        x_1 = ra_1 + decoder_1_ra
        lateral_map_3 = F.interpolate(
            x_1, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        
        # ------------------- atten-four -----------------------
        decoder_0 = F.interpolate(
            x_1, size=x1_size, mode="bilinear", align_corners=False
        )
        # cfp_out_4 = self.CFP_0(x1)
        decoder_0_ra = -1 * (torch.sigmoid(decoder_0)) + 1
        # aa_atten_0 = self.aa_kernel_0(cfp_out_4)
        # aa_atten_0 += cfp_out_4
        aa_atten_0_o = decoder_0_ra.expand(-1, 64, -1, -1).mul(fuse_3)

        ra_0 = self.ra0_conv1(aa_atten_0_o)
        ra_0 = self.ra0_conv2(ra_0)
        ra_0 = self.ra0_conv3(ra_0)

        x_0 = ra_0 + decoder_0
        lateral_map_4 = F.interpolate(
            x_0, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        lateral_map_1 = torch.sigmoid(lateral_map_1)
        lateral_map_2 = torch.sigmoid(lateral_map_2)
        lateral_map_3 = torch.sigmoid(lateral_map_3)
        lateral_map_4 = torch.sigmoid(lateral_map_4)
        # lateral_map_5 = torch.sigmoid(lateral_map_5)

        return lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1
