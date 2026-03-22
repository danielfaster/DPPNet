import torch.nn as nn
from .Decoder import Decoder,DepthMapDecoder
import numpy as np
from .Transformer import Transformer
from .Transformer import token_Transformer
import torch.nn.functional as F
import torch
import torchvision
from .mae_vit import vit_large_patch16
import  math
import cv2
from  Models.basic_module import DPPNet_Decoder,Depth_Pixel_wise_Potential_aware_Module,Feature_Alignment_Module,Reduction,RGB_Feature_Enhancement_Module,Depth_Feature_Enhancement_Module,Adaptive_Multiple_Fusion_Module,ConvBNReLU,Conver2Output,MaxPooling
from Models.res2net_v1b_base import Res2Net_model
class DPPNet(nn.Module):
    def __init__(self, args):
        super(DPPNet, self).__init__()
        output_channel = 64
        # VST Encoder
        self.rgb_backbone = vit_large_patch16(pre_model_path=args.pre_model_path)
        self.dppm = Depth_Pixel_wise_Potential_aware_Module(Res2Net_model(50), output_channel)
        # Feature_Alignment_Module
        self.fam = Feature_Alignment_Module(1024,output_channel)

        self.saliency_feature_convertor_s1 = Reduction(output_channel*4, output_channel)
        self.saliency_feature_convertor_s2 = Reduction(output_channel*8, output_channel)
        self.saliency_feature_convertor_s3 = Reduction(output_channel*16, output_channel)
        self.saliency_feature_convertor_s4 = Reduction(output_channel*32, output_channel)

        self.edge_feature_convertor_e1 = Reduction(output_channel*4, output_channel)
        self.edge_feature_convertor_e2 = Reduction(output_channel*8, output_channel)
        self.edge_feature_convertor_e3 = Reduction(output_channel*16, output_channel)
        self.edge_feature_convertor_e4 = Reduction(output_channel*32, output_channel)

        self.dfem = Depth_Feature_Enhancement_Module(output_channel)

        self.rfem_1 = RGB_Feature_Enhancement_Module(output_channel)
        self.rfem_2 = RGB_Feature_Enhancement_Module(output_channel)
        self.rfem_3 = RGB_Feature_Enhancement_Module(output_channel)
        self.rfem_4 = RGB_Feature_Enhancement_Module(output_channel)

        self.amfm_1_32 = Adaptive_Multiple_Fusion_Module(output_channel, output_channel, MaxPooling(output_channel))
        self.amfm_1_16 = Adaptive_Multiple_Fusion_Module(output_channel, output_channel, MaxPooling(output_channel))
        self.amfm_1_8 = Adaptive_Multiple_Fusion_Module(output_channel, output_channel, MaxPooling(output_channel))
        self.amfm_1_4 = Adaptive_Multiple_Fusion_Module(output_channel, output_channel, MaxPooling(output_channel))

        self.conv_amfm_1_32 = ConvBNReLU(2 * output_channel, output_channel, 3, 1, 1)
        self.conv_amfm_1_16 = ConvBNReLU(2 * output_channel, output_channel, 3, 1, 1)
        self.conv_amfm_1_8 = ConvBNReLU(3 * output_channel, output_channel, 3, 1, 1)
        self.conv_amfm_1_4 = ConvBNReLU(4 * output_channel, output_channel, 3, 1, 1)

        # decoder
        self.depth_decoder = DPPNet_Decoder(output_channel, return_middle_layer_final=True)
        self.salient_decoder = DPPNet_Decoder(output_channel, return_middle_layer_final=True)

        self.depth_quality_decoder = DPPNet_Decoder(output_channel, return_middle_layer_final=True)
        self.fuse_salient_decoder = DPPNet_Decoder(output_channel, return_middle_layer_final=True)
        self.edge_decoder = DPPNet_Decoder(output_channel, return_middle_layer_final=True)

        self.conv_depht_1_32 = nn.Conv2d(output_channel, 1, 1)
        self.conv_depht_1_16 = nn.Conv2d(output_channel, 1, 1)
        self.conv_depht_1_8 = nn.Conv2d(output_channel, 1, 1)
        self.conv_depht_1_4 = nn.Conv2d(output_channel, 1, 1)

        self.depth_conv = nn.Conv2d(1, 3, kernel_size=1)
        self.convert_output_sal = Conver2Output(output_channel)
        self.convert_output_depth = Conver2Output(output_channel)
        self.convert_output_depth_quality = Conver2Output(output_channel)


    def forward(self, image_Input, depth_Input):

        B, _, _, _ = image_Input.shape

        # VST Encoder
        x_trans = self.rgb_backbone(image_Input)

        rgb_fea_1_32_o, rgb_fea_1_16_o, rgb_fea_1_8_o, rgb_fea_1_4_o = self.fam(x_trans)

        rgb_fea_1_32_backbone = self.saliency_feature_convertor_s4(rgb_fea_1_32_o)
        rgb_fea_1_16_backbone = self.saliency_feature_convertor_s3(rgb_fea_1_16_o)
        rgb_fea_1_8_backbone = self.saliency_feature_convertor_s2(rgb_fea_1_8_o)
        rgb_fea_1_4_backbone = self.saliency_feature_convertor_s1(rgb_fea_1_4_o)

        edge_1_32_backbone = self.edge_feature_convertor_e4(rgb_fea_1_32_o)
        edge_1_16_backbone = self.edge_feature_convertor_e3(rgb_fea_1_16_o)
        edge_1_8_backbone = self.edge_feature_convertor_e2(rgb_fea_1_8_o)
        edge_1_4_backbone = self.edge_feature_convertor_e1(rgb_fea_1_4_o)

        # depth Encoder
        depth_Input = self.depth_conv(depth_Input)
        depth_fea_1_32_backbone, depth_fea_1_16_backbone, depth_fea_1_8_backbone, depth_fea_1_4_backbone, \
        depth_quality_1_32_backbone, depth_quality_1_16_backbone, depth_quality_1_8_backbone, depth_quality_1_4_backbone = self.dppm(depth_Input)


        #  dfem
        depth_fea_1_32, depth_fea_1_16, depth_fea_1_8, depth_fea_1_4 = self.dfem(depth_fea_1_32_backbone, depth_fea_1_16_backbone, depth_fea_1_8_backbone, depth_fea_1_4_backbone,
                                                                                                depth_quality_1_32_backbone, depth_quality_1_16_backbone, depth_quality_1_8_backbone, depth_quality_1_4_backbone)

        # rfem
        rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, edge_1_32, edge_1_16, edge_1_8, edge_1_4 = self.rfem_1(
            rgb_fea_1_32_backbone, rgb_fea_1_16_backbone, rgb_fea_1_8_backbone, rgb_fea_1_4_backbone,edge_1_32_backbone, edge_1_16_backbone, edge_1_8_backbone, edge_1_4_backbone)

        rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, edge_1_32, edge_1_16, edge_1_8, edge_1_4 = self.rfem_2(
            rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4,edge_1_32, edge_1_16, edge_1_8, edge_1_4)

        rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, edge_1_32, edge_1_16, edge_1_8, edge_1_4 = self.rfem_3(
            rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4,edge_1_32, edge_1_16, edge_1_8, edge_1_4)

        rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, edge_1_32, edge_1_16, edge_1_8, edge_1_4 = self.rfem_4(
            rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4,edge_1_32, edge_1_16, edge_1_8, edge_1_4)

        # amfm
        fuse_rgb_fea_1_32 = self.amfm_1_32(rgb_fea_1_32, depth_fea_1_32, depth_quality_1_32_backbone)
        fuse_rgb_fea_1_32 = self.conv_amfm_1_32(torch.cat([fuse_rgb_fea_1_32, rgb_fea_1_32], dim=1))

        fuse_rgb_fea_1_16 = self.amfm_1_16(rgb_fea_1_16, depth_fea_1_16, depth_quality_1_16_backbone)
        fuse_rgb_fea_1_16 = self.conv_amfm_1_16(torch.cat([fuse_rgb_fea_1_16,
                                                         F.interpolate(rgb_fea_1_32, scale_factor=2, mode='bilinear',
                                                                       align_corners=True)], dim=1))

        fuse_rgb_fea_1_8 = self.amfm_1_8(rgb_fea_1_8, depth_fea_1_8, depth_quality_1_8_backbone)
        fuse_rgb_fea_1_8 = self.conv_amfm_1_8(torch.cat([fuse_rgb_fea_1_8,
                                                       F.interpolate(rgb_fea_1_32, scale_factor=4, mode='bilinear',
                                                                     align_corners=True),
                                                       F.interpolate(rgb_fea_1_16, scale_factor=2, mode='bilinear',
                                                                     align_corners=True)
                                                       ], dim=1))

        fuse_rgb_fea_1_4 = self.amfm_1_4(rgb_fea_1_4, depth_fea_1_4, depth_quality_1_4_backbone)
        fuse_rgb_fea_1_4 = self.conv_amfm_1_4(torch.cat([fuse_rgb_fea_1_4,
                                                       F.interpolate(rgb_fea_1_32, scale_factor=8, mode='bilinear',
                                                                     align_corners=True),
                                                       F.interpolate(rgb_fea_1_16, scale_factor=4, mode='bilinear',
                                                                     align_corners=True),
                                                       F.interpolate(rgb_fea_1_8, scale_factor=2, mode='bilinear',
                                                                     align_corners=True)], dim=1))

        # decoder
        salient_final = self.salient_decoder(rgb_fea_1_32, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4)

        edge_final = self.edge_decoder(edge_1_32, edge_1_16, edge_1_8, edge_1_4)

        depth_final = self.depth_decoder(depth_fea_1_32,depth_fea_1_16,depth_fea_1_8,depth_fea_1_4)

        depth_quality = self.depth_quality_decoder(depth_quality_1_32_backbone, depth_quality_1_16_backbone,
                                                   depth_quality_1_8_backbone, depth_quality_1_4_backbone)

        fuse_salient = self.fuse_salient_decoder(fuse_rgb_fea_1_32, fuse_rgb_fea_1_16, fuse_rgb_fea_1_8,fuse_rgb_fea_1_4)

        return fuse_salient[1:],edge_final[1:],depth_quality[1:],depth_final[-1],salient_final[-1]

