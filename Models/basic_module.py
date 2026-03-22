import torch.nn as nn
import torch.nn.functional as F
import torch
import  numpy as np
import  math
from torch.nn import BatchNorm2d


def gaussian_kernel(kernel_size, sigma):
    kernel_size = kernel_size // 2 * 2 + 1
    x = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    kernel = np.exp(-x**2 / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.tensor(kernel, dtype=torch.float32)

def apply_gaussian_filter(input_features, kernel_size=5, sigma=0.5):

    input_channels = input_features.size(1)
    gaussian_kernel_1d = gaussian_kernel(kernel_size, sigma)
    gaussian_kernel_2d = gaussian_kernel_1d.unsqueeze(0) * gaussian_kernel_1d.unsqueeze(1)
    gaussian_kernel_2d = gaussian_kernel_2d / gaussian_kernel_2d.sum()
    weights = gaussian_kernel_2d.unsqueeze(0).unsqueeze(0).expand(input_channels, 1, -1, -1)

    weights = weights.to(input_features.device)
    filtered_features = F.conv2d(input_features,
                                 weight=weights,
                                 padding=kernel_size // 2,
                                 stride=1,groups=input_channels)

    return  filtered_features

# convert the channel to 1
class Conver2Output(nn.Module):
    def __init__(self,input_channel):
        super(Conver2Output,self).__init__()
        self.output_1_32 = nn.Conv2d(input_channel, 1, 1)
        self.output_1_16 = nn.Conv2d(input_channel, 1, 1)
        self.output_1_8 = nn.Conv2d(input_channel, 1, 1)
        self.output_1_4 = nn.Conv2d(input_channel, 1, 1)

    def forward(self,inputs):
        output_4 = self.output_1_32(inputs[1])
        output_3 = self.output_1_16(inputs[2])
        output_2 = self.output_1_8(inputs[3])
        output_1 = self.output_1_4(inputs[4])

        return inputs[0],output_4,output_3,output_2,output_1

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel,outchannel,kernel_size=1):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, outchannel, kernel_size)

    def forward(self, x, targetSize):

        if type(targetSize) == list:
            if x.size()[2:] != targetSize[2:]:
                x = self.conv(F.interpolate(x, size=targetSize[2:], mode='bilinear', align_corners=True))
        else:
            if x.size()[2:] != targetSize.size()[2:]:
                 x = self.conv(F.interpolate(x, size=targetSize.size()[2:], mode='bilinear', align_corners=True))

        return x

class conv_upsample_2(nn.Module):
    def __init__(self, channel):
        super(conv_upsample_2, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, leaky_relu=False, use_bn=True, frozen=False,  prelu=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            if leaky_relu is True:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu is True:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x

class SoftPool(torch.nn.Module):
    def __init__(self, kernel_size):
        super(SoftPool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.avg_pool2d(x**2, self.kernel_size)**0.5

class MaxPooling(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.basic_conv = BasicConv2d(in_planes=in_channel, out_planes=in_channel, kernel_size=3, stride=1, padding=1)
        self.softpool_1 = SoftPool(kernel_size=2)
        self.softpool_2 = SoftPool(kernel_size=2)
        self.softpool_3 = SoftPool(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, f1, f2):
        f = f1.mul(f2)
        f = self.basic_conv(f)
        f = self.softpool_1(self.upsample(f))

        f1 = f + f1
        f2 = f + f2
        f1 = self.softpool_2(self.upsample(f1))
        f2 = self.softpool_3(self.upsample(f2))

        f = f1 + f2
        return f, f1, f2

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2*in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, kernel_size=x.size(2))
        max_pool = F.max_pool2d(x, kernel_size=x.size(2))

        pool = torch.cat([avg_pool, max_pool], dim=1)

        attention_weights = self.conv(pool)

        attention_weights = torch.sigmoid(attention_weights)

        attended_feature = x * attention_weights

        return attended_feature

class Depth_Potentia_aware_Module(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, depth, depth_quality):
        ################################
        x_depth = self.layer_10(depth)
        x_depth_quality = self.layer_20(depth_quality)
        depth_w = nn.Sigmoid()(x_depth_quality)
        ##
        x_depth_w = depth.mul(depth_w)
        x_depth_r = x_depth_w + x_depth
        ## fusion
        x_depth_r_r = self.layer_11(x_depth_r)

        return x_depth_r_r


class Adaptive_Multiple_Fusion_Module(nn.Module):
    def __init__(self, in_dim,  out_dim, maxpooling):
        super().__init__()

        self.layer_10 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_11 = BasicConv2d(out_dim * 1, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_21 = BasicConv2d(out_dim * 1, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_12 = BasicConv2d(out_dim * 1, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_22 = BasicConv2d(out_dim * 1, out_dim, kernel_size=3, stride=1, padding=1)

        self.maxpooling = maxpooling
        self.layer_ful1 = BasicConv2d(out_dim * 4, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth,depth_potentail):
        ################################

        x_rgb = self.layer_10(rgb)
        x_dep_r = self.layer_20(depth)

        x_dep = x_dep_r

        rgb_w = nn.Sigmoid()(x_rgb)
        dep_w = nn.Sigmoid()(x_dep)

        ##
        x_rgb_w = rgb.mul(dep_w)
        x_dep_w = depth.mul(rgb_w)

        x_rgb_r = x_rgb_w + rgb
        x_dep_r = x_dep_w + depth

        ## fusion
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)

        dp_w = nn.Sigmoid()(depth_potentail)
        # branch 1
        ful_mul = torch.mul(x_rgb_r, x_dep_r)

        # branch 2
        x_in1 = torch.reshape(x_rgb_r, [x_rgb_r.shape[0], 1, x_rgb_r.shape[1], x_rgb_r.shape[2], x_rgb_r.shape[3]])
        x_in2 = torch.reshape(x_dep_r, [x_dep_r.shape[0], 1, x_dep_r.shape[1], x_dep_r.shape[2], x_dep_r.shape[3]])
        x_cat = torch.cat((x_in1, x_in2), dim=1)
        ful_max = x_cat.max(dim=1)[0]

        # branch 3
        ful_cie, _, _ = self.maxpooling(x_rgb_r, x_dep_r)

        # branch 4

        ful_adaptive_add = x_rgb_r + x_dep_r * dp_w

        ful_out = torch.cat((ful_mul, ful_cie, ful_max, ful_adaptive_add), dim=1)

        out1 = self.layer_ful1(ful_out)

        return out1

#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class DPPNet_Decoder(nn.Module):
    def __init__(self, channel,return_middle_layer=False,return_middle_layer_final=False):
        super().__init__()
        self.conv_upsample1 = conv_upsample(channel, channel,kernel_size=1)
        self.conv_upsample2 = conv_upsample(channel, channel,kernel_size=1)
        self.conv_upsample3 = conv_upsample(channel, channel,kernel_size=1)


        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.return_middle_layer = return_middle_layer
        self.return_middle_layer_final = return_middle_layer_final

        self.gcm_1_32 = GCM(channel, channel)
        self.gcm_1_16 = GCM(channel, channel)
        self.gcm_1_8 = GCM(channel, channel)
        self.gcm_1_4 = GCM(channel, channel)

        if return_middle_layer_final:
            self.output_1_32 =  nn.Conv2d(channel, 1, 1)
            self.output_1_16 =  nn.Conv2d(channel, 1, 1)
            self.output_1_8 =  nn.Conv2d(channel, 1, 1)
            self.output_1_4 =  nn.Conv2d(channel, 1, 1)



    def forward(self, x4, x3, x2, x1):
        # x4, x3, x2, x1  1_32, 1_16, 1_8, 1_4
        x4 = self.gcm_1_32(x4)

        x3 = torch.cat((x3, self.conv_upsample1(x4,x3)), 1)
        x3 = self.conv_cat1(x3)

        x3 = self.gcm_1_16(x3)

        x2 = torch.cat((x2, self.conv_upsample2(x3,x2)), 1)
        x2 = self.conv_cat2(x2)
        x2 = self.gcm_1_8(x2)

        x1 = torch.cat((x1, self.conv_upsample3(x2,x1)), 1)
        x1 = self.conv_cat3(x1)
        x1 = self.gcm_1_4(x1)

        x = self.output(x1)

        x = F.interpolate(x,scale_factor=4,mode='bilinear',align_corners=True)

        if self.return_middle_layer_final:
            return self.output_1_32(x4), self.output_1_16(x3), self.output_1_8(x2), self.output_1_4(x1),x
        elif self.return_middle_layer :
            return x, x4, x3, x2, x1
        else:
            return x



class Depth_Pixel_wise_Potential_aware_Module(nn.Module):

    def __init__(self,backbone,outputChannel):
        super().__init__()
        self.resnet = backbone
        self.depth_convertor_1_4 =  Reduction(256,outputChannel)
        self.depth_convertor_1_8  =  Reduction(512,outputChannel)
        self.depth_convertor_1_16  =  Reduction(1024,outputChannel)
        self.depth_convertor_1_32 =  Reduction(2048,outputChannel)

        self.depth_potential_convertor_1_4 = Reduction(256, outputChannel)
        self.depth_potential_convertor_1_8 = Reduction(512, outputChannel)
        self.depth_potential_convertor_1_16 = Reduction(1024, outputChannel)
        self.depth_potential_convertor_1_32 = Reduction(2048, outputChannel)

    def  forward(self,x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_1_4 = self.resnet.layer1(x)  # 256 x 64 x 64  1/4
        x_1_8 = self.resnet.layer2(x_1_4)  # 512 x 32 x 32  1/8
        x_1_16 = self.resnet.layer3(x_1_8)  # 1024 x 16 x 16  1/16
        x_1_32 = self.resnet.layer4(x_1_16)  # 2048 x 8 x 8  1/32

        # compression channel
        x_1_4_dq = self.depth_convertor_1_4(x_1_4)
        x_1_8_dq = self.depth_convertor_1_8(x_1_8)
        x_1_16_dq = self.depth_convertor_1_16(x_1_16)
        x_1_32_dq = self.depth_convertor_1_32(x_1_32)

        x_1_4_d = self.depth_potential_convertor_1_4(x_1_4)
        x_1_8_d = self.depth_potential_convertor_1_8(x_1_8)
        x_1_16_d = self.depth_potential_convertor_1_16(x_1_16)
        x_1_32_d = self.depth_potential_convertor_1_32(x_1_32)


        return x_1_32_d, x_1_16_d, x_1_8_d, x_1_4_d,x_1_32_dq,x_1_16_dq,x_1_8_dq,x_1_4_dq

class ResNet_Backbone_v2(nn.Module):

    def __init__(self,backbone,outputChannel):
        super(ResNet_Backbone_v2, self).__init__()
        self.resnet = backbone
        self.reduction_1_4_depth =  Reduction(256,outputChannel)
        self.reduction_1_8_depth  =  Reduction(512,outputChannel)
        self.reduction_1_16_depth  =  Reduction(1024,outputChannel)
        self.reduction_1_32_depth  =  Reduction(2048,outputChannel)

    def  forward(self,x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_1_4 = self.resnet.layer1(x)  # 256 x 64 x 64  1/4
        x_1_8 = self.resnet.layer2(x_1_4)  # 512 x 32 x 32  1/8
        x_1_16 = self.resnet.layer3(x_1_8)  # 1024 x 16 x 16  1/16
        x_1_32 = self.resnet.layer4(x_1_16)  # 2048 x 8 x 8  1/32

        # compression channel
        x_1_4_d = self.reduction_1_4_depth(x_1_4)
        x_1_8_d = self.reduction_1_8_depth(x_1_8)
        x_1_16_d = self.reduction_1_16_depth(x_1_16)
        x_1_32_d = self.reduction_1_32_depth(x_1_32)


        return x_1_32_d, x_1_16_d, x_1_8_d, x_1_4_d

class PVT_Backbone_v1(nn.Module):

    def __init__(self,backbone,outputChannel):
        super(PVT_Backbone_v1, self).__init__()
        self.pvtNet = backbone
        self.reduction_1_4_depth =  Reduction(64,outputChannel)
        self.reduction_1_8_depth  =  Reduction(128,outputChannel)
        self.reduction_1_16_depth  =  Reduction(320,outputChannel)
        self.reduction_1_32_depth  =  Reduction(512,outputChannel)

        self.reduction_1_4_depth_quality = Reduction(64, outputChannel)
        self.reduction_1_8_depth_quality = Reduction(128, outputChannel)
        self.reduction_1_16_depth_quality = Reduction(320, outputChannel)
        self.reduction_1_32_depth_quality = Reduction(512, outputChannel)

    def  forward(self,x):
        x_1_4, x_1_8, x_1_16, x_1_32 =self.pvtNet(x)

        # compression channel
        x_1_4_dq = self.reduction_1_4_depth_quality(x_1_4)
        x_1_8_dq = self.reduction_1_8_depth_quality(x_1_8)
        x_1_16_dq = self.reduction_1_16_depth_quality(x_1_16)
        x_1_32_dq = self.reduction_1_32_depth_quality(x_1_32)

        x_1_4_d = self.reduction_1_4_depth(x_1_4)
        x_1_8_d = self.reduction_1_8_depth(x_1_8)
        x_1_16_d = self.reduction_1_16_depth(x_1_16)
        x_1_32_d = self.reduction_1_32_depth(x_1_32)


        return x_1_32_d, x_1_16_d, x_1_8_d, x_1_4_d,x_1_32_dq,x_1_16_dq,x_1_8_dq,x_1_4_dq

class Depth_Feature_Enhancement_Module(nn.Module):

    def __init__(self, output_channel):
        super().__init__()

        self.spacial_attn_1_32 = SpatialAttention(output_channel)
        self.spacial_attn_1_16 = SpatialAttention(output_channel)
        self.spacial_attn_1_8 = SpatialAttention(output_channel)
        self.spacial_attn_1_4 = SpatialAttention(output_channel)

        self.dpam_1_32 = Depth_Potentia_aware_Module(output_channel, output_channel)
        self.dpam_1_16 = Depth_Potentia_aware_Module(output_channel, output_channel)
        self.dpam_1_8 = Depth_Potentia_aware_Module(output_channel, output_channel)
        self.dpam_1_4 = Depth_Potentia_aware_Module(output_channel, output_channel)

        self.layer_2 = nn.Sequential(nn.Conv2d(2 * output_channel, output_channel, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True), )
        self.layer_3 = nn.Sequential(nn.Conv2d(3 * output_channel, output_channel, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True), )
        self.layer_4 = nn.Sequential(nn.Conv2d(4 * output_channel, output_channel, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True), )

    def forward(self, depth_fea_1_32, depth_fea_1_16, depth_fea_1_8, depth_fea_1_4,
                depth_quality_1_32, depth_quality_1_16, depth_quality_1_8, depth_quality_1_4):
        depth_fea_1_32 = apply_gaussian_filter(depth_fea_1_32, kernel_size=3, sigma=0.5)
        depth_fea_1_16 = apply_gaussian_filter(depth_fea_1_16, kernel_size=3, sigma=0.5)
        depth_fea_1_8 = apply_gaussian_filter(depth_fea_1_8, kernel_size=3, sigma=0.5)
        depth_fea_1_4 = apply_gaussian_filter(depth_fea_1_4, kernel_size=5, sigma=0.5)

        depth_fea_1_32 = self.dpam_1_32(depth_quality_1_32, depth_fea_1_32)

        depth_fea_1_16 = self.dpam_1_16(depth_quality_1_16, depth_fea_1_16)
        depth_fea_1_16 = self.layer_2(torch.cat((depth_fea_1_16,
                                                 F.interpolate(depth_fea_1_32, scale_factor=2, mode='bilinear',
                                                               align_corners=True)), dim=1))

        depth_fea_1_8 = self.dpam_1_8(depth_quality_1_8, depth_fea_1_8)
        depth_fea_1_8 = self.layer_3(torch.cat((depth_fea_1_8,
                                                F.interpolate(depth_fea_1_32, scale_factor=4, mode='bilinear',
                                                              align_corners=True),
                                                F.interpolate(depth_fea_1_16, scale_factor=2, mode='bilinear',
                                                              align_corners=True)), dim=1))

        depth_fea_1_4 = self.dpam_1_4(depth_quality_1_4, depth_fea_1_4)
        depth_fea_1_4 = self.layer_4(torch.cat((depth_fea_1_4,
                                                F.interpolate(depth_fea_1_32, scale_factor=8, mode='bilinear',
                                                              align_corners=True),
                                                F.interpolate(depth_fea_1_16, scale_factor=4, mode='bilinear',
                                                              align_corners=True),
                                                F.interpolate(depth_fea_1_8, scale_factor=2, mode='bilinear',
                                                              align_corners=True)), dim=1))
        depth_fea_1_32 = self.spacial_attn_1_32(depth_fea_1_32)
        depth_fea_1_16 = self.spacial_attn_1_16(depth_fea_1_16)
        depth_fea_1_8 = self.spacial_attn_1_8(depth_fea_1_8)
        depth_fea_1_4 = self.spacial_attn_1_4(depth_fea_1_4)

        return depth_fea_1_32, depth_fea_1_16, depth_fea_1_8, depth_fea_1_4


class RGB_Feature_Enhancement_Module(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = conv_upsample_2(channel)
        self.conv2 = conv_upsample_2(channel)
        self.conv3 = conv_upsample_2(channel)
        self.conv4 = conv_upsample_2(channel)
        self.conv5 = conv_upsample_2(channel)
        self.conv6 = conv_upsample_2(channel)
        self.conv7 = conv_upsample_2(channel)
        self.conv8 = conv_upsample_2(channel)
        self.conv9 = conv_upsample_2(channel)
        self.conv10 = conv_upsample_2(channel)
        self.conv11 = conv_upsample_2(channel)
        self.conv12 = conv_upsample_2(channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(5 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):


        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1,
                                               self.conv1(x_e2, x_s1),
                                               self.conv2(x_e3, x_s1),
                                               self.conv3(x_e4, x_s1)), 1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2, x_e2,
                                               self.conv4(x_e3, x_s2),
                                               self.conv5(x_e4, x_s2)), 1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3,
                                               self.conv6(x_e4, x_s3)), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.conv_f5(x_e1 * x_sf1 *
                                    self.conv7(x_sf2, x_e1) *
                                    self.conv8(x_sf3, x_e1) *
                                    self.conv9(x_sf4, x_e1))
        x_ef2 = x_e2 + self.conv_f6(x_e2 * x_sf2 *
                                    self.conv10(x_sf3, x_e2) *
                                    self.conv11(x_sf4, x_e2))
        x_ef3 = x_e3 + self.conv_f7(x_e3 * x_sf3 *
                                    self.conv12(x_sf4, x_e3))
        x_ef4 = x_e4 + self.conv_f8(x_e4 * x_sf4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class  Feature_Alignment_Module(nn.Module):

    def __init__(self,inchannel=768,outchannel=64):
        super().__init__()
        self.conv1 = BasicConv2d(inchannel,outchannel*32,kernel_size=3,stride=2,padding=1)
        self.conv2 = BasicConv2d(inchannel,outchannel*16,kernel_size=1,stride=1)
        self.up_conv3 = conv_upsample(inchannel,outchannel*8)
        self.up_conv4 = conv_upsample(inchannel,outchannel*4)

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x

    def forward(self,x):

        x = self.to_2D(x)
        b,c,h,w = x.shape
        x4 = self.conv1(x)
        x3 = self.conv2(x)
        x2 = self.up_conv3(x,[b,c,2*h,2*w])
        x1 = self.up_conv4(x,[b,c,4*h,4*w])

        return x4,x3,x2,x1





