import torch.nn as nn
import torch
from .token_performer import Token_performer
from .Transformer import saliency_token_inference, contour_token_inference, token_TransformerEncoder,depth_quality_token_inference
from .Transformer import  token_TransformerEncoder

class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens, contour_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = torch.cat((fea, contour_tokens), dim=1)
        # [B, 1 + H*W + 1, 384]

        fea = self.encoderlayer(fea)
        # fea [B, 1 + H*W + 1, 384]
        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        contour_tokens = fea[:, -1, :].unsqueeze(1)

        saliency_fea = self.saliency_token_pre(fea)
        # saliency_fea [B, H*W, 384]
        contour_fea = self.contour_token_pre(fea)
        # contour_fea [B, H*W, 384]

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        contour_fea = self.mlp2_c(self.norm2_c(contour_fea))

        return saliency_fea, contour_fea, fea, saliency_tokens, contour_tokens



class depth_quality_token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(depth_quality_token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.depth_quality_token_pre = depth_quality_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, depth_quality_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((depth_quality_tokens, fea), dim=1)
        # [B, 1 + H*W + 1, 384]

        fea = self.encoderlayer(fea)
        # fea [B, 1 + H*W + 1, 384]
        depth_quality_tokens = fea[:, 0, :].unsqueeze(1)

        depth_quality_fea = self.depth_quality_token_pre(fea)
        # saliency_fea [B, H*W, 384]

        # reproject back to 64 dim
        depth_quality_fea = self.mlp2(self.norm2(depth_quality_fea))


        return depth_quality_fea, fea, depth_quality_tokens


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)

        return dec_fea


class DepthMapDecoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, num_heads=6, img_size=224, mlp_ratio=0.3):
        super(DepthMapDecoder, self).__init__()
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)

        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1,
                                       kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)


        # token based multi-task predictions
        self.depth_quality_token_trans_1_8 = depth_quality_token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.depth_quality_token_trans_1_4 = depth_quality_token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)

        self.depth_map_quality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.depth_map_token_pre_1_16 = depth_quality_token_inference(dim=embed_dim, num_heads=1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, token_fea_1_16, origin_depth_quality_fea_1_8, origin_depth_quality_fea_1_4):

        B, _, _, = token_fea_1_16.size()

        depth_map_quality_token = self.depth_map_quality_token.expand(B, -1, -1)
        depth_quality_fea_1_16 = torch.cat((token_fea_1_16,depth_map_quality_token), dim=1)
        # fea_1_16 [B, 1 + 14*14, 384]
        depth_quality_fea_1_16 = self.encoderlayer(depth_quality_fea_1_16)
        # fea_1_16 [B, 1 + 14*14, 384]
        depth_map_quality_token = depth_quality_fea_1_16[:, 0, :].unsqueeze(1)


        pre_depth_quality_1_16 = self.depth_map_token_pre_1_16(depth_quality_fea_1_16)

        pre_depth_quality_1_16 = self.mlp(self.norm(pre_depth_quality_1_16))

        pre_depth_quality_1_16 = self.pre_1_16(pre_depth_quality_1_16)
        pre_depth_quality_1_16 = pre_depth_quality_1_16[:, :, :].transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)


        # contour_fea_1_16 = self.mlp_c(self.norm_c(contour_fea_1_16))
        # # contour_fea_1_16 [B, 14*14, 64]
        # contour_1_16 = self.pre_1_16_c(contour_fea_1_16)
        # contour_1_16 = contour_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        fea_1_8 = self.decoder1(token_fea_1_16, origin_depth_quality_fea_1_8)

        # token prediction
        depth_quality_fea_1_8, token_fea_1_8, depth_quality_tokens = self.depth_quality_token_trans_1_8(fea_1_8, depth_map_quality_token)

        # predict depth quality maps
        pre_depth_quality_1_8 = self.pre_1_8(depth_quality_fea_1_8)
        pre_depth_quality_1_8 = pre_depth_quality_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)


        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(token_fea_1_8[:,1:,:], origin_depth_quality_fea_1_4)


        # token prediction
        depth_quality_fea_1_4, token_fea_1_4, depth_quality_tokens = self.depth_quality_token_trans_1_4(fea_1_4,
                                                                                                        depth_map_quality_token)

        # predict depth quality maps
        pre_depth_quality_1_4 = self.pre_1_4(depth_quality_fea_1_4)
        pre_depth_quality_1_4 = pre_depth_quality_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)


        # 1/4 -> 1
        fea_1_1 = self.decoder3(depth_quality_fea_1_4)
        pre_depth_quality_1 = self.pre_1_1(fea_1_1)
        pre_depth_quality_1 = pre_depth_quality_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)


        return [pre_depth_quality_1_16, pre_depth_quality_1_8, pre_depth_quality_1_4, pre_depth_quality_1]


class DepthMapDecoder_V2(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, num_heads=6, img_size=224, mlp_ratio=0.3):
        super(DepthMapDecoder_V2, self).__init__()
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=784, token_dim=token_dim, img_size=img_size, ratio=4,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1,
                                       kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)


        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)

        # self.depth_map_quality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.depth_map_token_pre_1_16 = saliency_token_inference(dim=embed_dim, num_heads=1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, depth_quality_fea_1_16, depth_quality_fea_1_8, depth_quality_fea_1_4):
    # def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8,
    #             rgb_fea_1_4):
        # saliency_fea_1_16 [B, 14*14, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain saliency token and contour token)

        # saliency_tokens [B, 1, 384]
        # contour_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 28*28, 64]

        B, _, _, = depth_quality_fea_1_16.size()

        # contour_fea_1_16 = self.contour_token_pre(fea_1_16)
        # return saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens

        # saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # # saliency_fea_1_16 [B, 14*14, 64]
        # mask_1_16 = self.pre_1_16(saliency_fea_1_16)

        pre_depth_quality_1_16 = self.depth_map_token_pre_1_16(depth_quality_fea_1_16)
        pre_depth_quality_1_16 = self.mlp(self.norm(pre_depth_quality_1_16))
        pre_depth_quality_1_16 = self.pre_1_16(pre_depth_quality_1_16)
        pre_depth_quality_1_16 = pre_depth_quality_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)



        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature

        fea_1_8 = self.decoder1(depth_quality_fea_1_16, depth_quality_fea_1_8)

        # predict depth quality maps
        pre_depth_quality_1_8 = self.pre_1_16(fea_1_8)
        pre_depth_quality_1_8 = pre_depth_quality_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)


        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(fea_1_8, depth_quality_fea_1_4)

        # predict depth quality maps
        pre_depth_quality_1_4 = self.pre_1_4(fea_1_4)
        pre_depth_quality_1_4 = pre_depth_quality_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)


        # 1/4 -> 1
        fea_1_1 = self.decoder3(fea_1_4)
        pre_depth_quality_1 = self.pre_1_1(fea_1_1)
        pre_depth_quality_1 = pre_depth_quality_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)


        return [pre_depth_quality_1_16, pre_depth_quality_1_8, pre_depth_quality_1_4, pre_depth_quality_1]


class DepthMapDecoder_V1(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, num_heads=6, img_size=224, mlp_ratio=0.3):
        super(DepthMapDecoder_V1, self).__init__()
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1,
                                       kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)


        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)

        self.depth_map_quality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.depth_map_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, depth_quality_1_16, depth_quality_1_8, depth_quality_1_4):
    # def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8,
    #             rgb_fea_1_4):
        # saliency_fea_1_16 [B, 14*14, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain saliency token and contour token)

        # saliency_tokens [B, 1, 384]
        # contour_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 28*28, 64]

        B, _, _, = depth_quality_1_16.size()

        depth_map_quality_token = self.depth_map_quality_token.expand(B, -1, -1)

        depth_quality_1_16 = torch.cat((depth_map_quality_token, depth_quality_1_16), dim=1)

        # fea_1_16 [B, 1 + 14*14 + 1, 384]

        depth_quality_1_16 = self.encoderlayer(depth_quality_1_16)
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        depth_quality_tokens = depth_quality_1_16[:, 0, :].unsqueeze(1)


        depth_quality_fea_1_16 = self.depth_map_token_pre(fea_1_16)
        # contour_fea_1_16 = self.contour_token_pre(fea_1_16)
        # return saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens


        # saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # # saliency_fea_1_16 [B, 14*14, 64]
        # mask_1_16 = self.pre_1_16(saliency_fea_1_16)

        mask_1_16 = depth_quality_1_16

        # contour_fea_1_16 = self.mlp_c(self.norm_c(contour_fea_1_16))
        # # contour_fea_1_16 [B, 14*14, 64]
        # contour_1_16 = self.pre_1_16_c(contour_fea_1_16)
        # contour_1_16 = contour_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature

        # saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        fea_1_8 = self.decoder1(depth_quality_1_16, depth_quality_1_8)


        # predict saliency and contour maps
        mask_1_8 = self.pre_1_16(fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)


        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(fea_1_8, depth_quality_1_4)

        # predict saliency maps and contour maps
        mask_1_4 = self.pre_1_8(fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)


        # 1/4 -> 1
        fea_1_1 = self.decoder3(fea_1_4)

        mask_1_1 = self.pre_1_4(fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)


        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1]

class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim,  img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_c = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        # predict contour maps
        self.pre_1_16_c = nn.Linear(token_dim, 1)
        self.pre_1_8_c = nn.Linear(token_dim, 1)
        self.pre_1_4_c = nn.Linear(token_dim, 1)
        self.pre_1_1_c = nn.Linear(token_dim, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4):
        # saliency_fea_1_16 [B, 14*14, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain saliency token and contour token)

        # saliency_tokens [B, 1, 384]
        # contour_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 28*28, 64]

        B, _, _, = token_fea_1_16.size()

        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # saliency_fea_1_16 [B, 14*14, 64]
        mask_1_16 = self.pre_1_16(saliency_fea_1_16)
        mask_1_16 = mask_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        contour_fea_1_16 = self.mlp_c(self.norm_c(contour_fea_1_16))
        # contour_fea_1_16 [B, 14*14, 64]
        contour_1_16 = self.pre_1_16_c(contour_fea_1_16)
        contour_1_16 = contour_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        fea_1_8 = self.decoder1(token_fea_1_16[:, 1:-1, :], rgb_fea_1_8)

        # token prediction
        saliency_fea_1_8, contour_fea_1_8, token_fea_1_8, saliency_tokens, contour_tokens = self.token_pre_1_8(fea_1_8, saliency_tokens, contour_tokens)

        # predict saliency and contour maps
        mask_1_8 = self.pre_1_8(saliency_fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        contour_1_8 = self.pre_1_8_c(contour_fea_1_8)
        contour_1_8 = contour_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(token_fea_1_8[:, 1:-1, :], rgb_fea_1_4)

        # token prediction
        saliency_fea_1_4, contour_fea_1_4, token_fea_1_4, saliency_tokens, contour_tokens = self.token_pre_1_4(fea_1_4, saliency_tokens, contour_tokens)

        # predict saliency maps and contour maps
        mask_1_4 = self.pre_1_4(saliency_fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        contour_1_4 = self.pre_1_4_c(contour_fea_1_4)
        contour_1_4 = contour_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        saliency_fea_1_1 = self.decoder3(saliency_fea_1_4)
        contour_fea_1_1 = self.decoder3_c(contour_fea_1_4)

        mask_1_1 = self.pre_1_1(saliency_fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        contour_1_1 = self.pre_1_1_c(contour_fea_1_1)
        contour_1_1 = contour_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1], [contour_1_16, contour_1_8, contour_1_4, contour_1_1]

import torch
if __name__ == '__main__':
    depth_map_decoder = DepthMapDecoder(embed_dim=384, token_dim=64, depth=2, img_size=224)

    # depth_map_decoder = DepthMapDecoder(decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
    #                                    kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True))

    depth_quality_1_16 = torch.randn([4,196,384])
    depth_quality_1_8 = torch.randn([4,784,64])
    depth_quality_1_4 = torch.randn([4,3136,64])
    depth_map_decoder(depth_quality_1_16, depth_quality_1_8, depth_quality_1_4)