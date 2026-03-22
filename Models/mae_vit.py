import torch.nn as nn
from functools import partial
import timm.models.vision_transformer
import torch



class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, fully_conn_output=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.fully_conn_output = fully_conn_output
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.fully_conn_output is False:
            return x[:, 1:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome



def vit_base_patch16(pre_model_path="",**kwargs):
    embed_dim =768
    vit_model = VisionTransformer(
        patch_size=16, embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,num_classes=embed_dim,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), fully_conn_output=False, **kwargs)
    # load pre-trained model
    if torch.cuda.is_available():
        vit_model.cuda()
    if pre_model_path !="":
        print("Load pre-trained checkpoint from: %s" % pre_model_path)
        checkpoint = torch.load(pre_model_path)
        checkpoint_model = checkpoint['model']
        vit_model.load_state_dict(checkpoint_model, strict=False)


    return vit_model

def vit_large_patch16(pre_model_path="",**kwargs):
    embed_dim =1024
    # init and load vit_model
    vit_model = VisionTransformer(
               patch_size=16, embed_dim=embed_dim, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,num_classes=embed_dim,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), fully_conn_output=False, **kwargs)

    if torch.cuda.is_available():
        vit_model.cuda()
    if pre_model_path != "":
        # load pre-trained model
        print("Load pre-trained checkpoint from: %s" % pre_model_path)
        checkpoint = torch.load(pre_model_path)
        checkpoint_model = checkpoint['model']
        vit_model.load_state_dict(checkpoint_model, strict=False)

    return vit_model
