__all__ = ['vitae_lite', 'vitae_base', 'vitae_large']


import torch
import torch.nn as nn

from functools import partial
from .util.pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Block


class CNNDecBlock(nn.Module):

    def __init__(self, in_chans, out_chans, norm_layer=nn.BatchNorm2d):
        super(CNNDecBlock, self).__init__()

        layers = [nn.Conv2d(in_chans, out_chans, 3, padding=1)]
        if norm_layer is not None:
            layers.append(norm_layer(out_chans))
        layers.append(nn.LeakyReLU(0.02, inplace=True))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class ViTAutoEncoder(nn.Module):

    def __init__(
        self, input_size, in_chans, patch_size,
        enc_chans=1, enc_dim=128, enc_depth=8, enc_num_heads=8,
        enc_mlp_ratio=4., enc_norm_layer=nn.LayerNorm,
        dec_dims=[16, 16, 16, 16, 16], dec_norm_layer=nn.BatchNorm2d
    ):
        super(ViTAutoEncoder, self).__init__()

        # ViT encoder specifics
        self.in_chans = in_chans
        self.enc_chans = enc_chans
        self.patch_embed = PatchEmbed(input_size, patch_size, in_chans, enc_dim)

        if isinstance(input_size, int):
            self.grid_size = (input_size // patch_size,) * 2
        elif isinstance(input_size, (tuple, list)):
            self.grid_size = (input_size[0] // patch_size, input_size[1] // patch_size)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, enc_dim), requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            Block(
                enc_dim, enc_num_heads, enc_mlp_ratio,
                qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0,
                norm_layer=enc_norm_layer
            )
            for i in range(enc_depth)
        ])
        self.norm = enc_norm_layer(enc_dim)
        self.encoder_out = nn.Conv2d(enc_chans, 1, 1, padding=0)

        # CNN decoder specifics
        self.decoder_embed = nn.Linear(enc_dim, patch_size ** 2 * enc_chans, bias=True)
        dec_dims = [enc_chans] + dec_dims
        decoder_cnn_blocks = [
            CNNDecBlock(dec_dims[i], dec_dims[i + 1], norm_layer=dec_norm_layer)
            for i in range(len(dec_dims) - 1)
        ]
        self.decoder_cnn = nn.Sequential(*decoder_cnn_blocks)
        self.decoder_out = nn.Conv2d(dec_dims[-1], 1, 1, padding=0)

        self.initialize_weights()

    def initialize_weights(self):        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        ph, pw = self.patch_embed.patch_size
        h, w = self.grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, self.enc_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.enc_chans, h * ph, w * pw))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed
        x = x + self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)
        # reshape to 2d
        x_enc = self.unpatchify(x)
        pred_enc = self.encoder_out(x_enc)
        # 2d conv
        x_dec = self.decoder_cnn(x_enc)
        pred_dec = self.decoder_out(x_dec)
        return pred_dec, pred_enc

    def forward(self, feature):
        latent = self.forward_encoder(feature)
        pred_dec, pred_enc = self.forward_decoder(latent)
        return pred_dec, pred_enc


# ------------------------------------------------------------------------------


def vitae_lite(input_size, in_chans, patch_size):
    model = ViTAutoEncoder(
        input_size, in_chans, patch_size,
        enc_chans=16, enc_dim=32, enc_depth=8, enc_num_heads=8,
        enc_mlp_ratio=4, enc_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        dec_dims=[16, 16, 16, 16, 16], dec_norm_layer=nn.BatchNorm2d,
    )
    return model


def vitae_base(input_size, in_chans, patch_size):
    model = ViTAutoEncoder(
        input_size, in_chans, patch_size,
        enc_chans=32, enc_dim=64, enc_depth=8, enc_num_heads=8,
        enc_mlp_ratio=4, enc_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        dec_dims=[32, 32, 32, 32, 32], dec_norm_layer=nn.BatchNorm2d,
    )
    return model


def vitae_large(input_size, in_chans, patch_size):
    model = ViTAutoEncoder(
        input_size, in_chans, patch_size,
        enc_chans=64, enc_dim=128, enc_depth=8, enc_num_heads=8,
        enc_mlp_ratio=4, enc_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        dec_dims=[64, 64, 64, 64, 64], dec_norm_layer=nn.BatchNorm2d,
    )
    return model
