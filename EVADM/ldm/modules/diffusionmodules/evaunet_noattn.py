# -*- coding: utf-8 -*-
"""
Created on 2023/7/29 22:39
U-Net w/o attention

@author: LU
"""
import numpy as np
import torch
import torch as th
import torch.nn as nn
from einops import rearrange

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer

from ldm.modules.diffusionmodules.openaimodel import \
    TimestepBlock, TimestepEmbedSequential, Upsample, Downsample
from torchinfo import summary

def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # self.h_upd = self.x_upd = nn.Identity()
        ## 2. VarAttn
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(channels // 8, self.out_channels, bias=False))
        self.vp1 = nn.Linear(channels, self.out_channels, bias=False)
        self.vp2 = nn.Linear(channels, self.out_channels, bias=False)
        ## 2. AvgAttn / ChannelAttn
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(channels // 8, self.out_channels, bias=False))

        ## 3. DWConv
        self.dwconv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1,
                                 groups=self.out_channels)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """
        if not hasattr(self.fc1, 'init_weights'):
            kaiming_init(self.in_layers, a=0, nonlinearity='leaky_relu')
            kaiming_init(self.fc1, a=0, nonlinearity='leaky_relu')
            kaiming_init(self.fc2, a=0, nonlinearity='leaky_relu')
            kaiming_init(self.dwconv, a=0, nonlinearity='leaky_relu')
            kaiming_init(self.emb_layers, a=0, nonlinearity='leaky_relu')

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb)
    def var_pooling(self, x, is_std_norm=True):
        # var_pool = torch.var(x, dim=[1, 2], keepdim=True)
        # [batch, channel, height, width]
        var_pool = torch.var(x, dim=[2, 3])
        if not is_std_norm:
            return var_pool
        else:  # use std_norm
            var_var, mean_var = torch.var_mean(var_pool, dim=1, keepdim=True)
            var_var = torch.where(torch.isnan(var_var), torch.full_like(var_var, 0), var_var)
            var_pool_std_norm = (var_pool - mean_var) / torch.sqrt(var_var + 1e-5)
            return var_pool_std_norm

    def _forward(self, x, emb):
        # 1 conv
        h = self.in_layers(x)
        # 1 var_attn
        var_pool = self.var_pooling(x)
        var_pool_fc = self.fc1(var_pool)
        var_attn = self.sig(self.vp1(var_pool) + var_pool_fc).unsqueeze(-1).unsqueeze(-1)
        # 1 avg_attn
        b, c, _, _ = x.size()
        avg_pool = self.avgpool(x).view(b, c)
        avg_pool_fc = self.fc2(avg_pool)#.view(b, c, 1, 1)
        b, c = avg_pool_fc.size()
        avg_attn = self.sig(self.vp2(avg_pool) + avg_pool_fc).view(b, c, 1, 1)

        var_avg_attn = var_attn+avg_attn
        x1 = h * var_avg_attn

        # 1 timestep_embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        x2 = x1 + emb_out

        # 2 spatial_attn
        dw_attn = self.sig(self.dwconv(x2))
        x3 = self.out_layers(x2) * dw_attn
        # 2 conv
        return self.skip_connection(x) / (2 ** 0.5) + x3

class EVA_UNetModel_noattn(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.

    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_head_channels=32,
            legacy=True,
            num_heads=-1,
            num_heads_upsample=-1,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dtype = th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # X if self.num_classes is not None:

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels  # model_channels = 160
        ds = 1
        for level, mult in enumerate(channel_mult):  # channel_mult= ->[1,2,2,4]
            for _ in range(num_res_blocks):  # num_res_blocks=2
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,  # time_embed_dim = model_channels * 4 = 640
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                    )
                ]
                ch = mult * model_channels  # ch = ->[160, 320, 320, 640]

                # X if ds in attention_resolutions:

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            # FOR DownSample
            if level != len(channel_mult) - 1:  # 非最后一块
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # X if num_head_channels == -1:
        num_heads = ch // num_head_channels
        dim_head = num_head_channels

        if legacy:
            # num_heads = 1
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
            ),
            # X AttentionBlock(

            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
            ),
            # X AttentionBlock(

            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                    )
                ]
                ch = model_channels * mult

                # X if ds in attention_resolutions:

                # FOR UpSample
                if level and i == num_res_blocks:  # 非第0块 且为该块最后一层
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        # X if self.predict_codebook_ids:

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        # X if self.predict_codebook_ids:
        return self.out(h)


if __name__ == "__main__":
    # ldmx4
    unet_model1 = EVA_UNetModel_noattn(
        image_size=64,
        in_channels=6,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        attention_resolutions=(),
        num_head_channels=32,
    ).to('cuda')
    unet_model1.eval()
    feat1 = torch.randn((16, 6, 64, 64), device='cuda')
    feat2 = torch.tensor([1] * 16, device='cuda')
    t = unet_model1(feat1, feat2)
    summary(unet_model1, input_size=(1, 6, 64, 64), timesteps=torch.tensor([1]))
    # Trainable params: 98,122,499
    # Non-trainable params: 0
    # Total mult-adds (G): 29.39

    # x4
    EVAUNET_FULL_noattn = EVA_UNetModel_noattn(
        image_size=64,
        in_channels=6,
        model_channels=160,
        out_channels=3,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(64,),
        num_head_channels=32,
    ).to('cpu')
    EVAUNET_FULL_noattn.eval()
    summary(EVAUNET_FULL_noattn, input_size=(1, 6, 64, 64), timesteps=torch.tensor([1]))
    # Trainable params: 63,203,683
    # Total mult-adds (G): 35.91

    # x2
    EVAUNET_FULL_noattn = EVA_UNetModel_noattn(
        image_size=128,
        in_channels=6,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        channel_mult=(1, 2, 2),
        attention_resolutions=(),
        num_head_channels=32,
    ).to('cpu')
    EVAUNET_FULL_noattn.eval()
    summary(EVAUNET_FULL_noattn, input_size=(1, 6, 64, 64), timesteps=torch.tensor([1]))

    # x8
    EVAUNET_FULL_noattn = EVA_UNetModel_noattn(
        image_size=64,
        in_channels=7,
        model_channels=160,
        out_channels=4,
        num_res_blocks=2,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(),
        num_head_channels=32,
    ).to('cpu')
    EVAUNET_FULL_noattn.eval()
    summary(EVAUNET_FULL_noattn, input_size=(1, 7, 64, 64), timesteps=torch.tensor([1]))
    # Trainable params: 127,458,084
    # Total mult-adds (G): 40.14