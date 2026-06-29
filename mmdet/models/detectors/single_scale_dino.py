# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Optional

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig
from ..layers import (DetrTransformerEncoder, DinoTransformerDecoder,
                      SinePositionalEncoding)
from .dino import DINO

@MODELS.register_module()
class DetrSingleScaleEncoder(BaseModule):
    """Transformer Encoder for single-scale DINO baseline.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the encode layer.
        in_channels (List[int]): The input channels of the feature maps. Defaults to [256, 256, 256].
        use_encoder_idx (List[int]): The indices of the encoder layers to use. Defaults to [2].
        num_encoder_layers (int): The number of encoder layers. Defaults to 6.
        pe_temperature (float): The temperature of the positional encoding. Defaults to 10000.
    """

    def __init__(self,
                 layer_cfg: ConfigType,
                 in_channels: list = [256, 256, 256],
                 use_encoder_idx: list = [2],
                 num_encoder_layers: int = 6,
                 pe_temperature: float = 10000.0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature

        self.transformer_blocks = nn.ModuleList([
            DetrTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))
        ])

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.,
        device=None,
    ) -> Tensor:
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, ('Embed dimension must be divisible by 4 '
                                    'for 2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device)
        omega = temperature**(omega / -pos_dim)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pos_embd = [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ]
        return torch.cat(pos_embd, axis=1)[None, :, :]

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            inputs (tuple[Tensor]): Input features.

        Returns:
            tuple[Tensor]: Encoded features.
        """
        assert len(inputs) == len(self.in_channels)
        outs = list(inputs)

        # encoder
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = outs[enc_ind].shape[2:]
            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = outs[enc_ind].flatten(2).permute(0, 2,
                                                           1).contiguous()
            pos_embed = self.build_2d_sincos_position_embedding(
                w,
                h,
                embed_dim=self.in_channels[enc_ind],
                temperature=self.pe_temperature,
                device=src_flatten.device)
            memory = self.transformer_blocks[i](
                src_flatten, query_pos=pos_embed, key_padding_mask=None)
            outs[enc_ind] = memory.permute(0, 2, 1).contiguous().reshape(
                -1, self.in_channels[enc_ind], h, w)

        return tuple(outs)

@MODELS.register_module()
class DetrDINO(DINO):
    r"""Baseline DINO with a single-scale DETR Encoder.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrSingleScaleEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        
        # embed_dims from decoder because encoder might not expose it directly in the same way
        self.embed_dims = self.decoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer."""
        encoder_inputs_dict, decoder_inputs_dict = super().pre_transformer(
            mlvl_feats, batch_data_samples)
        
        # Inject mlvl_feats so our customized forward_encoder receives it
        encoder_inputs_dict['mlvl_feats'] = mlvl_feats
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, mlvl_feats: Tuple[Tensor],
                        **kwargs) -> Dict:
        """Forward with Transformer encoder."""
        mlvl_feats = self.encoder(mlvl_feats)

        feat_flatten = []
        for feat in mlvl_feats:
            batch_size, c, h, w = feat.shape
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            feat_flatten.append(feat)

        memory = torch.cat(feat_flatten, 1)

        encoder_outputs_dict = dict(
            memory=memory, memory_mask=kwargs.get('feat_mask', None), spatial_shapes=kwargs.get('spatial_shapes'))
        return encoder_outputs_dict
