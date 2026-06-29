import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, det, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

class FEAdaptive(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FEAdaptive, self).__init__()
        # Shared RGBD and BN
        self.rgdb = RGBD(in_channels)
        self.spatial_rout = SpatialRouting(feature_dim=in_channels, label_nc=num_classes)
        self.norm0 = nn.BatchNorm2d(in_channels)

    def forward(self, features, spatial_shapes, g_map=None):
        x0, x1, x2 = decouple_value_to_features(features, spatial_shapes)
        x0 = self.rgdb(x0)

        use_guidance = (g_map is not None) and torch.any(g_map)
        if use_guidance:
            g0, _, _ = decouple_value_to_features(g_map, spatial_shapes)
            x0 = self.spatial_rout(x0, g0)

        x0 = self.norm0(x0)
        features = combine_features_to_value((x0, x1, x2), spatial_shapes)
        return features

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels):
        super(RGBD, self).__init__()
        self.dense = ConvLeakyRelu2d(in_channels, in_channels)
        self.sobelconv = Sobelxy(in_channels)
    def forward(self,x):
        x1=self.dense(x)
        x2=self.sobelconv(x)
        return F.leaky_relu(x1+x2,negative_slope=0.1)
    
class SpatialRouting(nn.Module):
    def __init__(self, feature_dim, label_nc=1):
        super(SpatialRouting, self).__init__()
        # We aggregate label channels by mean, so projection expects a single-channel mask
        self.proj = DepthwiseSeparableConv(feature_dim, feature_dim, dilation=2)

    def forward(self, x, obj_mask): # (b,dim,h,w), (b, label_nc, h, w)
        # Average across label dimension to get a single-channel guidance map
        # obj_mask_mean: (b, 1, h, w)
        obj_mask = obj_mask.mean(dim=1, keepdim=True)
        # Min-max normalize per sample to [0,1]
        B, _, H, W = obj_mask.shape
        mask_flat = obj_mask.view(B, 1, -1)
        min_vals = mask_flat.min(dim=2, keepdim=True)[0]
        max_vals = mask_flat.max(dim=2, keepdim=True)[0]
        denom = (max_vals - min_vals).clamp(min=1e-6)
        obj_mask = (mask_flat - min_vals) / denom
        obj_mask = obj_mask.view(B, 1, H, W)

        masked_x = x * obj_mask
        return self.proj(masked_x) + x

    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        assert isinstance(in_ch, int), "Input channels must be integer"
        
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_ch
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1
        )
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, lrelu=True):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
        self.lrelu = lrelu

    def forward(self,x):
        # print(x.size())
        if self.lrelu:
            return F.leaky_relu(self.conv(x), negative_slope=0.2, inplace=True)
        else:
            return self.conv(x)
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale

        
def decouple_value_to_features(
    value: torch.Tensor,
    src_shape: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decouple flattened value tensor into three feature levels.
    
    Args:
        value (Tensor): The input value tensor with shape (B, num_value, C)
        src_shape (Tensor): The spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).
            
    Returns:
        Tuple[Tensor, Tensor, Tensor]: Three feature tensors with shapes:
            - (B, C, H1, W1) for first level
            - (B, C, H2, W2) for second level
            - (B, C, H3, W3) for third level
    """
    B, L, C = value.shape
    num_levels = src_shape.size(0)
    
    # Calculate start indices for each level
    level_start_index = torch.cat((
        src_shape.new_zeros((1, )),
        src_shape.prod(1).cumsum(0)[:-1]))
    
    # Split value into separate levels
    features = []
    for lvl in range(num_levels):
        h, w = src_shape[lvl].tolist()
        start_idx = level_start_index[lvl]
        end_idx = start_idx + h * w
        
        # Extract and reshape the feature for this level
        feat = value[:, start_idx:end_idx, :]  # (B, H*W, C)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, C, H*W)
        feat = feat.view(B, C, h, w)  # (B, C, H, W)
        features.append(feat)
    
    return tuple(features)

def combine_features_to_value(
    features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    src_shape: torch.Tensor,
) -> torch.Tensor:
    """Combine three feature levels into a single flattened value tensor.
    
    Args:
        features (Tuple[Tensor, Tensor, Tensor]): Three feature tensors with shapes:
            - (B, C, H1, W1) for first level
            - (B, C, H2, W2) for second level
            - (B, C, H3, W3) for third level
        src_shape (Tensor): The spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).
            
    Returns:
        Tensor: The combined value tensor with shape (B, num_value, C)
    """
    B, C, _, _ = features[0].shape
    num_levels = src_shape.size(0)
    
    # Flatten each feature level
    flattened_features = []
    for lvl in range(num_levels):
        feat = features[lvl]  # (B, C, H, W)
        h, w = src_shape[lvl].tolist()
        feat = feat.view(B, C, -1)  # (B, C, H*W)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, H*W, C)
        flattened_features.append(feat)
    
    # Concatenate all flattened features
    value = torch.cat(flattened_features, dim=1)  # (B, num_value, C)
    return value

def gen_class_guidance_map_max_prob_threshold(
    output_cls: torch.Tensor,
    reference_points: torch.Tensor,
    src_size: list,
    src_shape: torch.Tensor,
    threshold: float = None,
    topk: Optional[Union[int, float]] = 10,
    gaussian_sigma: float = 0.9,
) -> torch.Tensor:
    """
    Generate class guidance maps with Gaussian falloff inside each query box.
    Queries are selected either by top-k or by probability threshold.

    Args:
        output_cls:    (B, num_queries, num_classes)
        reference_points: (B, num_queries, 4) in (cx, cy, w, h), normalized [0,1]
        src_size:      list of len B, src_size[b][lvl] = [h_feat, w_feat]
        src_shape:     (num_levels, 2), each row = (H_lvl, W_lvl)
        topk:          if int/float, use top-k selection; if None, use thresholding
        threshold:     probability threshold if topk is None
        gaussian_sigma: sigma for Gaussian falloff (lower = sharper center)

    Returns:
        Tensor of shape (B, L, num_classes)
    """
    with torch.no_grad():
        B, num_queries, num_classes = output_cls.shape
        num_levels = src_shape.size(0)
        device = output_cls.device
        per_level_guidance = []

        for lvl in range(num_levels):
            H_lvl, W_lvl = src_shape[lvl].tolist()
            guidance_lvl = output_cls.new_zeros((B, num_classes, H_lvl, W_lvl))

            for b in range(B):
                h_feat, w_feat = src_size[b][lvl]
                ref_b = reference_points[b]
                ref_xyxy = bbox_cxcywh_to_xyxy(ref_b)

                # Scale to feature map coordinates
                xmin = torch.clamp(torch.floor(ref_xyxy[:, 0] * w_feat), min=0, max=w_feat - 1).long()
                ymin = torch.clamp(torch.floor(ref_xyxy[:, 1] * h_feat), min=0, max=h_feat - 1).long()
                xmax = torch.clamp(torch.ceil(ref_xyxy[:, 2] * w_feat), min=1, max=w_feat).long()
                ymax = torch.clamp(torch.ceil(ref_xyxy[:, 3] * h_feat), min=1, max=h_feat).long()

                # Coordinate grids
                y_grid = torch.arange(h_feat, device=device).view(1, -1, 1)
                x_grid = torch.arange(w_feat, device=device).view(1, 1, -1)

                xmin_exp = xmin.view(-1, 1, 1)
                xmax_exp = xmax.view(-1, 1, 1)
                ymin_exp = ymin.view(-1, 1, 1)
                ymax_exp = ymax.view(-1, 1, 1)

                x_mask = (x_grid >= xmin_exp) & (x_grid < xmax_exp)
                y_mask = (y_grid >= ymin_exp) & (y_grid < ymax_exp)
                box_mask = y_mask & x_mask

                valid_boxes = (xmax > xmin) & (ymax > ymin)
                box_mask = box_mask & valid_boxes.view(-1, 1, 1)

                # Max class + prob
                max_probs, max_class_idx = output_cls[b].max(dim=1)

                # Selection (topk or threshold)
                max_class_one_hot = torch.zeros((num_queries, num_classes), device=device)
                if topk is not None:
                    if isinstance(topk, float):
                        k = math.ceil(num_queries * max(0.0, min(1.0, topk)))
                    else:
                        k = int(topk)
                    k = max(0, min(k, num_queries))
                    if k > 0:
                        _, topk_idx = torch.topk(max_probs, k, largest=True, sorted=False)
                        max_class_one_hot[topk_idx, max_class_idx[topk_idx]] = 1.0
                else:
                    valid_queries = max_probs >= threshold
                    if valid_queries.any():
                        idx = torch.where(valid_queries)[0]
                        max_class_one_hot[idx, max_class_idx[idx]] = 1.0

                # Apply probability weight
                max_class_probs = max_class_one_hot * max_probs.unsqueeze(1)

                # Gaussian weighting
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                cx_exp = center_x.view(-1, 1, 1)
                cy_exp = center_y.view(-1, 1, 1)
                dx = (x_grid - cx_exp) / (xmax_exp - xmin_exp + 1e-6)
                dy = (y_grid - cy_exp) / (ymax_exp - ymin_exp + 1e-6)
                dist_sq = dx**2 + dy**2
                gaussian_falloff = torch.exp(-dist_sq / (2 * gaussian_sigma**2))

                box_mask_falloff = box_mask.float() * gaussian_falloff

                # Accumulate
                max_class_probs = max_class_probs.view(num_queries, num_classes, 1, 1)
                box_mask_falloff = box_mask_falloff.unsqueeze(1)
                guidance_small = (max_class_probs * box_mask_falloff).sum(dim=0)

                guidance_lvl[b, :, :h_feat, :w_feat] = guidance_small

            flat = guidance_lvl.view(B, num_classes, H_lvl * W_lvl)
            flat = flat.permute(0, 2, 1).contiguous()
            per_level_guidance.append(flat)

        return torch.cat(per_level_guidance, dim=1)

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

class ChannelAttention(nn.Module):
    def __init__(self, channels, k=16):
        super().__init__()
        self.channels = channels
        self.k = k
        # Grouped 1x1 convs for each path
        self.conv_gap = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.conv_gmp = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        # Final group 1x1 conv after concat
        self.conv_final = nn.Conv2d(2 * channels, channels, kernel_size=1, groups=channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        k = min(self.k, H, W)  # Ensure k does not exceed spatial dims
        # GAP and GMP to (B, C, k, k)
        gap = nn.functional.adaptive_avg_pool2d(x, (k, k))
        gmp = nn.functional.adaptive_max_pool2d(x, (k, k))
        # ReLU
        gap = nn.functional.relu(gap)
        gmp = nn.functional.relu(gmp)
        # Sum across spatial dims -> (B, C, 1, 1)
        gap_sum = gap.sum(dim=[2, 3], keepdim=True)
        gmp_sum = gmp.sum(dim=[2, 3], keepdim=True)
        # Grouped 1x1 convs (channel-wise)
        gap_score = self.conv_gap(gap_sum)
        gmp_score = self.conv_gmp(gmp_sum)
        # Concatenate along channel dim
        cat = torch.cat([gap_score, gmp_score], dim=1)  # (B, 2C, 1, 1)
        # Final group 1x1 conv
        attn = self.conv_final(cat)  # (B, C, 1, 1)
        # Optionally, apply sigmoid to get weights in [0, 1]
        attn = torch.sigmoid(attn)
        return attn

class SpatialAttention(nn.Module):
    def __init__(self,):
        super().__init__()
        kernel_size = 1
        self.compress = ChannelPool()
        self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale