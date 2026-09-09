import torch
from torch import nn
from torch import Tensor
import math
import numpy as np
from torch.nn import functional as F
import numbers
from einops import rearrange
from timm.models.layers import DropPath
from torch.nn.init import trunc_normal_
from torch.nn import init


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False


def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while int(factor) != factor:
        start += 1
        factor = integer / start
    return int(factor), start

def activation(act='ReLU'):
    if act == 'ReLU':
        return nn.ReLU()
    elif act == 'LeakyReLU':
        return nn.LeakyReLU()
    elif act == 'ELU':
        return nn.ELU()
    elif act == 'PReLU':
        return nn.PReLU()
    else:
        return nn.Identity()


def norm_layer3d(norm_type, num_features):
    if norm_type == 'batchnorm':
        return nn.BatchNorm3d(num_features=num_features, momentum=0.05)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm3d(num_features=num_features)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=num_features // 4, num_channels=num_features)
    else:
        return nn.Identity()


class StdConv3d(nn.Conv3d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, eps=1e-6):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.view(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class StdConvTranspose3d(nn.ConvTranspose3d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, eps=1e-6):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size,num_spatial_dims=3)
        weight = F.batch_norm(
            self.weight.permute(1, 0, 2, 3, 4).reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).permute(1, 0, 2).reshape_as(self.weight)
        x = F.conv_transpose3d(
            x, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        return x


class ClsHead(nn.Module):
    def __init__(self, in_channels, num_anchors=1, num_classes=1, feature_size=96, conv_num=2,
                 norm_type='groupnorm', act_type='ReLU'):
        super(ClsHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        cls_convs = []
        for i in range(conv_num):
            if i == 0:
                cls_convs.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                cls_convs.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_output = nn.Conv3d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.cls_act = nn.Sigmoid()

    def forward(self, x):
        x_cls = self.cls_convs(x)
        x_cls = self.cls_act(self.cls_output(x_cls))

        # out is B x C x Z x Y x X, with C = n_classes * n_anchors
        x_cls = x_cls.permute(0, 2, 3, 4, 1)
        batch_size, zz, yy, xx, channels = x_cls.shape
        x_cls = x_cls.view(batch_size, zz, yy, xx, self.num_anchors, self.num_classes)

        return x_cls.contiguous().view(x_cls.shape[0], -1, self.num_classes)


class RegHead(nn.Module):
    def __init__(self, in_channels, num_anchors=1, num_classes=1, feature_size=96, conv_num=2,
                 norm_type='groupnorm', act_type='ReLU'):
        super(RegHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        reg_convs = []
        for i in range(conv_num):
            if i == 0:
                reg_convs.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                reg_convs.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_output = nn.Conv3d(feature_size, num_anchors * 6, kernel_size=3, padding=1)

    def forward(self, x):
        x_reg = self.reg_convs(x)
        x_reg = self.reg_output(x_reg)
        x_reg = x_reg.permute(0, 2, 3, 4, 1)

        return x_reg.contiguous().view(x.shape[0], -1, 6)


class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, feature_size=16,
                 norm_type='groupnorm', act_type='ReLU'):
        super(SegHead, self).__init__()

        self.num_classes = num_classes

        self.up_conv = UpsamplingDeconvBlock(in_channels, feature_size, stride=2,
                                             norm_type=norm_type, act_type=act_type)
        self.seg_output = nn.Conv3d(feature_size, num_classes, kernel_size=3, padding=1)
        self.seg_act = nn.Sigmoid()

    def forward(self, x):
        x_seg = self.up_conv(x)
        x_seg = self.seg_act(self.seg_output(x_seg))
        return x_seg


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1,
                 norm_type='groupnorm', act_type='ReLU'):
        super(ConvBlock, self).__init__()

        self.conv = StdConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                              padding=kernel_size // 2 + dilation - 1, dilation=dilation, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class HeightPositionEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_encoding = None

    def forward(self, feature_map):
        B, C, D, H, W = feature_map.shape

        if C != self.channels:
            raise ValueError(f"Input channels {C} do not match expected channels {self.channels}")

        if self.pos_encoding is None or self.pos_encoding.shape[3] != H:
            self.pos_encoding = nn.Parameter(torch.zeros(1, self.channels, 1, H, 1).to(feature_map.device))
            trunc_normal_(self.pos_encoding, std=0.02)

        pos_enc_expanded = self.pos_encoding.expand(B, self.channels, D, -1, W)

        return feature_map + pos_enc_expanded

class WidthPositionEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_encoding = None

    def forward(self, feature_map):
        B, C, D, H, W = feature_map.shape

        if C != self.channels:
            raise ValueError(f"Input channels {C} do not match expected channels {self.channels}")

        if self.pos_encoding is None or self.pos_encoding.shape[4] != W:
            self.pos_encoding = nn.Parameter(torch.zeros(1, self.channels, 1, 1, W).to(feature_map.device))
            trunc_normal_(self.pos_encoding, std=0.02)

        pos_enc_expanded = self.pos_encoding.expand(B, self.channels, D, H, -1)

        return feature_map + pos_enc_expanded

class PEConv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.hconv = HeightPositionEncoding(self.dim_conv3)
        self.wconv = WidthPositionEncoding(self.dim_untouched)


    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.hconv(x1)
        x2 = self.wconv(x2)
        x = torch.cat((x1, x2), 1)
        return x

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_type='groupnorm', act_type='ReLU'):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               act_type=act_type, norm_type=norm_type)

        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, stride=1,
                               act_type='none', norm_type=norm_type)

        if in_channels == out_channels and stride == 1:
            self.res = nn.Identity()
        elif in_channels != out_channels and stride == 1:
            self.res = ConvBlock(in_channels, out_channels, kernel_size=1, act_type='none', norm_type=norm_type)
        elif in_channels != out_channels and stride > 1:
            self.res = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                ConvBlock(in_channels, out_channels, kernel_size=1, act_type='none', norm_type=norm_type))

        self.act = activation(act_type)
        self.peconv = PEConv(in_channels)

    def forward(self, x):
        identity = self.res(x)

        x = self.peconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        x = self.act(x)
        return x


class LayerBasic(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels, stride=1, norm_type='groupnorm', act_type='ReLU'):
        super(LayerBasic, self).__init__()
        self.n_stages = n_stages
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = in_channels
                stride = stride
            else:
                input_channel = out_channels
                stride = 1
            ops.append(BasicBlock(input_channel, out_channels, stride=stride, norm_type=norm_type, act_type=act_type))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature):
        B, C, D, H, W = Q_feature.shape
        x = Q_feature

        Q_ch = x.mean(dim=(2, 3, 4))  # [B, C]
        K_ch = x.mean(dim=(2, 3, 4))  # [B, C]
        V_ch = x.mean(dim=(2, 3, 4))  # [B, C]

        Q = self.query(Q_ch)  # [B, C]
        K = self.key(K_ch)  # [B, C]
        V = self.value(V_ch)  # [B, C]

        d_k = C ** 0.5
        attention_scores = torch.matmul(Q.unsqueeze(-1), K.unsqueeze(1)) / d_k  # [B, C, C]
        attention_weights = self.softmax(attention_scores)

        attended_channels = torch.matmul(attention_weights, V.unsqueeze(-1)).squeeze(-1)  # [B, C]


        out = x + x * attended_channels.view(B, C, 1, 1, 1) # 方案A: 门控+残差
        return out

class SpatialSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SpatialSelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, Q_feature):
        B, C, D, H, W = Q_feature.shape
        L = D * H * W
        x = Q_feature

        Q_flat = x.permute(0, 2, 3, 4, 1).reshape(B, L, C)
        K_flat = x.permute(0, 2, 3, 4, 1).reshape(B, L, C)

        Q_flat = self.norm(Q_flat)
        K_flat = self.norm(K_flat)

        Q = self.query(Q_flat)  # [B, L, C]
        K = self.key(K_flat)  # [B, L, C]
        V = self.value(K_flat)  # [B, L, C]

        d_k = C ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # [B, L, L]
        attention_weights = self.softmax(attention_scores)
        attended_features = torch.matmul(attention_weights, V)  # [B, L, C]

        out_flat = attended_features.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        out = x + out_flat  # [B, C, D, H, W]
        return out


class PerformanceFirstBottleneck(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_refine = ChannelSelfAttention(in_channels)
        self.spatial_transform = SpatialSelfAttention(in_channels)
        self.residual_conv = nn.Identity()

    def forward(self, x):
        x_res = self.residual_conv(x)
        x_refined = self.channel_refine(x)
        x_transformed = self.spatial_transform(x_refined)
        out = x_res + x_transformed
        return out

class DownsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_type='groupnorm', act_type='ReLU'):
        super(DownsamplingConvBlock, self).__init__()

        self.conv = StdConv3d(in_channels, out_channels, kernel_size=2, padding=0, stride=stride, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=2, pool_type='max',
                 norm_type='groupnorm', act_type='ReLU'):
        super(DownsamplingBlock, self).__init__()

        if pool_type == 'avg':
            self.down = nn.AvgPool3d(kernel_size=stride, stride=stride)
        else:
            self.down = nn.MaxPool3d(kernel_size=stride, stride=stride)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.down(x)
        if hasattr(self, 'conv'):
            x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_type='groupnorm', act_type='ReLU'):
        super(UpsamplingDeconvBlock, self).__init__()

        self.conv = StdConvTranspose3d(in_channels, out_channels, kernel_size=stride, padding=0, stride=stride,
                                       bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=2, mode='trilinear', norm_type='groupnorm',
                 act_type='ReLU'):
        super(UpsamplingBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=stride, mode=mode)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        if hasattr(self, 'conv'):
            x = self.conv(x)
        x = self.up(x)
        return x

class GroupBatchNorm3d(nn.Module):

    def __init__(self, num_channels: int,
                 num_groups: int = 16,
                 eps: float = 1e-10):
        super().__init__()
        assert num_channels >= num_groups
        self.num_groups = num_groups
        self.scale = nn.Parameter(torch.randn(num_channels, 1, 1, 1))
        self.shift = nn.Parameter(torch.zeros(num_channels, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        grouped_feat = x.view(B, self.num_groups, -1)

        group_mean = grouped_feat.mean(dim=2, keepdim=True)
        group_std = grouped_feat.std(dim=2, keepdim=True)

        normalized_feat = (grouped_feat - group_mean) / (group_std + self.eps)
        normalized_feat = normalized_feat.view(B, C, D, H, W)

        return normalized_feat * self.scale + self.shift

class InfoGatedReconstructUnit(nn.Module):

    def __init__(self,
                 in_channels: int,
                 gate_threshold: float = 0.5):
        super().__init__()
        self.group_norm = GroupBatchNorm3d(in_channels, num_groups=in_channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_feat = self.group_norm(x)

        channel_weights = self.group_norm.scale / self.group_norm.scale.sum()

        info_scores = self.sigmoid(norm_feat * channel_weights)

        high_info_mask = info_scores >= self.gate_threshold
        low_info_mask = info_scores < self.gate_threshold

        high_info_feat = high_info_mask * x
        low_info_feat = low_info_mask * x
        x1_proj = self.conv1(high_info_feat)
        x2_proj = self.conv2(low_info_feat)
        z = torch.sigmoid(x1_proj + x2_proj)
        out = z * high_info_feat + (1 - z) * low_info_feat

        return out
class VarianceAttentionModule(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        spatial_var = torch.var(x, dim=(-3, -2, -1), keepdim=True).pow(2)  # shape: [B, C, 1, 1, 1]

        global_var_norm = spatial_var.sum(dim=[2, 3, 4], keepdim=True) / (D * H * W - 1 + self.eps)

        attention_coef = spatial_var / (4 * (global_var_norm + self.eps)) + 0.5

        attention_weight = self.sigmoid(attention_coef)  # shape: [B, C, 1, 1, 1]
        return x * attention_weight
class ASPP2(nn.Module):
    def __init__(self, channels, out_channels, ratio=4,
                 dilations=[1, 2, 3, 4, 5],
                 norm_type='groupnorm', act_type='ReLU'):
        super(ASPP2, self).__init__()
        inner_channels = channels // ratio
        cat_channels = inner_channels * 5
        self.aspp0 = ConvBlock(channels, inner_channels, kernel_size=1,
                               dilation=dilations[0], norm_type=norm_type, act_type=act_type)
        self.aspp1 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[1], norm_type=norm_type, act_type=act_type)
        self.aspp2 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[2], norm_type=norm_type, act_type=act_type)
        self.aspp3 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[3], norm_type=norm_type, act_type=act_type)
        self.aspp4 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[4], norm_type=norm_type, act_type=act_type)
        self.transition = ConvBlock(cat_channels, out_channels, kernel_size=1,
                                    dilation=dilations[0], norm_type=norm_type, act_type=act_type)
        self.asp_vam = VarianceAttentionModule()

    def forward(self, input):
        aspp0 = self.aspp0(input)
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        aspp4 = self.aspp4(input)

        out = torch.cat((self.asp_vam(aspp0), self.asp_vam(aspp1), self.asp_vam(aspp2), self.asp_vam(aspp3), self.asp_vam(aspp4)), dim=1)
        out = self.transition(out)
        return out