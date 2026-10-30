import math
import time
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM as DWConv
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmcv.cnn import ConvModule
#from natten import NeighborhoodAttention2D, use_fused_na, use_gemm_na
from natten.functional import na2d, na2d_av, na2d_qk, natten2dav, natten2dqkrpb
# from timm.models.vision_transformer import _cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from natten.flops import qk_2d_rpb_flop, av_2d_flop, add_natten_handle
from triton.ops import attention

try:
    from csm_triton import CrossScanTriton, CrossMergeTriton
except:
    from .csm_triton import CrossScanTriton, CrossMergeTriton

import selective_scan_cuda_oflex
import select


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

class VSSM(nn.Module):

    def __init__(
            self,
            d_model=96,
            d_state=1,
            expansion_ratio=1,
            dt_rank="auto",
            norm_layer=LayerNorm2d,
            dropout=0.0,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            k_groups=16,
            **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)

        self.expansion_ratio = expansion_ratio

        if self.expansion_ratio != 1.0:
            self.proj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True)  # (K * D)

        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None,
                        backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)

    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)

    def forward(self, x, to_dtype=False, force_fp32=False):
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        # xs = torch.stack([x, x.flip([-1])], dim=1).reshape(B, -1, L)
        # xs = self._cross_scan(x).reshape(B, -1, L)
        if self.expansion_ratio != 1.0:
            x = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            xs = self._cross_scan(x)  # size b, 4, embed_dim, L
            xs = xs.reshape(B, -1, L)
        else:
            xs = self._cross_scan(x).reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)

        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys = self._selective_scan(xs, dts, As, Bs,
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)

        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, L)

        if self.expansion_ratio != 1.0:
            y = self.yproj(y.permute(0, 2, 1)).permute(0, 2, 1)

        if to_dtype:
            y = y.to(x.dtype)

        return y
def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    print(x.shape)
    print(sin.shape)
    print(cos.shape)
    return (x * cos) + (rotate_every_two(x) * sin)

# def theta_shift_3d(x, sin, cos):
#     """
#     三维旋转位置编码
#     x: (B, N, S, D) 其中S=D*H*W展开的序列长度
#     sin/cos: (D, H, W, dim)
#     """
#     x = x.view(*x.shape[:-1], -1, 3)  # 拆分为三个空间维度
#     x_rot = torch.stack([
#         x[..., 0] * cos[..., 0::3] - x[..., 1] * sin[..., 0::3],
#         x[..., 1] * cos[..., 1::3] + x[..., 0] * sin[..., 1::3],
#         x[..., 2] * cos[..., 2::3]
#     ], dim=-1)
#     return x_rot.flatten(-2)

class RoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        '''
        初始化函数，用于设置角度的计算公式并注册为buffer。

        参数：
        embed_dim (int): 嵌入维度的大小
        num_heads (int): 注意力头的数量

        解释：
        1. 通过 `torch.linspace` 生成从0到1均匀分布的数字，并根据这个数字来计算角度（angle）。
        2. `angle` 是基于位置的编码角度值，后续会用来生成位置的正弦和余弦值。
        '''
        super().__init__()
        # 根据给定的embed_dim和num_heads计算角度
        # 角度范围从 1.0 到 1/(10000^0.25)
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        # 将angle的形状扩展并重复，用于后续的位置编码计算
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # 将角度值注册为模块的buffer，避免在反向传播时更新它
        self.register_buffer('angle', angle)

    def forward(self, slen):
        '''
        前向传播函数，计算位置编码。

        参数：
        slen (tuple): 输入的形状 (h, w)，其中h是高度，w是宽度
            - h * w == l，表示位置编码的总长度

        返回：
        (sin, cos): 返回正弦和余弦的编码结果，分别是 `(h, w, d1)` 形状的张量
        '''
        # 获取h和w的索引（分别对应行和列的序列）
        index_h = torch.arange(slen[0]).to(self.angle)  # 高度（行）索引
        index_w = torch.arange(slen[1]).to(self.angle)  # 宽度（列）索引

        # 计算行的正弦值（sin_h）
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h, d1//2)
        # 计算列的正弦值（sin_w）
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w, d1//2)

        # 扩展正弦值的形状，重复以匹配宽度和高度
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h, w, d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h, w, d1//2)

        # 拼接行和列的正弦值
        sin = torch.cat([sin_h, sin_w], -1)  # (h, w, d1)

        # 计算行的余弦值（cos_h）
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h, d1//2)
        # 计算列的余弦值（cos_w）
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w, d1//2)

        # 扩展余弦值的形状，重复以匹配宽度和高度
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h, w, d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h, w, d1//2)

        # 拼接行和列的余弦值
        cos = torch.cat([cos_h, cos_w], -1)  # (h, w, d1)

        # 返回正弦和余弦值
        return (sin, cos)


class RoPE3D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        三维旋转位置编码 (Rotary Position Embedding)
        参数：
            embed_dim: 总嵌入维度 (需能被3*num_heads整除)
            num_heads: 注意力头数
        """
        super().__init__()
        # 计算每个维度的编码维度
        dim_per_axis = embed_dim // (3 * num_heads) * 2  # 每个空间轴分配相同维度
        self.dim_per_axis = dim_per_axis

        # 生成基础频率 (覆盖三个空间维度)
        theta = 1.0 / (10000 ** torch.linspace(0, 1, dim_per_axis // 2))
        self.register_buffer('theta', theta)

    def forward(self, slen):
        """
        生成三维位置编码
        参数：
            slen: 空间维度元组 (depth, height, width)
        返回：
            (sin, cos): 形状为 (D, H, W, embed_dim)
        """
        D, H, W = slen
        device = self.theta.device

        # 为三个空间维度生成位置索引
        grid_d = torch.arange(D, device=device)
        grid_h = torch.arange(H, device=device)
        grid_w = torch.arange(W, device=device)

        # 生成各维度的旋转分量
        def get_rot_component(grid, theta):
            pos = grid[:, None] * theta[None, :]  # (L, d/2)
            return torch.cat([pos.sin(), pos.cos()], dim=-1)  # (L, d)

        # 各维度独立编码
        rot_d = get_rot_component(grid_d, self.theta)  # (D, d)
        rot_h = get_rot_component(grid_h, self.theta)  # (H, d)
        rot_w = get_rot_component(grid_w, self.theta)  # (W, d)

        # 组合三维编码 (使用爱因斯坦求和约定)
        sin = torch.einsum('d i, h j, w k -> d h w i j k',
                           rot_d[..., :self.dim_per_axis // 2],
                           rot_h[..., :self.dim_per_axis // 2],
                           rot_w[..., :self.dim_per_axis // 2])
        sin = sin.reshape(D, H, W, -1)  # 展平最后三个维度

        cos = torch.einsum('d i, h j, w k -> d h w i j k',
                           rot_d[..., self.dim_per_axis // 2:],
                           rot_h[..., self.dim_per_axis // 2:],
                           rot_w[..., self.dim_per_axis // 2:])
        cos = cos.reshape(D, H, W, -1)

        return sin, cos


def theta_shift_3d(x, sin, cos):
    """
    三维旋转位置编码应用
    参数：
        x: 输入张量 (B, N, D, H, W, C)
        sin/cos: 位置编码 (D, H, W, C)
    """
    # 添加广播维度
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W, C)
    cos = cos.unsqueeze(0).unsqueeze(0)

    # 分解通道为三个空间分量
    x_d, x_h, x_w = torch.chunk(x, 3, dim=-1)

    # 应用旋转矩阵
    rot_d = x_d * cos[..., :x_d.size(-1)] + rotate_half(x_d) * sin[..., :x_d.size(-1)]
    rot_h = x_h * cos[..., x_d.size(-1):x_d.size(-1) + x_h.size(-1)] + rotate_half(x_h) * sin[...,
                                                                                          x_d.size(-1):x_d.size(
                                                                                              -1) + x_h.size(-1)]
    rot_w = x_w * cos[..., x_d.size(-1) + x_h.size(-1):] + rotate_half(x_w) * sin[..., x_d.size(-1) + x_h.size(-1):]

    return torch.cat([rot_d, rot_h, rot_w], dim=-1)


def rotate_half(x):
    """将输入的后半部分取反实现旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    def __init__(self,
                 embed_dim,  # 嵌入维度
                 num_heads,  # 注意力头的数量
                 window_size,  # 窗口大小
                 window_dilation,  # 窗口扩张
                 global_mode=False,  # 是否使用全局模式
                 image_size=None,  # 图像尺寸（仅在全局模式下使用）
                 use_rpb=False,  # 是否使用相对位置偏置
                 sr_ratio=1,  # 空间缩减比例
                 fused_na=False,  # 是否融合非局部操作
                 ssm_ratio=1,
                 temporal_stride=1):  # 空间记忆比例

        super().__init__()
        window_size = to_3tuple(window_size)  # 转换为二维元组
        window_dilation = to_3tuple(window_dilation)  # 转换为二维元组

        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头的数量
        self.temporal_stride = temporal_stride
        self.head_dim = self.embed_dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子
        self.window_size = window_size  # 窗口大小
        self.window_dilation = window_dilation  # 窗口扩张
        self.global_mode = global_mode  # 是否使用全局模式
        self.sr_ratio = sr_ratio  # 空间缩减比例
        self.image_size = image_size  # 图像尺寸
        self.fused_na = fused_na  # 是否融合非局部操作
        self.rope = RoPE3D(embed_dim, num_heads)
        # 定义卷积层用于生成查询、键、值
        self.qkv = nn.Conv3d(embed_dim, embed_dim*3 , kernel_size=1)
        # 定义卷积层用于局部位置编码
        self.lepe = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        # 定义卷积层用于输出投影
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # 如果不使用全局模式，则定义额外的卷积层和SSM模块
        if not global_mode:
            self.dwconv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.BatchNorm2d(embed_dim),
            )
            self.ssm = VSSM(d_model=embed_dim, expansion_ratio=ssm_ratio)  # 空间记忆模块
            self.norm = LayerNorm2d(embed_dim)  # 层归一化

        # 如果使用相对位置偏置，则定义相关参数
        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads, (2 * window_size[0] - 1), (2 * window_size[1] - 1)),
                                     requires_grad=True)]
            if global_mode:
                rpb_list.append(
                    nn.Parameter(torch.empty(num_heads, image_size[0] * image_size[1], image_size[0] * image_size[1]),
                                 requires_grad=True))

            self.rpb = nn.ParameterList(rpb_list)

        # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Xavier初始化卷积层权重
        nn.init.xavier_normal_(self.qkv.weight, gain=2 ** -2.5)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_normal_(self.proj.weight, gain=2 ** -2.5)
        nn.init.zeros_(self.proj.bias)

        # 如果使用相对位置偏置，则初始化
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item)  # 可以考虑使用 `trunc_normal_` 初始化

    def forward(self, x, pos_enc):
        B, C, D, H, W = x.shape  # 输入形状 (Batch, Channel, Depth, Height, Width)
        x = x.permute(0, 4, 1, 2, 3)
        # 处理时空下采样
        if self.temporal_stride > 1:
            x = x[:, :, ::self.temporal_stride, :, :]
            D = x.shape[2]

        # QKV投影
        qkv = self.qkv(x)  # (B, 3*C, D, H, W)
        print(qkv.shape)
        # lepe = self.lepe(qkv[:, C:, ...])  # (B, C, D, H, W)
        # print(lepe.shape)

        # 拆分QKV并处理多头
        q, k, v = rearrange(qkv, 'b (m c) d h w -> m b d h w c', m=3)
        q = rearrange(q, 'b d h w (n g) -> b n (d h w) g', n=self.num_heads)
        k = rearrange(k, 'b d h w (n g) -> b n (d h w) g', n=self.num_heads)
        v = rearrange(v, 'b d h w (n g) -> b n (d h w) g', n=self.num_heads)


        # 3D位置编码
        sin, cos = self.rope((D, H, W))  # 生成3D位置编码
        q = theta_shift_3d(q, sin, cos) * self.scale
        k = theta_shift_3d(k, sin, cos)

        # 如果使用相对位置偏置，则提取
        if hasattr(self, 'rpb'):
            rpb = self.rpb[0]
        else:
            rpb = None

        # 如果启用融合的非局部操作，则进行处理
        if self.fused_na:
            q = rearrange(q, 'b n h w c -> b h w n c')
            k = rearrange(k, 'b n h w c -> b h w n c')
            v = rearrange(v, 'b n h w c -> b h w n c')

            # 使用融合的非局部操作
            x = na2d(q, k, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation,
                     scale=float(q.size(-1) ** 0.5))
            q = rearrange(q, 'b h w n c -> b n h w c')
            k = rearrange(k, 'b h w n c -> b n h w c')
            x = rearrange(x, 'b h w n c -> b n h w c')

        else:
            # 计算注意力
            attn = na2d_qk(q, k, kernel_size=toodd(self.window_size), dilation=self.window_dilation, rpb=rpb)
            attn = torch.softmax(attn, dim=-1)  # 计算注意力分数
            # 使用注意力对值进行加权
            x = na2d_av(attn, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation)

        # 如果不使用全局模式，则执行进一步的处理
        if not self.global_mode:
            q = rearrange(q, 'b n h w c -> b n c h w')
            k = rearrange(k, 'b n h w c -> b n c h w')
            v = rearrange(x, 'b n h w c -> b n c h w')

            v_r = v.flatten(1, 2)
            v = self.dwconv(v_r)  # 使用深度卷积进行处理
            v = F.silu(v)  # 激活函数
            v = self.ssm(v)  # 空间记忆处理

            v = self.norm(v.reshape(B, -1, H, W))  # 归一化
            # 将处理后的结果与原值相加
            x = v + v_r

        else:
            # 全局模式下，查询、键、值的形状变化
            q = rearrange(q, 'b n h w c -> b n (h w) c')
            k = rearrange(k, 'b n h w c -> b n (h w) c')
            v = rearrange(x, 'b n h w c -> b n (h w) c')

            # 计算全局注意力
            attn = einsum(q, k, 'b n l c, b n m c -> b n l m')

            if hasattr(self, 'rpb'):
                attn = attn + self.rpb[-1]  # 添加相对位置偏置

            attn = torch.softmax(attn, dim=-1)  # 计算注意力分数
            x = einsum(attn, v, 'b n l m, b n m c -> b n c l').reshape(B, -1, H, W)

        # 加入局部位置编码，并通过投影层得到最终结果
        # x = x + lepe
        x = self.proj(x)

        return x  # 返回最终输出


def toodd(size):
    size = to_2tuple(size)
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size

# if __name__ == '__main__':
#     # 初始化 Attention 模块
#     attention = Attention(
#         embed_dim=64,          # 嵌入维度
#         num_heads=2,           # 注意力头的数量
#         window_size=7,         # 窗口大小
#         window_dilation=1,     # 窗口扩张
#         global_mode=False,     # 是否使用全局模式
#         image_size=(224, 224), # 图像尺寸（仅在全局模式下使用）
#         use_rpb=False,          # 是否使用相对位置偏置
#         sr_ratio=1,            # 空间缩减比例
#         fused_na=False,        # 是否融合非局部操作
#         ssm_ratio=1            # 空间记忆比例
#     ).to('cuda')
#     x=torch.rand(4, 64, 224, 224)
#     # 创建一个假设的输入张量 (batch_size, channels, height, width)
#     input_tensor = x.to('cuda')  # batch_size=4, channels=96, height=224, width=224
#     # 假设的正弦和余弦位置编码 (num_heads, height, width)
#     #pos_enc = (torch.randn(8, 224, 224).to('cuda'), torch.randn(8, 224, 224).to('cuda'))
#     pos_enc = attention.rope((x.shape[2:]))
#     # 前向传播
#     output = attention(input_tensor, pos_enc)
#     # 输出尺寸
#     print("Input size:", input_tensor.size())
#     print("Output size:", output.size())
