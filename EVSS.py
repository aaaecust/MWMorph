import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.init import constant_
import numbers
from einops import rearrange, repeat
import math
import pkg_resources
import random
import pywt
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm import Mamba
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np
from typing import Callable
from timm.models.layers import DropPath, trunc_normal_, to_3tuple


class _ScaleModule3D(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dims) * init_scale)

    def forward(self, x):
        return x * self.scale



def create_3d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)

    # 直接创建8个3D滤波器，类似2D版本的4个滤波器
    dec_filters = []
    for i in range(2):  # D dimension: L/H
        for j in range(2):  # H dimension: L/H
            for k in range(2):  # W dimension: L/H
                d_filter = dec_lo if i == 0 else dec_hi
                h_filter = dec_lo if j == 0 else dec_hi
                w_filter = dec_lo if k == 0 else dec_hi

                # 使用外积创建3D滤波器，类似2D版本
                filter_3d = torch.einsum('d,h,w->dhw', d_filter, h_filter, w_filter)
                dec_filters.append(filter_3d)

    # 堆叠并扩展维度，类似2D版本
    dec_filters = torch.stack(dec_filters, dim=0)  # (8, d_size, h_size, w_size)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1, 1)  # (8*in_size, 1, d, h, w)

    # 重构滤波器，简化处理
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])

    rec_filters = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                d_filter = rec_lo if i == 0 else rec_hi
                h_filter = rec_lo if j == 0 else rec_hi
                w_filter = rec_lo if k == 0 else rec_hi

                filter_3d = torch.einsum('d,h,w->dhw', d_filter, h_filter, w_filter)
                rec_filters.append(filter_3d)

    rec_filters = torch.stack(rec_filters, dim=0)  # (8, d_size, h_size, w_size)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1, 1)  # (8*out_size, 1, d, h, w)

    return dec_filters, rec_filters


def wavelet_transform_3d(x, filters):
    """3D小波变换"""
    b, c, d, h, w = x.shape
    pad_d = (filters.shape[2] // 2 - 1,) * 2
    pad_h = (filters.shape[3] // 2 - 1,) * 2
    pad_w = (filters.shape[4] // 2 - 1,) * 2

    x = F.pad(x, pad_w + pad_h + pad_d)
    x = F.conv3d(x, filters, stride=2, groups=c)
    x = x.reshape(b, c, 8, d // 2, h // 2, w // 2)
    return x


def inverse_wavelet_transform_3d(x, filters):
    b, c, _, d_half, h_half, w_half = x.shape
    pad_d = (filters.shape[2] // 2 - 1,) * 2
    pad_h = (filters.shape[3] // 2 - 1,) * 2
    pad_w = (filters.shape[4] // 2 - 1,) * 2

    # 类似2D版本的简单重塑
    x = x.reshape(b, c * 8, d_half, h_half, w_half)
    x = F.conv_transpose3d(x, filters, stride=2, groups=c, padding=(pad_d[0], pad_h[0], pad_w[0]))
    return x


class _ScaleModule3D(nn.Module):
    """3D缩放模块"""

    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule3D, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)



class MBWTSS3D(nn.Module):
    _global_base_factorization = None
    _global_base_length = None
    _global_layer_dimensions = None
    _global_is_initialized = False

    def __init__(self,
                 d_model=96,
                 d_state=16,
                 d_conv=3,
                 expand=2,
                 wt_levels=1,
                 wt_type='haar',
                 kernel_size=3,
                 depths=[2,2,6,2],
                 stride=1,
                 **kwargs):
        super(MBWTSS3D, self).__init__()

        self.d_model = d_model
        self.wt_levels = wt_levels
        self.stride = stride
        # 小波变换滤波器
        self.wt_filter, self.iwt_filter = create_3d_wavelet_filter(
            wt_type, d_model, d_model, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform_3d, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_3d, filters=self.iwt_filter)
        self.depths = depths
        self.num_layers = len(depths)

        # SS3D全局注意力
        self.global_atten = SS3D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )

        self.base_scale = _ScaleModule3D([1, d_model, 1, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv3d(d_model * 8, d_model * 8, kernel_size,
                      padding=kernel_size // 2, stride=1, groups=d_model * 8, bias=False)
            for _ in range(self.wt_levels)
        ])

        self.wavelet_scale = nn.ModuleList([
            _ScaleModule3D([1, d_model * 8, 1, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

    def find_3d_factorization(self, length):

        cube_root = int(round(length ** (1 / 3)))
        if cube_root ** 3 == length:
            return (cube_root, cube_root, cube_root)

        min_padding = float('inf')
        best_factorization = None
        search_min = max(1, cube_root // 2)
        search_max = cube_root * 2

        for d in range(search_min, search_max + 1):
            remaining = length / d
            h = int(math.sqrt(remaining))
            w = math.ceil(remaining / h)
            padding = d * h * w - length

            # 如果找到更少的填充，更新最佳分解
            if padding < min_padding:
                min_padding = padding
                best_factorization = (d, h, w)

            h += 1
            w = math.ceil(remaining / h)
            padding = d * h * w - length

            if padding < min_padding:
                min_padding = padding
                best_factorization = (d, h, w)

        return best_factorization

    def forward(self, x):
        B, L, C = x.shape

        D, H, W = self.find_3d_factorization(L)
        total_length = D * H * W
        if total_length > L:
            padding_length = total_length - L
            padding = torch.zeros((B, padding_length, C), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :total_length, :]


        x_3d = x_padded.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x_3d


        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)


            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0) or (curr_shape[4] % 2 > 0):
                curr_pads = (0, curr_shape[4] % 2, 0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :, :]


            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 8, shape_x[3], shape_x[4], shape_x[5])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:8, :, :, :])


        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)


            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3], :curr_shape[4]]


        x_tag = next_x_ll.permute(0, 2, 3, 4, 1).contiguous()


        x_global = self.global_atten(x)


        x_global_3d = x_global.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x_global_scaled = self.base_scale(x_global_3d)
        x_global_final = x_global_scaled.permute(0, 2, 3, 4, 1)

        x = x_global_final + x_tag
        return x




class SS3D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = to_3tuple(d_conv)
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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
        elif dt_init == "normal":
            nn.init.normal_(dt_proj.weight, mean=dt_init_std, std=dt_init_std * 0.1)
        elif dt_init == "trunc_normal":
            # 使用截断正态分布，控制上下界，避免极端值
            nn.init.trunc_normal_(dt_proj.weight,
                                  mean=dt_init_std,
                                  std=dt_init_std * 0.2,  # 较小的标准差
                                  a=0,  # 下界为0，确保正值
                                  b=dt_init_std * 2)  # 上界控制在合理范围内
        elif dt_init == "orthogonal":
            nn.init.orthogonal_(dt_proj.weight)
            dt_proj.weight.data *= dt_init_std
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
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def find_3d_factorization(self, length):

        cube_root = int(round(length ** (1 / 3)))
        if cube_root ** 3 == length:
            return (cube_root, cube_root, cube_root)


        min_padding = float('inf')
        best_factorization = None
        search_min = max(1, cube_root // 2)
        search_max = cube_root * 2

        for d in range(search_min, search_max + 1):
            remaining = length / d
            h = int(math.sqrt(remaining))
            w = math.ceil(remaining / h)
            padding = d * h * w - length

            # 如果找到更少的填充，更新最佳分解
            if padding < min_padding:
                min_padding = padding
                best_factorization = (d, h, w)

            h += 1
            w = math.ceil(remaining / h)
            padding = d * h * w - length

            if padding < min_padding:
                min_padding = padding
                best_factorization = (d, h, w)

        return best_factorization

    def forward_corev0(self, x: torch.Tensor, spatial_shape=None):
        self.selective_scan = selective_scan_fn if x.is_cuda else selective_scan_ref

        B, C, D, H, W = x.shape
        L = D * H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)

        direction_features = [
            out_y[:, 0],
            inv_y[:, 0],
            wh_y,
            invwh_y
        ]

        y = sum(direction_features)
        y = torch.transpose(y, dim0=1, dim1=2)
        y = y.contiguous().view(B, D, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, L, C = x.shape

        D, H, W = self.find_3d_factorization(L)
        total_length = D * H * W

        if total_length > L:
            padding_length = total_length - L
            padding = torch.zeros((B, padding_length, C), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x

        x_3d = x_padded.view(B, D, H, W, C)
        xz = self.in_proj(x_3d)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))

        y = self.forward_core(x, spatial_shape=(D, H, W))
        gate = F.silu(z)
        y = y * gate
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class WaveletModule(nn.Module):


    def __init__(self, d_model=96, wt_levels=1, wt_type='haar', d_state=16):
        super().__init__()
        self.d_model = d_model
        self.wt_levels = wt_levels

        # 小波变换滤波器
        self.wt_filter, self.iwt_filter = create_3d_wavelet_filter(
            wt_type, d_model, d_model, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform_3d, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform_3d, filters=self.iwt_filter)

        # 小波处理层
        self.wavelet_convs = nn.ModuleList([
            nn.Conv3d(d_model * 8, d_model * 8, 3,
                      padding=1, stride=1, groups=d_model * 8, bias=False)
            for _ in range(self.wt_levels)
        ])

        self.wavelet_scale = nn.ModuleList([
            _ScaleModule3D([1, d_model * 8, 1, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

    def find_3d_factorization(self, length):
        cube_root = int(round(length ** (1 / 3)))
        if cube_root ** 3 == length:
            return (cube_root, cube_root, cube_root)

        min_padding = float('inf')
        best_factorization = None
        search_min = max(1, cube_root // 2)
        search_max = cube_root * 2

        for d in range(search_min, search_max + 1):
            remaining = length / d
            h = int(math.sqrt(remaining))
            w = math.ceil(remaining / h)
            padding = d * h * w - length

            if padding < min_padding:
                min_padding = padding
                best_factorization = (d, h, w)

            h += 1
            w = math.ceil(remaining / h)
            padding = d * h * w - length

            if padding < min_padding:
                min_padding = padding
                best_factorization = (d, h, w)

        return best_factorization




    def forward(self, x, H=None, W=None, T=None):
        B, L, C = x.shape

        if H is not None and W is not None and T is not None:
            D, H_use, W_use = T, H, W
        else:
            D, H_use, W_use = self.find_3d_factorization(L)

        total_length = D * H_use * W_use
        if total_length > L:
            padding_length = total_length - L
            padding = torch.zeros((B, padding_length, C), device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :total_length, :]

        x_3d = x_padded.view(B, D, H_use, W_use, C).permute(0, 4, 1, 2, 3).contiguous()

        # 小波变换处理
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x_3d

        # 前向小波分解
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0) or (curr_shape[4] % 2 > 0):
                curr_pads = (0, curr_shape[4] % 2, 0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 8, shape_x[3], shape_x[4], shape_x[5])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))

            #curr_x_tag = self.apply_sknet_fusion(curr_x_tag, i)

            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:8, :, :, :])

        # 逆向小波重构
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3], :curr_shape[4]]

        x_out = next_x_ll.permute(0, 2, 3, 4, 1).contiguous()
        return x_out
    # def __init__(self, d_model=96, levels=1, d_state=16):
    #     super().__init__()
    #     self.d_model = d_model
    #     self.levels = levels
    #
    #     # 多尺度卷积分支
    #     self.multi_scale_convs = nn.ModuleList()
    #     self.scale_modules = nn.ModuleList()
    #     self.residual_convs = nn.ModuleList()
    #
    #     for i in range(self.levels):
    #         # 主要卷积分支 - 使用不同kernel size模拟不同频率分量
    #         conv_branch = nn.ModuleList([
    #             # 低频分量 (大kernel)
    #             nn.Conv3d(d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=False),
    #             # 中频分量 (中等kernel)
    #             nn.Conv3d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
    #             # 高频分量 (小kernel)
    #             nn.Conv3d(d_model, d_model, kernel_size=1, padding=0, bias=False),
    #         ])
    #         self.multi_scale_convs.append(conv_branch)
    #
    #         # 特征融合卷积
    #         fusion_conv = nn.Conv3d(d_model * 3, d_model, kernel_size=1, bias=False)
    #         self.residual_convs.append(fusion_conv)
    #
    #         # 缩放模块
    #         scale_module = _ScaleModule3D([1, d_model, 1, 1, 1], init_scale=0.1)
    #         self.scale_modules.append(scale_module)
    #
    #     # 最终输出投影
    #     self.output_norm = nn.LayerNorm(d_model)
    #     self.output_proj = nn.Linear(d_model, d_model)
    #
    # def find_3d_factorization(self, length):
    #     """与原始模块相同的3D因式分解方法"""
    #     cube_root = int(round(length ** (1 / 3)))
    #     if cube_root ** 3 == length:
    #         return (cube_root, cube_root, cube_root)
    #
    #     min_padding = float('inf')
    #     best_factorization = None
    #     search_min = max(1, cube_root // 2)
    #     search_max = cube_root * 2
    #
    #     for d in range(search_min, search_max + 1):
    #         remaining = length / d
    #         h = int(math.sqrt(remaining))
    #         w = math.ceil(remaining / h)
    #         padding = d * h * w - length
    #
    #         if padding < min_padding:
    #             min_padding = padding
    #             best_factorization = (d, h, w)
    #
    #         h += 1
    #         w = math.ceil(remaining / h)
    #         padding = d * h * w - length
    #
    #         if padding < min_padding:
    #             min_padding = padding
    #             best_factorization = (d, h, w)
    #
    #     return best_factorization
    #
    # def forward(self, x, H=None, W=None, T=None):
    #     B, L, C = x.shape
    #
    #     # 与原始模块相同的reshape逻辑
    #     if H is not None and W is not None and T is not None:
    #         D, H_use, W_use = T, H, W
    #     else:
    #         D, H_use, W_use = self.find_3d_factorization(L)
    #
    #     total_length = D * H_use * W_use
    #     if total_length > L:
    #         padding_length = total_length - L
    #         padding = torch.zeros((B, padding_length, C), device=x.device, dtype=x.dtype)
    #         x_padded = torch.cat([x, padding], dim=1)
    #     else:
    #         x_padded = x[:, :total_length, :]
    #
    #     # 转换为3D张量
    #     x_3d = x_padded.view(B, D, H_use, W_use, C).permute(0, 4, 1, 2, 3).contiguous()
    #
    #     # 多尺度CNN处理
    #     residual = x_3d
    #
    #     for i in range(self.levels):
    #         # 多尺度卷积分支
    #         features = []
    #
    #         # 低频、中频、高频分量处理
    #         for conv in self.multi_scale_convs[i]:
    #             feat = conv(residual)
    #             feat = F.gelu(feat)  # 激活函数
    #             features.append(feat)
    #
    #         # 特征融合
    #         multi_scale_feat = torch.cat(features, dim=1)  # (B, C*3, D, H, W)
    #         fused_feat = self.residual_convs[i](multi_scale_feat)  # (B, C, D, H, W)
    #
    #         # 缩放和残差连接
    #         scaled_feat = self.scale_modules[i](fused_feat)
    #         residual = residual + scaled_feat
    #
    #     # 输出处理
    #     x_out = residual.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
    #
    #     # 应用归一化和投影（可选，保持特征质量）
    #     original_shape = x_out.shape
    #     x_flat = x_out.view(-1, C)
    #     x_flat = self.output_norm(x_flat)
    #     x_flat = self.output_proj(x_flat)
    #     x_out = x_flat.view(original_shape)
    #
    #     return x_out




class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.1,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        modeling_type: str = 'mamba',  # 'mamba' 或 'wavelet'
        d_conv: int = 3,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.modeling_type = modeling_type
        self.ln_1 = norm_layer(hidden_dim)
        if modeling_type == 'mamba':
            self.self_attention = SS3D(
                d_model=hidden_dim,
                dropout=attn_drop_rate,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        elif modeling_type == 'wavelet':
            self.self_attention = WaveletModule(
                d_model=hidden_dim,
                d_state=d_state
            )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, H=None, W=None, T=None):
        if self.modeling_type == 'wavelet' and H is not None:
            a = self.drop_path(self.self_attention(self.ln_1(x), H, W, T))
        elif self.modeling_type == 'mamba':
            a = self.drop_path(self.self_attention(self.ln_1(x)))
        else:
            a = self.drop_path(self.self_attention(self.ln_1(x)))
        B, H, W, T, C = a.shape
        a = a.view(B, H * W * T, C)
        x = x + a
        return x










