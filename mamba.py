import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
import pdb
from einops import rearrange, repeat
from functools import partial
from models_abdomen.EVSS import *
from timm.models.layers import trunc_normal_, DropPath
#from EVSS import Dynamic_Adaptive_Scan
import gc
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # 通道维度扩展
        x = self.expand(x)
        x = rearrange(x, 'b h w d (p1 p2 p3 c) -> b (h p1) (w p2) (d p3) c', p1=2, p2=2, p3=2)
        x = self.norm(x)
        return x





class MambaLayer(nn.Module):
    def __init__(self, depths,dim, kernel_size,d_state=16, d_conv=3, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.dt_modulator = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.dt_modulator.weight)


        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand, )  # Block expansion factor)

        # 降采样
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None



    def forward(self, x, H, W, T):
        B, L, C = x.shape
        residual = x
        
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        
        x_mamba = self.mamba(x_norm)
        x = residual + x_mamba.type(x.dtype)  # 残差连接
        
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T




class VMambaLayer(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, kernel_size=3,layer_idx=0, total_layers=4, downsample=None,mamba_layers=[0,1], wavelet_layers=[2,3]):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.norm = nn.LayerNorm(dim)

        self.mamba_layers = mamba_layers
        self.wavelet_layers = wavelet_layers
        self.use_mamba = layer_idx in mamba_layers
        self.use_wavelet = layer_idx in wavelet_layers

        if self.use_mamba:
            # Mamba建模模块
            self.norm = nn.LayerNorm(dim)
            self.mamba = nn.ModuleList([
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=0.,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0.,
                    d_state=d_state,
                    modeling_type='mamba',  # 新增参数
                    d_conv=kernel_size,
                    expand=expand,
                )

                for i in range(depths)
            ])

        if self.use_wavelet:
            # 小波建模模块
            self.norm = nn.LayerNorm(dim)
            self.wavelet_blocks = nn.ModuleList([
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=0.,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0.,
                    d_state=d_state,
                    modeling_type='wavelet',  # 新增参数
                    d_conv=d_conv,
                    expand=expand,
                )
                for i in range(depths)
            ])


        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None
            # 打印配置信息
        print(f"Layer {layer_idx}: {'Mamba' if self.use_mamba else 'Wavelet' if self.use_wavelet else 'None'}")
        print(f"  - Depths: {depths}")
        print(f"  - Created {depths} {'mamba' if self.use_mamba else 'wavelet'} blocks")

    def forward(self, x, H, W, T):
        """
        前向传播函数：
        输入：
        - x: 输入特征张量，形状为 (B, H*W*T, C)
        - H, W, T: 输入特征的空间/时间维度
        输出：
        - x: 原特征的输出
        - H, W, T: 原特征的维度（未变化）
        - x_down: 降采样后的特征（如果降采样层存在）
        - Wh, Ww, Wt: 降采样后的维度
        """
        B, C = x.shape[0], x.shape[-1]  # 获取批量大小和输入特征维度
        assert C == self.dim  # 确保输入的特征维度与初始化时设置的维度一致

        x_norm = self.norm(x)  # 对输入特征进行归一化处理
        if x_norm.dtype == torch.float16:  # 如果数据类型为 float16，将其转换为 float32
            x_norm = x_norm.type(torch.float32)

        if self.use_mamba:
            x_hidden = x_norm
            for blk in self.mamba:
                if self.training:
                    x_hidden = checkpoint.checkpoint(lambda inp: blk(inp), x_hidden, use_reentrant=False)
                else:
                    x_hidden = blk(x_hidden)
            x = x_hidden.type(x.dtype)

        elif self.use_wavelet:
            x_hidden = x_norm
            for blk in self.wavelet_blocks:
                if self.training:
                    x_hidden = checkpoint.checkpoint(
                        lambda inp, blk=blk: blk(inp, H, W, T),
                        x_hidden,
                        use_reentrant=False,
                    )
                else:
                    x_hidden = blk(x_hidden, H, W, T)
            x = x_hidden.type(x.dtype)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)  # 降采样后的特征
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


        

