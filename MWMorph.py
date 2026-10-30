import torch.nn as nn
from torch.distributions.normal import Normal
import gc
import timm
from typing import List, Tuple, Union
from torch.nn.modules.utils import _pair, _triple
import torch.nn as nn
from torch.distributions.normal import Normal

import models_abdomen.configs_MWMorph as configs
from models_abdomen.EVSS import *
from models_abdomen.EVSS import _ScaleModule3D
from models_abdomen.mamba import *

#from models_abdomen.attention import Attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WO_SELF_ATT=False
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        drop = self.drop(x)
        x = drop
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

def filter_dilated_rows(
    tensor: torch.Tensor,
    dilation: Tuple[int, int, int],
    dilated_kernel_size: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
):
    """
    A helper function that removes extra rows created during the process of
    implementing dilation.

    Args:
        tensor: A tensor containing the output slices resulting from unfolding
                the input tensor to `unfold3d()`.
                Shape is ``(B, C, D_out, H_out, W_out, dilated_kernel_size[0],
                dilated_kernel_size[1], dilated_kernel_size[2])``.
        dilation: The dilation given to `unfold3d()`.
        dilated_kernel_size: The size of the dilated kernel.
        kernel_size: The size of the kernel given to `unfold3d()`.

    Returns:
        A tensor of shape (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
        For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`.

    Example:
        >>> tensor = torch.zeros([1, 1, 3, 3, 3, 5, 5, 5])
        >>> dilation = (2, 2, 2)
        >>> dilated_kernel_size = (5, 5, 5)
        >>> kernel_size = (3, 3, 3)
        >>> filter_dilated_rows(tensor, dilation, dilated_kernel_size, kernel_size).shape
        torch.Size([1, 1, 3, 3, 3, 3, 3, 3])
    """

    kernel_rank = len(kernel_size)

    indices_to_keep = [
        list(range(0, dilated_kernel_size[i], dilation[i])) for i in range(kernel_rank)
    ]

    tensor_np = tensor.numpy()

    axis_offset = len(tensor.shape) - kernel_rank

    for dim in range(kernel_rank):
        tensor_np = np.take(tensor_np, indices_to_keep[dim], axis=axis_offset + dim)

    return torch.Tensor(tensor_np)
def unfold3d(
    tensor: torch.Tensor,
    *,
    kernel_size: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]] = 0,
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
):
    r"""
    Extracts sliding local blocks from an batched input tensor.

    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs

    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.

    Returns:
        A tensor of shape ``(B, C * np.product(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details

    Example:
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1, B*C*D*H*W + 1.).view(B, C, D, H, W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])
    """

    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(padding, int):
        padding = (padding, padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    if padding == "same":
        total_pad_D = dilation[0] * (kernel_size[0] - 1)
        total_pad_H = dilation[1] * (kernel_size[1] - 1)
        total_pad_W = dilation[2] * (kernel_size[2] - 1)
        pad_D_left = math.floor(total_pad_D / 2)
        pad_D_right = total_pad_D - pad_D_left
        pad_H_left = math.floor(total_pad_H / 2)
        pad_H_right = total_pad_H - pad_H_left
        pad_W_left = math.floor(total_pad_W / 2)
        pad_W_right = total_pad_W - pad_W_left

    elif padding == "valid":
        pad_D_left, pad_D_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
    else:
        pad_D_left, pad_D_right, pad_H_left, pad_H_right, pad_W_left, pad_W_right = (
            padding[0],
            padding[0],
            padding[1],
            padding[1],
            padding[2],
            padding[2],
        )

    batch_size, channels, _, _, _ = tensor.shape

    # Input shape: (B, C, D, H, W)
    tensor = F.pad(
        tensor,
        (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right),
    )
    # Output shape: (B, C, D+pad_W_left+pad_W_right, H+pad_H_left+pad_H_right, W+pad_D_left+pad_D_right)

    dilated_kernel_size = (
        kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1),
        kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1),
        kernel_size[2] + (kernel_size[2] - 1) * (dilation[2] - 1),
    )

    tensor = tensor.unfold(dimension=2, size=dilated_kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=dilated_kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=dilated_kernel_size[2], step=stride[2])

    if dilation != (1, 1, 1):
        tensor = filter_dilated_rows(tensor, dilation, dilated_kernel_size, kernel_size)

    # Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`

    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(
        1, 2
    )

    # Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]

    return tensor


# 示例：填充到偶数尺寸
def pad_to_even(x):
    """填充张量使d,h,w为偶数"""
    d, h, w = x.shape[-3:]
    pad_d = (2 - d % 2) % 2
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
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





class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class GradientMagnitude3D(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

        dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

        return torch.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self.eps)



class MambaBlock(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): 输入图像大小，默认 224
        patch_size (int | tuple(int)): 每个补丁的大小，默认 4
        in_chans (int): 输入图像的通道数，默认 3
        embed_dim (int): 补丁嵌入维度，默认 96
        depths (tuple(int)): 每个 Swin Transformer 层的深度。
        drop_rate (float): Dropout 率，默认 0
        norm_layer (nn.Module): 归一化层，默认 nn.LayerNorm。
        ape (bool): 如果为 True，则将绝对位置嵌入添加到补丁嵌入，默认 False
        patch_norm (bool): 如果为 True，则在补丁嵌入后添加归一化，默认 True
        use_checkpoint (bool): 是否使用检查点以节省内存，默认 False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 kernel_size=[1, 3, 3, 5],
                 ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.kernel_size = kernel_size
        self.conv_blocks = nn.ModuleList()

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
                pretrain_img_size[2] // patch_size[2],
            ]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2])
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            current_kernel_size = self.kernel_size[i_layer]
            layer = VMambaLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depths=self.depths[i_layer],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                kernel_size=current_kernel_size,
                layer_idx=i_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        for i_layer in out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def _embed_tokens(self, x):
        x = self.patch_embed(x)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        return x, Wh, Ww, Wt

    def forward(self, x):
        """前向传播函数。"""
        x = self.patch_embed(x)  # 将输入图像转换为补丁嵌入

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)  # 获取补丁的空间和时间维度

        if self.ape:
            # 将位置嵌入插值到相应的大小
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # 添加位置嵌入并调整维度
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)  # 添加正弦位置嵌入并调整维度
        else:
            x = x.flatten(2).transpose(1, 2)  # 仅调整维度
        x = self.pos_drop(x)  # 应用 Dropout

        outs = []  # 用于存储输出
        for i in range(self.num_layers):
            layer = self.layers[i]  # 获取当前层
            # 进行前向传播，返回输出和其他信息
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')  # 获取归一化层
                x_out = norm_layer(x_out)  # 应用归一化

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()  # 调整输出维度
                outs.append(out)  # 存储输出
        return outs  # 返回所有层的输出

    def train(self, mode=True):
        super(MambaBlock, self).train(mode)
        self._freeze_stages()







class PMEStage(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2,
                 kernel_size=3, enable_cross=False, downsample=True):
        super().__init__()
        # 修改：每个编码 stage 内部同时学习局部几何和全局上下文。
        self.local_branch = nn.Sequential(
            Conv3dReLU(dim, dim, 3, 1, use_batchnorm=False),
            Conv3dReLU(dim, dim, 3, 1, use_batchnorm=False),
        )
        self.global_branch = VMambaLayer(
            dim=dim,
            depths=depths,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            kernel_size=kernel_size,
            layer_idx=0,
            total_layers=1,
            mamba_layers=[0],
            wavelet_layers=[],
            downsample=None,
        )
        self.stream_fuse = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(inplace=True),
        )
        self.pair_fuse = nn.Sequential(
            nn.Conv3d(dim * 3, dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(inplace=True),
        )
        self.enable_cross = enable_cross
        if self.enable_cross:
            # 修改：从中层开始做跨流交互，最浅层保持解耦以保留模态特异纹理。
            self.cross_gate = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.cross_gate = None
        self.downsample = PatchMerging(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4) if downsample else None

    @staticmethod
    def _tokens_to_feature(x, H, W, T):
        B, _, C = x.shape
        return x.view(B, H, W, T, C).permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _feature_to_tokens(x):
        B, C, H, W, T = x.shape
        return x.permute(0, 2, 3, 4, 1).contiguous().view(B, H * W * T, C)

    @staticmethod
    def _pack_vis_tensor(x):
        if x is None:
            return None
        return x.detach().to(dtype=torch.float16).cpu().contiguous()

    def _forward_stream(self, tokens, H, W, T):
        local_feat = self.local_branch(self._tokens_to_feature(tokens, H, W, T))
        global_tokens, _, _, _, _, _, _, _ = self.global_branch(tokens, H, W, T)
        global_feat = self._tokens_to_feature(global_tokens, H, W, T)
        fused_feat = self.stream_fuse(torch.cat([local_feat, global_feat], dim=1))
        return fused_feat

    def forward(self, src_tokens, tgt_tokens, H, W, T, return_vis=False):
        src_stream_fuse = self._forward_stream(src_tokens, H, W, T)
        tgt_stream_fuse = self._forward_stream(tgt_tokens, H, W, T)

        pair_diff_before_gate = torch.abs(src_stream_fuse - tgt_stream_fuse)
        pair_feat_before_gate = self.pair_fuse(torch.cat([src_stream_fuse, tgt_stream_fuse, pair_diff_before_gate], dim=1))

        src_feat = src_stream_fuse
        tgt_feat = tgt_stream_fuse
        cross_weight = None
        gated_update = None
        pair_feat = pair_feat_before_gate

        if self.cross_gate is not None:
            cross_weight = self.cross_gate(pair_feat_before_gate)
            gated_update = cross_weight * pair_feat_before_gate
            src_feat = src_stream_fuse + gated_update
            tgt_feat = tgt_stream_fuse + gated_update
            pair_diff_after_gate = torch.abs(src_feat - tgt_feat)
            pair_feat = self.pair_fuse(torch.cat([src_feat, tgt_feat, pair_diff_after_gate], dim=1))

        next_H, next_W, next_T = H, W, T
        src_tokens = self._feature_to_tokens(src_feat)
        tgt_tokens = self._feature_to_tokens(tgt_feat)

        if self.downsample is not None:
            src_tokens = self.downsample(src_tokens, H, W, T)
            tgt_tokens = self.downsample(tgt_tokens, H, W, T)
            next_H, next_W, next_T = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2

        vis_info = None
        if return_vis:
            vis_info = {
                "cross_enabled": self.cross_gate is not None,
                "spatial_shape_before_downsample": (H, W, T),
                "spatial_shape_after_downsample": (next_H, next_W, next_T),
                "src_stream_fuse": self._pack_vis_tensor(src_stream_fuse),
                "tgt_stream_fuse": self._pack_vis_tensor(tgt_stream_fuse),
                "pair_diff_before_gate": self._pack_vis_tensor(pair_diff_before_gate),
                "pair_feat_before_gate": self._pack_vis_tensor(pair_feat_before_gate),
                "cross_gate": self._pack_vis_tensor(cross_weight),
                "gated_update": self._pack_vis_tensor(gated_update),
                "pair_feat_after_gate": self._pack_vis_tensor(pair_feat),
            }

        if return_vis:
            return pair_feat, src_tokens, tgt_tokens, next_H, next_W, next_T, vis_info
        return pair_feat, src_tokens, tgt_tokens, next_H, next_W, next_T


class PME(nn.Module):
    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 kernel_size=[1, 3, 3, 5]):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.kernel_size = kernel_size

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
                pretrain_img_size[2] // patch_size[2],
            ]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2])
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # self.stages = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     self.stages.append(
        #         PMEStage(
        #             dim=self.num_features[i_layer],
        #             depths=self.depths[i_layer],
        #             d_state=d_state,
        #             d_conv=d_conv,
        #             expand=expand,
        #             kernel_size=self.kernel_size[i_layer],
        #             enable_cross=i_layer >= 1,
        #             downsample=i_layer < self.num_layers - 1,
        #         )
        #     )
        self.stages = nn.ModuleList()
        cross_stage_indices = (1,2,3)# 改这里就行
        for i_layer in range(self.num_layers):
            self.stages.append(
                PMEStage(
                    dim=self.num_features[i_layer],
                    depths=self.depths[i_layer],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    kernel_size=self.kernel_size[i_layer],
                    enable_cross=i_layer in cross_stage_indices,
                    downsample=i_layer < self.num_layers - 1,
                )
            )


    def _embed_single(self, x):
        x = self.patch_embed(x)
        H, W, T = x.size(2), x.size(3), x.size(4)
        if self.ape:
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear')
            x = x + absolute_pos_embed
        elif self.spe:
            x = x + self.pos_embd(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        return x, H, W, T

    def forward(self, src, tgt, return_vis=False):
        # 修改：编码器改为共享参数的双流输入，而不是先拼接再走单流主干。
        src_tokens, H, W, T = self._embed_single(src)
        tgt_tokens, _, _, _ = self._embed_single(tgt)
        outs = []
        vis_list = []
        for i, stage in enumerate(self.stages):
            if return_vis:
                pair_feat, src_tokens, tgt_tokens, H, W, T, vis_info = stage(src_tokens, tgt_tokens, H, W, T, return_vis=True)
                vis_info["stage_index"] = i
                vis_list.append(vis_info)
            else:
                pair_feat, src_tokens, tgt_tokens, H, W, T = stage(src_tokens, tgt_tokens, H, W, T)
            if i in self.out_indices:
                outs.append(pair_feat)
        if return_vis:
            return outs, vis_list
        return outs


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)


    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


def pixel_shuffle_3d(x, upscale_factor):
    """
    3D Pixel Shuffle implementation
    Args:
        x: input tensor of shape (batch, channels, depth, height, width)
        upscale_factor: upsampling scale factor
    Returns:
        upsampled tensor
    """
    batch_size, channels, in_depth, in_height, in_width = x.size()

    # channels should be divisible by upscale_factor^3
    assert channels % (upscale_factor ** 3) == 0

    out_channels = channels // (upscale_factor ** 3)
    out_depth = in_depth * upscale_factor
    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    # Reshape and permute
    x = x.view(batch_size, out_channels, upscale_factor, upscale_factor, upscale_factor,
               in_depth, in_height, in_width)

    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

    return x.view(batch_size, out_channels, out_depth, out_height, out_width)


class PixelShuffle3d(nn.Module):
    """3D Pixel Shuffle layer"""

    def __init__(self, upscale_factor):
        super(PixelShuffle3d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle_3d(x, self.upscale_factor)


class PatchExpand(nn.Module):
    """与PatchMerging对称的上采样层"""

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        # 上采样：将通道数从dim扩展到2*dim，空间尺寸翻倍
        self.expansion = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        x = self.norm(x)
        x = self.expansion(x)  # B, H*W*T, 2*C

        x = x.view(B, H, W, T, 2, C)
        x = x.permute(0, 1, 2, 3, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, T * 2, C)  # 空间尺寸翻倍，通道数减半
        x = x.view(B, -1, C)

        return x






class WGRDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0,
                 d_state=16, d_conv=3, expand=2, kernel_size=3,
                 modeling_type='mamba'):  # 新增参数，默认mamba
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

        # 上采样层
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # 跳跃连接处理
        if skip_channels > 0:
            self.skip_conv = nn.Conv3d(in_channels + skip_channels, in_channels,
                                     kernel_size=1, padding=0, bias=False)
            self.skip_norm = nn.BatchNorm3d(in_channels)
            self.skip_align = nn.Conv3d(skip_channels, in_channels, kernel_size=1, padding=0, bias=False)
            self.local_gate = nn.Sequential(
                nn.Conv3d(in_channels + skip_channels, in_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(in_channels),
                nn.Sigmoid(),
            )
        else:
            self.local_gate = None

        # 根据modeling_type参数设置mamba_layers和wavelet_layers
        if modeling_type == 'mamba':
            mamba_layers = [0]
            wavelet_layers = []
        elif modeling_type == 'wavelet':
            mamba_layers = []
            wavelet_layers = [0]
        else:
            raise ValueError(f"modeling_type must be 'mamba' or 'wavelet', got {modeling_type}")

        # 使用 VMambaLayer 进行处理
        self.vmamba_layer = VMambaLayer(
            dim=in_channels,
            depths=1,  # 使用单个深度
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            kernel_size=kernel_size,
            layer_idx=0,
            total_layers=1,
            mamba_layers=mamba_layers,      # 根据modeling_type设置
            wavelet_layers=wavelet_layers,  # 根据modeling_type设置
            downsample=None,
        )

        # 输出通道调整
        if in_channels != out_channels:
            self.output_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.output_norm = nn.BatchNorm3d(out_channels)
            self.output_act = nn.LeakyReLU(inplace=True)
        else:
            self.output_conv = None

    # def forward(self, x, skip=None):
    #     """
    #     Args:
    #         x: 输入特征 (B, C, H, W, T)
    #         skip: 跳跃连接特征 (B, C_skip, H*2, W*2, T*2) 或 None
    #     Returns:
    #         输出特征 (B, C_out, H*2, W*2, T*2)
    #     """
    #     x = self.up(x)  # (B, C, H*2, W*2, T*2)

    #     if skip is not None and self.skip_channels > 0:
    #         if x.shape[-3:] != skip.shape[-3:]:
    #             skip = F.interpolate(skip, size=x.shape[-3:], mode='trilinear', align_corners=False)

    #         local_residual = torch.cat([x, skip], dim=1)
    #         if self.local_gate is not None:
    #             gate = self.local_gate(local_residual)
    #             x = x + gate * self.skip_align(skip)
    #         x = self.skip_conv(local_residual)
    #         x = self.skip_norm(x)

    #     B, C, H, W, T = x.shape
    #     x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)

    #     x_out, _, _, _, x_processed, _, _, _ = self.vmamba_layer(x_flat, H, W, T)

    #     x = x_processed.view(B, H, W, T, C).permute(0, 4, 1, 2, 3).contiguous()

    #     if self.output_conv is not None:
    #         x = self.output_conv(x)
    #         x = self.output_norm(x)
    #         x = self.output_act(x)

    #     return x
    @staticmethod
    def _pack_vis_tensor(x):
        if x is None:
            return None
        return x.detach().to(dtype=torch.float16).cpu().contiguous()

    def forward(self, x, skip=None, return_vis=False):
        x = self.up(x)

        if skip is not None and self.skip_channels > 0:
            if x.shape[-3:] != skip.shape[-3:]:
                skip = F.interpolate(skip, size=x.shape[-3:], mode='trilinear', align_corners=False)

            local_residual = torch.cat([x, skip], dim=1)
            if self.local_gate is not None:
                gate = self.local_gate(local_residual)
                x = x + gate * self.skip_align(skip)
            x = self.skip_conv(local_residual)
            x = self.skip_norm(x)

        # WGRD / decoder refinement 前
        x_before = x

        B, C, H, W, T = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)
        x_out, _, _, _, x_processed, _, _, _ = self.vmamba_layer(x_flat, H, W, T)

        # WGRD / decoder refinement 后（注意：这里还没过 output_conv，便于和 before 对齐）
        x_after = x_processed.view(B, H, W, T, C).permute(0, 4, 1, 2, 3).contiguous()

        vis = None
        if return_vis:
            vis = {
                "before": self._pack_vis_tensor(x_before),
                "after": self._pack_vis_tensor(x_after),
                "delta": self._pack_vis_tensor((x_after - x_before).abs().mean(1, keepdim=True)),
                "spatial_shape": (H, W, T),
            }

        x = x_after

        if self.output_conv is not None:
            x = self.output_conv(x)
            x = self.output_norm(x)
            x = self.output_act(x)

        if return_vis:
            return x, vis
        return x


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel=3,
                 stride=1, padding=1, norm=False, act=True):
        super().__init__()
        self.norm = norm
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel, stride, padding)
        self.act = act
        if self.norm:
            BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
            self.batch_norm = BatchNorm(out_channels)
        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        torch.nn.init.kaiming_normal_(self.main.weight)
        torch.nn.init.constant_(self.main.bias, 0)

    def forward(self, x):
        out = self.main(x)
        if self.norm:
            out = self.batch_norm(out)
        if self.act:
            out = self.activation(out)
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels.
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 norm=False,
                 conv_1_1=False):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                if prev_nf != nf and conv_1_1:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                    convs.append(ConvBlock(ndims, nf, nf, norm=norm))
                else:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                if prev_nf != nf and conv_1_1:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                    convs.append(ConvBlock(ndims, nf, nf, norm=norm))
                else:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            if prev_nf != nf and conv_1_1:
                self.remaining.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                self.remaining.append(ConvBlock(ndims, nf, nf, norm=norm))
            else:
                self.remaining.append(ConvBlock(ndims, prev_nf, nf, norm=norm))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x




class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, inshape=None, infeats=None, nb_features=None,
                 half_res=False, variant='nano'):
        """
        显存优化的ConvNeXt - 只保存必要的backbone参数
        """

        super().__init__()

        if infeats is None: infeats = 1
        if inshape is None: inshape = (64, 256, 256)
        if nb_features is None: nb_features = default_unet_features()

        self.half_res = half_res

        # 确定输出通道数
        if isinstance(nb_features, (list, tuple)) and len(nb_features) >= 2:
            enc_nf, dec_nf = nb_features
            output_channels = dec_nf[-1] if dec_nf else 16
        else:
            output_channels = 16

        # 创建ConvNeXt但立即进行显存优化
        self.backbone = timm.create_model(
            f'convnext_{variant}',
            pretrained=True,
            in_chans=infeats,
            features_only=True
        )
        # self.backbone  = timm.create_model(
        # 'convnext_tiny.dinov3_lvd1689m',
        # pretrained=True,
        # features_only=True,
        # )
        # 立即冻结backbone的所有参数以节省显存
        for param in self.backbone.parameters():
            param.requires_grad = False


        self.backbone.eval()

        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, infeats, inshape[1], inshape[2])
            features = self.backbone(dummy)

        self.feature_dims = [f.shape[1] for f in features]
        print(f"ConvNeXt {variant} feature dims: {self.feature_dims}")

        # 只使用第一层特征（显存最少）
        self.selected_channels = self.feature_dims[0]

        # 极简的特征处理层 - 只保留必要组件
        self.feature_processor = nn.Sequential(

            nn.Conv3d(self.selected_channels, output_channels, 1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.GELU()
        )
        
        
        self.final_nf = output_channels


        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def forward(self, x):
        B, C, D, H, W = x.shape

        # 分片处理以节省显存 - 每次处理更多深度切片以提高吞吐
        slice_size = 8
        features_list = []

        # backbone部分完全不需要梯度
        with torch.no_grad():
            for i in range(0, D, slice_size):
                end_idx = min(i + slice_size, D)
                x_slice = x[:, :, i:end_idx, :, :].reshape(-1, C, H, W)

                # 只使用第一层特征
                feat = self.backbone(x_slice)[0]

                # 如果空间维度不匹配，调整到原始尺寸
                if feat.shape[-2:] != (H, W):
                    feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)

                features_list.append(feat)

                del x_slice

        # 重新组合特征
        all_features = torch.cat(features_list, dim=0)
        feat_3d = all_features.view(B, D, self.selected_channels, H, W).permute(0, 2, 1, 3, 4)

        # 清理中间变量
        del features_list, all_features

        # 特征处理（这部分需要梯度）
        output = self.feature_processor(feat_3d)

        if self.half_res:
            output = F.avg_pool3d(output, 2, 2)

        return output

class StructuralStream(nn.Module):
    """
    结构流：在全分辨率下运行，使用轻量级3D卷积提取高频结构信息（边缘、纹理）
    不进行下采样，保证配准精度。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 简单的3D卷积，捕捉空间结构
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels // 2),
            nn.LeakyReLU(inplace=True),
            # 深度卷积节省显存
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
class SemanticStream(nn.Module):
    """
    语义流：在低分辨率下运行。
    - 普通预训练backbone: 冻结 + features_only
    - plain_cnn: 不用预训练，作为公平的无预训练语义分支基线
    """
    def __init__(self, backbone_name='convnext_tiny', in_channels=1, out_channels=16, scale_factor=0.25):
        super().__init__()
        self.scale_factor = scale_factor
        self.backbone_name = backbone_name
        self.use_plain_cnn = backbone_name == 'plain_cnn'

        if self.use_plain_cnn:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
            )
            in_feat_dim = 96
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                in_chans=in_channels,
                features_only=True,
                out_indices=(0,)
            )

            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

            with torch.no_grad():
                dummy = torch.randn(1, in_channels, 64, 64)
                feats = self.backbone(dummy)
                in_feat_dim = feats[0].shape[1]

        self.adapter = nn.Sequential(
            nn.Conv2d(in_feat_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # (B, C, D, H, W) -> (B*D, C, H, W)
        x_2d = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # 只缩 H, W
        h_small = int(H * self.scale_factor)
        w_small = int(W * self.scale_factor)
        x_small = F.interpolate(x_2d, size=(h_small, w_small), mode='bilinear', align_corners=False)

        # backbone
        if self.use_plain_cnn:
            feat = self.backbone(x_small)
        else:
            with torch.no_grad():
                feat = self.backbone(x_small)[0]

        # 通道适配
        feat = self.adapter(feat)

        # 还原 3D
        _, c_out, h_f, w_f = feat.shape
        feat_3d = feat.view(B, D, c_out, h_f, w_f).permute(0, 2, 1, 3, 4)

        # 上采样回原分辨率
        feat_up = F.interpolate(feat_3d, size=(D, H, W), mode='trilinear', align_corners=False)

        return feat_up



class HFE(nn.Module):
    """
    终极方案：快速混合特征提取器
    """
    def __init__(self, inshape=None, infeats=1, nb_features=None, variant='nano'):
        super().__init__()
        
        # 配置输出通道
        if isinstance(nb_features, (list, tuple)) and len(nb_features) >= 2:
            output_channels = nb_features[-1][-1] if nb_features[-1] else 16
        else:
            output_channels = 16
        
        # 1. 结构流 (高频, High-Res, Learnable)
        self.structural = StructuralStream(in_channels=infeats, out_channels=output_channels)
        
        # 2. 语义流 (低频, Low-Res, Pretrained Frozen)
        # scale_factor=0.25 意味着长宽各缩小4倍，计算量减少16倍
        self.semantic = SemanticStream(
            backbone_name=f'convnext_nano',
            in_channels=infeats,
            out_channels=output_channels,
            scale_factor=0.375
        )
        
        # 3. 创新点融合模块 (Attention-based Fusion)
        self.fusion = nn.Sequential(
            nn.Conv3d(output_channels * 2, output_channels, 1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.Sigmoid() # 作为注意力门控
        )
        self.semantic_weight = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(output_channels * 2, max(output_channels // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(output_channels // 4, 4), output_channels, 1),
            nn.Sigmoid()
        )

        self.final_nf = output_channels
        self.final_conv = nn.Conv3d(output_channels, output_channels, 1)
    def forward(self, x):
        # 1. 获取结构特征 (B, C, D, H, W)
        f_struct = self.structural(x)
        
        # 2. 获取语义特征 (B, C, D, H, W) - 内部已经处理了降采样/上采样
        f_sem = self.semantic(x)
        
        # 3. 融合创新
        # 简单拼接
        cat = torch.cat([f_struct, f_sem], dim=1)
        
        # 计算融合门控
        gate = self.fusion(cat)
        semantic_scale = self.semantic_weight(cat)

        # 最终特征：结构特征 + 空间门控语义特征，并由通道自适应权重调节语义注入强度
        out = f_struct + semantic_scale * f_sem * gate

        return self.final_conv(out)












class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        # 旧版单层 head：
        # conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        # conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        # super().__init__(conv3d)

        hidden_channels = max(in_channels // 2, out_channels)
        conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        norm1 = nn.InstanceNorm3d(hidden_channels)
        act1 = nn.LeakyReLU(inplace=True)
        conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        nn.init.kaiming_normal_(conv1.weight)
        nn.init.constant_(conv1.bias, 0)
        conv2.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2.weight.shape))
        conv2.bias = nn.Parameter(torch.zeros(conv2.bias.shape))

        super().__init__(conv1, norm1, act1, conv2)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.size = tuple(size)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def _get_grid(self, shape, device, dtype):
        if tuple(shape) == self.size:
            return self.grid.to(device=device, dtype=dtype)

        vectors = [torch.arange(0, s, device=device, dtype=dtype) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        return torch.stack(grids).unsqueeze(0)

    def forward(self, src, flow):
        shape = flow.shape[2:]
        grid = self._get_grid(shape, flow.device, flow.dtype)
        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


class MWMorph(nn.Module):


    def __init__(self, config):

        super().__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        self.embed_dim = config.embed_dim
        self.cond_patch_size = to_3tuple(config.patch_size)
        embed_dim = config.embed_dim
        img_size = config.img_size

        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.read_out_head = nn.ModuleList()
        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1] // 2, 3))

        self.feature_extractor = HFE(
            inshape=config.img_size,
            infeats=1,
            nb_features=nb_feat_extractor,
            variant='nano'
        )
        self.transformer = PME(patch_size=config.patch_size,
                                      in_chans=nb_feat_extractor[-1][-1],
                                      embed_dim=config.embed_dim,
                                      depths=config.depths,
                                      drop_rate=config.drop_rate,
                                      ape=config.ape,
                                      spe=config.spe,
                                      patch_norm=config.patch_norm,
                                      out_indices=config.out_indices,
                                      d_state=config.d_state,
                                      d_conv=config.d_conv,
                                      expand=config.expand,
                                      #kernel_size=config.kernel_size
                                     )

        self.up0 = WGRDBlock(
            in_channels=self.embed_dim * 8,
            out_channels=self.embed_dim * 4,
            skip_channels=self.embed_dim * 4 if self.if_transskip else 0,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            kernel_size=5,
            modeling_type="wavelet"

        )

        self.up1 = WGRDBlock(
            in_channels=self.embed_dim * 4,
            out_channels=self.embed_dim * 2,
            skip_channels=self.embed_dim * 2 if self.if_transskip else 0,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            kernel_size=3,
            modeling_type="mamba"

        )

        self.up2 = WGRDBlock(
            in_channels=self.embed_dim * 2,
            out_channels=self.embed_dim,
            skip_channels=self.embed_dim if self.if_transskip else 0,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            kernel_size=3,  # 对应编码器第二层的kernel_size=3
            modeling_type="wavelet"    # 对应编码器第二层的depths=2

        )


        self.up3 = WGRDBlock(
            in_channels=self.embed_dim,
            out_channels=config.reg_head_chan,
            skip_channels=self.embed_dim // 2 if self.if_convskip else 0,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            kernel_size=1,  # 对应编码器第一层的kernel_size=1
            modeling_type="wavelet"   # 对应编码器第一层的depths=2

        )

        # 修改：每个解码层单独预测残差形变，做 coarse-to-fine refinement。
        self.flow0 = RegistrationHead(in_channels=self.embed_dim * 4, out_channels=3, kernel_size=3)
        self.flow1 = RegistrationHead(in_channels=self.embed_dim * 2, out_channels=3, kernel_size=3)
        self.flow2 = RegistrationHead(in_channels=self.embed_dim, out_channels=3, kernel_size=3)
        self.flow3 = RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3)


        # 卷积跳跃连接
        #self.c1 = Conv3dReLU(2, embed_dim // 2, 1, 0, use_batchnorm=False)
        self.c1 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, embed_dim // 2, 3, 1, use_batchnorm=False)


        # 空间变换器和辅助模块
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)



    def forward(self, x, return_vis=False):

        source = x[:, 0:1, :, :]
        target = x[:, 1:2, :, :, :]

        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        # 双流编码器分别接收 source/target 特征
        if return_vis:
            out_feats, pme_vis = self.transformer(source_feat, target_feat, return_vis=True)
        else:
            out_feats = self.transformer(source_feat, target_feat)
            pme_vis = None

        # ConvSkip
        if self.if_convskip:
            x_skip = torch.cat([source_feat, target_feat], dim=1)
            x_s1 = self.avg_pool(x_skip)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        # Transformer skip
        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = f2 = f3 = None

        wgrd_vis = []

        # Decoder stage 0
        if return_vis:
            x0, vis0 = self.up0(out_feats[-1], f1, return_vis=True)
            vis0["stage_index"] = 0
            wgrd_vis.append(vis0)
        else:
            x0 = self.up0(out_feats[-1], f1)
        flow0 = self.flow0(x0)

        # Decoder stage 1
        if return_vis:
            x1, vis1 = self.up1(x0, f2, return_vis=True)
            vis1["stage_index"] = 1
            wgrd_vis.append(vis1)
        else:
            x1 = self.up1(x0, f2)
        flow1 = self.up(flow0) + self.flow1(x1)

        # Decoder stage 2
        if return_vis:
            x2, vis2 = self.up2(x1, f3, return_vis=True)
            vis2["stage_index"] = 2
            wgrd_vis.append(vis2)
        else:
            x2 = self.up2(x1, f3)
        flow2 = self.up(flow1) + self.flow2(x2)

        # Decoder stage 3
        if return_vis:
            x3, vis3 = self.up3(x2, f4, return_vis=True)
            vis3["stage_index"] = 3
            wgrd_vis.append(vis3)
        else:
            x3 = self.up3(x2, f4)
        flow3 = self.up(flow2) + self.flow3(x3)

        # 最终全分辨率 flow
        flow = self.up(flow3)

        out = self.spatial_trans(source, flow)

        if return_vis:
            vis_dict = {
                "pme": pme_vis,
                "wgrd": wgrd_vis,
                "source_feature": source_feat.detach().to(dtype=torch.float16).cpu().contiguous(),
                "target_feature": target_feat.detach().to(dtype=torch.float16).cpu().contiguous(),
            }
            return out, flow, vis_dict

        return out, flow






CONFIGS = {
    'MWMorph': configs.get_MWMorph_config(),
    'MWMorph-Large': configs.get_MWMorphLarge_config(),
    'MWMorph-Small': configs.get_MWMorphSmall_config(),
}

