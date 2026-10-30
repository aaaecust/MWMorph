import torch 
import torch.nn as nn
import torch.nn.functional as F
#from nn_util import *
from utils import *
from DWT_IDWT.DWT_IDWT_Functions import *
# wavelet='haar' # for Brain
# wavelet='db4' # for Cardiac

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils import *
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Norm, Act
from monai.networks.blocks import UpSample
from monai.utils import InterpolateMode, UpsampleMode
from monai.networks.layers.utils import get_act_layer

Act.add_factory_callable("softsign", lambda: nn.modules.Softsign)







class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)
        return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)




class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, mov_image, flow, mod='bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)], indexing='ij')
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]
        # Softsign
        # disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        # disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        # disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = F.grid_sample(mov_image, sample_grid, mode=mod, align_corners=True)

        return sample_grid, warped


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)], indexing='ij')
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)

            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)

            flow = flow + F.grid_sample(flow, deformation, mode='bilinear', padding_mode="border", align_corners=True)

        return flow


class DiffeomorphicTransform2(nn.Module):
    def __init__(self, shape, time_step=7, device='cuda:0'):
        super(DiffeomorphicTransform2, self).__init__()
        self.time_step = time_step
        d2, h2, w2 = shape
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)], indexing='ij')
        grid_h = grid_h.to(device).float()
        grid_d = grid_d.to(device).float()
        grid_w = grid_w.to(device).float()
        self.grid_d = nn.Parameter(grid_d, requires_grad=False)
        self.grid_w = nn.Parameter(grid_w, requires_grad=False)
        self.grid_h = nn.Parameter(grid_h, requires_grad=False)

    def forward(self, flow):
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (self.grid_d + flow_d).squeeze(1)
            disp_h = (self.grid_h + flow_h).squeeze(1)
            disp_w = (self.grid_w + flow_w).squeeze(1)

            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border",
                                                          align_corners=True)

        return flow


class STN(nn.Module):
    def __init__(self, device='cuda', norm=True):
        super(STN, self).__init__()
        self.dev = device
        self.norm = norm

    def reference_grid(self, shape):
        dhw = shape[2:]
        if self.norm:
            grid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, s) for s in dhw], indexing='ij'), dim=0)
        else:
            grid = torch.stack(torch.meshgrid([torch.arange(0, s) for s in dhw], indexing='ij'), dim=0)
        grid = nn.Parameter(grid.float(), requires_grad=False).to(self.dev)
        return grid

    def forward(self, image, ddf, mode='bilinear', p_mode='zeros'):
        spatial_dims = len(image.shape) - 2
        grid = ddf + self.reference_grid(image.shape)
        if not self.norm:
            for i in range(len(spatial_dims)):
                grid[:, i, ...] = 2 * (grid[:, i, ...] / (spatial_dims[i] - 1) - 0.5)
        grid = grid.movedim(1, -1)
        idx_order = list(range(spatial_dims - 1, -1, -1))
        grid = grid[..., idx_order]  # z, y, x -> x, y, z
        warped = F.grid_sample(image, grid, mode=mode, padding_mode=p_mode, align_corners=True)
        return grid, warped


class svf_exp(nn.Module):
    def __init__(self, time_step=7, device='cuda'):
        super(svf_exp, self).__init__()
        self.time_step = time_step
        self.warp = STN(device=device)

    def forward(self, flow):
        flow = flow / (2 ** self.time_step)
        for _ in range(self.time_step):
            flow = flow + self.warp(image=flow, ddf=flow, mode='bilinear', p_mode="border")[1]
        return flow


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


class WaveletEncoderBlock(nn.Module):
    """
    小波编码器块：完全仿照WaveletDecoderBlock的结构，但是做下采样
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
            wavelet='haar'
    ):
        super().__init__()

        # 下采样（对应解码器的上采样）
        self.down = nn.Sequential(
            nn.MaxPool3d(2, stride=2),  # 或者用Conv3d(stride=2)
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        )

        # 第一个卷积（对应解码器的conv1）
        self.conv1 = Conv3dReLU(
            in_channels,  # 编码器不需要skip_channels，所以直接用in_channels
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # 第二个卷积（对应解码器的conv2）
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # 小波卷积（和解码器完全一样）
        self.wavelet_conv = WaveletConv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            wavelet=wavelet
        )

    def forward(self, x):
        # 下采样
        x = self.down(x)

        # 卷积处理
        x = self.conv1(x)
        x = self.conv2(x)

        # 小波处理（可选，根据需要启用）
        # x = self.wavelet_conv(x)  # 暂时注释掉，避免DWT问题

        return x


class WaveletConv3dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 use_batchnorm=True, wavelet='haar'):
        super().__init__()
        # 初始化3D DWT（GPU支持）
        self.dwt = DWTForward(wave=wavelet, J=1)  # J=1表示单层分解

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels * 8, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        )
        self.norm = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def _wavelet_decomposition(self, x):
        """纯GPU的小波分解"""
        # 反射填充（处理奇数尺寸）
        pad_d = (2 - x.shape[2] % 2) % 2
        pad_h = (2 - x.shape[3] % 2) % 2
        pad_w = (2 - x.shape[4] % 2) % 2
        x_pad = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='reflect')

        # 执行DWT（自动在GPU上运行）
        yl, yh = self.dwt(x_pad)  # yl: 低频, yh: 高频子带列表

        # 拼接所有子带 [yl, yh[0][:,:,0], yh[0][:,:,1], ...]
        # yh[0]的形状: [batch, 7, channels, D/2, H/2, W/2]
        yh = yh[0].unbind(dim=1)  # 解包7个高频子带
        return torch.cat([yl] + list(yh), dim=1)  # 输出通道数: in_channels*8

    def forward(self, x):
        x_wavelet = self._wavelet_decomposition(x)
        return self.relu(self.norm(self.conv(x_wavelet)))


class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, wavelet='haar', dwt_fn=1, up_train =True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ch = start_channel
        act = "leakyrelu"
        # act = "prelu"
        # act =  ("leakyrelu", {"negative_slope": 0.2})  
     
        norm_enc  = None # batch | instance
        norm_dec  = None # batch | instance
        
        if dwt_fn == 1: 
            from DWT_IDWT.DWT_IDWT_layer  import DWT_3D,IDWT_3D
        else:
            from DWT_IDWT.DWT_IDWT_layer2 import DWT_3D,IDWT_3D

        super(UNet, self).__init__()

        self.DWT  =  DWT_3D(wavename=wavelet)
        self.iDWT = IDWT_3D(wavename=wavelet)
        
        self.layer1 =  conv_twice(8*self.in_channel, self.ch*4,  s=1,  act=act, norm='INSTANCE')
        
        self.layer2 =  conv_twice(self.ch*4, self.ch*4,  s=2,  act=act,norm=norm_enc) 
        self.layer3 =  conv_twice(self.ch*4, self.ch*8,  s=2,  act=act,norm=norm_enc)  
        self.layer4 =  conv_twice(self.ch*8, self.ch*8,  s=2,  act=act,norm=norm_enc)  
        
        self.up_conv1 = Up_conv(self.ch*8, self.ch*8, train=up_train, act=act, norm=norm_dec, skip_c=self.ch*8)
        self.f_1_8 = self.to_out(self.ch*8, 8*3)
        
        self.up_conv2 = Up_conv(self.ch*8, self.ch*8, train=up_train, act=act, norm=norm_dec, skip_c=self.ch*4)
        self.f_1_4 = self.to_out(self.ch*8, 7*3)
        
        self.up_conv3 = Up_conv(self.ch*8, self.ch*4, train=up_train, act=act, norm=norm_dec, skip_c=self.ch*4)
        self.f_1_2 = self.to_out(self.ch*4, 7*3)

        self.group_conv1 = nn.Sequential(nn.Conv3d(21*2,21*2,3,1,1, groups=7),
                                         get_act_layer(act),
                                         nn.Conv3d(21*2,21, 3,1,1,  groups=7),
                                         )
        self.group_conv2 = nn.Sequential(nn.Conv3d(21*2,21*2,3,1,1, groups=7),
                                         get_act_layer(act),
                                         nn.Conv3d(21*2,21, 3,1,1,  groups=7)
                                         )
        
        self.conv1 = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)
        self.conv2 = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)
        self.conv3 = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)
       
    def to_out(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                )
        return layer

    def cat_h(self, LLH, LHL, LHH, HLL, HLH, HHL, HHH,  
                    LLH_,LHL_,LHH_,HLL_,HLH_,HHL_,HHH_, scale=2):
        LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.split_ch(self.up(torch.cat([LLH, LHL, LHH, HLL, HLH, HHL, HHH],dim=1), scale), 7)
        return torch.cat([LLH, LLH_, 
                          LHL, LHL_, 
                          LHH, LHH_,
                          HLL, HLL_, 
                          HLH, HLH_, 
                          HHL, HHL_, 
                          HHH, HHH_], dim=1)
        
    # def cat_h(self, LLH, LHL, LHH, HLL, HLH, HHL, HHH,  
    #                 LLH_,LHL_,LHH_,HLL_,HLH_,HHL_,HHH_, scale=2):
    #     return torch.cat([self.up(LLH,scale), LLH_, 
    #                       self.up(LHL,scale), LHL_, 
    #                       self.up(LHH,scale), LHH_,
    #                       self.up(HLL,scale), HLL_, 
    #                       self.up(HLH,scale), HLH_, 
    #                       self.up(HHL,scale), HHL_, 
    #                       self.up(HHH,scale), HHH_], dim=1)
        
    
    def up(self, x, scale=2):
        return F.interpolate(x, scale_factor=scale, mode='trilinear', align_corners=True) 
    
    def split_ch(self, a, n):
        return  [a[:,3*i:(3*i+3),...] for i in range(n)]
    

    def forward(self, x, y, multi=False):
        e0 = torch.cat((*self.DWT(x), *self.DWT(y)), 1)
        
        e1 = self.layer1(e0) # 64 64 48   | [80, 96, 112]
        e2 = self.layer2(e1) # 32 32 24   | [40, 48, 56]
        e3 = self.layer3(e2) # 16 16 12   | [20, 24, 28]
        e4 = self.layer4(e3) # 8 8 6      | [10, 12, 14]
                             # 4 4 3      | [5, 6, 7]
       
        d0 = self.up_conv1(e4, e3)
        idwt_1_8 = self.f_1_8(d0)
        
        d1 = self.up_conv2(d0, e2)
        idwt_1_4 = self.f_1_4(d1)
       
        d2 = self.up_conv3(d1, e1) 
        idwt_1_2 = self.f_1_2(d2)
       
        LLL_8, LLH_8, LHL_8, LHH_8, HLL_8, HLH_8, HHL_8, HHH_8 = self.split_ch(idwt_1_8, 8)
        LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4        = self.split_ch(idwt_1_4, 7)
        LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2        = self.split_ch(idwt_1_2, 7)
        
        xy_LLL_1_4 = self.conv1(self.iDWT(LLL_8, LLH_8, LHL_8, LHH_8, HLL_8, HLH_8, HHL_8, HHH_8))
     
        cat_value = self.cat_h(LLH_8,LHL_8,LHH_8,HLL_8,HLH_8,HHL_8,HHH_8,
                               LLH_4,LHL_4,LHH_4,HLL_4,HLH_4,HHL_4,HHH_4, scale=2)
    
        group_cov = self.group_conv1(cat_value)   
        LLH_4_, LHL_4_, LHH_4_, HLL_4_, HLH_4_, HHL_4_, HHH_4_ = self.split_ch(group_cov, 7)
    
        xy_LLL_1_2 = self.conv2(self.iDWT(xy_LLL_1_4, LLH_4_, LHL_4_, LHH_4_, HLL_4_, HLH_4_, HHL_4_, HHH_4_))
  
        cat_value2 = self.cat_h(LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4, 
                                LLH_2,LHL_2,LHH_2,HLL_2,HLH_2,HHL_2,HHH_2, scale=2)
        
        group_cov2 = self.group_conv2(cat_value2)   
        LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_ = self.split_ch(group_cov2, 7)
         
        xy_LLL_1_1 = self.conv3(self.iDWT(xy_LLL_1_2, LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_ ))
        f_xy = xy_LLL_1_1
        
        # if multi:
        #     return  xy_LLL_1_1, xy_LLL_1_2, xy_LLL_1_4, [LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_], [LLH_4_, LHL_4_, LHH_4_, HLL_4_, HLH_4_, HHL_4_, HHH_4_],[LLH_8, LHL_8, LHH_8, HLL_8, HLH_8, HHL_8, HHH_8]
        
        return f_xy
    
