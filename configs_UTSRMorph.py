import ml_collections
'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): 是否启用来自 Transformer 块的跳跃连接。
if_convskip (bool): 是否启用来自卷积块的跳跃连接。
patch_size (int | tuple(int)): 补丁大小，默认为 4。
in_chans (int): 输入图像的通道数，默认为 2（用于移动图像和固定图像）。
embed_dim (int): 补丁嵌入维度，默认为 96。
depths (tuple(int)): 每个 Swin Transformer 层的深度。
num_heads (tuple(int)): 各层的注意力头数。
window_size (tuple(int)): 图像大小应能被窗口大小整除，例如，如果图像大小为 (160, 192, 224)，则窗口大小可以为 (5, 6, 7)。
mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比率，默认为 4。
pat_merg_rf (int): 补丁合并时嵌入维度的缩小因子，例如，NC->N/4C（如果设置为 4）。默认为 4。
qkv_bias (bool): 如果为 True，则为查询、键和值添加可学习的偏置。默认为 True。
drop_rate (float): Dropout 比率，默认为 0。
drop_path_rate (float): 随机深度比率，默认为 0.1。
ape (bool): 启用可学习的位置嵌入。默认为 False。
spe (bool): 启用正弦位置嵌入。默认为 False。
patch_norm (bool): 如果为 True，则在补丁嵌入后添加归一化。默认为 True。
use_checkpoint (bool): 是否使用检查点来节省内存。默认为 False（从 Swin Transformer 继承，通常不需要）。
out_indices (tuple(int)): 要输出特征的 Transformer 块的索引，默认为 (0, 1, 2, 3)。
reg_head_chan (int): 配准头（即最终卷积层）中的通道数。
img_size (int | tuple(int)): 输入图像的大小，例如 (160, 192, 224)。
'''
def get_UTSRMorph_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 6,2)
    config.num_heads = (4, 4, 8, 8)
    #config.window_size = (6, 5, 6)
    config.window_size = (5, 6, 6)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2,3)
    config.reg_head_chan = 16
    #config.img_size = (192, 160, 192)
    config.img_size = (160, 192, 192)
    config.d_state = 16
    config.d_conv = 3
    config.expand = 2
    config.int_steps = 5
    config.wavelet='haar'
    config.out_fmaps=['P4','P3','P2','P1']
    # 添加跨层注意力相关配置
    config.cross_attn_config = ml_collections.ConfigDict()
    # 配置跨层连接：键为目标层索引，值为源层索引
    config.cross_attn_config.connections = {2: 1, 3: 2}  # 第3层使用第2层特征，第4层使用第3层特征
    # 配置特征适配器的维度压缩比例：键为目标层索引，值为压缩比例
    config.cross_attn_config.adapter_ratios = {2: 0.5, 3: 0.5}
    # 配置跨层注意力的头数，默认与对应层的num_heads相同
    config.cross_attn_config.cross_heads = {}
    # 跨层特征融合权重
    config.cross_attn_config.fusion_weights = {2: 0.5, 3: 0.5}  # 默认各层权重均为0.5
    return config
def get_UTSRMorphSmall_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 48
    config.depths = (2, 2, 2, 2)
    config.num_heads = (4, 4, 4, 4)
    config.window_size = (6, 5, 6)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (192, 160, 192)
    return config
def get_UTSRMorphLarge_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (2, 2, 18, 2)
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (6, 5, 6)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (192, 160, 192)
    config.d_state = 16
    config.d_conv = 4
    config.expand = 2
    config.wavelet = 'haar'
    config.out_fmaps = ['P4', 'P3', 'P2', 'P1']
    # 添加跨层注意力相关配置
    config.cross_attn_config = ml_collections.ConfigDict()
    # 配置跨层连接：键为目标层索引，值为源层索引
    config.cross_attn_config.connections = {2: 1, 3: 2}  # 第3层使用第2层特征，第4层使用第3层特征
    # 配置特征适配器的维度压缩比例：键为目标层索引，值为压缩比例
    config.cross_attn_config.adapter_ratios = {2: 0.5, 3: 0.5}
    # 配置跨层注意力的头数，默认与对应层的num_heads相同
    config.cross_attn_config.cross_heads = {}
    # 跨层特征融合权重
    config.cross_attn_config.fusion_weights = {2: 0.5, 3: 0.5}  # 默认各层权重均为0.5
    return config



