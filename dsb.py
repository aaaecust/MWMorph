import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# === 测试 1: 纯 Module 属性（应该能识别）===
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv2 = nn.Conv3d(128, 256, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# === 测试 2: 使用 F.conv3d（无法识别）===
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(128, 64, 3, 3, 3))
        self.weight2 = nn.Parameter(torch.randn(256, 128, 3, 3, 3))

    def forward(self, x):
        x = F.conv3d(x, self.weight1, stride=1, padding=1)
        x = F.conv3d(x, self.weight2, stride=1, padding=1)
        return x


# === 测试 3: 混合使用 ===
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(64, 128, 3, padding=1)
        self.weight_custom = nn.Parameter(torch.randn(256, 128, 3, 3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = F.conv3d(x, self.weight_custom, stride=1, padding=1)
        return x


# === 测试 4: 动态创建层（无法识别）===
class Model4(nn.Module):
    def forward(self, x):
        conv = nn.Conv3d(64, 128, 3, padding=1).to(x.device)
        x = conv(x)
        return x


# === 测试 5: 循环中的卷积 ===
class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        # 方式1: 使用 ModuleList（✅ 能识别）
        self.convs = nn.ModuleList([
            nn.Conv3d(64, 64, 3, padding=1) for _ in range(3)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class Model6(nn.Module):
    def __init__(self):
        super().__init__()
        # 方式2: 单个 weight，循环调用 F.conv3d（❌ 可能统计错误）
        self.weight = nn.Parameter(torch.randn(64, 64, 3, 3, 3))

    def forward(self, x):
        for _ in range(3):
            x = F.conv3d(x, self.weight, stride=1, padding=1)
        return x


# === 运行测试 ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(1, 64, 32, 32, 32).to(device)

models = [
    ("Model1: 纯Module属性", Model1()),
    ("Model2: 纯F.conv3d", Model2()),
    ("Model3: 混合使用", Model3()),
    ("Model4: 动态创建", Model4()),
    ("Model5: ModuleList循环", Model5()),
    ("Model6: F.conv3d循环", Model6()),
]

print("=" * 80)
print(f"{'模型':<30} | {'手动参数':<15} | {'thop参数':<15} | {'thop FLOPs':<15}")
print("=" * 80)

for name, model in models:
    model = model.to(device)

    # 手动统计参数
    params_manual = sum(p.numel() for p in model.parameters())

    # thop 统计
    try:
        flops, params_thop = profile(model, inputs=(x,), verbose=False)
        flops_str = f"{flops / 1e9:.3f}G"
        params_thop_str = f"{params_thop / 1e6:.3f}M"
    except Exception as e:
        flops_str = "ERROR"
        params_thop_str = "ERROR"

    # 对比
    params_manual_str = f"{params_manual / 1e6:.3f}M"
    match = "✅" if abs(params_manual - params_thop) < 100 else "❌"

    print(f"{name:<30} | {params_manual_str:<15} | {params_thop_str:<15} | {flops_str:<15} {match}")

print("=" * 80)
