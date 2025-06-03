import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # 224x224 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layer1 = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),  # 112x112
            InvertedResidual(16, 24, 2, 6),  # 56x56
            InvertedResidual(24, 24, 1, 6),
        )

        self.layer2 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),  # 28x28
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
        )

        self.layer3 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),  # 14x14
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 1, 6),  # Channel expansion
        )

        self.layer4 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),  # 7x7
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 320, 1, 6),  # Final expansion
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(320 + 96 + 32, 256, 1, bias=False),  # Fuse multi-scale features
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)

        x1 = self.layer1(x)  # 56x56, 24 channels
        x2 = self.layer2(x1)  # 28x28, 32 channels
        x3 = self.layer3(x2)  # 14x14, 96 channels
        x4 = self.layer4(x3)  # 7x7, 320 channels

        x2_up = F.interpolate(
            x2, size=x4.shape[2:], mode="bilinear", align_corners=False
        )
        x3_up = F.interpolate(
            x3, size=x4.shape[2:], mode="bilinear", align_corners=False
        )

        fused = torch.cat([x4, x3_up, x2_up], dim=1)
        fused = self.fusion_conv(fused)

        x = self.global_pool(fused).flatten(1)
        x = self.dropout(x)

        return x


class Head(nn.Module):
    def __init__(
        self,
        in_features: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 1,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)

        return torch.sigmoid(x)


class MultiTaskModel(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.backbone = EfficientBackbone()

        self.heads = nn.ModuleList(
            [
                Head(
                    in_features=256,
                    hidden_dim=128,
                )
                for i in range(num_tasks)
            ]
        )

        self.num_tasks = num_tasks

    def forward(self, x: Tensor) -> List[Tensor]:
        features = self.backbone(x)

        outputs = [head(features) for head in self.heads]
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    model = MultiTaskModel(num_tasks=5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    outputs = model(x)

    print(f"Input shape: {x.shape}")
    for i, output in enumerate(outputs):
        print(f"Task {i} output shape: {output.shape}")
