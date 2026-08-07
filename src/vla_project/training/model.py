"""ResNet-18 backbone + 可切换的 action head。

支持三种动作表示：
- regression: 7-d 连续 absolute_q (SmoothL1)
- absolute_q_32: 7×32 分类 (CrossEntropy per joint)
- delta_q_64: 7×64 分类 (CrossEntropy per joint)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_resnet_backbone():
    """返回去掉 fc 层的 ResNet-18 (ImageNet 预训练)。"""
    import torchvision.models as tv_models

    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    # 去掉最后的 fc 和 avgpool，保留 feature extractor
    modules = list(resnet.children())[:-2]  # 保留到 layer4, 去掉 avgpool+fc
    backbone = nn.Sequential(*modules)
    return backbone, 512  # ResNet-18 layer4 输出 512 通道


class RegressionHead(nn.Module):
    """连续回归头：7-d absolute_q + auxiliary (gripper, terminate)。"""

    def __init__(self, feature_dim=512, hidden_dim=256, num_joints=7):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_joints),
        )
        self.aux_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # gripper, terminate
        )

    def forward(self, features):
        action = self.action_net(features)
        aux = self.aux_net(features)
        return action, aux


class ClassificationHead(nn.Module):
    """每关节独立分类头：7 × num_bins 类 + auxiliary。"""

    def __init__(self, feature_dim=512, hidden_dim=256, num_joints=7, num_bins=32):
        super().__init__()
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.action_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_joints * num_bins),
        )
        self.aux_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, features):
        logits_flat = self.action_net(features)  # (B, 7*num_bins)
        logits = logits_flat.view(-1, self.num_joints, self.num_bins)
        aux = self.aux_net(features)
        return logits, aux


class BCModel(nn.Module):
    """BC 模型：ResNet-18 backbone + 可切换 action head。

    Args:
        action_representation: "regression" | "absolute_q_32" | "delta_q_64"
    """

    def __init__(self, action_representation="regression"):
        super().__init__()
        self.action_representation = action_representation
        self.backbone, feature_dim = _make_resnet_backbone()

        if action_representation == "regression":
            self.head = RegressionHead(feature_dim)
            self._is_classification = False
        elif action_representation == "absolute_q_32":
            self.head = ClassificationHead(feature_dim, num_bins=32)
            self._is_classification = True
        elif action_representation == "delta_q_64":
            self.head = ClassificationHead(feature_dim, num_bins=64)
            self._is_classification = True
        else:
            raise ValueError(
                f"未知的动作表示: {action_representation}"
            )

    def forward(self, images):
        """返回 (action_pred, aux_pred)。"""
        # ResNet backbone: (B, 3, 224, 224) → (B, 512, 7, 7)
        feat_map = self.backbone(images)
        # Global average pooling → (B, 512)
        features = feat_map.mean(dim=[2, 3])
        return self.head(features)

    def is_classification(self):
        return self._is_classification
