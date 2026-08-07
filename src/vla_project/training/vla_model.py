"""VLA 模型：ResNet-18 视觉 + MiniLM 文本编码 + 融合 + RegressionHead。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import RegressionHead, _make_resnet_backbone


def _make_text_encoder():
    """加载 all-MiniLM-L6-v2，返回 (encoder, embedding_dim)。"""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder, encoder.get_sentence_embedding_dimension()  # 384


class VLAModel(nn.Module):
    """图片 + 文本指令 → 关节动作。

    Args:
        action_representation: 当前仅支持 "regression"。
    """

    def __init__(self, action_representation="regression"):
        super().__init__()
        if action_representation != "regression":
            raise ValueError(
                f"VLAModel 当前仅支持 regression，收到: {action_representation}"
            )
        self.action_representation = action_representation
        self._is_classification = False

        # 视觉 backbone（与 BCModel 共用）
        self.visual_backbone, visual_dim = _make_resnet_backbone()

        # 文本编码器
        self.text_encoder, text_dim = _make_text_encoder()

        # 融合层
        fused_dim = visual_dim + text_dim  # 512 + 384 = 896
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
        )

        # 动作头（复用它）
        self.head = RegressionHead(feature_dim=256)

    def encode_texts(self, texts):
        """将一批指令字符串编码为嵌入向量。

        Args:
            texts: list[str] 或单个 str

        Returns:
            torch.Tensor: (B, 384) 在 self.text_encoder 所在设备上
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.text_encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings

    def forward(self, images, text_embeddings):
        """前向传播。

        Args:
            images: (B, 3, 224, 224) RGB 张量
            text_embeddings: (B, 384) 文本嵌入

        Returns:
            (action_pred, aux_pred): 7 维关节角 + 2 维辅助
        """
        # 视觉特征: (B, 512, 7, 7) → GAP → (B, 512)
        feat_map = self.visual_backbone(images)
        visual_feat = feat_map.mean(dim=[2, 3])

        # 融合
        fused = torch.cat([visual_feat, text_embeddings], dim=1)
        features = self.fusion(fused)

        return self.head(features)

    def is_classification(self):
        return self._is_classification
