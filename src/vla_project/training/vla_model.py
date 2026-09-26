"""VLA 模型：ResNet-18 视觉 + MiniLM 文本编码 + 融合 + RegressionHead。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import RegressionHead, _make_resnet_backbone


def _make_text_encoder():
    """加载 paraphrase-multilingual-MiniLM-L12-v2，返回 (encoder, embedding_dim)。

    all-MiniLM-L6-v2 是纯英文模型，会把中文"红色"/"蓝色"都映射为 [UNK]，
    两条指令的文本嵌入完全一致。multilingual 版本支持中文，可区分。
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        local_files_only=True,
    )
    try:
        dim = encoder.get_embedding_dimension()
    except AttributeError:
        dim = encoder.get_sentence_embedding_dimension()
    return encoder, dim  # 384


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

        # 目标色辅助头（P1）：从融合特征直接预测"目标是红还是蓝"。
        # 用真实目标色监督它，强制模型在训练中利用语言区分红/蓝，
        # 使融合特征编码语言条件，动作分支随之学会随指令变化。
        self.color_head = nn.Linear(256, 2)

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

    def encode_texts_grad(self, texts):
        """训练用：走完整 transformer 前向，文本嵌入带梯度。

        与 encode_texts() 的区别：encode_texts() 内部使用
        SentenceTransformer.encode()，其输出 detached（requires_grad=False），
        语言分支在训练中拿不到梯度；encode_texts_grad() 通过 dict 协议走
        SentenceTransformer 前向，输出与 encode() 逐位一致但保留计算图，
        使语言编码器参数可以参与反向传播。推理路径仍用 encode_texts()。
        """
        if isinstance(texts, str):
            texts = [texts]
        device = next(self.parameters()).device
        tokens = self.text_encoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_encoder.max_seq_length,
            return_tensors="pt",
        )
        features = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.text_encoder(features)
        return outputs["sentence_embedding"]

    def forward(self, images, text_embeddings):
        """前向传播。

        Args:
            images: (B, 3, 224, 224) RGB 张量
            text_embeddings: (B, 384) 文本嵌入

        Returns:
            (action_pred, aux_pred, color_logits):
                7 维关节角 + 2 维辅助（gripper/terminate）+ 目标色 logits(红/蓝)
        """
        # 视觉特征: (B, 512, 7, 7) → GAP → (B, 512)
        feat_map = self.visual_backbone(images)
        visual_feat = feat_map.mean(dim=[2, 3])

        # 融合
        fused = torch.cat([visual_feat, text_embeddings], dim=1)
        features = self.fusion(fused)

        action_pred, aux_pred = self.head(features)
        color_logits = self.color_head(features)
        return action_pred, aux_pred, color_logits

    def is_classification(self):
        return self._is_classification
