"""保护 VLA 文本编码器可在冻结缓存下离线复现。"""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from vla_project.training import vla_model


class OfflineTextEncoderTests(unittest.TestCase):
    def test_text_encoder_never_checks_network(self):
        encoder = Mock()
        encoder.get_embedding_dimension.return_value = 384
        encoder.get_sentence_embedding_dimension.return_value = 384
        constructor = Mock(return_value=encoder)
        fake_module = types.SimpleNamespace(SentenceTransformer=constructor)

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            actual, dimension = vla_model._make_text_encoder()

        constructor.assert_called_once_with(
            "paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
        self.assertIs(actual, encoder)
        self.assertEqual(dimension, 384)


if __name__ == "__main__":
    unittest.main()
