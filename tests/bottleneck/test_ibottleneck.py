import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from nd_llm.bottleneck import IBottleneck
from nd_llm.encoders import Encoder, LayoutEncoder, TextEncoder
from nd_llm.registry import Registry


class MockEncoder:
    def __init__(self, embeddings):
        dim = len(embeddings[0]) if embeddings else 0
        if any(len(vec) != dim for vec in embeddings):
            raise ValueError("embeddings must share dimension")
        self.embedding_dim = dim
        self._embeddings = [list(vec) for vec in embeddings]

    def encode(self, batch):  # noqa: D401 - protocol hook
        if len(batch) != len(self._embeddings):
            raise AssertionError("mock encoder expected different batch size")
        return self._embeddings


def test_topk_selection_respects_budget():
    fields = {
        "text": ["alpha", "beta", "gamma"],
        "layout": ["b0", "b1"],
    }
    encoders = {
        "text": MockEncoder([[0.1], [0.5], [0.9]]),
        "layout": MockEncoder([[0.2], [0.7]]),
    }

    bottleneck = IBottleneck(target_budget=3)
    result = bottleneck.compress(fields, encoders)

    # The highest norms are: text[2], layout[1], text[1]
    assert result.telemetry.selected_indices["text"] == [1, 2]
    assert result.telemetry.selected_indices["layout"] == [1]

    kept = sum(len(v) for v in result.telemetry.selected_indices.values())
    assert kept == 3
    assert result.metrics["information_bound"] == pytest.approx(3 / 5)

    reconstructed = bottleneck.decompress(result)
    assert reconstructed == result.compressed_fields


def test_encoder_protocol_and_registry_compatibility():
    text_encoder = TextEncoder()
    layout_encoder = LayoutEncoder()

    assert isinstance(text_encoder, Encoder)
    assert isinstance(layout_encoder, Encoder)

    registry = Registry()
    registry.register_encoder("text", text_encoder)
    registry.register_encoder("layout", layout_encoder)

    fields = {
        "text": ["hello", "world"],
        "layout": [{"xyxy": (0, 0, 1, 1)}, {"xyxy": (1, 1, 2, 2)}],
    }

    bottleneck = IBottleneck(target_budget=2)
    result = bottleneck.compress(fields, registry.encoders)

    assert set(result.telemetry.selected_indices.keys()) == {"text", "layout"}
    assert all(isinstance(v, list) for v in result.compressed_fields.values())
