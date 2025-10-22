import pytest
import torch

from nd_llm.bottleneck import (
    IBottleneck,
    LearnableTokenScorer,
    QueryDotProductScoringStrategy,
    configure_scorer,
)
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


def test_topk_selection_respects_budget_and_telemetry():
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

    assert result.telemetry.selected_indices["text"] == [1, 2]
    assert result.telemetry.selected_indices["layout"] == [1]
    assert result.telemetry.field_budgets["text"] == 2
    assert result.telemetry.field_budgets["layout"] == 1
    assert result.metrics["information_bound"] == pytest.approx(3 / 5)
    assert result.telemetry.dropped_indices["text"] == [0]

    reconstructed = bottleneck.decompress(result)
    assert reconstructed["kept"] == result.compressed_fields
    assert set(reconstructed["regenerated"]) == {"text", "layout"}
    assert len(reconstructed["regenerated"]["text"]) == 1
    assert reconstructed["metrics"]["retained_ratio"] == pytest.approx(3 / 5)
    assert reconstructed["metrics"]["fields"]["text"]["mse"] >= 0.0


def test_query_conditioned_scoring_prefers_query_aligned_tokens():
    fields = {"text": ["zero", "one", "two"]}
    encoders = {"text": MockEncoder([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])}
    query_vector = [1.0, 0.0]

    bottleneck = IBottleneck(
        target_budget=1,
        scorer=QueryDotProductScoringStrategy(mix_weight=1.0),
    )
    result = bottleneck.compress(fields, encoders, context={"query_embedding": query_vector})

    assert result.telemetry.selected_indices["text"] == [0]
    assert result.telemetry.field_budgets["text"] == 1
    assert result.telemetry.selected_scores["text"][0] == pytest.approx(1.0)


def test_budget_allocator_respects_salience_metadata():
    registry = Registry()
    registry.add_field("salient", keys=["doc_id"], salience=True)
    registry.add_field("context", keys=["doc_id"])

    fields = {
        "salient": ["s0", "s1", "s2"],
        "context": ["c0", "c1", "c2", "c3"],
    }
    encoders = {
        "salient": MockEncoder([[0.6], [0.7], [0.8]]),
        "context": MockEncoder([[0.1], [0.2], [0.3], [0.4]]),
    }

    bottleneck = IBottleneck(target_budget=3)
    result = bottleneck.compress(fields, encoders, registry=registry)

    assert result.telemetry.field_budgets["salient"] == 2
    assert result.telemetry.field_budgets["context"] == 1
    assert result.telemetry.allocation_weights["salient"] > result.telemetry.allocation_weights["context"]
    assert sum(len(v) for v in result.telemetry.selected_indices.values()) == 3


def test_metrics_report_information_proxies():
    fields = {"text": ["low", "high"]}
    encoders = {"text": MockEncoder([[1.0], [3.0]])}

    bottleneck = IBottleneck(target_budget=1)
    result = bottleneck.compress(fields, encoders)

    metrics = result.metrics
    assert metrics["information_bound"] == pytest.approx(0.5)
    assert metrics["ib_proxy"] == pytest.approx(0.9)
    assert metrics["rd_proxy"] == pytest.approx(0.5)
    assert metrics["embedding_reconstruction_error"] == pytest.approx(2.0)
    assert result.telemetry.dropped_indices["text"] == [0]
    assert result.loss_terms == {}


def test_scorer_configuration_allows_strategy_swapping():
    fields = {"text": ["zero", "one", "two"]}
    encoders = {"text": MockEncoder([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])}

    norm_fn, norm_module = configure_scorer("norm")
    assert callable(norm_fn)
    assert norm_module is None

    default_bottleneck = IBottleneck(target_budget=1)
    default_result = default_bottleneck.compress(fields, encoders)

    norm_bottleneck = IBottleneck(target_budget=1, scorer_config="norm")
    norm_result = norm_bottleneck.compress(fields, encoders)

    assert norm_result.telemetry.selected_indices == default_result.telemetry.selected_indices
    assert norm_result.loss_terms == {}

    learn_config = {
        "type": "learnable",
        "embedding_dim": 2,
        "hidden_dims": (4,),
        "context_dim": 2,
    }
    learn_bottleneck = IBottleneck(target_budget=1, scorer_config=learn_config)
    assert isinstance(learn_bottleneck.learnable_scorer, LearnableTokenScorer)
    learn_result = learn_bottleneck.compress(fields, encoders, context={"query_embedding": [1.0, 0.0]})
    assert set(learn_result.loss_terms.keys()) >= {"ib_proxy", "rd_proxy"}
    _, helper_module = configure_scorer({**learn_config, "type": "learnable"})
    assert isinstance(helper_module, LearnableTokenScorer)


def test_learnable_scorer_yields_trainable_loss_terms():
    torch.manual_seed(0)
    fields = {"text": ["a", "b", "c"]}
    encoders = {"text": MockEncoder([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])}
    scorer = LearnableTokenScorer(embedding_dim=2, context_dim=2, hidden_dims=(8,))
    bottleneck = IBottleneck(target_budget=2, learnable_scorer=scorer)

    result = bottleneck.compress(fields, encoders, context={"query_embedding": [0.6, 0.4]})

    assert "ib_proxy" in result.loss_terms
    assert "rd_proxy" in result.loss_terms
    combined_loss = result.loss_terms["rd_proxy"] - result.loss_terms["ib_proxy"]
    combined_loss.backward()
    grads = [param.grad for param in scorer.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)


def test_encoder_protocol_and_registry_compatibility():
    text_encoder = TextEncoder()
    layout_encoder = LayoutEncoder()

    assert isinstance(text_encoder, Encoder)
    assert isinstance(layout_encoder, Encoder)

    registry = Registry()
    registry.add_field("text", keys=["doc_id", "span_id"], salience=True)
    registry.add_field("layout", keys=["doc_id", "span_id"])
    registry.register_encoder("text", text_encoder)
    registry.register_encoder("layout", layout_encoder)

    fields = {
        "text": ["hello", "world"],
        "layout": [{"xyxy": (0, 0, 1, 1)}, {"xyxy": (1, 1, 2, 2)}],
    }

    bottleneck = IBottleneck(target_budget=2)
    result = bottleneck.compress(fields, registry.encoders, registry=registry)

    assert set(result.telemetry.selected_indices.keys()) == {"text", "layout"}
    assert all(isinstance(v, list) for v in result.compressed_fields.values())
    assert sum(result.telemetry.field_budgets.values()) == 2
