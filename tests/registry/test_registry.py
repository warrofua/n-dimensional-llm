import textwrap

import pytest

from nd_llm.registry import Registry


def test_add_field_success():
    registry = Registry()
    spec = registry.add_field("text", keys=["doc_id", "span_id"], salience=True)

    assert registry.fields["text"] is spec
    assert spec.salience is True
    assert spec.keys == ["doc_id", "span_id"]


def test_add_field_duplicate():
    registry = Registry()
    registry.add_field("text", keys=["doc_id"])

    with pytest.raises(ValueError):
        registry.add_field("text", keys=["doc_id"])


def test_add_affinity_validation():
    registry = Registry()
    registry.add_field("text", keys=["doc_id", "span_id"])
    registry.add_field("bbox", keys=["doc_id"])

    with pytest.raises(ValueError):
        registry.add_affinity("text", "bbox", keys=["doc_id", "span_id"])


def test_add_affinity_requires_fields():
    registry = Registry()
    registry.add_field("text", keys=["doc_id"])

    with pytest.raises(ValueError):
        registry.add_affinity("text", "bbox", keys=["doc_id"])


def test_yaml_round_trip_from_readme():
    yaml_text = textwrap.dedent(
        """
        fields:
          text:
            keys: [doc_id, span_id]
            salience: true
          bbox:
            keys: [doc_id, span_id]
          timestamp:
            keys: [doc_id, frame_id, session_id, t]
          audio_chunk:
            keys: [session_id, t]
        affinity:
          - [text, bbox, {by: [doc_id, span_id]}]
          - [text, timestamp, {by: [doc_id]}]
          - [audio_chunk, timestamp, {by: [session_id, t]}]
        """
    ).strip()

    registry = Registry.from_yaml(yaml_text)
    assert set(registry.fields) == {"text", "bbox", "timestamp", "audio_chunk"}
    assert registry.fields["text"].salience is True
    assert registry.affinities[0].keys == ["doc_id", "span_id"]

    serialized = registry.to_yaml()
    registry_roundtrip = Registry.from_yaml(serialized)
    assert registry_roundtrip.to_dict() == registry.to_dict()
