from __future__ import annotations

import pytest

from nd_llm.stm import STM
from nd_llm.utils import STMConfig


def test_superposition_write_and_read(tmp_path) -> None:
    stm = STM(STMConfig(storage_dir=tmp_path))

    stm.write_superposition("usage", [1.0, 2.0], weight=1.0, metadata={"task": "alpha"})
    stm.write_superposition("usage", [3.0, 1.0], weight=2.0)

    vector, metadata = stm.read_superposition("usage")
    assert pytest.approx(vector[0], rel=1e-6) == (1.0 + 6.0) / 3.0
    assert pytest.approx(vector[1], rel=1e-6) == (2.0 + 2.0) / 3.0
    assert metadata["channel"] == "usage"
    assert pytest.approx(metadata["weight"], rel=1e-6) == 3.0

    raw_vector, _ = stm.read_superposition("usage", normalize=False)
    assert raw_vector == pytest.approx([7.0, 4.0])
