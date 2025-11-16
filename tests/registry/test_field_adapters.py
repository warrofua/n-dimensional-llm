from __future__ import annotations

from nd_llm.registry import (
    FieldAdapter,
    FieldAdapterRegistry,
    LayoutAligner,
    quad_to_box,
)


def test_field_adapter_registry_adds_coords() -> None:
    def builder(document):
        yield {
            "quad": {
                "x1": 0,
                "y1": 0,
                "x2": 50,
                "y2": 0,
                "x3": 50,
                "y3": 100,
                "x4": 0,
                "y4": 100,
            }
        }

    registry = FieldAdapterRegistry()
    registry.register(
        FieldAdapter(name="layout", builder=builder, aligner=LayoutAligner())
    )

    transformed = registry.transform({"width": 100, "height": 200})
    assert "layout" in transformed
    entry = transformed["layout"][0]
    assert "coords" in entry
    assert entry["coords"] == [0.0, 0.0, 0.5, 0.5]
    assert entry["xyxy"] == entry["coords"]


def test_quad_to_box_handles_complete_values() -> None:
    box = quad_to_box(
        {"x1": 10, "y1": 20, "x2": 30, "y2": 20, "x3": 30, "y3": 80, "x4": 10, "y4": 80}
    )
    assert box == [10.0, 20.0, 30.0, 80.0]
