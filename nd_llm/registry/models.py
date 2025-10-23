"""Data models and registry utilities for ND-LLM field schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
)

from nd_llm.encoders import Encoder

try:  # pragma: no cover - import shim for optional dependency
    import yaml as _pyyaml  # type: ignore[import-not-found,import-untyped]
except ModuleNotFoundError:  # pragma: no cover - dependency injection point
    _pyyaml = None  # type: ignore[assignment]


def _split_flow_items(text: str) -> List[str]:
    items: List[str] = []
    depth = 0
    token: List[str] = []
    in_string: Optional[str] = None
    for idx, ch in enumerate(text):
        if in_string:
            token.append(ch)
            if ch == in_string and (idx == 0 or text[idx - 1] != "\\"):
                in_string = None
            continue
        if ch in {"'", '"'}:
            in_string = ch
            token.append(ch)
        elif ch in "[{":
            depth += 1
            token.append(ch)
        elif ch in "]}":
            depth -= 1
            token.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(token).strip())
            token = []
        else:
            token.append(ch)
    if token:
        items.append("".join(token).strip())
    return [item for item in items if item]


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "":
        return ""
    if token[0] == token[-1] and token[0] in {"'", '"'} and len(token) >= 2:
        return token[1:-1]
    lowered = token.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if token.startswith("0") and token != "0":
            raise ValueError
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def _parse_flow_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return _parse_flow_sequence(text)
    if text.startswith("{") and text.endswith("}"):
        return _parse_flow_mapping(text)
    return _parse_scalar(text)


def _parse_flow_sequence(text: str) -> List[Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    return [_parse_flow_value(part) for part in _split_flow_items(inner)]


def _parse_flow_mapping(text: str) -> Dict[str, Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return {}
    mapping: Dict[str, Any] = {}
    for part in _split_flow_items(inner):
        key, _, value = part.partition(":")
        if not _:
            raise ValueError("Invalid inline mapping entry in YAML fallback parser")
        mapping[str(_parse_scalar(key))] = _parse_flow_value(value)
    return mapping


def _prepare_lines(text: str) -> List[tuple[int, str]]:
    lines: List[tuple[int, str]] = []
    for raw in str(text).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    return lines


def _parse_block(
    lines: List[tuple[int, str]], index: int, indent: int
) -> tuple[Any, int]:
    if index >= len(lines):
        return None, index
    indent_level, content = lines[index]
    if indent_level < indent:
        return None, index

    if content.startswith("-"):
        result: List[Any] = []
        while index < len(lines):
            line_indent, line_content = lines[index]
            if line_indent < indent or not line_content.startswith("-"):
                break
            value_text = line_content[1:].strip()
            index += 1
            if value_text:
                result.append(_parse_flow_value(value_text))
            else:
                nested, index = _parse_block(lines, index, line_indent + 2)
                result.append(nested)
        return result, index

    result_dict: Dict[str, Any] = {}
    while index < len(lines):
        line_indent, line_content = lines[index]
        if line_indent < indent or line_content.startswith("-"):
            break
        if ":" not in line_content:
            raise ValueError("Invalid mapping entry in YAML fallback parser")
        key_part, _, value_part = line_content.partition(":")
        key = str(_parse_scalar(key_part))
        value_part = value_part.strip()
        index += 1
        if value_part:
            result_dict[key] = _parse_flow_value(value_part)
        else:
            if index < len(lines) and lines[index][0] > line_indent:
                nested, index = _parse_block(lines, index, line_indent + 2)
                result_dict[key] = nested
            else:
                result_dict[key] = None
    return result_dict, index


def _fallback_safe_load(stream: Any) -> Any:
    if stream is None:
        return None
    text = stream.decode() if isinstance(stream, (bytes, bytearray)) else str(stream)
    lines = _prepare_lines(text)
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index < len(lines):
        remaining = lines[index:]
        raise ValueError(
            f"Unable to parse YAML fallback beyond line {index}: {remaining!r}"
        )
    return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return "''"
    if any(ch in text for ch in "{}[],:#") or text.strip() != text or " " in text:
        return repr(text)
    return text


def _dump_block(value: Any, indent: int, sort_keys: bool) -> List[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        unordered_items = value.items()
        items: Iterable[tuple[Any, Any]]
        if sort_keys:
            items = sorted(unordered_items)
        else:
            items = unordered_items
        for key, item in items:
            key_text = _format_scalar(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_dump_block(item, indent + 2, sort_keys))
            else:
                lines.append(f"{prefix}{key_text}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_dump_block(item, indent + 2, sort_keys))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return lines
    return [f"{prefix}{_format_scalar(value)}"]


def _fallback_safe_dump(data: Any, sort_keys: bool = False) -> str:
    lines = _dump_block(data, 0, sort_keys)
    return "\n".join(lines) + ("\n" if lines else "")


class _YamlAPI(Protocol):
    """Subset of the PyYAML API used by the registry helpers."""

    def safe_load(self, stream: Any) -> Any: ...

    def safe_dump(self, data: Any, **kwargs: Any) -> str: ...


class _FallbackYaml:
    """Drop-in object that mirrors the subset of PyYAML we rely on."""

    def safe_load(self, stream: Any) -> Any:
        return _fallback_safe_load(stream)

    def safe_dump(self, data: Any, **kwargs: Any) -> str:
        sort_keys = bool(kwargs.get("sort_keys", False))
        return _fallback_safe_dump(data, sort_keys=sort_keys)


if _pyyaml is None:  # pragma: no cover - exercised in environments without PyYAML
    yaml: _YamlAPI = _FallbackYaml()
else:  # pragma: no cover - prefer system PyYAML when available
    yaml = cast(_YamlAPI, _pyyaml)


@dataclass(slots=True)
class FieldSpec:
    """Specification describing a field tracked in the registry."""

    name: str
    keys: List[str] = field(default_factory=list)
    salience: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.keys = list(dict.fromkeys(self.keys))  # deduplicate while preserving order
        if not all(isinstance(k, str) and k for k in self.keys):
            raise ValueError(
                f"Field '{self.name}' has invalid key entries: {self.keys!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"keys": list(self.keys)}
        if self.salience:
            data["salience"] = self.salience
        if self.metadata:
            data.update(self.metadata)
        return data

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "FieldSpec":
        keys = data.get("keys", [])
        salience = bool(data.get("salience", False))
        known_keys = {"keys", "salience"}
        metadata = {k: v for k, v in data.items() if k not in known_keys}
        return cls(name=name, keys=list(keys), salience=salience, metadata=metadata)


@dataclass(slots=True)
class AffinityRule:
    """Describes an alignment constraint between two registry fields."""

    source: str
    target: str
    keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.keys = list(dict.fromkeys(self.keys))
        if not self.keys:
            raise ValueError("Affinity rules require at least one alignment key")
        if not all(isinstance(k, str) and k for k in self.keys):
            raise ValueError(f"Affinity rule has invalid key entries: {self.keys!r}")

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "fields": [self.source, self.target],
            "by": list(self.keys),
        }
        if self.metadata:
            data.update(self.metadata)
        return data

    @classmethod
    def from_raw(cls, raw: Union[Sequence[Any], Mapping[str, Any]]) -> "AffinityRule":
        if isinstance(raw, Mapping):
            fields = raw.get("fields")
            if not isinstance(fields, Sequence) or len(fields) != 2:
                raise ValueError("Affinity mapping must contain exactly two fields")
            source, target = fields
            by_keys = raw.get("by")
            if by_keys is None:
                raise ValueError("Affinity mapping missing 'by' key list")
            mapping_metadata = {
                k: v for k, v in raw.items() if k not in {"fields", "by"}
            }
            return cls(
                source=str(source),
                target=str(target),
                keys=list(by_keys),
                metadata=mapping_metadata,
            )

        if isinstance(raw, Sequence):
            if len(raw) < 3:
                raise ValueError(
                    "Affinity sequence must include fields and key mapping"
                )
            source = str(raw[0])
            target = str(raw[1])
            keys: Optional[Iterable[str]] = None
            metadata: Dict[str, Any] = {}
            for item in raw[2:]:
                if isinstance(item, Mapping):
                    if "by" in item:
                        keys = item["by"]
                    else:
                        metadata.update(item)
                else:
                    raise ValueError("Unexpected positional entry in affinity rule")
            if keys is None:
                raise ValueError("Affinity sequence missing 'by' key list")
            return cls(source=source, target=target, keys=list(keys), metadata=metadata)

        raise TypeError("Affinity definition must be a mapping or sequence")


@dataclass
class Registry:
    """Registry storing declared fields and alignment affinities."""

    fields: Dict[str, FieldSpec] = field(default_factory=dict)
    affinities: List[AffinityRule] = field(default_factory=list)
    _encoders: Dict[str, Encoder] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def add_field(
        self,
        name: str,
        *,
        keys: Iterable[str],
        salience: bool | None = None,
        **metadata: Any,
    ) -> FieldSpec:
        if name in self.fields:
            raise ValueError(f"Field '{name}' is already registered")
        if salience is None:
            salience = bool(metadata.pop("salience", False))
        spec = FieldSpec(
            name=name, keys=list(keys), salience=bool(salience), metadata=metadata
        )
        self.fields[name] = spec
        return spec

    def add_affinity(
        self,
        source: str,
        target: str,
        *,
        keys: Iterable[str],
        **metadata: Any,
    ) -> AffinityRule:
        self._ensure_field_exists(source)
        self._ensure_field_exists(target)
        rule = AffinityRule(
            source=source, target=target, keys=list(keys), metadata=metadata
        )
        self._validate_affinity_keys(rule)
        self.affinities.append(rule)
        return rule

    def validate(self) -> None:
        for name, field_spec in self.fields.items():
            if not field_spec.keys:
                raise ValueError(
                    f"Field '{name}' must declare at least one alignment key"
                )
        for rule in self.affinities:
            self._validate_affinity_keys(rule)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": {name: spec.to_dict() for name, spec in self.fields.items()},
            "affinity": [rule.to_dict() for rule in self.affinities],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Registry":
        fields_data = data.get("fields", {})
        if not isinstance(fields_data, Mapping):
            raise TypeError("'fields' must be a mapping of field specifications")
        registry = cls()
        for name, spec_data in fields_data.items():
            if not isinstance(spec_data, Mapping):
                raise TypeError(f"Field '{name}' specification must be a mapping")
            registry.fields[str(name)] = FieldSpec.from_dict(str(name), spec_data)

        affinity_data = data.get("affinity", [])
        if isinstance(affinity_data, Sequence):
            for raw in affinity_data:
                rule = AffinityRule.from_raw(raw)
                registry._ensure_field_exists(rule.source)
                registry._ensure_field_exists(rule.target)
                registry._validate_affinity_keys(rule)
                registry.affinities.append(rule)
        elif affinity_data is not None:
            raise TypeError("'affinity' must be a sequence of affinity rules")

        registry.validate()
        return registry

    def to_yaml(self) -> str:
        self._ensure_yaml_available()
        return yaml.safe_dump(
            self.to_dict(), sort_keys=False
        )  # type: ignore[union-attr]

    @classmethod
    def from_yaml(cls, source: Union[str, Path, Any]) -> "Registry":
        data = cls._load_yaml(source)
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise TypeError("YAML registry definition must produce a mapping")
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Encoder registration helpers
    # ------------------------------------------------------------------

    def register_encoder(self, field: str, encoder: Encoder) -> None:
        """Associate ``encoder`` with ``field`` for downstream compression."""

        if not isinstance(field, str) or not field:
            raise ValueError("field name must be a non-empty string")
        if not isinstance(encoder, Encoder):
            raise TypeError("encoder must implement the Encoder protocol")
        self._encoders[field] = encoder

    def get_encoder(self, field: str) -> Encoder:
        try:
            return self._encoders[field]
        except KeyError as exc:  # pragma: no cover - defensive safeguard
            raise KeyError(f"encoder not registered for field '{field}'") from exc

    @property
    def encoders(self) -> Mapping[str, Encoder]:
        """Return a shallow copy of the registered encoder mapping."""

        return dict(self._encoders)

    @staticmethod
    def _load_yaml(source: Union[str, Path, Any]) -> Any:
        Registry._ensure_yaml_available()
        if hasattr(source, "read"):
            content = source.read()
            return yaml.safe_load(content)  # type: ignore[union-attr]

        if isinstance(source, (str, Path)):
            path = Path(source)
            if isinstance(source, str) and (
                "\n" in source or ":" in source or source.strip().startswith("{")
            ):
                return yaml.safe_load(source)  # type: ignore[union-attr]
            if path.exists():
                return yaml.safe_load(path.read_text())  # type: ignore[union-attr]
            return yaml.safe_load(str(source))  # type: ignore[union-attr]

        raise TypeError("Unsupported YAML source type")

    @staticmethod
    def _ensure_yaml_available() -> None:
        return None

    def _ensure_field_exists(self, name: str) -> None:
        if name not in self.fields:
            raise ValueError(f"Field '{name}' is not registered")

    def _validate_affinity_keys(self, rule: AffinityRule) -> None:
        source_keys = set(self.fields[rule.source].keys)
        target_keys = set(self.fields[rule.target].keys)
        missing_source = [k for k in rule.keys if k not in source_keys]
        missing_target = [k for k in rule.keys if k not in target_keys]
        if missing_source or missing_target:
            raise ValueError(
                "Affinity keys must be present in both fields: "
                f"missing_from_source={missing_source}, "
                f"missing_from_target={missing_target}"
            )
