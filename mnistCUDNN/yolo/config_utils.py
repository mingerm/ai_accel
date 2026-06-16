from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config_section(path: str | Path, section: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    data = load_config(config_path)
    value = data.get(section, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section must be a mapping: {section}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return parse_simple_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data)


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            without_comment = raw.split("#", 1)[0].rstrip()
            if not without_comment.strip() or ":" not in without_comment:
                continue

            indent = len(without_comment) - len(without_comment.lstrip())
            key, value = without_comment.strip().split(":", 1)
            key = key.strip()
            value = value.strip()

            if indent == 0:
                if value == "":
                    current_section = {}
                    data[key] = current_section
                else:
                    data[key] = parse_scalar(value)
                    current_section = None
            elif current_section is not None:
                current_section[key] = parse_scalar(value)

    return data


def parse_scalar(value: str) -> Any:
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
