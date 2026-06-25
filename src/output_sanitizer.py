"""Utilities for removing local filesystem paths from visible outputs."""

from __future__ import annotations

import copy
import re
from typing import Any


PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/][^\]\)\s\r\n]+|[^\]\)\s\r\n]*data[\\/]raw[\\/][^\]\)\s\r\n]+)"
)


def sanitize_source_value(value: str) -> str:
    text = str(value or "")
    if not text:
        return text
    normalized = text.replace("\\", "/")
    return normalized.rstrip("/").split("/")[-1]


def sanitize_text(text: str) -> str:
    value = str(text or "")

    def replace_path(match: re.Match[str]) -> str:
        return sanitize_source_value(match.group(0))

    return PATH_PATTERN.sub(replace_path, value)


def sanitize_sources(sources: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for source in sources or []:
        item = copy.deepcopy(source)
        for key, value in list(item.items()):
            if isinstance(value, str):
                if "\\" in value or "/" in value:
                    item[key] = sanitize_source_value(value)
                else:
                    item[key] = sanitize_text(value)
        cleaned.append(item)
    return cleaned


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    return value


def sanitize_result_dict(result: dict) -> dict:
    sanitized = copy.deepcopy(result or {})
    if sanitized.get("answer"):
        sanitized["answer"] = sanitize_text(sanitized["answer"])
    if sanitized.get("final_answer"):
        sanitized["final_answer"] = sanitize_text(sanitized["final_answer"])
    if sanitized.get("sources"):
        sanitized["sources"] = sanitize_sources(sanitized["sources"])
    if sanitized.get("evidence"):
        sanitized["evidence"] = _sanitize_value(sanitized["evidence"])
    return sanitized
