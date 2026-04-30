from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from typing import Any, List, Mapping, Optional, Iterable


def clean_text(value: Any) -> str:
    """
    Clean and normalize text values for consistent processing.

    This function handles None values, converts to string, removes excess whitespace,
    and strips leading/trailing whitespace.

    Args:
        value (Any): Input value to clean.

    Returns:
        str: Cleaned text string. Empty string if input is None.
    """
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_token(value: Any) -> str:
    """
    Clean and normalize text for use as tokens or identifiers.

    This function cleans text and then normalizes it for use as tokens by:
    - Converting to lowercase
    - Replacing whitespace with underscores
    - Removing invalid characters (keeping only alphanumeric, underscore, colon, hyphen, dot, plus)
    - Stripping leading/trailing underscores

    Args:
        value (Any): Input value to tokenize.

    Returns:
        str: Clean token string. Empty string if input is None or becomes empty after cleaning.
    """
    text = clean_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_:/\-\.\+]", "", text)
    return text.strip("_")


def to_list(value: Any) -> List[Any]:
    """
    Convert a value to a list if it isn't already.

    Handles None, single values, and existing iterables.

    Args:
        value (Any): Value to convert to list.

    Returns:
        List[Any]: List containing the value(s). Empty list if value is None.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def unique_tokens(values: Iterable[Any]) -> List[str]:
    """
    Extract unique, cleaned tokens from an iterable of values.

    Processes each value through clean_token(), removes duplicates while
    preserving order, and filters out empty tokens.

    Args:
        values (Iterable[Any]): Values to process into tokens.

    Returns:
        List[str]: List of unique, non-empty tokens in order of first appearance.
    """
    out: List[str] = []
    seen = set()
    for v in values:
        token = clean_token(v)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out
