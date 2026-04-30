from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .text_utils import clean_text, clean_token


def bow_to_text(
    bow_vector: Optional[Mapping[Any, Any]],
    vocab: Optional[Mapping[Any, str]],
    max_tokens: int = 256,
    per_word_cap: int = 5,
) -> str:
    """Convert a BoW vector into deterministic pseudo-text for embedding."""
    if not bow_vector:
        return ""

    vocab = vocab or {}
    pieces: List[str] = []
    for raw_wid, raw_count in sorted(bow_vector.items(), key=lambda x: str(x[0])):
        try:
            wid = int(raw_wid)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue

        word = vocab.get(wid) or vocab.get(str(wid)) or f"unk_{wid}"
        token = clean_token(word) or f"unk_{wid}"
        pieces.extend([token] * min(count, max(1, per_word_cap)))
        if len(pieces) >= max_tokens:
            break

    return " ".join(pieces[:max_tokens])


def build_retrieval_text(
    track_document: Mapping[str, Any],
    mxm_vocab: Optional[Mapping[Any, str]] = None,
    max_bow_tokens: int = 256,
) -> str:
    """
    Build the v1 retrieval text for a track document.

    v1 policy:
    - Prefer precomputed `retrieval_text.embedding_text`
    - Otherwise build text from `lyrics.bow_vector` only
    - Do not inject genre/tag/artist/title metadata into the vector text
    """
    td = dict(track_document or {})
    retrieval_text = td.get("retrieval_text") or {}
    precomputed = clean_text(retrieval_text.get("embedding_text"))
    if precomputed:
        return precomputed

    lyrics = td.get("lyrics") or {}
    return bow_to_text(lyrics.get("bow_vector"), mxm_vocab, max_tokens=max_bow_tokens)
