"""
Simple vector embedding utilities for EchoAgent.

What this file does:
- Manage ChromaDB collections (`track_text`, `lyrics`)
- Build embeddings from TrackDocument dicts
- Convert MXM bag-of-words (BoW) into compact text
- Load and apply Last.fm seed genre from `msd_lastfm_map.cls` / `.zip`

"""

from __future__ import annotations

import os
import re
import zipfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import chromadb
from chromadb.config import Settings
import numpy as np


DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_token(value: Any) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_:/\-\.\+]", "", text)
    return text.strip("_")


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _unique_tokens(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        token = _clean_token(v)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def parse_msd_lastfm_map_line(line: str, min_strength: int = 20) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse one line from `msd_lastfm_map.cls`.

    Format:
      trackId<TAB>seed_genre<TAB>tag1<TAB>strength1<TAB>tag2<TAB>strength2...

    We keep:
    - `seed_genre` (genre label)
    - tag/strength pairs (filtered by `min_strength`)
    """
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split("\t")
    if len(parts) < 2:
        return None

    track_id = parts[0].strip()
    seed_genre = _clean_token(parts[1]) or None

    tags: List[str] = []
    tag_strengths: Dict[str, int] = {}
    tail = parts[2:]

    # Tail is tag/strength pairs.
    for i in range(0, len(tail) - 1, 2):
        tag = _clean_token(tail[i])
        try:
            strength = int(tail[i + 1])
        except (TypeError, ValueError):
            continue
        if not tag or strength < min_strength:
            continue
        if tag not in tag_strengths:
            tags.append(tag)
        tag_strengths[tag] = strength

    return track_id, {"seed_genre": seed_genre, "tags": tags, "tag_strengths": tag_strengths}


def load_msd_lastfm_map(
    path: str,
    min_strength: int = 20,
    max_tags_per_track: Optional[int] = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Load Last.fm genre/tag labels from `.cls` or `.zip`.

    Returns:
      {msd_track_id: {"seed_genre": str|None, "tags": [...], "tag_strengths": {...}}}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out: Dict[str, Dict[str, Any]] = {}

    def consume_lines(lines: Iterable[str]) -> None:
        for line in lines:
            parsed = parse_msd_lastfm_map_line(line, min_strength=min_strength)
            if parsed is None:
                continue
            track_id, item = parsed
            if max_tags_per_track is not None:
                kept = item.get("tags", [])[:max_tags_per_track]
                item["tags"] = kept
                keep_set = set(kept)
                item["tag_strengths"] = {
                    k: v for k, v in item.get("tag_strengths", {}).items() if k in keep_set
                }
            out[track_id] = item

    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            cls_names = [n for n in names if n.lower().endswith(".cls")]
            for name in (cls_names or names):
                with zf.open(name, "r") as fh:
                    consume_lines((b.decode("utf-8", errors="ignore") for b in fh))
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            consume_lines(fh)

    return out


def merge_lastfm_tags_into_track_document(
    track_document: Mapping[str, Any],
    lastfm_entry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Merge one Last.fm mapping entry into a TrackDocument-like dict.
    `seed_genre` is used for genres. Tags/strengths are preserved on the document
    for later use/debugging, but not used in the current embedding text builder.
    """
    td = dict(track_document or {})
    td.setdefault("genres", [])
    td.setdefault("tags", [])
    td.setdefault("provenance", {})

    if not lastfm_entry:
        return td

    genres = _unique_tokens(td.get("genres", []))
    tags = _unique_tokens(td.get("tags", []))
    seed_genre = _clean_token(lastfm_entry.get("seed_genre"))
    if seed_genre and seed_genre not in genres:
        genres.append(seed_genre)
    for tag in _unique_tokens(lastfm_entry.get("tags", [])):
        if tag not in tags:
            tags.append(tag)

    td["genres"] = genres
    td["tags"] = tags
    td["provenance"] = dict(td.get("provenance") or {})
    td["provenance"]["genre_source"] = "msd_lastfm_map"
    if lastfm_entry.get("tags"):
        td["provenance"]["tag_source"] = "msd_lastfm_map"
    if lastfm_entry.get("tag_strengths"):
        td["provenance"]["lastfm_tag_strengths"] = dict(lastfm_entry["tag_strengths"])
    return td


def load_mxm_vocab(path: str) -> Dict[int, str]:
    """
    Load MXM vocab from the `%...` header line in an MXM dataset file.

    MXM word ids in the data lines are 1-based positions into this list.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%"):
                words = line[1:].split(",")
                return {i + 1: w for i, w in enumerate(words)}
    raise ValueError(f"Could not find MXM vocab header line (%) in: {path}")


def iter_mxm_bow_rows(path: str):
    """
    Yield MXM rows as simple dicts.

    Each row contains:
    - `track_id`  (MSD track id)
    - `mxm_tid`   (musiXmatch track id, string)
    - `bow_vector` (dict[word_id -> count])
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            parts = line.split(",")
            if len(parts) < 2:
                continue

            track_id = parts[0].strip()
            mxm_tid = parts[1].strip()
            bow_vector: Dict[int, int] = {}

            for pair in parts[2:]:
                if ":" not in pair:
                    continue
                wid_s, count_s = pair.split(":", 1)
                try:
                    wid = int(wid_s)
                    count = int(count_s)
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    bow_vector[wid] = count

            yield {
                "track_id": track_id,
                "mxm_tid": mxm_tid,
                "bow_vector": bow_vector,
            }


class EmbeddingManager:
    """
    Small manager for Chroma + embedding models.

    Main TrackDocument workflow:
      1) `build_track_embedding_text(track_doc, mxm_vocab=...)`
      2) `generate_track_embedding(track_doc, mxm_vocab=...)`
      3) `upsert_track_document_embedding(track_doc, mxm_vocab=...)`
      4) `query_track_embeddings("rock guitar heartbreak")`
    """

    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory is None:
            persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # New primary collection (TrackDocument text embeddings).
        self.track_collection = self.client.get_or_create_collection(
            name="track_text_embeddings",
            metadata={"description": "TrackDocument text embeddings"},
        )

        # Legacy lyrics collection kept so existing code paths do not break.
        self.lyrics_collection = self.client.get_or_create_collection(
            name="lyrics_embeddings",
            metadata={"description": "Song lyrics embeddings"},
        )

        self._text_model = None
        self._text_model_name = None

    # --------------------------
    # Model loading
    # --------------------------

    def _get_text_model(self, model_name: Optional[str] = None):
        model_name = model_name or os.getenv("LYRICS_EMBEDDING_MODEL", DEFAULT_TEXT_MODEL)
        if self._text_model is None or self._text_model_name != model_name:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(model_name)
            self._text_model_name = model_name
        return self._text_model

    # --------------------------
    # TrackDocument helpers
    # --------------------------

    def bow_to_text(
        self,
        bow_vector: Optional[Mapping[Any, Any]],
        vocab: Optional[Mapping[Any, str]],
        max_tokens: int = 256,
        per_word_cap: int = 5,
    ) -> str:
        """
        Convert MXM BoW like {"12": 3, "98": 1} into compact pseudo-text.

        Why: many tracks do not have full lyrics text, but BoW still gives
        lexical signal for semantic retrieval.
        """
        if not bow_vector:
            return ""

        vocab = vocab or {}
        pieces: List[str] = []

        # Sort word ids to keep output deterministic.
        for raw_wid, raw_count in sorted(bow_vector.items(), key=lambda x: str(x[0])):
            try:
                wid = int(raw_wid)
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue

            word = vocab.get(wid) or vocab.get(str(wid)) or f"unk_{wid}"
            token = _clean_token(word) or f"unk_{wid}"

            # Cap repeats so one frequent word does not dominate the embedding text.
            repeat = min(count, max(1, per_word_cap))
            pieces.extend([token] * repeat)
            if len(pieces) >= max_tokens:
                break

        return " ".join(pieces[:max_tokens])

    def build_track_embedding_text(
        self,
        track_document: Mapping[str, Any],
        mxm_vocab: Optional[Mapping[Any, str]] = None,
        include_bow: bool = True,
        max_bow_tokens: int = 256,
    ) -> str:
        """
        Build deterministic text from a TrackDocument for text embeddings.
        Current policy: use only Last.fm genre + MXM BOW text.

        If `retrieval_text.embedding_text` already exists, use it directly.
        """
        td = dict(track_document or {})
        td.setdefault("lyrics", {})
        td.setdefault("retrieval_text", {})

        # Allow ingestion code to precompute a final embedding string.
        precomputed = _clean_text((td["retrieval_text"] or {}).get("embedding_text"))
        if precomputed:
            return precomputed

        tokens: List[str] = []

        # Use only genre labels (primarily from Last.fm seed genre merge).
        for genre in _unique_tokens(_to_list(td.get("genres"))):
            tokens.append(f"genre:{genre}")
            tokens.append(genre)

        if include_bow:
            lyrics = td.get("lyrics") or {}
            bow_text = self.bow_to_text(lyrics.get("bow_vector"), mxm_vocab, max_tokens=max_bow_tokens)
            if bow_text:
                tokens.append(bow_text)

        return re.sub(r"\s+", " ", " ".join(tokens)).strip()

    def track_metadata(self, track_document: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Build Chroma-safe metadata from TrackDocument.

        Chroma metadata is easiest to query when values are scalars, so list
        fields are stored as pipe-separated strings.
        """
        td = dict(track_document or {})
        td.setdefault("lyrics", {})
        td.setdefault("source_ids", {})
        td.setdefault("provenance", {})

        metadata: Dict[str, Any] = {}

        track_id = _clean_text(td.get("track_id"))
        if track_id:
            metadata["track_id"] = track_id
            metadata["song_id"] = track_id  # legacy compatibility

        for key in ["title", "artist_name"]:
            value = _clean_text(td.get(key))
            if value:
                metadata[key] = value

        genres = _unique_tokens(_to_list(td.get("genres")))
        if genres:
            metadata["genres_csv"] = "|".join(genres)
        tags = _unique_tokens(_to_list(td.get("tags")))
        if tags:
            metadata["tags_csv"] = "|".join(tags)

        lyrics = td.get("lyrics") or {}
        metadata["has_bow"] = bool(lyrics.get("bow_vector"))
        metadata["has_lyrics"] = bool(lyrics.get("full_text"))

        source_ids = td.get("source_ids") or {}
        for key in ["spotify_id", "mxm_tid", "msd_track_id"]:
            if source_ids.get(key) is not None:
                metadata[key] = str(source_ids[key])

        provenance = td.get("provenance") or {}
        for key in ["metadata_source", "genre_source", "tag_source", "lyrics_source"]:
            if provenance.get(key):
                metadata[key] = str(provenance[key])

        return metadata

    # --------------------------
    # Embedding generation/store
    # --------------------------

    def generate_text_embedding(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        model = self._get_text_model(model_name)
        return model.encode(text, convert_to_numpy=True)

    def generate_lyrics_embedding(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        # Kept for older call sites.
        return self.generate_text_embedding(text, model_name=model_name)

    def generate_bow_embedding(
        self,
        bow_vector: Mapping[Any, Any],
        vocab: Mapping[Any, str],
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        text = self.bow_to_text(bow_vector, vocab)
        return self.generate_text_embedding(text, model_name=model_name)

    def generate_track_embedding(
        self,
        track_document: Mapping[str, Any],
        mxm_vocab: Optional[Mapping[Any, str]] = None,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        # Core TrackDocument -> text -> vector step.
        text = self.build_track_embedding_text(track_document, mxm_vocab=mxm_vocab)
        if not text:
            raise ValueError("No embedding text could be built from TrackDocument")
        return self.generate_text_embedding(text, model_name=model_name)

    def _upsert(self, collection, item_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        md = dict(metadata or {})
        md.setdefault("song_id", item_id)
        collection.upsert(
            ids=[item_id],
            embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)],
            metadatas=[md],
        )

    def store_track_embedding(self, track_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        md = dict(metadata or {})
        md["track_id"] = track_id
        md.setdefault("song_id", track_id)
        self._upsert(self.track_collection, track_id, embedding, md)

    def store_lyrics_embedding(self, song_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        md = dict(metadata or {})
        md["song_id"] = song_id
        self._upsert(self.lyrics_collection, song_id, embedding, md)

    def upsert_track_document_embedding(
        self,
        track_document: Mapping[str, Any],
        mxm_vocab: Optional[Mapping[Any, str]] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Convenience method: build vector and store it in `track_text_embeddings`.
        """
        track_id = _clean_text((track_document or {}).get("track_id"))
        if not track_id:
            raise ValueError("track_document.track_id is required")

        embedding = self.generate_track_embedding(track_document, mxm_vocab=mxm_vocab, model_name=model_name)
        md = self.track_metadata(track_document)
        if metadata:
            md.update(metadata)
        self.store_track_embedding(track_id, embedding, md)
        return embedding

    # --------------------------
    # Query
    # --------------------------

    def _format_query_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not results.get("ids"):
            return out
        ids = results["ids"][0]
        dists = results.get("distances", [[]])[0] if results.get("distances") else []
        metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        for i, item_id in enumerate(ids):
            out.append(
                {
                    "id": item_id,
                    "distance": dists[i] if i < len(dists) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return out

    def query_track_embeddings(
        self,
        query_text: str,
        top_k: int = 10,
        model_name: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the TrackDocument text collection.
        """
        query_embedding = self.generate_text_embedding(query_text, model_name=model_name).tolist()
        results = self.track_collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        return self._format_query_results(results)

    def query_by_semantics(
        self,
        query_text: str,
        collection: str = "lyrics",
        top_k: int = 10,
        model_name: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible query entry point used by `db.py`.

        Supported collections:
        - `lyrics`
        - `track`, `track_text`, `tracks`
        """
        collection_key = (collection or "lyrics").strip().lower()
        if collection_key in {"track", "track_text", "tracks"}:
            return self.query_track_embeddings(query_text, top_k=top_k, model_name=model_name, where=where)

        if collection_key == "lyrics":
            model = self._get_text_model(model_name)
            coll = self.lyrics_collection
        else:
            raise ValueError("collection must be one of: lyrics, track_text")

        query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()
        results = coll.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        return self._format_query_results(results)

    def upsert_all_from_mxm_and_lastfm(
        self,
        *,
        mxm_train_path: str,
        lastfm_map_path: str,
        mxm_test_path: Optional[str] = None,
        min_lastfm_strength: int = 20,
        max_lastfm_tags_per_track: Optional[int] = 20,
        limit: Optional[int] = None,
        progress_every: int = 5000,
        model_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Batch-build embeddings for the intersection of:
        - MXM BOW tracks
        - Last.fm-labeled MSD tracks

        You do not need to pass `track_id` manually; it is read from MXM rows.
        """
        lastfm_map = load_msd_lastfm_map(
            lastfm_map_path,
            min_strength=min_lastfm_strength,
            max_tags_per_track=max_lastfm_tags_per_track,
        )
        mxm_vocab = load_mxm_vocab(mxm_train_path)

        stats = {
            "seen_mxm_rows": 0,
            "matched_lastfm": 0,
            "embedded": 0,
            "skipped_missing_lastfm": 0,
            "skipped_empty_bow": 0,
        }
        seen_track_ids = set()

        def process_file(path: str) -> bool:
            for row in iter_mxm_bow_rows(path):
                stats["seen_mxm_rows"] += 1

                track_id = row["track_id"]
                if not track_id or track_id in seen_track_ids:
                    continue
                seen_track_ids.add(track_id)

                if not row["bow_vector"]:
                    stats["skipped_empty_bow"] += 1
                    continue

                lastfm_entry = lastfm_map.get(track_id)
                if not lastfm_entry:
                    stats["skipped_missing_lastfm"] += 1
                    continue

                stats["matched_lastfm"] += 1

                track_doc = {
                    "track_id": track_id,
                    "genres": [],
                    "lyrics": {
                        "bow_vector": row["bow_vector"],
                    },
                    "source_ids": {
                        "mxm_tid": row["mxm_tid"],
                        "msd_track_id": track_id,
                    },
                    "provenance": {
                        "lyrics_source": "mxm",
                    },
                }

                track_doc = merge_lastfm_tags_into_track_document(track_doc, lastfm_entry)
                self.upsert_track_document_embedding(
                    track_doc,
                    mxm_vocab=mxm_vocab,
                    model_name=model_name,
                )
                stats["embedded"] += 1

                if progress_every and stats["embedded"] % progress_every == 0:
                    print(f"Embedded {stats['embedded']} tracks...")

                if limit is not None and stats["embedded"] >= limit:
                    return True
            return False

        stop = process_file(mxm_train_path)
        if not stop and mxm_test_path:
            process_file(mxm_test_path)

        return stats


# Module-level singleton (kept for convenience and existing imports)
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(persist_directory: Optional[str] = None, chroma_path: Optional[str] = None) -> EmbeddingManager:
    # `chroma_path` is kept in the signature for backward compatibility.
    del chroma_path
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(persist_directory=persist_directory)
    return _embedding_manager


# Simple wrapper functions (optional convenience API)
def query_by_semantics(
    query_text: str,
    collection: str = "lyrics",
    top_k: int = 10,
    model_name: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return get_embedding_manager().query_by_semantics(
        query_text=query_text,
        collection=collection,
        top_k=top_k,
        model_name=model_name,
        where=where,
    )


def query_track_embeddings(
    query_text: str,
    top_k: int = 10,
    model_name: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return get_embedding_manager().query_track_embeddings(
        query_text=query_text,
        top_k=top_k,
        model_name=model_name,
        where=where,
    )


def generate_bow_embedding(
    bow_vector: Mapping[Any, Any],
    vocab: Mapping[Any, str],
    model_name: Optional[str] = None,
) -> np.ndarray:
    return get_embedding_manager().generate_bow_embedding(bow_vector, vocab, model_name=model_name)


def generate_track_embedding(
    track_document: Mapping[str, Any],
    mxm_vocab: Optional[Mapping[Any, str]] = None,
    model_name: Optional[str] = None,
) -> np.ndarray:
    return get_embedding_manager().generate_track_embedding(
        track_document,
        mxm_vocab=mxm_vocab,
        model_name=model_name,
    )


def upsert_track_document_embedding(
    track_document: Mapping[str, Any],
    mxm_vocab: Optional[Mapping[Any, str]] = None,
    model_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return get_embedding_manager().upsert_track_document_embedding(
        track_document,
        mxm_vocab=mxm_vocab,
        model_name=model_name,
        metadata=metadata,
    )


def upsert_all_from_mxm_and_lastfm(
    *,
    mxm_train_path: str,
    lastfm_map_path: str,
    mxm_test_path: Optional[str] = None,
    min_lastfm_strength: int = 20,
    max_lastfm_tags_per_track: Optional[int] = 20,
    limit: Optional[int] = None,
    progress_every: int = 5000,
    model_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Module-level convenience wrapper for batch embedding generation.
    """
    return get_embedding_manager().upsert_all_from_mxm_and_lastfm(
        mxm_train_path=mxm_train_path,
        lastfm_map_path=lastfm_map_path,
        mxm_test_path=mxm_test_path,
        min_lastfm_strength=min_lastfm_strength,
        max_lastfm_tags_per_track=max_lastfm_tags_per_track,
        limit=limit,
        progress_every=progress_every,
        model_name=model_name,
    )
