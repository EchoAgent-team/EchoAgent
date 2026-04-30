from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .text_utils import clean_text, clean_token, to_list, unique_tokens
from .retrieval_text import build_retrieval_text


def parse_msd_lastfm_map_line(line: str, min_strength: int = 20) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse a single line from the MSD Last.fm mapping file.

    The file format is tab-separated with:
    track_id<TAB>seed_genre<TAB>tag1<TAB>strength1<TAB>tag2<TAB>strength2...

    This function extracts the track ID, seed genre, and tag/strength pairs,
    filtering tags by minimum strength.

    Args:
        line (str): Raw line from the mapping file.
        min_strength (int): Minimum tag strength to include. Default 20.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Tuple of (track_id, mapping_dict) if valid,
            None if line is invalid or empty. The mapping_dict contains:
            - 'seed_genre': Cleaned genre string or None
            - 'tags': List of tag names (filtered by strength)
            - 'tag_strengths': Dict of tag -> strength mappings
    """
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split("\t")
    if len(parts) < 2:
        return None

    track_id = parts[0].strip()
    seed_genre = clean_token(parts[1]) or None

    tags: List[str] = []
    tag_strengths: Dict[str, int] = {}
    tail = parts[2:]

    # Tail is tag/strength pairs.
    for i in range(0, len(tail) - 1, 2):
        tag = clean_token(tail[i])
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
    Load Last.fm genre and tag mappings from MSD dataset files.

    Supports both plain .cls files and .zip archives containing .cls files.
    Processes each line to extract track IDs, seed genres, and tag/strength pairs.

    Args:
        path (str): Path to .cls file or .zip archive containing .cls files.
        min_strength (int): Minimum tag strength to include. Default 20.
        max_tags_per_track (Optional[int]): Maximum tags to keep per track. Default 20.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of track_id to genre/tag data.
            Each value dict contains 'seed_genre', 'tags', and 'tag_strengths'.

    Raises:
        FileNotFoundError: If the specified path does not exist.
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
        # Handle zip archives containing .cls files
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            cls_names = [n for n in names if n.lower().endswith(".cls")]
            for name in (cls_names or names):
                with zf.open(name, "r") as fh:
                    consume_lines((b.decode("utf-8", errors="ignore") for b in fh))
    else:
        # Handle plain .cls files
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            consume_lines(fh)

    return out


def merge_lastfm_tags_into_track_document(
    track_document: Mapping[str, Any],
    lastfm_entry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Merge Last.fm genre and tag data into a track document.

    This function enriches a track document with genre and tag information from
    Last.fm mappings. It adds seed genres and tags to the document while preserving
    existing data and tracking provenance.

    Merging behavior:
    - Adds Last.fm seed genre to genres list if not already present
    - Adds Last.fm tags to tags list if not already present
    - Preserves existing genres and tags
    - Records provenance information about data sources

    Args:
        track_document (Mapping[str, Any]): Base track document to enrich.
        lastfm_entry (Optional[Mapping[str, Any]]): Last.fm data with 'seed_genre',
            'tags', and 'tag_strengths'. If None, returns track_document unchanged.

    Returns:
        Dict[str, Any]: Enriched track document with merged Last.fm data.
    """
    td = dict(track_document or {})
    td.setdefault("genres", [])
    td.setdefault("tags", [])
    td.setdefault("provenance", {})

    if not lastfm_entry:
        return td

    genres = unique_tokens(td.get("genres", []))
    tags = unique_tokens(td.get("tags", []))
    seed_genre = clean_token(lastfm_entry.get("seed_genre"))
    if seed_genre and seed_genre not in genres:
        genres.append(seed_genre)
    for tag in unique_tokens(lastfm_entry.get("tags", [])):
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
    Load the musiXmatch (MXM) vocabulary from a dataset file.

    The MXM vocabulary is stored in a header line starting with '%' followed by
    comma-separated words. Word IDs are 1-based indices into this list.

    Args:
        path (str): Path to MXM dataset file containing the vocabulary header.

    Returns:
        Dict[int, str]: Mapping of word_id (1-based) to word string.

    Raises:
        ValueError: If no vocabulary header line (%) is found in the file.
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
    Iterate over bag-of-words rows from an MXM dataset file.

    Parses each data line into a dictionary containing track ID, MXM track ID,
    and bag-of-words vector. Skips header lines and comments.

    Each row format: track_id,mxm_tid,word_id:count,word_id:count,...

    Args:
        path (str): Path to MXM dataset file.

    Yields:
        Dict[str, Any]: Row dictionaries with keys:
            - 'track_id': MSD track ID (string)
            - 'mxm_tid': musiXmatch track ID (string)
            - 'bow_vector': Dict[int, int] of word_id -> count
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


def track_id_to_msd_h5_path(track_id: str, msd_root: str) -> Path:
    """
    Resolve an MSD track ID to its expected HDF5 path inside MillionSongSubset.

    MSD subset files are stored as:
    <root>/<track_id[2]>/<track_id[3]>/<track_id[4]>/<track_id>.h5
    """
    clean_track_id = clean_text(track_id)
    if len(clean_track_id) < 5:
        raise ValueError(f"Invalid MSD track_id: {track_id!r}")
    return Path(msd_root) / clean_track_id[2] / clean_track_id[3] / clean_track_id[4] / f"{clean_track_id}.h5"


def upsert_all_from_mxm_and_lastfm(
    *,
    mxm_train_path: str,
    lastfm_map_path: str,
    mxm_test_path: Optional[str] = None,
    msd_subset_root: Optional[str] = None,
    min_lastfm_strength: int = 20,
    max_lastfm_tags_per_track: Optional[int] = 20,
    limit: Optional[int] = None,
    progress_every: int = 5000,
    model_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Batch process MXM BoW data plus Last.fm metadata and upsert track embeddings.

    This indexing flow lives outside `embeddings.py` so the vector layer stays focused
    on embedder loading, vector-store access, upsert, and semantic query.
    """
    from .embeddings import upsert_track_embedding

    lastfm_map = load_msd_lastfm_map(
        lastfm_map_path,
        min_strength=min_lastfm_strength,
        max_tags_per_track=max_lastfm_tags_per_track,
    )
    mxm_vocab = load_mxm_vocab(mxm_train_path)
    msd_root = Path(msd_subset_root) if msd_subset_root else None
    total_msd_tracks = sum(1 for _ in msd_root.rglob("*.h5")) if msd_root is not None else 0

    stats = {
        "seen_mxm_rows": 0,
        "total_msd_tracks": total_msd_tracks,
        "matched_msd_subset": 0,
        "matched_lastfm": 0,
        "embedded": 0,
        "missing_lastfm": 0,
        "skipped_empty_bow": 0,
        "skipped_missing_msd_subset": 0,
    }
    seen_track_ids = set()

    def process_file(path: str) -> bool:
        for row in iter_mxm_bow_rows(path):
            stats["seen_mxm_rows"] += 1

            track_id = row["track_id"]
            if not track_id or track_id in seen_track_ids:
                continue
            seen_track_ids.add(track_id)

            if msd_root is not None:
                h5_path = track_id_to_msd_h5_path(track_id, str(msd_root))
                if not h5_path.exists():
                    stats["skipped_missing_msd_subset"] += 1
                    continue
                stats["matched_msd_subset"] += 1

            if not row["bow_vector"]:
                stats["skipped_empty_bow"] += 1
                continue

            lastfm_entry = lastfm_map.get(track_id)
            if not lastfm_entry:
                stats["missing_lastfm"] += 1
            else:
                stats["matched_lastfm"] += 1

            track_doc = {
                "track_id": track_id,
                "genres": [],
                "lyrics": {
                    "bow_vector": row["bow_vector"],
                },
                "retrieval_text": {},
                "source_ids": {
                    "mxm_tid": row["mxm_tid"],
                    "msd_track_id": track_id,
                },
                "provenance": {
                    "lyrics_source": "mxm",
                },
            }

            if lastfm_entry:
                track_doc = merge_lastfm_tags_into_track_document(track_doc, lastfm_entry)

            track_doc["retrieval_text"]["embedding_text"] = build_retrieval_text(
                track_doc,
                mxm_vocab=mxm_vocab,
            )
            upsert_track_embedding(
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
