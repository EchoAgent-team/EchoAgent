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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
import numpy as np


DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"


def _clean_text(value: Any) -> str:
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


def _clean_token(value: Any) -> str:
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
    text = _clean_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_:/\-\.\+]", "", text)
    return text.strip("_")


def _to_list(value: Any) -> List[Any]:
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


def _unique_tokens(values: Iterable[Any]) -> List[str]:
    """
    Extract unique, cleaned tokens from an iterable of values.

    Processes each value through _clean_token(), removes duplicates while
    preserving order, and filters out empty tokens.

    Args:
        values (Iterable[Any]): Values to process into tokens.

    Returns:
        List[str]: List of unique, non-empty tokens in order of first appearance.
    """
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
    clean_track_id = _clean_text(track_id)
    if len(clean_track_id) < 5:
        raise ValueError(f"Invalid MSD track_id: {track_id!r}")
    return Path(msd_root) / clean_track_id[2] / clean_track_id[3] / clean_track_id[4] / f"{clean_track_id}.h5"


class EmbeddingManager:
    """
    Manages vector embeddings for track documents using ChromaDB and sentence transformers.

    This class provides a comprehensive interface for handling text embeddings in the EchoAgent
    music recommendation system. It manages two ChromaDB collections: 'track_text_embeddings'
    for full track document embeddings and 'lyrics_embeddings' for legacy lyrics-only embeddings.

    Key functionalities:
    - Loading and managing sentence transformer models for text embedding generation
    - Converting track documents (including metadata, genres, and bag-of-words lyrics) into
      embedding vectors
    - Storing and querying embeddings in ChromaDB collections
    - Batch processing of MXM (musiXmatch) bag-of-words data with Last.fm genre mappings
    - Providing backward-compatible APIs for existing code

    The primary workflow involves:
    1. Building embedding text from track documents using genres and MXM BoW data
    2. Generating embeddings using sentence transformers
    3. Storing embeddings with metadata in ChromaDB
    4. Querying for semantically similar tracks

    Attributes:
        client (chromadb.PersistentClient): Persistent ChromaDB client for database operations
        track_collection: ChromaDB collection for track text embeddings
        lyrics_collection: ChromaDB collection for legacy lyrics embeddings
        _text_model: Cached sentence transformer model instance
        _text_model_name: Name of the currently loaded model

    Example:
        manager = EmbeddingManager()
        # Build and store embedding for a track
        track_doc = {
            "track_id": "TR123",
            "title": "Song Title",
            "artist_name": "Artist Name",
            "genres": ["rock", "pop"],
            "lyrics": {"bow_vector": {1: 5, 2: 3}}
        }
        embedding = manager.upsert_track_document_embedding(track_doc, mxm_vocab=vocab)

        # Query for similar tracks
        results = manager.query_track_embeddings("rock guitar solo", top_k=5)
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the EmbeddingManager with ChromaDB collections.

        Args:
            persist_directory (Optional[str]): Directory path for ChromaDB persistence.
                If None, defaults to 'chroma_db' subdirectory in the same directory as this file.
                The directory will be created if it doesn't exist.

        Raises:
            OSError: If the persist_directory cannot be created.
        """
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
        """
        Load and cache a sentence transformer model for text embedding.

        This method implements lazy loading and caching of sentence transformer models.
        If the requested model is already loaded, it returns the cached instance.
        Otherwise, it loads the model from Hugging Face and caches it.

        Args:
            model_name (Optional[str]): Name of the sentence transformer model to load.
                If None, uses the environment variable 'LYRICS_EMBEDDING_MODEL' or
                defaults to DEFAULT_TEXT_MODEL ('all-MiniLM-L6-v2').

        Returns:
            SentenceTransformer: The loaded sentence transformer model instance.

        Note:
            Model loading can be expensive, so this method caches models to avoid
            reloading the same model multiple times.
        """
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
        Convert a bag-of-words (BoW) vector into a compact text representation.

        This method transforms MXM (musiXmatch) bag-of-words data into pseudo-text
        that can be embedded. It handles word frequency capping to prevent dominant
        words from overwhelming the embedding, and limits total tokens for efficiency.

        The conversion process:
        1. Sorts word IDs for deterministic output
        2. Maps word IDs to vocabulary terms (or uses 'unk_{id}' for unknowns)
        3. Repeats words based on their frequency (capped at per_word_cap)
        4. Joins tokens with spaces, limited to max_tokens total

        Args:
            bow_vector (Optional[Mapping[Any, Any]]): Dictionary mapping word IDs to counts.
                Can be None or empty, in which case an empty string is returned.
            vocab (Optional[Mapping[Any, str]]): Vocabulary mapping word IDs to words.
                If None, uses 'unk_{id}' format for all words.
            max_tokens (int): Maximum number of tokens in the output text. Default 256.
            per_word_cap (int): Maximum times a single word can be repeated. Default 5.
                This prevents frequent words from dominating the embedding.

        Returns:
            str: Space-separated text representation of the bag-of-words data.

        Example:
            bow = {1: 10, 2: 3, 3: 1}
            vocab = {1: "love", 2: "heart", 3: "song"}
            result = manager.bow_to_text(bow, vocab, max_tokens=10, per_word_cap=3)
            # Result: "love love love heart heart heart song"
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
        Build a deterministic text string from a track document for embedding generation.

        This method constructs embedding text by combining genre information and
        bag-of-words lyrics data. The current policy prioritizes Last.fm genre labels
        and MXM bag-of-words data for semantic retrieval.

        The text building process:
        1. Checks for precomputed embedding text in track_document.retrieval_text.embedding_text
        2. Adds genre tokens in "genre:{genre}" and plain "{genre}" formats
        3. Optionally includes bag-of-words text converted via bow_to_text()
        4. Joins all tokens with spaces and normalizes whitespace

        Args:
            track_document (Mapping[str, Any]): Track document dictionary containing
                metadata, genres, and lyrics information.
            mxm_vocab (Optional[Mapping[Any, str]]): MXM vocabulary for BoW conversion.
                Required if include_bow is True and BoW data is present.
            include_bow (bool): Whether to include bag-of-words lyrics in the text.
                Default True.
            max_bow_tokens (int): Maximum tokens from BoW conversion. Default 256.

        Returns:
            str: Normalized text string suitable for embedding generation.

        Note:
            If track_document.retrieval_text.embedding_text exists, it takes precedence
            over dynamic text building, allowing for custom embedding strategies.
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
        Extract and format metadata from a track document for ChromaDB storage.

        ChromaDB requires metadata values to be scalars for efficient querying.
        This method converts list fields to pipe-separated strings and extracts
        relevant scalar metadata from the track document.

        Extracted metadata includes:
        - Basic track info: track_id, title, artist_name
        - Genres and tags as CSV strings (pipe-separated)
        - Lyrics availability flags (has_bow, has_lyrics)
        - Source IDs from various music databases
        - Provenance information about data sources

        Args:
            track_document (Mapping[str, Any]): Track document containing metadata
                fields like track_id, title, artist_name, genres, tags, etc.

        Returns:
            Dict[str, Any]: ChromaDB-compatible metadata dictionary with scalar values.

        Note:
            List fields like genres and tags are converted to "|"-separated strings
            for ChromaDB compatibility. Use the CSV fields for querying.
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
        """
        Generate an embedding vector from input text using a sentence transformer model.

        This is the core method for converting text into dense vector representations
        that capture semantic meaning. The method uses the cached sentence transformer
        model to encode the text.

        Args:
            text (str): Input text to embed. Should be non-empty for meaningful results.
            model_name (Optional[str]): Name of the model to use. If None, uses the
                default or cached model.

        Returns:
            np.ndarray: Dense embedding vector as a NumPy array.

        Raises:
            ValueError: If text is empty or model loading fails.
        """
        model = self._get_text_model(model_name)
        return model.encode(text, convert_to_numpy=True)

    def generate_lyrics_embedding(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate an embedding for lyrics text (legacy method).

        This method is maintained for backward compatibility with existing code
        that specifically handles lyrics embeddings. It delegates to the general
        text embedding method.

        Args:
            text (str): Lyrics text to embed.
            model_name (Optional[str]): Model name to use for embedding.

        Returns:
            np.ndarray: Embedding vector for the lyrics text.
        """
        # Kept for older call sites.
        return self.generate_text_embedding(text, model_name=model_name)

    def generate_bow_embedding(
        self,
        bow_vector: Mapping[Any, Any],
        vocab: Mapping[Any, str],
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate an embedding from a bag-of-words vector.

        This method converts a BoW vector to text using bow_to_text() and then
        generates an embedding from that text. Useful for embedding MXM data
        directly without building a full track document.

        Args:
            bow_vector (Mapping[Any, Any]): Bag-of-words dictionary (word_id -> count).
            vocab (Mapping[Any, str]): Vocabulary mapping word IDs to words.
            model_name (Optional[str]): Model name for embedding generation.

        Returns:
            np.ndarray: Embedding vector for the BoW data.
        """
        text = self.bow_to_text(bow_vector, vocab)
        return self.generate_text_embedding(text, model_name=model_name)

    def generate_track_embedding(
        self,
        track_document: Mapping[str, Any],
        mxm_vocab: Optional[Mapping[Any, str]] = None,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate an embedding vector from a complete track document.

        This is the primary method for creating embeddings from track documents.
        It builds embedding text from the track document and converts it to a vector.

        Args:
            track_document (Mapping[str, Any]): Complete track document with metadata,
                genres, and lyrics information.
            mxm_vocab (Optional[Mapping[Any, str]]): MXM vocabulary for BoW conversion.
                Required if the track has BoW lyrics data.
            model_name (Optional[str]): Model name for embedding generation.

        Returns:
            np.ndarray: Dense embedding vector representing the track.

        Raises:
            ValueError: If no embedding text can be built from the track document.
        """
        # Core TrackDocument -> text -> vector step.
        text = self.build_track_embedding_text(track_document, mxm_vocab=mxm_vocab)
        if not text:
            raise ValueError("No embedding text could be built from TrackDocument")
        return self.generate_text_embedding(text, model_name=model_name)

    def _upsert(self, collection, item_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Internal method to upsert an embedding into a ChromaDB collection.

        This method handles the low-level ChromaDB upsert operation, converting
        NumPy arrays to lists as required by ChromaDB.

        Args:
            collection: ChromaDB collection object to upsert into.
            item_id (str): Unique identifier for the item.
            embedding (np.ndarray): Embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Metadata dictionary to store with the embedding.
        """
        md = dict(metadata or {})
        md.setdefault("song_id", item_id)
        collection.upsert(
            ids=[item_id],
            embeddings=[embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)],
            metadatas=[md],
        )

    def store_track_embedding(self, track_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a track embedding in the track_text_embeddings collection.

        Args:
            track_id (str): Unique identifier for the track.
            embedding (np.ndarray): Embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Additional metadata to store.
        """
        md = dict(metadata or {})
        md["track_id"] = track_id
        md.setdefault("song_id", track_id)
        self._upsert(self.track_collection, track_id, embedding, md)

    def store_lyrics_embedding(self, song_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a lyrics embedding in the legacy lyrics collection.

        Args:
            song_id (str): Unique identifier for the song.
            embedding (np.ndarray): Embedding vector to store.
            metadata (Optional[Dict[str, Any]]): Additional metadata to store.
        """
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
        Convenience method to generate and store a track document embedding.

        This method combines embedding generation and storage into a single operation.
        It extracts the track_id from the document, generates the embedding, and stores
        it with automatically extracted metadata.

        Args:
            track_document (Mapping[str, Any]): Complete track document.
            mxm_vocab (Optional[Mapping[Any, str]]): MXM vocabulary for BoW processing.
            model_name (Optional[str]): Model name for embedding generation.
            metadata (Optional[Dict[str, Any]]): Additional metadata to merge with
                automatically extracted metadata.

        Returns:
            np.ndarray: The generated embedding vector.

        Raises:
            ValueError: If track_document.track_id is missing.
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
        """
        Format raw ChromaDB query results into a standardized list format.

        This internal method converts ChromaDB's nested result structure into
        a flat list of dictionaries with consistent keys.

        Args:
            results (Dict[str, Any]): Raw results from ChromaDB query.

        Returns:
            List[Dict[str, Any]]: List of result dictionaries with keys:
                - 'id': The item ID
                - 'distance': Similarity distance (if available)
                - 'metadata': Associated metadata dictionary
        """
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
        Query the track text embeddings collection for semantically similar tracks.

        This method performs semantic search against the track_text_embeddings collection
        using the provided query text. Results are ranked by embedding similarity.

        Args:
            query_text (str): Natural language query text (e.g., "rock guitar solo").
            top_k (int): Number of top results to return. Default 10.
            model_name (Optional[str]): Model name for query embedding generation.
            where (Optional[Dict[str, Any]]): ChromaDB where clause for metadata filtering.
                Example: {"genres_csv": {"$contains": "rock"}}

        Returns:
            List[Dict[str, Any]]: List of similar tracks with similarity scores and metadata.
                Each dict contains 'id', 'distance', and 'metadata' keys.
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
        Backward-compatible semantic query method supporting multiple collections.

        This method provides a unified interface for querying different embedding
        collections. It routes queries to the appropriate collection based on the
        collection parameter.

        Args:
            query_text (str): Query text for semantic search.
            collection (str): Collection to query. Supported values:
                - "lyrics" or "lyric": Query lyrics collection
                - "track", "track_text", "tracks": Query track text collection
            top_k (int): Number of results to return. Default 10.
            model_name (Optional[str]): Model for query embedding.
            where (Optional[Dict[str, Any]]): Metadata filtering conditions.

        Returns:
            List[Dict[str, Any]]: Query results with similarity information.

        Raises:
            ValueError: If collection is not supported.
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
        msd_subset_root: Optional[str] = None,
        min_lastfm_strength: int = 20,
        max_lastfm_tags_per_track: Optional[int] = 20,
        limit: Optional[int] = None,
        progress_every: int = 5000,
        model_name: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Batch process and embed tracks from MXM BoW data with Last.fm genre mappings.

        This method performs bulk embedding generation for tracks that have both
        musiXmatch (MXM) bag-of-words lyrics data and Last.fm genre/tag labels.
        It processes train and optionally test datasets, merging genre information
        and generating embeddings for storage.

        The process:
        1. Load Last.fm genre mappings and MXM vocabulary
        2. Iterate through MXM rows, filtering for tracks with Last.fm data
        3. Build track documents with merged metadata
        4. Generate and store embeddings with progress reporting

        Args:
            mxm_train_path (str): Path to MXM train dataset file (.txt).
            lastfm_map_path (str): Path to Last.fm mapping file (.cls or .zip).
            mxm_test_path (Optional[str]): Path to MXM test dataset file. Default None.
            msd_subset_root (Optional[str]): If provided, only embed tracks that have
                a matching `.h5` file in this MillionSongSubset root directory.
            min_lastfm_strength (int): Minimum tag strength to include. Default 20.
            max_lastfm_tags_per_track (Optional[int]): Max tags per track. Default 20.
            limit (Optional[int]): Maximum tracks to embed. Default None (no limit).
            progress_every (int): Print progress every N embeddings. Default 5000.
            model_name (Optional[str]): Embedding model to use.

        Returns:
            Dict[str, int]: Processing statistics with keys:
                - 'seen_mxm_rows': Total MXM rows processed
                - 'total_msd_tracks': Total `.h5` tracks present in MillionSongSubset
                - 'matched_msd_subset': MXM tracks with a matching `.h5` in MillionSongSubset
                - 'matched_lastfm': Tracks with Last.fm data found
                - 'embedded': Successfully embedded tracks
                - 'missing_lastfm': Tracks without Last.fm data
                - 'skipped_empty_bow': Tracks with empty BoW data
                - 'skipped_missing_msd_subset': Tracks missing from MillionSongSubset
        """
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
            # Process a single MXM file, returning True if we hit the limit
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

                # Build track document with merged metadata
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

                if lastfm_entry:
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

#----------------------------
#----- Langchain ------------
#----------------------------

class SentenceTransformerEmbeddings:
    """
    LangChain-compatible embedding wrapper for SentenceTransformer models.

    This class adapts the EmbeddingManager's text embedding functionality
    to work with LangChain's embedding interface, enabling integration
    with LangChain-based applications and pipelines.
    """

    def __init__(self, manager):
        """
        Initialize with an EmbeddingManager instance.

        Args:
            manager: EmbeddingManager instance to delegate embedding calls to.
        """
        self.manager = manager
    
    def embed_documents(self, texts):
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (as lists of floats).
        """
        return [self.manager.generate_text_embedding(t).to_list() for t in texts]
    
    def embed_query(self, text):
        """
        Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.manager.generate_text_embedding(text).tolist()
    

# Module-level singleton (kept for convenience and existing imports)
_embedding_manager: Optional[EmbeddingManager] = None

def get_embedding_manager(persist_directory: Optional[str] = None, chroma_path: Optional[str] = None) -> EmbeddingManager:
    """
    Get or create the global EmbeddingManager singleton instance.

    This function maintains a single EmbeddingManager instance for the module,
    creating it on first access. Subsequent calls return the same instance.

    Args:
        persist_directory (Optional[str]): Directory for ChromaDB persistence.
        chroma_path (Optional[str]): Legacy parameter, kept for compatibility.

    Returns:
        EmbeddingManager: The global singleton instance.
    """
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
    """
    Convenience wrapper for semantic querying using the global manager.

    Args:
        query_text (str): Query text.
        collection (str): Collection to query ("lyrics", "track_text", etc.).
        top_k (int): Number of results to return.
        model_name (Optional[str]): Model for embedding.
        where (Optional[Dict[str, Any]]): Metadata filters.

    Returns:
        List[Dict[str, Any]]: Query results.
    """
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
    """
    Convenience wrapper for querying track embeddings using the global manager.

    Args:
        query_text (str): Query text.
        top_k (int): Number of results to return.
        model_name (Optional[str]): Model for embedding.
        where (Optional[Dict[str, Any]]): Metadata filters.

    Returns:
        List[Dict[str, Any]]: Query results from track collection.
    """
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
    """
    Convenience wrapper for generating BoW embeddings using the global manager.

    Args:
        bow_vector (Mapping[Any, Any]): Bag-of-words data.
        vocab (Mapping[Any, str]): Word vocabulary.
        model_name (Optional[str]): Model for embedding.

    Returns:
        np.ndarray: Embedding vector.
    """
    return get_embedding_manager().generate_bow_embedding(bow_vector, vocab, model_name=model_name)


def generate_track_embedding(
    track_document: Mapping[str, Any],
    mxm_vocab: Optional[Mapping[Any, str]] = None,
    model_name: Optional[str] = None,
) -> np.ndarray:
    """
    Convenience wrapper for generating track embeddings using the global manager.

    Args:
        track_document (Mapping[str, Any]): Track document data.
        mxm_vocab (Optional[Mapping[Any, str]]): MXM vocabulary.
        model_name (Optional[str]): Model for embedding.

    Returns:
        np.ndarray: Embedding vector.
    """
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
    """
    Convenience wrapper for upserting track embeddings using the global manager.

    Args:
        track_document (Mapping[str, Any]): Track document data.
        mxm_vocab (Optional[Mapping[Any, str]]): MXM vocabulary.
        model_name (Optional[str]): Model for embedding.
        metadata (Optional[Dict[str, Any]]): Additional metadata.

    Returns:
        np.ndarray: Generated embedding vector.
    """
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
    msd_subset_root: Optional[str] = None,
    min_lastfm_strength: int = 20,
    max_lastfm_tags_per_track: Optional[int] = 20,
    limit: Optional[int] = None,
    progress_every: int = 5000,
    model_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convenience wrapper for batch MXM/Last.fm processing using the global manager.

    Args:
        mxm_train_path (str): Path to MXM train data.
        lastfm_map_path (str): Path to Last.fm mappings.
        mxm_test_path (Optional[str]): Path to MXM test data.
        msd_subset_root (Optional[str]): MillionSongSubset root to filter against.
        min_lastfm_strength (int): Minimum tag strength.
        max_lastfm_tags_per_track (Optional[int]): Max tags per track.
        limit (Optional[int]): Maximum tracks to process.
        progress_every (int): Progress reporting frequency.
        model_name (Optional[str]): Model for embedding.

    Returns:
        Dict[str, int]: Processing statistics.
    """
    return get_embedding_manager().upsert_all_from_mxm_and_lastfm(
        mxm_train_path=mxm_train_path,
        lastfm_map_path=lastfm_map_path,
        mxm_test_path=mxm_test_path,
        msd_subset_root=msd_subset_root,
        min_lastfm_strength=min_lastfm_strength,
        max_lastfm_tags_per_track=max_lastfm_tags_per_track,
        limit=limit,
        progress_every=progress_every,
        model_name=model_name,
    )


#----------------------------
#----- Langchain ------------
#----------------------------
    
def get_chroma_store(persist_directory: Optional[str] = None):
    """
    Create a LangChain Chroma vector store using the global EmbeddingManager.

    This function sets up a Chroma vector store that integrates with LangChain,
    using the track text embeddings collection and SentenceTransformer embeddings.

    Args:
        persist_directory (Optional[str]): Directory for ChromaDB persistence.

    Returns:
        Chroma: LangChain Chroma vector store instance.
    """
    manager = get_embedding_manager(persist_directory=persist_directory)
    embedding_fn = SentenceTransformerEmbeddings(manager)
    return Chroma(
        collection_name="track_text_embeddings",
        persist_directory=manager.client._identifier if False else (persist_directory or os.path.join(os.path.dirname(__file__), "chroma_db")),
        embedding_function=embedding_fn,
    )
    
def get_embedder(model_name: Optional[str] = None):
    """
    Get the sentence transformer model used for text embedding.

    This function provides direct access to the underlying SentenceTransformer
    model instance used by the EmbeddingManager for generating embeddings.

    Args:
        model_name (Optional[str]): Name of the model to retrieve. If None,
            uses the default or cached model.

    Returns:
        SentenceTransformer: The loaded sentence transformer model instance.
    """
    return get_embedding_manager()._get_text_model(model_name=model_name)
    
def semantic_search(query: str, k: int = 50, model_name: Optional[str] = None, 
                    where: Optional[Dict[str, Any]] = None,):
    """
    Perform semantic search for tracks using natural language queries.

    This function searches the track text embeddings collection for tracks
    that are semantically similar to the provided query text. Results are
    ranked by embedding similarity.

    Args:
        query (str): Natural language query text (e.g., "rock guitar solo").
        k (int): Number of top results to return. Default 50.
        model_name (Optional[str]): Model name for query embedding generation.
        where (Optional[Dict[str, Any]]): ChromaDB metadata filtering conditions.
            Example: {"genres_csv": {"$contains": "rock"}}

    Returns:
        List[Dict[str, Any]]: List of similar tracks with similarity scores and metadata.
            Each dict contains 'id', 'distance', and 'metadata' keys.
    """
    return get_embedding_manager().query_track_embeddings(query_text=query,
                                                            top_k=k,
                                                            model_name=model_name,
                                                            where=where,)
    
def get_chroma_store(persist_directory: Optional[str] = None):
    """
    Get the ChromaDB collection for track text embeddings.

    This function provides direct access to the underlying ChromaDB collection
    used for storing track text embeddings. Useful for advanced ChromaDB operations
    that require direct collection access.

    Args:
        persist_directory (Optional[str]): Directory for ChromaDB persistence.
            If provided, ensures the manager is initialized with this directory.

    Returns:
        chromadb.Collection: The ChromaDB collection for track text embeddings.
    """
    return get_embedding_manager(persist_directory=persist_directory).track_collection
