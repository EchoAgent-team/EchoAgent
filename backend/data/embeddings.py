"""
Vector embedding utilities for EchoAgent.

This module is responsible for:
- embedder loading
- LangChain/Chroma vector-store access
- track embedding upsert
- semantic query
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
from .text_utils import clean_text, clean_token, to_list, unique_tokens
from .retrieval_text import bow_to_text

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma


DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """
    Manages vector embeddings for track documents using ChromaDB and sentence transformers.

    This class provides the vector-facing interface for embeddings in EchoAgent.
    It manages two ChromaDB collections: 'track_text_embeddings' for track retrieval
    and 'lyrics_embeddings' for legacy compatibility.

    Key functionalities:
    - Loading and caching sentence transformer models
    - Converting precomputed retrieval text into embedding vectors
    - Storing embeddings in ChromaDB
    - Querying for semantically similar tracks

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
            "retrieval_text": {"embedding_text": "love city rain drive tonight"}
        }
        embedding = manager.upsert_track_embedding(track_doc)

        # Query for similar tracks
        results = manager.query_track_embeddings("late night city drive", top_k=5)
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

        track_id = clean_text(td.get("track_id"))
        if track_id:
            metadata["track_id"] = track_id
            metadata["song_id"] = track_id  # legacy compatibility

        for key in ["title", "artist_name"]:
            value = clean_text(td.get(key))
            if value:
                metadata[key] = value

        genres = unique_tokens(to_list(td.get("genres")))
        if genres:
            metadata["genres_csv"] = "|".join(genres)
        tags = unique_tokens(to_list(td.get("tags")))
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
        text = bow_to_text(bow_vector, vocab)
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
        It expects retrieval text to be precomputed and stored on the document.

        Args:
            track_document (Mapping[str, Any]): Complete track document with metadata,
                genres, and lyrics information.
            mxm_vocab (Optional[Mapping[Any, str]]): Unused in strict mode.
                Kept temporarily for compatibility with older call sites.
            model_name (Optional[str]): Model name for embedding generation.

        Returns:
            np.ndarray: Dense embedding vector representing the track.

        Raises:
            ValueError: If retrieval_text.embedding_text is missing.
        """
        del mxm_vocab
        retrieval_text = (track_document or {}).get("retrieval_text") or {}
        text = clean_text(retrieval_text.get("embedding_text"))
        if not text:
            raise ValueError("track_document.retrieval_text.embedding_text is required")
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

    def upsert_track_embedding(
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
        track_id = clean_text((track_document or {}).get("track_id"))
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
        return [self.manager.generate_text_embedding(t).tolist() for t in texts]
    
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
    Get or create the global EmbeddingManager single instance.

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


def upsert_track_embedding(
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
    return get_embedding_manager().upsert_track_embedding(
        track_document,
        mxm_vocab=mxm_vocab,
        model_name=model_name,
        metadata=metadata,
    )


#----------------------------
#----- Langchain ------------
#----------------------------
    
def get_vector_store(persist_directory: Optional[str] = None):
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
    
