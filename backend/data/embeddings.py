"""
Vector embeddings and Chroma database setup for EchoAgent.

This module handles embedding generation for lyrics and album art,
and stores them in Chroma vector database for semantic search.
"""

import os
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
import numpy as np


class EmbeddingManager:
    """Manages embeddings and Chroma vector database."""
    
    def __init__(self, chroma_path: Optional[str] = None, persist_directory: Optional[str] = None):
        """
        Initialize Chroma client and collections.
        
        Args:
            chroma_path: Path to Chroma database (for client-server mode)
            persist_directory: Local directory for persistent storage (default: backend/data/chroma_db)
        """
        if persist_directory is None:
            # Default to backend/data/chroma_db
            base_dir = os.path.dirname(__file__)
            persist_directory = os.path.join(base_dir, "chroma_db")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collections
        self.lyrics_collection = self.client.get_or_create_collection(
            name="lyrics_embeddings",
            metadata={"description": "Embeddings for song lyrics"}
        )
        
        self.art_collection = self.client.get_or_create_collection(
            name="album_art_embeddings",
            metadata={"description": "Embeddings for album art images"}
        )
        
        # Embedding models (will be loaded lazily)
        self._lyrics_model = None
        self._art_model = None
        self._lyrics_model_name = None
        self._art_model_name = None
    
    def get_embedding_model(self, model_name: str, model_type: str = "text"):
        """
        Get or load an embedding model.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model - 'text' for lyrics, 'image' for album art
            
        Returns:
            Loaded model
        """
        if model_type == "text":
            if self._lyrics_model is None or self._lyrics_model_name != model_name:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._lyrics_model = SentenceTransformer(model_name)
                    self._lyrics_model_name = model_name
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required for text embeddings. "
                        "Install it with: pip install sentence-transformers"
                    )
            return self._lyrics_model
        elif model_type == "image":
            if self._art_model is None or self._art_model_name != model_name:
                try:
                    from sentence_transformers import SentenceTransformer
                    # CLIP models can handle both text and images
                    self._art_model = SentenceTransformer(model_name)
                    self._art_model_name = model_name
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required for image embeddings. "
                        "Install it with: pip install sentence-transformers"
                    )
            return self._art_model
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'text' or 'image'")
    
    def generate_lyrics_embedding(
        self, 
        text: str, 
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for lyrics text.
        
        Args:
            text: Lyrics text to embed
            model_name: Optional model name (defaults to environment variable or 'all-MiniLM-L6-v2')
            
        Returns:
            numpy array of embeddings
        """
        if model_name is None:
            model_name = os.getenv("LYRICS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        model = self.get_embedding_model(model_name, model_type="text")
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_art_embedding(
        self, 
        image_path: str, 
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for album art image.
        
        Args:
            image_path: Path to image file
            model_name: Optional model name (defaults to environment variable or CLIP model)
            
        Returns:
            numpy array of embeddings
        """
        if model_name is None:
            model_name = os.getenv("ART_EMBEDDING_MODEL", "clip-ViT-B-32")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        model = self.get_embedding_model(model_name, model_type="image")
        
        # Load and encode image
        from PIL import Image
        image = Image.open(image_path)
        embedding = model.encode(image, convert_to_numpy=True)
        return embedding
    
    def store_lyrics_embedding(
        self, 
        song_id: str, 
        embedding: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store lyrics embedding in Chroma.
        
        Args:
            song_id: Song ID (primary key)
            embedding: Embedding vector
            metadata: Optional metadata dict (e.g., {'title': 'Song Title', 'artist': 'Artist Name'})
        """
        if metadata is None:
            metadata = {}
        
        # Ensure song_id is in metadata
        metadata["song_id"] = song_id
        
        # Convert numpy array to list for Chroma
        embedding_list = embedding.tolist()
        
        # Add or update the embedding
        self.lyrics_collection.upsert(
            ids=[song_id],
            embeddings=[embedding_list],
            metadatas=[metadata]
        )
    
    def store_art_embedding(
        self, 
        song_id: str, 
        embedding: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store album art embedding in Chroma.
        
        Args:
            song_id: Song ID (primary key)
            embedding: Embedding vector
            metadata: Optional metadata dict (e.g., {'title': 'Song Title', 'artist': 'Artist Name'})
        """
        if metadata is None:
            metadata = {}
        
        # Ensure song_id is in metadata
        metadata["song_id"] = song_id
        
        # Convert numpy array to list for Chroma
        embedding_list = embedding.tolist()
        
        # Add or update the embedding
        self.art_collection.upsert(
            ids=[song_id],
            embeddings=[embedding_list],
            metadatas=[metadata]
        )
    
    def query_by_semantics(
        self,
        query_text: str,
        collection: str = "lyrics",
        top_k: int = 10,
        model_name: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query vector database by semantic similarity.
        
        Args:
            query_text: Text query for semantic search
            collection: Collection to search ('lyrics' or 'art')
            top_k: Number of results to return
            model_name: Optional model name for query embedding
            where: Optional metadata filter (e.g., {'artist': 'Artist Name'})
            
        Returns:
            List of results with 'id', 'distance', and 'metadata' keys
        """
        # Get the appropriate collection
        if collection == "lyrics":
            coll = self.lyrics_collection
            if model_name is None:
                model_name = os.getenv("LYRICS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            model = self.get_embedding_model(model_name, model_type="text")
        elif collection == "art":
            coll = self.art_collection
            if model_name is None:
                model_name = os.getenv("ART_EMBEDDING_MODEL", "clip-ViT-B-32")
            model = self.get_embedding_model(model_name, model_type="image")
        else:
            raise ValueError(f"Unknown collection: {collection}. Use 'lyrics' or 'art'")
        
        # Generate query embedding
        query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()
        
        # Query the collection
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, song_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": song_id,
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                })
        
        return formatted_results

    def generate_bow_embedding(
        self,
        bow_vector: Dict[int, int],
        vocab: Dict[int, str],
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding from MXM musiXmatch bag-of-words.
        """
        words = []
        for wid, count in sorted(bow_vector.items()):
            word = vocab.get(wid, f"unk_{wid}")
            words.extend([word] * min(count, 5))
        text = " ".join(words[:512])
        
        model = self.get_embedding_model(model_name or "all-MiniLM-L6-v2", model_type="text")
        return model.encode(text, convert_to_numpy=True)


# Global instance (can be initialized with custom settings)
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(
    chroma_path: Optional[str] = None,
    persist_directory: Optional[str] = None
) -> EmbeddingManager:
    """
    Get or create global EmbeddingManager instance.
    
    Args:
        chroma_path: Path to Chroma database
        persist_directory: Local directory for persistent storage
        
    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(
            chroma_path=chroma_path,
            persist_directory=persist_directory
        )
    return _embedding_manager


# Convenience functions for direct use
def generate_lyrics_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """Generate lyrics embedding using global EmbeddingManager."""
    manager = get_embedding_manager()
    return manager.generate_lyrics_embedding(text, model_name)


def generate_art_embedding(image_path: str, model_name: Optional[str] = None) -> np.ndarray:
    """Generate album art embedding using global EmbeddingManager."""
    manager = get_embedding_manager()
    return manager.generate_art_embedding(image_path, model_name)


def store_lyrics_embedding(
    song_id: str, 
    embedding: np.ndarray, 
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Store lyrics embedding using global EmbeddingManager."""
    manager = get_embedding_manager()
    manager.store_lyrics_embedding(song_id, embedding, metadata)


def store_art_embedding(
    song_id: str, 
    embedding: np.ndarray, 
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Store album art embedding using global EmbeddingManager."""
    manager = get_embedding_manager()
    manager.store_art_embedding(song_id, embedding, metadata)


def query_by_semantics(
    query_text: str,
    collection: str = "lyrics",
    top_k: int = 10,
    model_name: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Query vector database by semantic similarity using global EmbeddingManager."""
    manager = get_embedding_manager()
    return manager.query_by_semantics(query_text, collection, top_k, model_name, where)

def generate_bow_embedding(
    bow_vector: Dict[int, int], 
    vocab: Dict[int, str], 
    model_name: Optional[str] = None
) -> np.ndarray:
    """Generate BoW embedding using global EmbeddingManager."""
    manager = get_embedding_manager()
    return manager.generate_bow_embedding(bow_vector, vocab, model_name)
