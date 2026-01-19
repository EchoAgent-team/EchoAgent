"""
Database models and initialization for EchoAgent.

This module defines SQLAlchemy models for storing song metadata, audio features,
tags, lyrics, and album art in a relational database (SQLite/PostgreSQL).
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, UniqueConstraint,
    create_engine, and_, or_
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import os

Base = declarative_base()


class Artist(Base):
    """Artists table storing artist information."""
    __tablename__ = "artists"

    artist_id = Column(String, primary_key=True)  # Spotify ID or custom UUID
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    songs = relationship("Song", back_populates="artist")


class Album(Base):
    """Albums table storing album information."""
    __tablename__ = "albums"

    album_id = Column(String, primary_key=True)  # Spotify ID or custom UUID
    title = Column(String, nullable=False)
    art_url = Column(String, nullable=True)  # URL to album art (original source)
    art_file_path = Column(String, nullable=True)  # Local file path for album art (for embeddings)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    songs = relationship("Song", back_populates="album")


class Song(Base):
    """Main songs table storing basic song metadata."""
    __tablename__ = "songs"

    id = Column(String, primary_key=True)  # Spotify ID or custom UUID
    title = Column(String, nullable=False)
    artist_id = Column(String, ForeignKey("artists.artist_id", ondelete="CASCADE"), nullable=False)
    album_id = Column(String, ForeignKey("albums.album_id", ondelete="SET NULL"), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    spotify_id = Column(String, unique=True, nullable=True)
    spotify_url = Column(String, nullable=True)
    preview_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    artist = relationship("Artist", back_populates="songs")
    album = relationship("Album", back_populates="songs")
    audio_features = relationship("AudioFeature", back_populates="song", uselist=False, cascade="all, delete-orphan")
    lyrics = relationship("Lyric", back_populates="song", uselist=False, cascade="all, delete-orphan")
    tags = relationship("Tag", back_populates="song", cascade="all, delete-orphan")


class AudioFeature(Base):
    """Audio features extracted from Spotify or other sources."""
    __tablename__ = "audio_features"

    id = Column(String, ForeignKey("songs.id", ondelete="CASCADE"), primary_key=True)
    tempo = Column(Float, nullable=True)
    key = Column(Integer, nullable=True)  # 0-11, representing musical keys
    mode = Column(Integer, nullable=True)  # 0=minor, 1=major
    valence = Column(Float, nullable=True)  # 0.0-1.0
    energy = Column(Float, nullable=True)  # 0.0-1.0
    danceability = Column(Float, nullable=True)  # 0.0-1.0
    acousticness = Column(Float, nullable=True)  # 0.0-1.0
    instrumentalness = Column(Float, nullable=True)  # 0.0-1.0
    liveness = Column(Float, nullable=True)  # 0.0-1.0
    speechiness = Column(Float, nullable=True)  # 0.0-1.0
    loudness = Column(Float, nullable=True)  # dB
    time_signature = Column(Integer, nullable=True)

    # Relationship
    song = relationship("Song", back_populates="audio_features")


class Tag(Base):
    """Tags for songs (genre, mood, source, etc.)."""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    song_id = Column(String, ForeignKey("songs.id", ondelete="CASCADE"), nullable=False)
    tag_type = Column(String, nullable=False)  # 'genre', 'mood', 'source'
    tag_value = Column(String, nullable=False)

    # Relationship
    song = relationship("Song", back_populates="tags")

    # Unique constraint to prevent duplicate tags
    __table_args__ = (
        UniqueConstraint('song_id', 'tag_type', 'tag_value', name='uq_song_tag'),
    )


class Lyric(Base):
    """Full lyrics text for songs."""
    __tablename__ = "lyrics"

    id = Column(String, ForeignKey("songs.id", ondelete="CASCADE"), primary_key=True)
    full_text = Column(Text, nullable=False)
    source = Column(String, nullable=True)  # e.g., 'genius', 'spotify'
    language = Column(String, nullable=True)

    # Relationship
    song = relationship("Song", back_populates="lyrics")


# Database initialization
def get_database_url() -> str:
    """Get database URL from environment or use default SQLite."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    # Default to SQLite in backend/data directory
    db_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "echoagent.db")
    return f"sqlite:///{db_path}"


def init_db(database_url: Optional[str] = None) -> None:
    """
    Initialize the database by creating all tables.
    
    Args:
        database_url: Optional database URL. If not provided, uses get_database_url().
    """
    url = database_url or get_database_url()
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {url}")


def get_db_session(database_url: Optional[str] = None) -> Session:
    """
    Get a database session.
    
    Args:
        database_url: Optional database URL. If not provided, uses get_database_url().
        
    Returns:
        SQLAlchemy Session object
    """
    url = database_url or get_database_url()
    engine = create_engine(url, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


# Context manager for database sessions
class DBSession:
    """Context manager for database sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url()
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        self.session = self.SessionLocal()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type:
                self.session.rollback()
            else:
                self.session.commit()
            self.session.close()


# Query pipeline functions
def query_by_filters(
    session: Session,
    artist: Optional[str] = None,
    title: Optional[str] = None,
    album: Optional[str] = None,
    min_tempo: Optional[float] = None,
    max_tempo: Optional[float] = None,
    key: Optional[int] = None,
    mode: Optional[int] = None,
    min_valence: Optional[float] = None,
    max_valence: Optional[float] = None,
    min_energy: Optional[float] = None,
    max_energy: Optional[float] = None,
    min_danceability: Optional[float] = None,
    max_danceability: Optional[float] = None,
    tags: Optional[List[Dict[str, str]]] = None,  # [{'tag_type': 'genre', 'tag_value': 'rock'}]
    limit: Optional[int] = None
) -> List[Song]:
    """
    Query songs using SQL filters.
    
    Args:
        session: Database session
        artist: Filter by artist name (partial match)
        title: Filter by title (partial match)
        album: Filter by album name (partial match)
        min_tempo: Minimum tempo
        max_tempo: Maximum tempo
        key: Musical key (0-11)
        mode: Mode (0=minor, 1=major)
        min_valence: Minimum valence (0.0-1.0)
        max_valence: Maximum valence (0.0-1.0)
        min_energy: Minimum energy (0.0-1.0)
        max_energy: Maximum energy (0.0-1.0)
        min_danceability: Minimum danceability (0.0-1.0)
        max_danceability: Maximum danceability (0.0-1.0)
        tags: List of tag filters, each with 'tag_type' and 'tag_value'
        limit: Maximum number of results
        
    Returns:
        List of Song objects matching the filters
    """
    query = session.query(Song).join(AudioFeature, Song.id == AudioFeature.id, isouter=True)
    
    # Basic filters
    if artist:
        query = query.join(Artist, Song.artist_id == Artist.artist_id)
        query = query.filter(Artist.name.ilike(f"%{artist}%"))
    if title:
        query = query.filter(Song.title.ilike(f"%{title}%"))
    if album:
        query = query.join(Album, Song.album_id == Album.album_id, isouter=True)
        query = query.filter(Album.title.ilike(f"%{album}%"))
    
    # Audio feature filters
    if min_tempo is not None:
        query = query.filter(AudioFeature.tempo >= min_tempo)
    if max_tempo is not None:
        query = query.filter(AudioFeature.tempo <= max_tempo)
    if key is not None:
        query = query.filter(AudioFeature.key == key)
    if mode is not None:
        query = query.filter(AudioFeature.mode == mode)
    if min_valence is not None:
        query = query.filter(AudioFeature.valence >= min_valence)
    if max_valence is not None:
        query = query.filter(AudioFeature.valence <= max_valence)
    if min_energy is not None:
        query = query.filter(AudioFeature.energy >= min_energy)
    if max_energy is not None:
        query = query.filter(AudioFeature.energy <= max_energy)
    if min_danceability is not None:
        query = query.filter(AudioFeature.danceability >= min_danceability)
    if max_danceability is not None:
        query = query.filter(AudioFeature.danceability <= max_danceability)
    
    # Tag filters
    if tags:
        tag_conditions = []
        for tag_filter in tags:
            tag_type = tag_filter.get('tag_type')
            tag_value = tag_filter.get('tag_value')
            if tag_type and tag_value:
                tag_conditions.append(
                    and_(
                        Tag.tag_type == tag_type,
                        Tag.tag_value == tag_value
                    )
                )
        if tag_conditions:
            # Join with tags and filter
            query = query.join(Tag, Song.id == Tag.song_id)
            query = query.filter(or_(*tag_conditions))
            # Use distinct to avoid duplicates from multiple tags
            query = query.distinct()
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def query_by_semantics(
    session: Session,
    query_text: str,
    collection: str = "lyrics",
    top_k: int = 10,
    model_name: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None
) -> List[Song]:
    """
    Query songs using semantic similarity search in vector database.
    
    Args:
        session: Database session
        query_text: Text query for semantic search
        collection: Collection to search ('lyrics' or 'art')
        top_k: Number of results to return
        model_name: Optional model name for query embedding
        where: Optional metadata filter for Chroma
        
    Returns:
        List of Song objects ordered by similarity
    """
    # Import here to avoid circular dependency
    from .embeddings import query_by_semantics as vector_query
    
    # Query vector database
    vector_results = vector_query(
        query_text=query_text,
        collection=collection,
        top_k=top_k,
        model_name=model_name,
        where=where
    )
    
    # Get song IDs from vector results
    song_ids = [result["id"] for result in vector_results]
    
    if not song_ids:
        return []
    
    # Query SQL database for full song objects
    songs = session.query(Song).filter(Song.id.in_(song_ids)).all()
    
    # Sort by order from vector results (maintain similarity ranking)
    song_dict = {song.id: song for song in songs}
    ordered_songs = [song_dict[song_id] for song_id in song_ids if song_id in song_dict]
    
    return ordered_songs


def hybrid_query(
    session: Session,
    filters: Optional[Dict[str, Any]] = None,
    text_query: Optional[str] = None,
    collection: str = "lyrics",
    top_k: int = 10,
    model_name: Optional[str] = None,
    vector_where: Optional[Dict[str, Any]] = None
) -> List[Song]:
    """
    Combine SQL filters with vector similarity search.
    
    This function first applies SQL filters to get a candidate set,
    then performs vector search within that set, or vice versa depending on
    which is more selective.
    
    Args:
        session: Database session
        filters: Dictionary of SQL filter parameters (same as query_by_filters)
        text_query: Text query for semantic search
        collection: Collection to search ('lyrics' or 'art')
        top_k: Number of results to return
        model_name: Optional model name for query embedding
        vector_where: Optional metadata filter for Chroma
        
    Returns:
        List of Song objects matching both filters and semantic query
    """
    if not filters and not text_query:
        # No filters, return empty
        return []
    
    if filters and not text_query:
        # Only SQL filters
        return query_by_filters(session, **filters, limit=top_k)
    
    if text_query and not filters:
        # Only vector search
        return query_by_semantics(
            session, text_query, collection, top_k, model_name, vector_where
        )
    
    # Both filters and text query - query Chroma first, then filter in Python
    # This approach avoids Chroma's $in operator limits and scales better
    
    # Get candidate set from SQL filters
    candidate_songs = query_by_filters(session, **filters)
    candidate_ids = {song.id for song in candidate_songs}
    
    if not candidate_ids:
        return []
    
    # Query vector database first (semantic search)
    from .embeddings import query_by_semantics as vector_query
    
    # Query with larger top_k to ensure we get enough semantic matches
    # that overlap with SQL candidates. Scale top_k based on candidate set size.
    # For large candidate sets, query more results to increase overlap probability
    query_size = min(top_k * 10, max(top_k * 3, len(candidate_ids) // 10))
    query_size = max(query_size, top_k * 2)  # At least 2x top_k
    
    vector_results = vector_query(
        query_text=text_query,
        collection=collection,
        top_k=query_size,
        model_name=model_name,
        where=vector_where
    )
    
    # Filter vector results to only include candidates from SQL
    # This Python-side filtering avoids Chroma's $in operator limits
    filtered_results = [
        result for result in vector_results 
        if result["id"] in candidate_ids
    ][:top_k]
    
    # Get song IDs
    song_ids = [result["id"] for result in filtered_results]
    
    if not song_ids:
        return []
    
    # Query SQL database for full song objects
    songs = session.query(Song).filter(Song.id.in_(song_ids)).all()
    
    # Sort by order from vector results
    song_dict = {song.id: song for song in songs}
    ordered_songs = [song_dict[song_id] for song_id in song_ids if song_id in song_dict]
    
    return ordered_songs


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
