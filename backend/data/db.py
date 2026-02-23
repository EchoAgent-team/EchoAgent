"""
MSD-based database models and initialization for EchoAgent.
Schema: artists → albums → tracks + audio_features + lyrics (BoW JSON).

This module defines SQLAlchemy models for storing song metadata, audio features,
tags, lyrics, and album art in a relational database (SQLite/PostgreSQL).
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, UniqueConstraint,
    create_engine, and_, or_, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import os

Base = declarative_base()


class Artist(Base):
    """Artists table storing artist information."""
    __tablename__ = "artists"
    
    artist_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    terms = Column(JSON, nullable=True)  # artist_terms array
    terms_freq = Column(JSON, nullable=True)
    terms_weight = Column(JSON, nullable=True)
    
    # Relationships
    albums = relationship("Album", back_populates="artist")
    tracks = relationship("Track", back_populates="artist")

class Album(Base):
    """Albums table storing album information."""
    __tablename__ = "albums"
    
    release_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    year = Column(Integer, nullable=True)
    artist_id = Column(String, ForeignKey("artists.artist_id"), nullable=False)
    
    # Relationships
    artist = relationship("Artist", back_populates="albums")
    tracks = relationship("Track", back_populates="album")

class Song(Base):
    """Main songs table storing basic song metadata."""
    __tablename__ = "tracks"

    track_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    artist_id = Column(String, ForeignKey("artists.artist_id", ondelete="CASCADE"), nullable=False)
    release_id = Column(String, ForeignKey("albums.release_id", ondelete="CASCADE"), nullable=True)
    seed_genre = Column(String, nullable=True)  # From msd_lastfm_map.cls
    top_tags_json = Column(JSON, nullable=True)  # Last.fm + inferred: {"tag": weight}
    
    # Relationships
    artist = relationship("Artist", back_populates="tracks")
    album = relationship("Album", back_populates="tracks")
    audio_feature = relationship("AudioFeature", back_populates="track", uselist=False)
    lyrics = relationship("Lyrics", back_populates="track", uselist=False)

class AudioFeature(Base):
    __tablename__ = "audio_features"
    
    track_id = Column(String, ForeignKey("tracks.track_id", ondelete="CASCADE"), primary_key=True)
    duration = Column(Float, nullable=True)
    key = Column(Integer, nullable=True)  # 0-11
    key_confidence = Column(Float, nullable=True)
    loudness = Column(Float, nullable=True)
    mode = Column(Integer, nullable=True)  # 0=minor, 1=major
    mode_confidence = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)
    time_signature = Column(Integer, nullable=True)
    time_signature_confidence = Column(Float, nullable=True)
    danceability = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    
    # Relationship
    track = relationship("Track", back_populates="audio_feature")

class Lyrics(Base):
    __tablename__ = "lyrics"
    
    track_id = Column(String, ForeignKey("tracks.track_id", ondelete="CASCADE"), primary_key=True)
    bow_vector = Column(JSON, nullable=True)  # musiXmatch: {"word": count}
    
    # Relationship
    track = relationship("Track", back_populates="lyrics")

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
def query_tracks_by_filters(
    session: Session,
    artist_id: Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    seed_genre: Optional[str] = None,
    min_energy: Optional[float] = None,
    max_energy: Optional[float] = None,
    min_danceability: Optional[float] = None,
    max_danceability: Optional[float] = None,
    min_tempo: Optional[float] = None,
    max_tempo: Optional[float] = None,
    limit: Optional[int] = 100
) -> List[Track]:
    """
    Query tracks with filters. Use track_ids from vector DB for hybrid search.
    
    Args:
        session: DB session.
        artist_id: Exact MSD artist_id.
        min_year/max_year: Album year range.
        seed_genre: Exact match.
        min_energy etc.: Audio feature ranges.
        limit: Max results.
    
    Returns:
        List of Track objects.
    """
    query = session.query(Track).join(Album).outerjoin(AudioFeature)
    
    if artist_id:
        query = query.filter(Track.artist_id == artist_id)
    if seed_genre:
        query = query.filter(Track.seed_genre == seed_genre)
    if min_year:
        query = query.filter(Album.year >= min_year)
    if max_year:
        query = query.filter(Album.year <= max_year)
    if min_energy:
        query = query.filter(AudioFeature.energy >= min_energy)
    if max_energy:
        query = query.filter(AudioFeature.energy <= max_energy)
    if min_danceability:
        query = query.filter(AudioFeature.danceability >= min_danceability)
    if max_danceability:
        query = query.filter(AudioFeature.danceability <= max_danceability)
    if min_tempo:
        query = query.filter(AudioFeature.tempo >= min_tempo)
    if max_tempo:
        query = query.filter(AudioFeature.tempo <= max_tempo)
    
    return query.limit(limit).all()

# Example: Get embed-ready data for embeddings.py
def get_embed_data(session: Session) -> List[Dict]:
    """
    Get data for embeddings.py: tracks + tags + lyrics + audio.
    """
    return session.query(
        Track.track_id,
        Track.seed_genre,
        Track.top_tags_json,
        Lyrics.bow_vector,
        AudioFeature.energy,
        AudioFeature.tempo,
        # Add more as needed
    ).outerjoin(Lyrics).outerjoin(AudioFeature).all()

if __name__ == "__main__":
    init_db()
