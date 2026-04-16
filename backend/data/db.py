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
from sqlalchemy.engine import Engine
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

class Track(Base):
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

    db_dir = os.path.dirname(__file__)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "music_relational.db")
    return f"sqlite:///{db_path}"

def get_engine(database_url: Optional[str] = None) -> Engine:
    """
    Create and return a SQLAlchemy engine.
    """
    url = database_url or get_database_url()
    return create_engine(url, echo=False, future=True)

def get_session_factory(database_url: Optional[str] = None) -> sessionmaker:
    """
    Return a configured SQLAlchemy session factory.
    """
    engine = get_engine(database_url)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(database_url: Optional[str] = None) -> None:
    """
    Initialize the database by creating all tables.
    
    Args:
        database_url: Optional database URL. If not provided, uses get_database_url().
    """
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {database_url or get_database_url()}")


def get_db_session(database_url: Optional[str] = None) -> Session:
    """
    Get a database session.
    
    Args:
        database_url: Optional database URL. If not provided, uses get_database_url().
        
    Returns:
        SQLAlchemy Session object
    """
    SessionLocal = get_session_factory(database_url)
    return SessionLocal()


# Context manager for database sessions
class DBSession:
    """Context manager for database sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url()
        self.SessionLocal = get_session_factory(self.database_url)
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

def serialize_track(track: Track) -> Dict[str, Any]:
    """
    Convert a Track ORM object into a plain dictionary for downstream use.
    """
    return {
        "track_id": track.track_id,
        "title": track.title,
        "artist_id": track.artist_id,
        "artist_name": track.artist.name if track.artist else None,
        "release_id": track.release_id,
        "album_title": track.album.title if track.album else None,
        "year": track.album.year if track.album else None,
        "seed_genre": track.seed_genre,
        "top_tags_json": track.top_tags_json,
        "duration": track.audio_feature.duration if track.audio_feature else None,
        "tempo": track.audio_feature.tempo if track.audio_feature else None,
        "danceability": track.audio_feature.danceability if track.audio_feature else None,
        "energy": track.audio_feature.energy if track.audio_feature else None,
        "loudness": track.audio_feature.loudness if track.audio_feature else None,
        "mode": track.audio_feature.mode if track.audio_feature else None,
    }


def query_tracks(
    filters: Dict[str, Any],
    limit: int = 100,
    database_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Public retrieval helper used by the retrieval layer.

    Expected filter keys:
        artist_id
        min_year
        max_year
        seed_genre
        min_energy
        max_energy
        min_danceability
        max_danceability
        min_tempo
        max_tempo

    Returns:
        List of serialized track dictionaries.
    """
    SessionLocal = get_session_factory(database_url)

    with SessionLocal() as session:
        tracks = query_tracks_by_filters(
            session=session,
            artist_id=filters.get("artist_id"),
            min_year=filters.get("min_year"),
            max_year=filters.get("max_year"),
            seed_genre=filters.get("seed_genre"),
            min_energy=filters.get("min_energy"),
            max_energy=filters.get("max_energy"),
            min_danceability=filters.get("min_danceability"),
            max_danceability=filters.get("max_danceability"),
            min_tempo=filters.get("min_tempo"),
            max_tempo=filters.get("max_tempo"),
            limit=limit,
        )
        return [serialize_track(track) for track in tracks]

# Example: Get embed-ready data for embeddings.py
def get_embed_data(session: Session) -> List[Dict]:
    """
    Get data for embeddings.py: tracks + tags + lyrics + audio.
    """
    rows = (
        session.query(
            Track.track_id,
            Track.seed_genre,
            Track.top_tags_json,
            Lyrics.bow_vector,
            AudioFeature.energy,
            AudioFeature.tempo,
        )
        .outerjoin(Lyrics)
        .outerjoin(AudioFeature)
        .all()
    )

    return [
        {
            "track_id": row.track_id,
            "seed_genre": row.seed_genre,
            "top_tags_json": row.top_tags_json,
            "bow_vector": row.bow_vector,
            "energy": row.energy,
            "tempo": row.tempo,
        }
        for row in rows
    ]

if __name__ == "__main__":
    init_db()
