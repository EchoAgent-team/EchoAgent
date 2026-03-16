"""
ETL for MSD → relational DB.

Phases: HDF5 core → Last.fm/tags → musiXmatch.
Usage:
    python ingest_msd.py --h5_dir /path/to/MillionSongSubset \
        --db_url sqlite:///echoagent.db
"""

import os
import argparse
import json
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any
import sqlite3
from sqlalchemy.exc import IntegrityError

from db import (
    get_db_session,
    DBSession,
    Artist,
    Album,
    Track,
    AudioFeature,
    Lyrics,
)

def parse_h5_file(h5_path: Path, session: Any) -> None:
    """Parse single HDF5 → insert artist/album/track/audio."""

    tid = h5_path.stem  # Use HDF5 filename as track_id
    with h5py.File(h5_path, 'r') as f:
        songs = f['metadata/songs'][:]
        song_data = {k: songs[k][0] for k in songs.dtype.names if k != 'song_id'}
        
        # Artist (skip if no ID)
        artist_id = song_data.get('artist_id')
        if artist_id is not None:
            artist_id_str = str(artist_id).strip()
            artist_name = song_data.get('artist_name')
            artist_name_str = str(artist_name) if artist_name is not None else ''

            # Check if artist already exists and ingest if not
            existing = session.query(Artist).filter_by(artist_id=artist_id_str).first()
            if not existing:
                session.merge(
                    Artist(
                        artist_id=artist_id_str,
                        name=artist_name_str,
                        terms=(
                            song_data.get('artist_terms', None).tolist()
                            if song_data.get('artist_terms') is not None
                            else None
                        ),
                        terms_freq=(
                            song_data.get('artist_terms_freq', None).tolist()
                            if song_data.get('artist_terms_freq') is not None
                            else None
                        ),
                        terms_weight=(
                            song_data.get('artist_terms_weight', None).tolist()
                            if song_data.get('artist_terms_weight') is not None
                            else None
                        ),
                    )
                )
        
        # Album (surrogate ID: artist_release_title, since MSD has no stable album PK)
        release = song_data.get('release')
        release_str = str(release) if release is not None else ''
        year = int(song_data.get('year', 0)) if song_data.get('year') is not None else None
        
        release_id = None
        if artist_id_str and release_str:
            release_id = (
                f"{artist_id_str}_{release_str.replace(' ', '_')[:50]}"
            )  # Truncate long titles
            
            session.merge(Album(
                release_id=release_id,
                title=release_str,
                year=year,
                artist_id=artist_id_str
            ))
        
        # Track base row points to artist/album and leaves tag/genre fields for later phases
        title = song_data.get('title')
        title_str = str(title) if title is not None else ''
        
        track = Track(
            track_id=str(tid),
            title=title_str,
            artist_id=artist_id_str or '',
            release_id=release_id,
            seed_genre=None,  # Phase 2
            top_tags_json=None  # Phase 2
        )
        session.merge(track)
        
        # Audio features
        analysis_group = f.get('analysis', None)
        if analysis_group and 'songs' in analysis_group:
            try:
                analysis_songs = analysis_group['songs'][:]
                analysis_row = analysis_songs[0]
                af_data = {}
                for field in [
                    'duration',
                    'key',
                    'key_confidence',
                    'loudness',
                    'mode',
                    'mode_confidence',
                    'tempo',
                    'time_signature',
                    'time_signature_confidence',
                    'danceability',
                    'energy',
                ]:
                    val = analysis_row[field] if field in analysis_row.dtype.names else None
                    af_data[field] = float(val) if val is not None else None
                
                af = AudioFeature(track_id=str(tid), **af_data)
                session.merge(af)
            except Exception as e:
                print(f"Warning: Could not parse analysis for {tid}: {e}")

def ingest_h5_dir(h5_dir: Path, db_url: str, batch_size: int = 100) -> None:
    """Ingest all HDF5 files for core metadata + audio features."""
    # Walk the MSD subset tree and collect all HDF5 files once up front
    h5_files = list(h5_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} HDF5 files.")
    
    # Use a single DBSession; commit per file for safety (no batch-level data loss)
    with DBSession(db_url) as session:
        pbar = tqdm(h5_files, desc="HDF5 files")
        for h5_path in pbar:
            try:
                # Parse and upsert a single track worth of rows
                parse_h5_file(h5_path, session)
                # Commit immediately so that a failure only affects this file
                session.commit()
            except IntegrityError as e:
                # Most likely a duplicate PK or FK issue for this specific file
                session.rollback()
                print(f"⚠️ IntegrityError for {h5_path} — skipping file: {e}")
            except Exception as e:
                # Any other parsing/DB error is logged and the file is skipped
                session.rollback()
                print(f"Error parsing {h5_path}: {e}")
            finally:
                # Detach objects from the session to keep memory usage bounded
                session.expunge_all()
    print("✅ Ingested core metadata + audio features from MSD.")

def ingest_lastfm_json_subset(json_base_dir: Path, db_url: str) -> None:
    """
    Ingest Last.fm JSON from MSD subset structure:
    AdditionalFiles/A/AA0/AA0000000.json → tags for matching .h5 track.
    JSON format: {"track_id": "...", "tags": [...], "tag_weights": [...]}
    """
    # Accumulate tags per track_id before a single DB pass
    tags_data = {}
    
    # Mirror the MSD subset folder structure (A/B/C/...) for Last.fm JSON sidecar files
    json_files = list(json_base_dir.rglob("*.json"))
    print(f"Found {len(json_files)} Last.fm JSON files.")
    
    for json_path in tqdm(json_files, desc="JSON tags"):
        try:
            data_raw = json_path.read_text(errors='ignore')
            data = json.loads(data_raw)
            tid = data.get('track_id', '').strip()
            if not tid:
                continue
                
            tags = data.get('tags', [])
            if isinstance(tags, list) and tags:
                weights = data.get('tag_weights', [])
                tag_dict = {}
                # Keep at most the top 50 tags per track to bound JSON size
                for j, tag in enumerate(tags[:50]):
                    w = float(weights[j]) if j < len(weights) else 1.0
                    if w > 0:
                        tag_dict[str(tag)] = w
                
                if tag_dict:
                    tags_data[tid] = tag_dict
                    
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"Skip {json_path.name}: {type(e).__name__}")
    
    print(f"Loaded {len(tags_data)} tagged tracks from JSON.")
    
    # Single pass over the DB to attach Last.fm tags to existing tracks
    with DBSession(db_url) as session:
        tracks = {t.track_id: t for t in session.query(Track).all()}
        updates = 0
        for tid, tags in tqdm(tags_data.items(), desc="Updating JSON tags"):
            if tid in tracks:
                tracks[tid].top_tags_json = tags
                updates += 1
        session.commit()
    print(f"✅ Updated {updates} tracks with Last.fm JSON tags.")        

def ingest_musixmatch_sqlite(db_path: Path, msd_db_url: str) -> None:
    """Query MXM SQLite directly and attach lyrics BoW to known MSD tracks."""
    print(f"Loading lyrics from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Only request lyrics for tracks we already ingested into the relational DB
    with DBSession(msd_db_url) as session:
        track_ids = [t.track_id for t in session.query(Track.track_id).all()]
    
    # Build an in-memory mapping of track_id → {word: count}
    lyrics_data = {}
    for tid in tqdm(track_ids, desc="MXM SQLite query"):
        # MXM schema stores one (track_id, word, count) row per non-zero entry
        cursor.execute("""
            SELECT word, count FROM lyrics 
            WHERE track_id = ? AND count > 0
        """, (tid,))
        bow = dict(cursor.fetchall())
        if bow:
            lyrics_data[tid] = bow
    
    # Insert lyrics rows once we have the complete BoW for each track
    with DBSession(msd_db_url) as session:
        for tid, bow in tqdm(lyrics_data.items(), desc="Insert BoW"):
            session.merge(Lyrics(track_id=tid, bow_vector=bow))
        session.commit()
    
    conn.close()
    print(f"✅ Ingested {len(lyrics_data)} MXM BoW lyrics.")

def ingest_genres(cls_file: Path, db_url: str) -> None:
    """Ingest seed_genre from msd_lastfm_map.cls → Track.seed_genre."""
    print(f"Parsing genres from {cls_file}...")
    genres_data = {}
    
    with open(cls_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Last.fm map .cls")):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip comments/empty
                continue
                
            # Format: track_id genre [other_tags...]
            # e.g. "TRABC123 rock alternative indie"
            parts = line.split()
            if len(parts) >= 2:
                tid = parts[0]
                seed_genre = parts[1]  # First = seed genre
                genres_data[tid] = seed_genre
    
    print(f"Loaded {len(genres_data)} genre mappings.")
    
    with DBSession(db_url) as session:
        tracks = {t.track_id: t for t in session.query(Track).all()}
        updates = 0
        for tid, genre in tqdm(genres_data.items(), desc="Updating seed_genre"):
            if tid in tracks:
                tracks[tid].seed_genre = genre
                updates += 1
        session.commit()
        print(f"✅ Ingested {updates} tracks with seed_genre from tagtraum.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSD ETL Pipeline")
    parser.add_argument("--h5_dir", required=True, help="Path to HDF5 directory")
    parser.add_argument("--lastfm_json_dir", help="Dir with Last.fm JSON (subset, parallel to h5)")
    parser.add_argument("--mxm_sqlite", help="Path to mxm_dataset.db (REQUIRED for lyrics)")
    parser.add_argument("--lastfm_map", help="Path to msd_lastfm_map.cls (REQUIRED for genres)")
    parser.add_argument("--db_url", default="sqlite:///echoagent.db")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    
    # Phase 1: Core (always)
    ingest_h5_dir(Path(args.h5_dir), args.db_url, args.batch_size)
    
    # Phase 2: Tags (JSON subset only)
    if args.lastfm_json_dir and Path(args.lastfm_json_dir).exists():
        ingest_lastfm_json_subset(Path(args.lastfm_json_dir), args.db_url)
    else:
        print("⚠️ Skip tags: Need --lastfm_json_dir (Last.fm JSON subset)")
    
    # Phase 3: Lyrics (SQLite only)
    if args.mxm_sqlite and Path(args.mxm_sqlite).exists():
        ingest_musixmatch_sqlite(Path(args.mxm_sqlite), args.db_url)
    else:
        print("⚠️ Skip lyrics: --mxm_sqlite REQUIRED (mxm_dataset.db)")
        
    # Phase 4: Genres (cls file)
    if args.lastfm_map and Path(args.lastfm_map).exists():
        ingest_genres(Path(args.lastfm_map), args.db_url)
    else:
        print("⚠️ Skip genres: --lastfm_map Data/msd_lastfm_map.cls")

    print("✅ ETL complete!")
