"""
ETL for MSD → relational DB.

Phases: HDF5 core → Last.fm/tags → musiXmatch.
Updated parser for official lastfm_train.txt/test.txt (tab-separated).
Usage:
    python ingest_msd.py --h5_dir /path/to/MillionSongSubset \
        --db_url sqlite:///echoagent.db
"""

import os
import argparse
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
import tables  # For HDF5
import sqlite3

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
    with h5py.File(h5_path, 'r') as f:
        songs = f['metadata/songs'][:]
        track_ids = songs['track_id'].astype(str)
        
        for i, tid in enumerate(track_ids):
            song_data = {k.decode(): v[i] for k, v in songs.items() if k != 'track_id'}
            
            # Artist (skip if no ID)
            artist_id = song_data.get('artist_id')
            if artist_id is not None:
                artist_id_str = (
                    artist_id.decode('utf-8')
                    if isinstance(artist_id, bytes)
                    else str(artist_id)
                )
                artist_name = song_data.get('artist_name')
                artist_name_str = (
                    artist_name.decode('utf-8')
                    if isinstance(artist_name, bytes)
                    else str(artist_name)
                    if artist_name is not None
                    else ''
                )

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
            
            # Album (surrogate ID: artist_release_title)
            release = song_data.get('release')
            release_str = (
                release.decode('utf-8')
                if isinstance(release, bytes)
                else str(release)
                if release is not None
                else None
            )
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
            
            # Track base
            title = song_data.get('title')
            title_str = (
                title.decode('utf-8')
                if isinstance(title, bytes)
                else str(title)
                if title is not None
                else ''
            )
            
            track = Track(
                track_id=tid.decode('utf-8'),
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
                    analysis = analysis_group['songs'][i]
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
                        val = analysis.get(field, None)
                        af_data[field] = float(val) if val is not None else None
                    
                    af = AudioFeature(track_id=tid.decode('utf-8'), **af_data)
                    session.merge(af)
                except Exception as e:
                    print(f"Warning: Could not parse analysis for {tid}: {e}")

def ingest_h5_dir(h5_dir: Path, db_url: str, batch_size: int = 100) -> None:
    """Ingest all HDF5 files."""
    h5_files = list(h5_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} HDF5 files.")
    
    with DBSession(db_url) as session:
        pbar = tqdm(h5_files, desc="HDF5 files")
        for h5_path in pbar:
            parse_h5_file(h5_path, session)
            if len(session.new) >= batch_size:
                session.commit()
                session.expunge_all()
                pbar.set_postfix(batch=f"{len(session.new)}")
        session.commit()
    print("Phase 1 complete: Core metadata + audio features.")

def ingest_lastfm_tags(lastfm_dir: Path, db_url: str) -> None:
    """Ingest Last.fm tags from official train.txt + test.txt (tab-separated)."""
    tags_data = {}
    
    for txt_file in lastfm_dir.glob("lastfm_*.txt"):
        print(f"Parsing {txt_file.name}...")
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            pbar = tqdm(
                lines,
                desc=f"{txt_file.name}",
                unit='lines',
                total=len(lines),
            )
            for line in pbar:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    tid, artist, title, tags_str = parts[0:4]
                    
                    tags = {}
                    if tags_str and tags_str.strip() and tags_str != '""':
                        # Unquote if needed
                        if tags_str.startswith('"') and tags_str.endswith('"'):
                            tags_str = tags_str[1:-1]
                        # Parse "tag1:w1;tag2:w2"
                        for tag_w in tags_str.split(';'):
                            tag_w = tag_w.strip()
                            if ':' in tag_w:
                                tag, w_str = tag_w.rsplit(':', 1)
                                try:
                                    tags[tag.strip()] = float(w_str.strip())
                                except ValueError:
                                    continue
                    
                    if tags:  # Only store if has meaningful tags
                        tags_data[tid] = {k: v for k, v in tags.items() if v > 0}
        
        print(f"{txt_file.name}: {len(tags_data)} unique tagged tracks so far.")
    
    print(f"Total: {len(tags_data)} tagged tracks loaded.")
    
    # Update tracks
    with DBSession(db_url) as session:
        tracks = (
            session.query(Track.track_id, Track)
            .filter(Track.track_id.in_(list(tags_data.keys())))
            .all()
        )
        track_dict = {tid: track for tid, track in tracks}
        
        updates = 0
        for tid, tags in tqdm(
            tags_data.items(),
            desc="Updating tags",
        ):
            if tid in track_dict:
                track_dict[tid].top_tags_json = tags
                updates += 1
        
        session.commit()
        print(f"Updated {updates} tracks with Last.fm tags.")

def ingest_lastfm_json_subset(json_base_dir: Path, db_url: str) -> None:
    """
    Ingest Last.fm JSON from MSD subset structure:
    AdditionalFiles/A/AA0/AA0000000.json → tags for matching .h5 track.
    JSON format: {"track_id": "...", "tags": [...], "tag_weights": [...]}
    """
    tags_data = {}
    
    # Walk parallel to HDF5: A/AA0/AA0000000.json
    json_files = list(json_base_dir.rglob("*.json"))
    print(f"Found {len(json_files)} Last.fm JSON files.")
    
    for json_path in tqdm(json_files, desc="JSON tags"):
        try:
            data = pd.read_json(json_path, typ='series')  # Single track JSON
            
            tid = data.get('track_id', '')
            tags = data.get('tags', [])
            weights = data.get('tag_weights', [])
            
            if tid and tags and len(tags) == len(weights):
                tag_dict = {tag: float(w) for tag, w in zip(tags, weights) if float(w) > 0}
                if tag_dict:
                    tags_data[tid] = tag_dict
                    
        except Exception as e:
            print(f"Warning: Skip {json_path}: {e}")
    
    print(f"Loaded {len(tags_data)} tagged tracks from JSON.")
    
    # Update DB (same as text parser)
    with DBSession(db_url) as session:
        tracks = {t.track_id: t for t in session.query(Track).all()}
        updates = 0
        for tid, tags in tqdm(tags_data.items(), desc="Updating JSON tags"):
            if tid in tracks:
                tracks[tid].top_tags_json = tags
                updates += 1
        session.commit()
    print(f"Updated {updates} tracks with subset JSON tags.")        

def ingest_musixmatch_sqlite(db_path: Path, msd_db_url: str) -> None:
    """Query MXM SQLite directly for existing track_ids."""
    print(f"Loading lyrics from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get relational track_ids
    with DBSession(msd_db_url) as session:
        track_ids = [t.track_id for t in session.query(Track.track_id).all()]
    
    lyrics_data = {}
    for tid in tqdm(track_ids, desc="MXM SQLite query"):
        cursor.execute("""
            SELECT word, count FROM lyrics 
            WHERE track_id = ? AND count > 0
        """, (tid,))
        bow = dict(cursor.fetchall())
        if bow:
            lyrics_data[tid] = bow
    
    # Insert
    with DBSession(msd_db_url) as session:
        for tid, bow in tqdm(lyrics_data.items(), desc="Insert BoW"):
            session.merge(Lyrics(track_id=tid, bow_vector=bow))
        session.commit()
    
    conn.close()
    print(f"✅ Inserted {len(lyrics_data)} lyrics BoW.")


def ingest_genres(cls_file: Path, db_url: str) -> None:
    """Ingest seed_genre from msd_lastfm_map.cls (placeholder—adjust format)."""
    print("Genres ingestion: TBD (custom parser for .cls format needed).")
    # TODO: Implement based on exact .cls format from tagtraum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSD ETL Pipeline")
    parser.add_argument("--h5_dir", required=True, help="Path to HDF5 directory")
    parser.add_argument("--lastfm_dir", help="Dir with lastfm_*.txt (full MSD)")
    parser.add_argument("--lastfm_json_dir", help="Dir with Last.fm JSON (subset, parallel to h5)")
    parser.add_argument("--mxm_sqlite", help="Path to mxm_dataset.db (REQUIRED for lyrics)")
    parser.add_argument("--db_url", default="sqlite:///echoagent.db")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    
    # Phase 1: Core (always)
    ingest_h5_dir(Path(args.h5_dir), args.db_url, args.batch_size)
    
    # Phase 2: Tags (text OR JSON)
    if args.lastfm_dir and Path(args.lastfm_dir).exists():
        ingest_lastfm_tags(Path(args.lastfm_dir), args.db_url)
    elif args.lastfm_json_dir and Path(args.lastfm_json_dir).exists():
        ingest_lastfm_json_subset(Path(args.lastfm_json_dir), args.db_url)
    else:
        print("⚠️ Skip tags: Need --lastfm_dir OR --lastfm_json_dir")
    
    # Phase 3: Lyrics (SQLite only)
    if args.mxm_sqlite and Path(args.mxm_sqlite).exists():
        ingest_musixmatch_sqlite(Path(args.mxm_sqlite), args.db_url)
    else:
        print("⚠️ Skip lyrics: --mxm_sqlite REQUIRED (mxm_dataset.db)")
    
    print("✅ ETL complete!")
