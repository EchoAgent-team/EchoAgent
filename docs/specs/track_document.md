# TrackDocument Specification

## Purpose

`TrackDocument` is the canonical, flattened track representation used by
ingestion, embeddings, retrieval, reranking, and API responses.

It is a **data contract**, not necessarily a database table.

This allows EchoAgent to keep a multi-table relational schema internally while
using one consistent object shape across the retrieval pipeline.

## Design Goals

- Local retrieval-first (open/local metadata is the primary corpus).
- Null-safe optional fields (missing lyrics/audio/tags should not break flows).
- Stable `track_id` for vector storage and ranking.
- Easy to serialize as JSON.
- Easy to build from current relational tables or future schema changes.
- Spotify fields are optional playback hydration only.

## Canonical Shape (Top Level)

```json
{
  "track_id": "string",
  "title": "string",
  "artist_name": "string",
  "album_title": "string | null",
  "duration_ms": 0,
  "language": "string | null",
  "era": "string | null",
  "genres": [],
  "moods": [],
  "themes": [],
  "tags": [],
  "audio_features": {},
  "lyrics": {},
  "retrieval_text": {},
  "source_ids": {},
  "provenance": {},
  "playback": {}
}
```

## Required vs Optional

### Required fields

- `track_id`
- `title`
- `artist_name`

### Optional fields (recommended when available)

- `album_title`
- `duration_ms`
- `language`
- `era`
- `genres`
- `moods`
- `themes`
- `tags`
- `audio_features`
- `lyrics`
- `retrieval_text`
- `source_ids`
- `provenance`
- `playback`

## Field Definitions

### Identity and Core Metadata

- `track_id: string`
  - Stable internal identifier used across SQL + vector DB + API.
  - Must be unique.

- `title: string`
  - Track title.

- `artist_name: string`
  - Primary display artist.

- `album_title: string | null`
  - Album/release title if known.

- `duration_ms: number | null`
  - Track length in milliseconds.

- `language: string | null`
  - ISO-like language code or normalized language label when known.

- `era: string | null`
  - Normalized era bucket (examples: `"1970s"`, `"1990s"`, `"2010s"`).

### Tags and Semantic Metadata

All tag-like lists should be normalized, lowercase, and deduplicated.

- `genres: string[]`
  - Genre labels (from local metadata, Last.fm mapping, or curated sources).

- `moods: string[]`
  - Mood-like labels (e.g., `nostalgic`, `melancholic`, `uplifting`).

- `themes: string[]`
  - Theme/context labels (e.g., `night_drive`, `heartbreak`, `rainy_day`).

- `tags: string[]`
  - Generic normalized tags not captured elsewhere.
  - Useful as a catch-all for `msd_lastfm_map` labels.

## `audio_features` Object (Optional)

`audio_features` is optional and may be `{}` when unavailable.

Suggested keys:

```json
{
  "tempo": 102.4,
  "key": 0,
  "mode": 1,
  "energy": 0.58,
  "valence": 0.41,
  "danceability": 0.63,
  "acousticness": 0.07,
  "instrumentalness": 0.02,
  "liveness": 0.11,
  "speechiness": 0.04,
  "loudness": -8.2,
  "time_signature": 4
}
```

Null-safe rule:
- Missing individual keys are allowed.
- Do not invent numeric defaults.

## `lyrics` Object (Optional)

`lyrics` is optional and may be `{}` when lyrics are unavailable.

Suggested keys:

```json
{
  "source": "mxm",
  "language": "en",
  "full_text": null,
  "bow_vector": {
    "12": 3,
    "98": 1
  },
  "vocab_size": 5000
}
```

Notes:
- `full_text` may be `null` for copyright reasons.
- `bow_vector` is the preferred local lyrics representation for MXM ingestion.
- Keys may be strings or integers internally, but serialization should use JSON-safe keys.

## `retrieval_text` Object (Optional but Recommended)

This object stores text fields used to build embeddings and semantic search.

Suggested keys:

```json
{
  "semantic_text": "nostalgic night drive synth pop medium energy",
  "embedding_text": "title midnight drive artist example artist genres synthpop moods nostalgic night_drive tags neon city rain"
}
```

Rules:
- `embedding_text` should be deterministic and reproducible from the document.
- Exclude empty fields.
- Keep it compact; do not dump raw JSON.

## `source_ids` Object (Optional)

External and dataset identifiers.

Suggested keys:

```json
{
  "spotify_id": "string",
  "mxm_tid": 12345,
  "msd_track_id": "TRXXXX",
  "musicbrainz_recording_id": "uuid"
}
```

## `provenance` Object (Optional)

Tracks where each field came from for debugging/trust.

Suggested keys:

```json
{
  "metadata_source": "msd",
  "lyrics_source": "mxm",
  "tag_source": "lastfm_map",
  "audio_features_source": "spotify",
  "ingested_at": "2026-02-23T10:00:00Z"
}
```

## `playback` Object (Optional Hydration Only)

Spotify is optional and should not be required for retrieval.

Suggested keys:

```json
{
  "spotify_url": "https://open.spotify.com/track/...",
  "preview_url": null
}
```

## Null-Safety Rules

- Missing sections must be represented as empty objects `{}` or empty lists `[]`,
  not invalid placeholders.
- Unknown scalar values should be `null`, not guessed values.
- Retrieval and embeddings code must tolerate:
  - no `lyrics`
  - no `audio_features`
  - no `genres/tags`
  - no `playback`

## Mapping From Current Relational Schema (Transitional)

This spec is compatible with the current multi-table schema in
`backend/data/db.py` by flattening joined rows:

- `Song.id` -> `track_id`
- `Song.title` -> `title`
- `Artist.name` -> `artist_name`
- `Album.title` -> `album_title`
- `Song.duration_ms` -> `duration_ms`
- `AudioFeature.*` -> `audio_features.*`
- `Lyric.bow_vector` -> `lyrics.bow_vector`
- `Lyric.full_text` -> `lyrics.full_text`
- `Tag.tag_value` (by type) -> `genres` / `moods` / `tags`
- `Song.spotify_url`, `Song.preview_url` -> `playback.*`
- `Song.spotify_id` -> `source_ids.spotify_id`

## Embeddings Guidance (For `embeddings.py`)

`embeddings.py` should embed a `TrackDocument`-derived text representation, not
just lyrics.

Recommended embedding inputs (in priority order):

1. `retrieval_text.embedding_text` (if precomputed)
2. Deterministic composition from:
   - `title`
   - `artist_name`
   - `album_title`
   - `genres`
   - `moods`
   - `themes`
   - `tags` (including `msd_lastfm_map` labels)
   - selected audio buckets (e.g., `energy`, tempo bucket)
   - BOW-derived text (from `lyrics.bow_vector`) when available

This keeps semantic retrieval useful even when full lyrics text is absent.

## Example `TrackDocument`

```json
{
  "track_id": "TRABC123",
  "title": "Midnight Drive",
  "artist_name": "Example Artist",
  "album_title": "Neon Roads",
  "duration_ms": 234000,
  "language": "en",
  "era": "2010s",
  "genres": ["synthpop", "indie_pop"],
  "moods": ["nostalgic", "dreamy"],
  "themes": ["night_drive", "city_rain"],
  "tags": ["neon", "late_night", "melodic"],
  "audio_features": {
    "tempo": 102.4,
    "energy": 0.58,
    "valence": 0.41
  },
  "lyrics": {
    "source": "mxm",
    "full_text": null,
    "bow_vector": {
      "12": 3,
      "98": 1
    },
    "vocab_size": 5000
  },
  "retrieval_text": {
    "embedding_text": "midnight drive example artist synthpop indie_pop nostalgic dreamy night_drive city_rain neon late_night melodic medium energy"
  },
  "source_ids": {
    "msd_track_id": "TRABC123",
    "mxm_tid": 456789,
    "spotify_id": "7abc..."
  },
  "provenance": {
    "metadata_source": "msd",
    "lyrics_source": "mxm",
    "tag_source": "msd_lastfm_map"
  },
  "playback": {
    "spotify_url": "https://open.spotify.com/track/7abc...",
    "preview_url": null
  }
}
```

## Success Criteria

- A single `TrackDocument` can be produced from current DB rows without schema changes.
- `embeddings.py` can generate embeddings with or without lyrics text.
- `TrackDocument` remains valid when only partial metadata is available.
- API responses can reuse the same shape with optional field omission.
