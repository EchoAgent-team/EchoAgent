from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("FRONTEND_BACKEND_URL", "http://localhost:8000").rstrip("/")
EXAMPLE_PROMPTS = [
    "late-night rainy city drive",
    "warm nostalgic indie autumn walk",
    "high-energy gym rhythm, no sad songs",
]


def page_setup() -> None:
    st.set_page_config(
        page_title="EchoAgent",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .stApp {
            color-scheme: light dark;
        }
        .block-container {
            max-width: 920px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        .echo-subtitle {
            margin-top: -0.75rem;
            color: rgba(128, 128, 128, 0.95);
            font-size: 1.05rem;
        }
        .track-card {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.65rem;
            background: rgba(128, 128, 128, 0.06);
        }
        .track-title-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: baseline;
        }
        .track-title {
            font-weight: 700;
            font-size: 1rem;
            line-height: 1.35;
        }
        .track-artist {
            color: rgba(128, 128, 128, 0.98);
            margin-top: 0.15rem;
        }
        .track-meta {
            color: rgba(128, 128, 128, 0.9);
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }
        .match-badge {
            white-space: nowrap;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            border: 1px solid rgba(128, 128, 128, 0.28);
            font-size: 0.78rem;
            font-weight: 650;
        }
        .muted {
            color: rgba(128, 128, 128, 0.88);
            font-size: 0.82rem;
        }
        div[data-testid="stHorizontalBlock"] button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def recommend(prompt: str) -> dict[str, Any]:
    response = requests.post(
        f"{BACKEND_URL}/recommend",
        json={"prompt": prompt},
        timeout=120,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text or "The backend returned an empty response."}

    if response.status_code >= 400:
        raise RuntimeError(format_error(response.status_code, payload))

    return payload


def format_error(status_code: int, payload: dict[str, Any]) -> str:
    detail = payload.get("detail", payload)

    if isinstance(detail, dict) and detail.get("code") == "playlist_rejected":
        reason = detail.get("reason") or "The playlist did not pass the quality check."
        retry_count = detail.get("retry_count")
        attempts = f" after {retry_count} replans" if isinstance(retry_count, int) and retry_count else ""
        return (
            f"The playlist did not pass EchoAgent's quality check{attempts}. "
            f"{reason} Try revising the prompt with clearer mood, genre, energy, or exclusions."
        )

    message = detail if isinstance(detail, str) else json.dumps(detail, indent=2)
    lowered = message.lower()

    if status_code == 502 and ("429" in lowered or "rate" in lowered or "quota" in lowered):
        return "The model provider is rate-limited. Try again in a moment."

    if status_code == 404:
        return "No tracks matched this request. Try broadening the prompt."

    if status_code == 422:
        return f"EchoAgent could not understand that request. {message}"

    if status_code == 503:
        return f"A required backend service is unavailable. {message}"

    return f"EchoAgent could not generate a playlist. {message}"


def match_badge(score: float | int | None) -> str:
    if score is None:
        return "Match"
    if score >= 0.78:
        return "Strong match"
    if score >= 0.55:
        return "Good match"
    return "Wildcard"


def format_track_name(track: dict[str, Any]) -> tuple[str, str]:
    title = track.get("title")
    artist = track.get("artist_name")
    track_id = track.get("track_id") or "unknown id"

    if not title and not artist:
        return "Unknown track", track_id

    return title or "Unknown title", artist or "Unknown artist"


def compact(values: list[Any]) -> str:
    return " | ".join(str(value) for value in values if value not in (None, "", []))


def render_track(track: dict[str, Any], index: int) -> None:
    title, artist = format_track_name(track)
    score = float(track.get("score") or 0.0)
    album = track.get("album_title")
    year = track.get("year")
    genre = track.get("genre")
    tags = track.get("tags") or []
    tag_text = ", ".join(tags[:4])
    meta = compact([album, year, genre, tag_text])
    raw_score = f"{score:.3f}"
    safe_title = escape(str(title))
    safe_artist = escape(str(artist))
    safe_meta = escape(meta if meta else "No album, year, or genre metadata")
    safe_badge = escape(match_badge(score))

    st.markdown(
        f"""
        <div class="track-card">
          <div class="track-title-row">
            <div>
              <div class="track-title">{index}. {safe_title}</div>
              <div class="track-artist">{safe_artist}</div>
            </div>
            <div class="match-badge">{safe_badge}</div>
          </div>
          <div class="track-meta">{safe_meta}</div>
          <div class="muted">score {raw_score}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_debug(debug: dict[str, Any] | None) -> None:
    if not debug:
        return

    with st.expander("Debug", expanded=False):
        counts = {
            "relational candidates": debug.get("relational_candidate_count"),
            "vector candidates": debug.get("vector_candidate_count"),
            "fused candidates": debug.get("fused_candidate_count"),
            "retry count": debug.get("retry_count"),
            "builder fallback used": debug.get("builder_fallback_used"),
            "builder fallback reason": debug.get("builder_fallback_reason"),
        }
        st.write("Retrieval and build status")
        st.json(counts)
        st.write("Parsed intent")
        st.json(debug.get("intent") or {})
        st.write("Planner output")
        st.json(debug.get("plan") or {})
        st.write("Critic report")
        st.json(debug.get("critic_report") or {})


def main() -> None:
    page_setup()

    st.title("EchoAgent")
    st.markdown('<p class="echo-subtitle">Describe the mix you have in mind.</p>', unsafe_allow_html=True)

    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None

    cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, example in zip(cols, EXAMPLE_PROMPTS):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state.prompt = example
                st.session_state.result = None
                st.session_state.error = None
                st.rerun()

    prompt = st.text_area(
        "Prompt",
        key="prompt",
        placeholder="late-night rainy city drive, introspective but not depressing",
        height=110,
        label_visibility="collapsed",
    )

    generate = st.button("Generate playlist", type="primary", use_container_width=True)

    if generate:
        clean_prompt = prompt.strip()
        st.session_state.error = None
        st.session_state.result = None

        if not clean_prompt:
            st.session_state.error = "Describe the playlist you want first."
        else:
            with st.spinner("Building a playlist..."):
                try:
                    st.session_state.result = recommend(clean_prompt)
                except requests.exceptions.ConnectionError:
                    st.session_state.error = (
                        f"Could not reach the backend at {BACKEND_URL}. "
                        "Start FastAPI, then try again."
                    )
                except requests.exceptions.Timeout:
                    st.session_state.error = "The request took too long. Try a simpler prompt or retry in a moment."
                except RuntimeError as exc:
                    st.session_state.error = str(exc)

    if st.session_state.error:
        st.error(st.session_state.error)

    result = st.session_state.result
    if result:
        playlist = result.get("playlist") or []
        st.subheader("Playlist")
        st.caption(f"{len(playlist)} tracks for: {result.get('prompt', prompt)}")

        for index, track in enumerate(playlist, start=1):
            render_track(track, index)

        render_debug(result.get("debug"))


if __name__ == "__main__":
    main()
