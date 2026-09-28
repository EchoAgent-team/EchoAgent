from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("FRONTEND_BACKEND_URL", "http://localhost:8000").rstrip("/")
EXAMPLE_PROMPTS = [
    (":material/nights_stay:", "late-night rainy city drive"),
    (":material/park:", "warm nostalgic indie autumn walk"),
    (":material/bolt:", "high-energy gym rhythm, no sad songs"),
]

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def escape_md(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def render_background() -> None:
    theme_type = getattr(st.context.theme, "type", "light")
    if theme_type == "dark":
        glow = (
            "radial-gradient(640px circle at 10% -8%, rgba(167,139,250,0.22), transparent 62%),"
            "radial-gradient(560px circle at 108% 6%, rgba(34,211,238,0.14), transparent 60%),"
            "radial-gradient(680px circle at 50% 118%, rgba(244,114,182,0.10), transparent 60%)"
        )
    else:
        glow = (
            "radial-gradient(640px circle at 10% -8%, rgba(124,58,237,0.10), transparent 62%),"
            "radial-gradient(560px circle at 108% 6%, rgba(8,145,178,0.08), transparent 60%),"
            "radial-gradient(680px circle at 50% 118%, rgba(219,39,119,0.06), transparent 60%)"
        )

    st.html(
        f"""
        <style>
        .stApp {{
            background-image: {glow};
            background-attachment: fixed;
        }}
        </style>
        """
    )


def page_setup() -> None:
    st.set_page_config(
        page_title="EchoAgent",
        page_icon=":material/queue_music:",
        layout="centered",
    )
    render_background()


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


def match_badge(score: float | int | None) -> tuple[str, str]:
    if score is None:
        return "Match", "gray"
    if score >= 0.78:
        return "Strong match", "green"
    if score >= 0.55:
        return "Good match", "blue"
    return "Wildcard", "gray"


def format_track_name(track: dict[str, Any]) -> tuple[str, str]:
    title = track.get("title")
    artist = track.get("artist_name")
    track_id = track.get("track_id") or "unknown id"

    if not title and not artist:
        return "Unknown track", track_id

    return title or "Unknown title", artist or "Unknown artist"


def compact(values: list[Any]) -> str:
    return " · ".join(str(value) for value in values if value not in (None, "", []))


def render_track(track: dict[str, Any], index: int) -> None:
    title, artist = format_track_name(track)
    score = float(track.get("score") or 0.0)
    album = track.get("album_title")
    year = track.get("year")
    genre = track.get("genre")
    tags = track.get("tags") or []
    tag_text = ", ".join(tags[:4])
    meta = compact([album, year, genre, tag_text])
    label, color = match_badge(score)

    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.markdown(f"**{index}. {escape_md(str(title))}**")
            st.badge(label, color=color)
        st.caption(escape_md(str(artist)))
        st.caption(meta if meta else "No album, year, or genre metadata")
        st.progress(min(max(score, 0.0), 1.0), text=f"Match strength · {score:.3f}")


def render_debug(debug: dict[str, Any] | None) -> None:
    if not debug:
        return

    with st.expander("Debug", icon=":material/bug_report:", expanded=False):
        counts = {
            "relational candidates": debug.get("relational_candidate_count"),
            "vector candidates": debug.get("vector_candidate_count"),
            "fused candidates": debug.get("fused_candidate_count"),
            "retry count": debug.get("retry_count"),
            "builder fallback used": debug.get("builder_fallback_used"),
            "builder fallback reason": debug.get("builder_fallback_reason"),
        }
        st.markdown("**Retrieval and build status**")
        st.json(counts)
        st.markdown("**Parsed intent**")
        st.json(debug.get("intent") or {})
        st.markdown("**Planner output**")
        st.json(debug.get("plan") or {})
        st.markdown("**Critic report**")
        st.json(debug.get("critic_report") or {})


def main() -> None:
    page_setup()

    st.title("EchoAgent", icon=":material/graphic_eq:")
    st.markdown("Type a vibe. Get a playlist.")

    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None

    st.space("small")

    with st.container(border=True):
        cols = st.columns(len(EXAMPLE_PROMPTS))
        for col, (icon, example) in zip(cols, EXAMPLE_PROMPTS):
            with col:
                if st.button(example, icon=icon, width="stretch"):
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

        generate = st.button(
            "Generate playlist",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

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
        st.error(st.session_state.error, icon=":material/error:")

    result = st.session_state.result
    if result:
        playlist = result.get("playlist") or []
        st.subheader("Playlist", icon=":material/queue_music:")
        st.caption(f"{len(playlist)} tracks for: {result.get('prompt', prompt)}")

        for index, track in enumerate(playlist, start=1):
            render_track(track, index)

        render_debug(result.get("debug"))


if __name__ == "__main__":
    main()
