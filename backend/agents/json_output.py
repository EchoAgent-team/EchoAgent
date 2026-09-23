"""Shared structured-output diagnostics without storing request data on clients."""
from contextvars import ContextVar
import logging

logger = logging.getLogger(__name__)
_call_context = ContextVar("llm_call_context", default=("unknown", 0))


class JSONOutputError(ValueError):
    """Recoverable generation failure handled by an agent's bounded repair loop."""


def generate_json(client, system_prompt, user_input, agent, attempt):
    token = _call_context.set((agent, attempt))
    try:
        text = client.generate(
            system_prompt=system_prompt, user_input=user_input, json_mode=True,
        )
        if not isinstance(text, str) or not text.strip():
            raise JSONOutputError("LLM returned empty content.")
        return text
    finally:
        _call_context.reset(token)


def check_completion(content, finish_reason, usage, model, json_mode):
    agent, attempt = _call_context.get()
    def count(name):
        return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    logger.info(
        "LLM completion agent=%s attempt=%s model=%s finish_reason=%s "
        "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        agent, attempt, model, finish_reason,
        count("prompt_tokens"), count("completion_tokens"), count("total_tokens"),
    )
    if json_mode:
        if finish_reason in {"length", "max_tokens"}:
            raise JSONOutputError("LLM output truncated at the token limit (finish_reason=length).")
        if finish_reason not in {None, "stop", "eos_token"}:
            raise JSONOutputError(f"LLM output incomplete (finish_reason={finish_reason}).")
        if not isinstance(content, str) or not content.strip():
            raise JSONOutputError("LLM returned empty content.")
    return content


def log_repair(agent, attempt, error):
    # Error type is safe to log; validation messages may contain generated content.
    logger.warning("LLM repair agent=%s attempt=%s error_type=%s", agent, attempt, type(error).__name__)
