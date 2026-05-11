"""
Manual integration test for PlannerAgent.

Three backends are supported — pick one via --backend:

  --backend hf-api      (default) Calls a HuggingFace hosted model via the
                        Inference API. No local download. Requires HF_TOKEN.
                        Default model: meta-llama/Llama-3.1-8B-Instruct.

  --backend hf          Loads a model locally via transformers. No API key
                        needed for open models. Default: Qwen/Qwen2.5-7B-Instruct.
                        For a faster run use --model Qwen/Qwen2.5-3B-Instruct.

  --backend anthropic   Uses Claude via the Anthropic API.
                        Requires: ANTHROPIC_API_KEY env var.

Examples:
  # HF Inference API — Llama 3.1 8B Instruct (default)
  HF_TOKEN=hf_... conda run -n echoagent-env python tests/test_planner_agent.py

  # HF Inference API — different model
  HF_TOKEN=hf_... conda run -n echoagent-env \\
      python tests/test_planner_agent.py --model meta-llama/Llama-3.1-70B-Instruct

  # Local Qwen 3B (no token needed, ~6GB download)
  conda run -n echoagent-env \\
      python tests/test_planner_agent.py --backend hf --model Qwen/Qwen2.5-3B-Instruct

  # Anthropic Claude
  ANTHROPIC_API_KEY=sk-ant-... conda run -n echoagent-env \\
      python tests/test_planner_agent.py --backend anthropic
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.agents.vibe_intent import VibeIntent
from backend.agents.planner_agent import PlannerAgent


# ---------------------------------------------------------------------------
# LLM client backends
# ---------------------------------------------------------------------------

class AnthropicLLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_input: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        return message.content[0].text


class HuggingFaceInferenceLLMClient:
    """
    Calls a model hosted on HuggingFace via the Inference API.
    No local download — the model runs on HF's servers.
    Requires HF_TOKEN in the environment.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 1024, temperature: float = 0.1):
        from huggingface_hub import InferenceClient

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("ERROR: HF_TOKEN not set.")
            sys.exit(1)

        self._client = InferenceClient(model=model_name, token=hf_token)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, system_prompt: str, user_input: str) -> str:
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-6),
        )
        return response.choices[0].message.content.strip()


class HuggingFaceLLMClient:
    """
    Local HuggingFace text-generation client.

    Uses apply_chat_template so instruction-tuned models receive their native
    prompt format rather than a raw SYSTEM:/USER: string.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 1024, temperature: float = 0.1):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading {model_name} on {self.device} …")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self._model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(self, system_prompt: str, user_input: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0.0),
                temperature=max(self.temperature, 1e-6),
                pad_token_id=self._tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

USER_PROMPT = (
    "I want something for a late-night drive — atmospheric, a little melancholic, "
    "mostly electronic but with some live instruments. Nothing too mainstream, "
    "surprise me with some hidden gems. No hip-hop."
)

EXAMPLE_INTENT = VibeIntent(
    semantic_query="late-night drive atmospheric melancholic electronic with live instruments",
    hard_constraints={
        "genres": ["electronic", "ambient", "indie electronic"],
    },
    soft_preferences={
        "mood": "melancholic",
        "energy": "low",
        "instrumentation": ["synthesizer", "live instruments"],
        "era": "2000s-present",
    },
    exclusions={
        "genres": ["hip-hop", "rap"],
        "popularity": "mainstream",
    },
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PlannerAgent integration test")
    p.add_argument("--backend", choices=["anthropic", "hf", "hf-api"], default="hf-api")
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Model override. "
            "hf-api default: meta-llama/Llama-3.1-8B-Instruct. "
            "hf default: Qwen/Qwen2.5-7B-Instruct. "
            "anthropic default: claude-sonnet-4-6."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p.parse_args()


def build_client(args: argparse.Namespace):
    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        model = args.model or "claude-sonnet-4-6"
        print(f"Backend: Anthropic ({model})")
        return AnthropicLLMClient(model=model, max_tokens=args.max_new_tokens)

    if args.backend == "hf-api":
        model = args.model or "meta-llama/Llama-3.1-8B-Instruct"
        print(f"Backend: HuggingFace Inference API ({model})")
        return HuggingFaceInferenceLLMClient(model_name=model, max_new_tokens=args.max_new_tokens)

    model = args.model or "Qwen/Qwen2.5-7B-Instruct"
    print(f"Backend: HuggingFace local ({model})")
    return HuggingFaceLLMClient(model_name=model, max_new_tokens=args.max_new_tokens)


def main() -> None:
    args = parse_args()
    llm_client = build_client(args)

    print()
    print("=" * 60)
    print("USER PROMPT:")
    print(f"  {USER_PROMPT}\n")
    print("VIBE INTENT:")
    print(json.dumps({
        "semantic_query": EXAMPLE_INTENT.semantic_query,
        "hard_constraints": EXAMPLE_INTENT.hard_constraints,
        "soft_preferences": EXAMPLE_INTENT.soft_preferences,
        "exclusions": EXAMPLE_INTENT.exclusions,
    }, indent=2))
    print("=" * 60)
    print("Running PlannerAgent …\n")

    plan = PlannerAgent(llm_client=llm_client, max_retries=3).plan(
        user_prompt=USER_PROMPT,
        intent=EXAMPLE_INTENT,
    )

    print("PLAYLIST PLAN OUTPUT:")
    print(json.dumps(plan.to_dict(), indent=2))
    print("\nRATIONALE:")
    print(f"  {plan.rationale}")


if __name__ == "__main__":
    main()
