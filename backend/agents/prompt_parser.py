from __future__ import annotations

import json, re
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .vibe_intent import VibeIntent

class PromptParser:
    """
    Prompt -> VibeIntent

    Responsibilities:
      - extract constraints (hard / exclusions)
      - extract soft preferences (moods/tags)
      - produce semantic_query (embedding text)
    """
    def __init__(self, prompt_schema_path, user_input, llm_client, max_retries=3):
        
        self.prompt_schema_path = prompt_schema_path
        self.user_input = user_input
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.required_fields = {"semantic_query", "hard_constraints", "soft_preferences", "exclusions"}

        
        with open(self.prompt_schema_path, 'r') as file:
            prompt_schema = json.load(file)
            
        
        for my_keys in ["_definitions", "_output_rules", "_template"]:
            if my_keys not in prompt_schema:
                raise ValueError(f"Missing key '{my_keys}' in prompt schema.")
        
        self.template = prompt_schema["_template"]
        for k in self.required_fields:
            if k not in self.template:
                raise ValueError(f"Template missing required field: {k}")
            
        self._definitions = prompt_schema["_definitions"]
        self.output_rules = prompt_schema["_output_rules"]
        
        self._system_prompt = self._build_system_prompt()

    def parse(self):

        last_err: Optional[str] = None
        last_raw: Optional[str] = None
        
        for attempt in range(1, self.max_retries+1):
            if attempt==1:
                system_prompt = self._system_prompt
            else:
                # Repair prompt includes what went wrong + previous raw output
                system_prompt = self._build_repair_prompt(error_message=last_err or "unknown_validation_error", 
                                                        last_raw=last_raw or "",)

            raw = self.llm_client.generate(system_prompt=system_prompt, user_input=self.user_input)
            last_raw = raw
            
            try:
                parsed_output = self._validate_and_parse_output(raw)
                return parsed_output
            except ValueError as ve:
                last_err = str(ve)
                continue
        raise ValueError(f"Failed to parse prompt after {self.max_retries} attempts. Last error: {last_err}")

        

    # -------------------------
    # Extraction helpers
    # -------------------------

    def _extract_json_block(self, text: str) -> str:
        text = (text or "").strip()

        # remove ```json fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        # find first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output.")
        return text[start : end + 1]
    
    
    def _build_system_prompt(self):
        definitions_str = json.dumps(self._definitions, indent=2, ensure_ascii=True)
        rules_str = "\n".join([f"- {r}" for r in self.output_rules])
        template_str = json.dumps(self.template, indent=2, ensure_ascii=True)
        
        return (
            "You are an expert structured intent parser for a music recommendation system.\n"
            "Your task is to convert user's natural-language prompt into a VibeIntent JSON object.\n\n"
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1) Output JSON ONLY (no markdown, no prose).\n"
            "2) Output must match the template keys EXACTLY.\n"
            "3) Include ALL keys; DO NOT add extra keys.\n"
            "4) `semantic_query` must be a non-empty string.\n"
            "5) `hard_constraints`, `soft_preferences`, and `exclusions` must be JSON objects.\n"
            "6) If a section has no detected values, use {} for that section.\n"
            "7) Include low-level normalized tags in `semantic_query` when detectable "
            "(e.g., mood:nostalgic scene:night_city instrument:synth energy:medium).\n"
            "8) Do NOT encode exclusions as positive tags in `semantic_query`.\n"
            "9) Normalize values and prefer snake_case when appropriate.\n"
            "10) Do NOT output any keys starting with '_' (schema metadata must never appear).\n\n"
            f"OUTPUT RULES:\n{rules_str}\n\n"
            "FIELD DEFINITIONS:\n"
            f"{definitions_str}\n\n"
            "RETURN ONLY THIS JSON OBJECT (filled in):\n"
            f"{template_str}\n"
        )
        
    def _build_repair_prompt(self, error_message, last_raw):
        template_str = json.dumps(self.template, indent=2, ensure_ascii=True)
        return ("You are a structured intent parser for a music recommendation system.\n"
                "Your previous output was INVALID due to the following error:\n"
                f"{error_message}\n\n"
                "PREVIOUS OUTPUT (for reference):\n"
                f"{last_raw}\n\n"
                "Fix the output.\n"
                "STRICT REQUIREMENTS:\n"
                "- Output JSON ONLY (no markdown, no prose).\n"
                "- Return ONLY the corrected JSON object. No other text.\n"
                "- Output must match the template keys EXACTLY.\n"
                "- Include ALL keys; DO NOT add extra keys.\n"
                "- `semantic_query` must be a non-empty string.\n"
                "- `hard_constraints`, `soft_preferences`, and `exclusions` must be JSON objects.\n"
                "- Use {} for empty sections.\n"
                "- Include low-level normalized tags in `semantic_query` when detectable.\n"
                "- Do NOT encode exclusions as positive tags in `semantic_query`.\n"
                "- Normalize values and prefer snake_case when appropriate.\n\n"
                "RETURN ONLY THIS JSON OBJECT (filled in):\n"
                f"{template_str}\n"
        )
        
    def _validate_and_parse_output(self, generated_text):

        json_str = self._extract_json_block(generated_text)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object (dict).")
        
        # required keys present
        missing = [k for k in self.required_fields if k not in data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        # type checks at the JSON level (before VibeIntent.validate())
        if not isinstance(data.get("semantic_query"), str) or not data["semantic_query"].strip():
            raise ValueError("semantic_query must be a non-empty string.")
        for k in ("hard_constraints", "soft_preferences", "exclusions"):
            if not isinstance(data.get(k), dict):
                raise ValueError(f"{k} must be a JSON object/dict.")
        
        intent = VibeIntent(
        semantic_query=data.get("semantic_query", ""),
        hard_constraints=data.get("hard_constraints", {}),
        soft_preferences=data.get("soft_preferences", {}),
        exclusions=data.get("exclusions", {}),
        )
        
        intent.normalize()
        intent.validate()
        return intent

class LLMClient:
    def __init__(self, model_name, 
                device, api_key, 
                endpoint, temperature, 
                max_new_tokens):
        """
        Store model name, API keys, endpoints, temperature, etc.
        No network calls here.
        """
        self.model_name = model_name
        self.device = device
        self.api_key = api_key
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._tokens = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                        torch_dtype=torch.float16 if self.device == "cuda" else None,
                                                        device_map="auto" if self.device == "cuda" else None,)
        

    def generate(self, system_prompt, user_input):
        """
        Args:
            - system_prompt: the instruction built from schema
            - user_prompt: the user's natural language input

        Returns:
            - raw text output from the LLM (string)
        """
        
        text = ("SYSTEM:\n" + system_prompt.strip() + "\n\n"
                "USER:\n" + user_input.strip() + "\n\n"
                "ASSISTANT:\n")
        
        inputs = self._tokens(text, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            output = self._model.generate(
                                        **inputs,
                                        max_new_tokens=self.max_new_tokens,
                                        do_sample=(self.temperature > 0.0),
                                        temperature=max(self.temperature, 1e-6),
                                        )
            
        decoded_text = self._tokens.decode(output[0], skip_special_tokens=True)
        marker = "ASSISTANT:"
        return decoded_text.split(marker, 1)[-1].strip() if marker in decoded_text else decoded_text.strip()
        


# Convenience function if you want a functional API
def parse_prompt(prompt: str, prompt_schema_path: str, llm_client: "LLMClient") -> VibeIntent:
    return PromptParser(prompt_schema_path=prompt_schema_path,
                        user_input=prompt,
                        llm_client=llm_client).parse()
