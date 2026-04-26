from __future__ import annotations

import json, re
from typing import Optional
from dataclasses import dataclass
from .vibe_intent import VibeIntent

class IntentAdapter:
    def __init__(self, prompt_parser: PromptParser):
        self.prompt_parser = prompt_parser
    
    def adapt()