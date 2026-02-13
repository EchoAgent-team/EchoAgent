from __future__ import annotations

import json, re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

@dataclass
class VibeIntent:
    """structured contract between Prompt parsing (language understanding) and Retrieval + ranking (system logic).
    """
    semantic_query : str
    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    soft_preferences: Dict[str, Any] = field(default_factory=dict)
    exclusions: Dict[str, Any] = field(default_factory=dict)
    
    ALLOWED_HARD_KEYS: Set[str] = field(default_factory=set, init=False, repr=False)
    ALLOWED_SOFT_KEYS: Set[str] = field(default_factory=set, init=False, repr=False)
    ALLOWED_EXCL_KEYS: Set[str] = field(default_factory=set, init=False, repr=False)

    # Optional: expected types for validation (fill in later)
    HARD_KEY_TYPES: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    SOFT_KEY_TYPES: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    EXCL_KEY_TYPES: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _clean_key(self, key: Any) -> str:
        if isinstance(key, str):
            return self._clean_text(key).replace("-", "_").replace(" ", "_")
        return str(key)
    
    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return re.sub(r"\s+", " ", value.strip().lower())

    def _canonicalize(self, key: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if key == "energy":
            return {"med": "medium", "mid": "medium", "moderate": "medium", "hi": "high", "low_energy": "low"}.get(value, value)
        if key == "vocal_content":
            return {
                "no vocals": "no_vocals",
                "without vocals": "no_vocals",
                "instrumental": "no_vocals",
                "instrumental_only": "no_vocals",
            }.get(value, value)
        return value

    def _is_empty(self, value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}
    

    def _validate_keys(self, section: Dict[str, Any], allowed: Set[str], name: str) -> None:
        if not allowed:
            return
        unknown = [k for k in section if k not in allowed]
        if unknown:
            raise ValueError(f"{name} has unsupported keys: {unknown}")

    def _validate_value_types(self, section: Dict[str, Any], spec: Dict[str, Any], name: str) -> None:
        if not spec:
            return
        for key, value in section.items():
            expected = spec.get(key)
            if expected is None:
                continue

            valid = True
            if isinstance(expected, type):
                valid = isinstance(value, expected)
            elif isinstance(expected, tuple) and all(isinstance(t, type) for t in expected):
                valid = isinstance(value, expected)
            elif callable(expected):
                valid = bool(expected(value))

            if not valid:
                raise TypeError(
                    f"{name}.{key} expected {expected}, got {type(value).__name__}"
                )

    def _to_value_set(self, value: Any) -> Set[Any]:
        if isinstance(value, (list, tuple, set, frozenset)):
            out: Set[Any] = set()
            for item in value:
                try:
                    out.add(item)
                except TypeError:
                    out.add(json.dumps(item, sort_keys=True, default=str))
            return out
        try:
            return {value}
        except TypeError:
            return {json.dumps(value, sort_keys=True, default=str)}
        
    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        norm_out: Dict[str, Any] = {}

        for raw_key, raw_value in d.items():
            if raw_key is None:
                continue

            key = self._clean_key(raw_key)
            value = raw_value

            # string values
            if isinstance(value, str):
                v = self._clean_text(value)
                v = self._canonicalize(key, v)
                if not self._is_empty(v):
                    norm_out[key] = v

            # list values
            elif isinstance(value, list):
                cleaned = []
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, str):
                        v = self._clean_text(item)
                        v = self._canonicalize(key, v)
                        if v:
                            cleaned.append(v)
                    else:
                        cleaned.append(item)

                # dedupe preserving order
                seen = set()
                unique = []
                for item in cleaned:
                    if item not in seen:
                        seen.add(item)
                        unique.append(item)

                if not self._is_empty(unique):
                    norm_out[key] = unique

            # numeric / bool / other types
            else:
                if not self._is_empty(value):
                    norm_out[key] = value

        return norm_out
        
        
    def normalize(self) -> "VibeIntent":
        """
        Normalize internal representation
        """
        self.semantic_query = self._clean_text(self.semantic_query)        
        self.hard_constraints = self._normalize_dict(self.hard_constraints)
        self.soft_preferences = self._normalize_dict(self.soft_preferences)
        self.exclusions = self._normalize_dict(self.exclusions)
        return self    

    def validate(self) -> None:
        """
        Enforce invariants:
            - semantic_query non-empty
            - keys are allowed
            - values are correct type/shape
            - no overlap between hard/soft
            - exclusions override (no contradictions)
        """
        if not isinstance(self.semantic_query, str) or not self.semantic_query.strip():
            raise ValueError("semantic_query must be a non-empty string")

        if not isinstance(self.hard_constraints, dict):
            raise TypeError("hard_constraints must be a dict")
        if not isinstance(self.soft_preferences, dict):
            raise TypeError("soft_preferences must be a dict")
        if not isinstance(self.exclusions, dict):
            raise TypeError("exclusions must be a dict")

        self._validate_keys(self.hard_constraints, self.ALLOWED_HARD_KEYS, "hard_constraints")
        self._validate_keys(self.soft_preferences, self.ALLOWED_SOFT_KEYS, "soft_preferences")
        self._validate_keys(self.exclusions, self.ALLOWED_EXCL_KEYS, "exclusions")

        self._validate_value_types(self.hard_constraints, self.HARD_KEY_TYPES, "hard_constraints")
        self._validate_value_types(self.soft_preferences, self.SOFT_KEY_TYPES, "soft_preferences")
        self._validate_value_types(self.exclusions, self.EXCL_KEY_TYPES, "exclusions")

        overlap = set(self.hard_constraints).intersection(self.soft_preferences)
        if overlap:
            raise ValueError(f"keys cannot appear in both hard and soft constraints: {sorted(overlap)}")

        for key, ex_value in self.exclusions.items():
            ex_set = self._to_value_set(ex_value)
            hard_val = self.hard_constraints.get(key)
            if hard_val is not None and self._to_value_set(hard_val).intersection(ex_set):
                raise ValueError(f"exclusion contradicts hard constraint for key: {key}")

            soft_val = self.soft_preferences.get(key)
            if soft_val is not None and self._to_value_set(soft_val).intersection(ex_set):
                raise ValueError(f"exclusion contradicts soft preference for key: {key}")

        return None
