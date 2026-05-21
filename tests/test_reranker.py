from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.reranker import rerank_candidates, reranker_node
from backend.agents.vibe_intent import VibeIntent


class RerankerTests(unittest.TestCase):
    def test_boosts_soft_preference_matches(self):
        intent = VibeIntent(
            semantic_query="night drive",
            soft_preferences={"genres_prefer": ["rock"]},
        ).normalize()

        candidates = [
            {"track_id": "A", "score": 1.0, "metadata": {"seed_genre": "pop"}},
            {"track_id": "B", "score": 1.0, "metadata": {"seed_genre": "rock"}},
        ]

        ranked = rerank_candidates(candidates, intent)

        self.assertEqual(ranked[0]["track_id"], "B")

    def test_penalizes_exclusions(self):
        intent = VibeIntent(
            semantic_query="night drive",
            exclusions={"genres_exclude": ["edm"]},
        ).normalize()

        candidates = [
            {"track_id": "A", "score": 1.0, "metadata": {"seed_genre": "edm"}},
            {"track_id": "B", "score": 1.0, "metadata": {"seed_genre": "rock"}},
        ]

        ranked = rerank_candidates(candidates, intent)

        self.assertEqual(ranked[0]["track_id"], "B")

    def test_uses_vector_distance(self):
        intent = VibeIntent(semantic_query="dreamy synth").normalize()

        candidates = [
            {
                "track_id": "A",
                "score": 1.0,
                "vector_candidate": {"vector_distance": 0.8},
            },
            {
                "track_id": "B",
                "score": 1.0,
                "vector_candidate": {"vector_distance": 0.1},
            },
        ]

        ranked = rerank_candidates(candidates, intent)

        self.assertEqual(ranked[0]["track_id"], "B")

    def test_reranker_node_returns_ranked_candidates(self):
        intent = VibeIntent(
            semantic_query="warm jazz",
            soft_preferences={"genres_prefer": ["jazz"]},
        ).normalize()

        output = reranker_node(
            {
                "intent": intent,
                "fused_candidates": [
                    {"track_id": "A", "score": 1.0, "metadata": {"seed_genre": "rock"}},
                    {"track_id": "B", "score": 1.0, "metadata": {"seed_genre": "jazz"}},
                ],
            }
        )

        self.assertEqual(output["ranked_candidates"][0]["track_id"], "B")


if __name__ == "__main__":
    unittest.main()
