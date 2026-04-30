from __future__ import annotations

from pathlib import Path
import unittest

from backend.agents.vector_retrieval_agent import (
    retrieve_vector_bundle,
    retrieve_vector_candidates,
    vector_retrieval_node,
)
from backend.agents.vibe_intent import VibeIntent


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DB_DIR = REPO_ROOT / "database" / "chroma_db"


class VectorRetrievalAgentTests(unittest.TestCase):
    def test_retrieve_vector_candidates_uses_project_vector_db(self):
        intent = VibeIntent(semantic_query="late night rainy city")

        candidates = retrieve_vector_candidates(
            intent=intent,
            limit=3,
            persist_directory=str(CHROMA_DB_DIR),
        )

        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["source"], "vector")
        self.assertIn("track_id", candidates[0])
        self.assertIn("vector_rank", candidates[0])
        self.assertIn("vector_distance", candidates[0])
        self.assertIn("metadata", candidates[0])

    def test_retrieve_vector_bundle_returns_debug_fields(self):
        intent = VibeIntent(semantic_query="dreamy synth")

        bundle = retrieve_vector_bundle(
            intent=intent,
            limit=2,
            persist_directory=str(CHROMA_DB_DIR),
        )

        self.assertEqual(bundle["vector_query"], "dreamy synth")
        self.assertGreater(bundle["vector_candidate_count"], 0)
        self.assertLessEqual(bundle["vector_candidate_count"], 2)
        self.assertEqual(
            len(bundle["vector_candidates"]),
            bundle["vector_candidate_count"],
        )

    def test_vector_retrieval_node_accepts_graph_state(self):
        intent = VibeIntent(semantic_query="nostalgic road trip")

        output = vector_retrieval_node(
            {
                "intent": intent,
                "vector_limit": 2,
                "chroma_persist_directory": str(CHROMA_DB_DIR),
            }
        )

        self.assertEqual(output["vector_query"], "nostalgic road trip")
        self.assertGreater(output["vector_candidate_count"], 0)
        self.assertLessEqual(output["vector_candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
