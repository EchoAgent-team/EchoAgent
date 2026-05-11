from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.candidate_fuser import candidate_fusion_node, fuse_candidates


class CandidateFuserTests(unittest.TestCase):
    def test_merges_candidates_with_same_track_id(self):
        relational_candidates = [
            {
                "track_id": "TR001",
                "title": "Night Drive",
                "artist_name": "The Signals",
            }
        ]
        vector_candidates = [
            {
                "track_id": "TR001",
                "vector_rank": 1,
                "vector_distance": 0.12,
                "metadata": {"track_id": "TR001"},
            }
        ]

        fused = fuse_candidates(relational_candidates, vector_candidates)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["track_id"], "TR001")
        self.assertEqual(fused[0]["sources"], ["relational", "vector"])
        self.assertEqual(fused[0]["relational_candidate"]["title"], "Night Drive")
        self.assertEqual(fused[0]["vector_candidate"]["vector_rank"], 1)
        self.assertEqual(fused[0]["score"], 2.0)

    def test_keeps_relational_only_and_vector_only_candidates(self):
        relational_candidates = [{"track_id": "TR_REL", "title": "Structured Match"}]
        vector_candidates = [
            {
                "track_id": "TR_VEC",
                "vector_rank": 1,
                "metadata": {"title": "Semantic Match"},
            }
        ]

        fused = fuse_candidates(relational_candidates, vector_candidates)
        by_track_id = {track["track_id"]: track for track in fused}

        self.assertEqual(set(by_track_id), {"TR_REL", "TR_VEC"})
        self.assertEqual(by_track_id["TR_REL"]["sources"], ["relational"])
        self.assertEqual(by_track_id["TR_VEC"]["sources"], ["vector"])

    def test_uses_vector_rank_for_score(self):
        fused = fuse_candidates(
            relational_candidates=[],
            vector_candidates=[
                {"track_id": "TR_FIRST", "vector_rank": 1, "metadata": {}},
                {"track_id": "TR_SECOND", "vector_rank": 2, "metadata": {}},
            ],
        )

        self.assertEqual(fused[0]["track_id"], "TR_FIRST")
        self.assertEqual(fused[0]["score"], 1.0)
        self.assertEqual(fused[1]["score"], 0.5)

    def test_gets_track_id_from_vector_metadata(self):
        fused = fuse_candidates(
            relational_candidates=[],
            vector_candidates=[
                {
                    "vector_rank": 1,
                    "metadata": {"track_id": "TR_META"},
                }
            ],
        )

        self.assertEqual(fused[0]["track_id"], "TR_META")

    def test_candidate_fusion_node_uses_playlist_plan_weights(self):
        output = candidate_fusion_node(
            {
                "playlist_plan": {
                    "relational_weight": 2.0,
                    "semantic_weight": 3.0,
                },
                "relational_candidates": [{"track_id": "TR001"}],
                "vector_candidates": [{"track_id": "TR001", "vector_rank": 1}],
            }
        )

        self.assertEqual(output["fused_candidate_count"], 1)
        self.assertEqual(output["fused_candidates"][0]["track_id"], "TR001")
        self.assertEqual(output["fused_candidates"][0]["score"], 5.0)


if __name__ == "__main__":
    unittest.main()
