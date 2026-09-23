import unittest
from unittest.mock import Mock

from backend.agents.planner_agent import PlaylistPlan
from backend.agents.playlist_builder import build_playlist_node
from backend.agents.vibe_intent import VibeIntent


class PlaylistPoolTests(unittest.TestCase):
    def run_builder(self, size, candidates, fail=False):
        plan = PlaylistPlan(playlist_size=size)
        agent = Mock()

        def build(**kwargs):
            if fail:
                raise ValueError("invalid model output")
            return [c["track_id"] for c in kwargs["pool"][:kwargs["playlist_plan"].playlist_size]]

        agent.build.side_effect = build
        result = build_playlist_node({
            "ranked_candidates": candidates,
            "playlist_plan": plan,
            "intent": VibeIntent(semantic_query="jazz"),
            "playlist_builder_agent": agent,
            "user_prompt": "jazz",
        })
        self.assertEqual(plan.playlist_size, size)
        return result["playlist"], agent

    def test_requests_above_twenty(self):
        for size in (30, 50):
            with self.subTest(size=size):
                candidates = [{"track_id": str(i)} for i in range(60)]
                playlist, agent = self.run_builder(size, candidates)
                self.assertEqual(playlist, candidates[:size])
                self.assertEqual(len(agent.build.call_args.kwargs["pool"]), size)

    def test_short_pool_uses_available_unique_tracks(self):
        candidates = [{"track_id": "a"}, {"track_id": "a"}, {}, {"track_id": "b"}]
        playlist, agent = self.run_builder(30, candidates)
        self.assertEqual([c["track_id"] for c in playlist], ["a", "b"])
        self.assertEqual(agent.build.call_args.kwargs["playlist_plan"].playlist_size, 2)

    def test_empty_pool_skips_model(self):
        playlist, agent = self.run_builder(30, [])
        self.assertEqual(playlist, [])
        agent.build.assert_not_called()

    def test_fallback_can_return_fifty_tracks(self):
        candidates = [{"track_id": str(i)} for i in range(60)]
        with self.assertLogs("backend.agents.playlist_builder", level="WARNING"):
            playlist, _ = self.run_builder(50, candidates, fail=True)
        self.assertEqual(playlist, candidates[:50])


if __name__ == "__main__":
    unittest.main()
