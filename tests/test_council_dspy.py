import unittest
from unittest.mock import AsyncMock, patch

from backend import council


class Stage2DspyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage2_uses_dspy_ranker_when_enabled(self):
        stage1_results = [
            {"model": "model-a", "response": "First answer"},
            {"model": "model-b", "response": "Second answer"},
        ]

        fake_ranker = type("FakeRanker", (), {})()
        fake_ranker.enabled = True
        fake_ranker.rank_responses = AsyncMock(
            return_value={
                "ranking": [
                    {"model": "model-b", "score": 9.0},
                    {"model": "model-a", "score": 7.5},
                ],
                "reasoning": "DSPy preferred Response B.",
                "method": "dspy-chain-of-thought",
            }
        )

        with patch.object(council, "DspyCouncilRanker", return_value=fake_ranker), patch.object(
            council,
            "query_models_parallel",
            new=AsyncMock(side_effect=AssertionError("legacy ranking path should not run")),
        ):
            stage2_results, label_to_model = await council.stage2_collect_rankings(
                "Which answer is better?",
                stage1_results,
            )

        self.assertEqual(
            label_to_model,
            {"Response A": "model-a", "Response B": "model-b"},
        )
        self.assertEqual(len(stage2_results), 1)
        self.assertEqual(stage2_results[0]["model"], "DSPy Council Ranker")
        self.assertEqual(stage2_results[0]["parsed_ranking"], ["Response B", "Response A"])
        self.assertIn("FINAL RANKING:", stage2_results[0]["ranking"])

    async def test_stage2_falls_back_to_legacy_rankings_when_dspy_disabled(self):
        stage1_results = [
            {"model": "model-a", "response": "First answer"},
            {"model": "model-b", "response": "Second answer"},
        ]

        fake_ranker = type("FakeRanker", (), {})()
        fake_ranker.enabled = False

        with patch.object(council, "DspyCouncilRanker", return_value=fake_ranker), patch.object(
            council,
            "query_models_parallel",
            new=AsyncMock(
                return_value={
                    "judge-a": {"content": "FINAL RANKING:\n1. Response B\n2. Response A"},
                }
            ),
        ):
            stage2_results, label_to_model = await council.stage2_collect_rankings(
                "Which answer is better?",
                stage1_results,
            )

        self.assertEqual(
            label_to_model,
            {"Response A": "model-a", "Response B": "model-b"},
        )
        self.assertEqual(stage2_results[0]["model"], "judge-a")
        self.assertEqual(stage2_results[0]["parsed_ranking"], ["Response B", "Response A"])


if __name__ == "__main__":
    unittest.main()
