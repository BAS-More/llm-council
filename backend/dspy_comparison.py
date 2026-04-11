"""
DSPy MultiChainComparison integration for llm-council.
Replaces manual ranking prompts with DSPy's self-optimizing comparison module.

Usage:
    from backend.dspy_comparison import DspyCouncilRanker
    ranker = DspyCouncilRanker()
    ranking = await ranker.rank_responses(query, responses)

Requires: pip install dspy-ai
"""

import os
import json
from typing import List, Dict, Any, Optional

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class DspyCouncilRanker:
    """
    Uses DSPy MultiChainComparison to rank council responses.
    Falls back to standard ranking if DSPy is not available.
    """

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None):
        if not DSPY_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True

        # Configure DSPy LM
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.enabled = False
            return

        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            self.lm = dspy.LM(
                model=model or "anthropic/claude-sonnet-4-5-20250929",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
        else:
            self.lm = dspy.LM(
                model=model or "openai/gpt-4o-mini",
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )

        dspy.configure(lm=self.lm)

        # Define the comparison signature
        self.compare = dspy.ChainOfThought(
            "question, responses -> ranking, reasoning, scores"
        )

    async def rank_responses(
        self,
        query: str,
        responses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Rank council responses using DSPy MultiChainComparison.

        Args:
            query: The original user question
            responses: List of dicts with 'model' and 'response' keys

        Returns:
            Dict with 'ranking' (ordered list), 'reasoning', and 'scores'
        """
        if not self.enabled or not responses:
            return self._fallback_ranking(responses)

        # Format responses for comparison
        labels = [chr(65 + i) for i in range(len(responses))]
        responses_text = "\n\n".join([
            f"Response {label} ({resp['model']}):\n{resp['response'][:2000]}"
            for label, resp in zip(labels, responses)
        ])

        try:
            result = self.compare(
                question=query,
                responses=responses_text,
            )

            # Parse ranking from DSPy output
            ranking_text = result.ranking or ""
            reasoning = result.reasoning or "No reasoning provided"
            scores_text = result.scores or ""

            # Parse scores
            scores = {}
            for part in scores_text.split(","):
                part = part.strip()
                if ":" in part:
                    label, score = part.split(":", 1)
                    try:
                        scores[label.strip()] = float(score.strip())
                    except ValueError:
                        pass

            # Build ordered ranking
            ordered = []
            for line in ranking_text.split("\n"):
                line = line.strip()
                for i, label in enumerate(labels):
                    if f"Response {label}" in line and i < len(responses):
                        ordered.append({
                            "rank": len(ordered) + 1,
                            "model": responses[i]["model"],
                            "label": f"Response {label}",
                            "score": scores.get(f"Response {label}", 5.0),
                        })
                        break

            # Add any missing responses at the end
            ranked_models = {r["model"] for r in ordered}
            for resp in responses:
                if resp["model"] not in ranked_models:
                    ordered.append({
                        "rank": len(ordered) + 1,
                        "model": resp["model"],
                        "label": "Unranked",
                        "score": 0.0,
                    })

            return {
                "ranking": ordered,
                "reasoning": reasoning,
                "scores": scores,
                "method": "dspy-chain-of-thought",
            }

        except Exception as e:
            print(f"DSPy ranking failed: {e}, falling back to standard")
            return self._fallback_ranking(responses)

    def _fallback_ranking(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback: return responses in original order."""
        return {
            "ranking": [
                {"rank": i + 1, "model": r.get("model", "unknown"), "score": 5.0}
                for i, r in enumerate(responses)
            ],
            "reasoning": "Fallback ranking (DSPy unavailable)",
            "scores": {},
            "method": "fallback",
        }
