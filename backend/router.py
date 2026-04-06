"""
LLM Council — Router (Phase 3 of the BAS-More + Ruflo integration plan).

Epsilon-greedy multi-armed bandit that learns which models work best for
which task types. Persists state to JSON so it survives restarts.

This is a staged-in placeholder for Ruflo's Q-learning + MoE router. It
exposes the same API surface (recommend / feedback) so we can swap the
implementation without touching the Council orchestrator. When Ruflo's
@claude-flow/routing package is extractable, we replace just this file.

Storage: data/router_state.json
Schema: {
    "<task_kind>": {
        "<model_id>": { "n": int, "reward_sum": float, "last_used": "iso" }
    }
}
"""

import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(os.environ.get("COUNCIL_ROUTER_DATA", "data"))
STATE_FILE = DATA_DIR / "router_state.json"
DECISIONS_FILE = DATA_DIR / "router_decisions.json"

EPSILON = 0.15  # 15% exploration
MIN_PULLS_BEFORE_UCB = 3  # Use random for first N pulls of each arm


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Router:
    def __init__(self, all_models: List[str]):
        self.all_models = list(all_models)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.decisions = self._load_decisions()

    # ---------- persistence ----------

    def _load_state(self) -> Dict[str, Dict[str, dict]]:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _load_decisions(self) -> Dict[str, dict]:
        if DECISIONS_FILE.exists():
            try:
                return json.loads(DECISIONS_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_decisions(self) -> None:
        DECISIONS_FILE.write_text(json.dumps(self.decisions, indent=2))

    # ---------- bandit ----------

    def _arm(self, task_kind: str, model: str) -> dict:
        kind_state = self.state.setdefault(task_kind, {})
        return kind_state.setdefault(model, {"n": 0, "reward_sum": 0.0, "last_used": None})

    def _mean(self, arm: dict) -> float:
        return arm["reward_sum"] / arm["n"] if arm["n"] > 0 else 0.0

    def _ucb_score(self, arm: dict, total_pulls_kind: int) -> float:
        """UCB1 score for tie-breaking. Higher = better."""
        if arm["n"] == 0:
            return float("inf")
        exploration = math.sqrt(2 * math.log(max(total_pulls_kind, 1)) / arm["n"])
        return self._mean(arm) + exploration

    def recommend(
        self,
        task_kind: str,
        candidate_models: Optional[List[str]] = None,
        n: int = 3,
    ) -> Tuple[List[str], str]:
        """Pick `n` model ids for this task kind. Returns (models, decision_id, explanation)."""
        candidates = candidate_models or self.all_models
        if not candidates:
            return [], "", "no candidates"

        kind_state = self.state.get(task_kind, {})
        total_pulls_kind = sum(a.get("n", 0) for a in kind_state.values())

        # Score every candidate
        scored = []
        for m in candidates:
            arm = self._arm(task_kind, m)
            if arm["n"] < MIN_PULLS_BEFORE_UCB:
                # Force early exploration of unseen arms
                scored.append((float("inf") - random.random(), m, arm))
            else:
                scored.append((self._ucb_score(arm, total_pulls_kind), m, arm))

        # Epsilon-greedy: 15% pure random
        if random.random() < EPSILON:
            random.shuffle(candidates)
            picks = candidates[:n]
            mode = "explore"
        else:
            scored.sort(key=lambda x: x[0], reverse=True)
            picks = [m for _, m, _ in scored[:n]]
            mode = "exploit"

        # Record decision so feedback can update the right arms
        decision_id = str(uuid.uuid4())
        self.decisions[decision_id] = {
            "task_kind": task_kind,
            "models": picks,
            "mode": mode,
            "ts": _now_iso(),
        }
        self._save_decisions()

        # Compose explanation
        details = []
        for m in picks:
            arm = kind_state.get(m, {"n": 0, "reward_sum": 0.0})
            mean = self._mean(arm)
            details.append(f"{m} (n={arm['n']}, mean={mean:.2f})")
        explanation = f"mode={mode} task={task_kind} picks=[{', '.join(details)}]"

        return picks, decision_id, explanation

    def feedback(self, decision_id: str, scores: Dict[str, float]) -> dict:
        """Record per-model reward scores for a previous decision.
        Scores must be in [0, 1]. Returns updated arm stats."""
        decision = self.decisions.get(decision_id)
        if not decision:
            return {"error": "unknown decision_id"}

        task_kind = decision["task_kind"]
        updated = {}
        for model, score in scores.items():
            if model not in decision["models"]:
                continue
            score = max(0.0, min(1.0, float(score)))
            arm = self._arm(task_kind, model)
            arm["n"] += 1
            arm["reward_sum"] += score
            arm["last_used"] = _now_iso()
            updated[model] = {"n": arm["n"], "mean": self._mean(arm)}

        self._save_state()
        return {"task_kind": task_kind, "updated": updated}

    def stats(self, task_kind: Optional[str] = None) -> dict:
        """Return current arm state for inspection / dashboard."""
        if task_kind:
            kinds = {task_kind: self.state.get(task_kind, {})}
        else:
            kinds = self.state
        out = {}
        for k, arms in kinds.items():
            out[k] = sorted(
                [
                    {
                        "model": m,
                        "n": a["n"],
                        "mean": self._mean(a),
                        "last_used": a.get("last_used"),
                    }
                    for m, a in arms.items()
                ],
                key=lambda x: x["mean"],
                reverse=True,
            )
        return out
