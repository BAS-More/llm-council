"""Role-specialized 3-stage Council — Sprint 35 G24.

Parallel implementation to council.py that injects per-role system prompts (Stage 1),
per-role ranking criteria (Stage 2), and role-aware Chairman synthesis (Stage 3).
Reuses parse_ranking_from_text and calculate_aggregate_rankings from council.py.

This module is ADDITIVE — council.py is untouched. main.py routes the /decide endpoint
here by default; legacy callers passing explicit `models` still get council.py's flat path.
"""

import asyncio
from typing import List, Dict, Any, Tuple

from .providers import query_model
from .roles import ROLE_DEFINITIONS, CHAIRMAN_MODEL, CHAIRMAN_SYSTEM
from .council import parse_ranking_from_text, calculate_aggregate_rankings

# Bulletproofing (Sprint 45): a strong, honesty-disciplined model used (a) as the independent
# FABRICATION AUDITOR over Stage-1 answers, and (b) as the Chairman fallback if Opus is unavailable.
# Chosen because in red-team testing claude-sonnet-4-6 reliably tagged [UNVERIFIED] instead of inventing
# citations, whereas weaker role models fabricated sources even with a no-fabrication prompt.
AUDITOR_MODEL = "anthropic/claude-sonnet-4-6"
CHAIRMAN_FALLBACK_MODEL = "anthropic/claude-sonnet-4-6"

_AUDIT_SYSTEM = (
    "You are the FABRICATION AUDITOR for an LLM Council. Your ONLY job is to catch invented evidence "
    "before it reaches the Chairman, and you are deliberately strict.\n"
    "For EACH role's answer, scan every citation, source, paper title, author name, year, URL, statistic, "
    "quoted figure, or named benchmark. FLAG any that is likely FABRICATED or UNCHECKABLE -- not a real, "
    "well-known public reference you are confident actually exists. If you cannot personally confirm a "
    "cited paper/author/number is real, FLAG it (false positives are fine; missed fabrications are not). "
    "Do NOT flag honest [UNVERIFIED] tags, hedged language, or claims made WITHOUT a citation -- only flag "
    "specifics presented as real/sourced that are likely invented.\n"
    "Output ONLY this format:\n"
    "FABRICATION FLAGS:\n"
    "- [ROLE] \"<suspect citation or stat>\" -- short reason it is likely fabricated\n"
    "If there are genuinely none, output exactly: FABRICATION FLAGS: none"
)


async def audit_fabrication(user_query: str, stage1_results: List[Dict[str, Any]]) -> str:
    """Independent, model-agnostic fabrication gate: a strong honest model flags invented/uncheckable
    citations in the Stage-1 answers so the Chairman can discard them no matter which role's model
    produced them. Prompt rules alone do NOT stop weaker role models from inventing sources (red-team
    confirmed), so this auditor is the layer that actually guarantees fabrication carries no weight."""
    answers = "\n\n".join(
        f"[{r.get('role', '?').upper()}]:\n{r['response']}" for r in stage1_results
    )
    messages = [
        {"role": "system", "content": _AUDIT_SYSTEM},
        {"role": "user", "content": f"Question: {user_query}\n\nStage-1 answers to audit:\n\n{answers}"},
    ]
    response = await query_model(AUDITOR_MODEL, messages, timeout=120)
    if response is None:
        return ("FABRICATION AUDIT UNAVAILABLE (auditor did not respond) -- Chairman: independently "
                "scrutinize every [VERIFIED] claim and discard any you cannot confirm is real.")
    return response.get("content", "").strip() or "FABRICATION FLAGS: none"


async def stage1_roles(
    user_query: str,
    role_betas: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    """Stage 1: each role answers with its own system prompt, in parallel."""
    role_betas = role_betas or {}

    async def query_role(role_name: str, role_def: Dict[str, Any]):
        beta = role_betas.get(role_name, role_def["beta_default"])
        system = role_def["stage1_system"].format(beta=beta)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]
        response = await query_model(role_def["model"], messages)
        return role_name, role_def["model"], response

    tasks = [query_role(rn, rd) for rn, rd in ROLE_DEFINITIONS.items()]
    results = await asyncio.gather(*tasks)

    stage1 = []
    for role_name, model, response in results:
        if response is not None:
            stage1.append({
                "role": role_name,
                "model": model,
                "response": response.get("content", ""),
            })
    return stage1


async def stage2_roles(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    role_betas: Dict[str, float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Stage 2: each role ranks the anonymized Stage 1 answers by its own criterion."""
    labels = [chr(65 + i) for i in range(len(stage1_results))]
    label_to_model = {
        f"Response {label}": result["model"]
        for label, result in zip(labels, stage1_results)
    }
    # Truncate each answer for the ranking payload — Groq free-tier enforces a
    # request-size cap (observed 413 Payload Too Large on gpt-oss-120b / qwen3-32b
    # when 7 full role answers were concatenated). Rankers only need the thesis,
    # not the full essay; Chairman (Stage 3) still receives the untruncated answers.
    RANK_TRUNC = 1200
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response'][:RANK_TRUNC]}"
        + ("\n...[truncated for ranking]" if len(result['response']) > RANK_TRUNC else "")
        for label, result in zip(labels, stage1_results)
    ])

    async def rank_role(role_name: str, role_def: Dict[str, Any]):
        criterion = role_def["stage2_criterion"]
        ranking_prompt = f"""You are the {role_name.upper()} member of an LLM Council, evaluating responses to:

Question: {user_query}

Here are the responses (anonymized):

{responses_text}

Rank these responses by YOUR specific criterion: {criterion}.

First, evaluate each response against that criterion in 1-2 sentences. Then provide your final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then a numbered list from best to worst, each line "N. Response X" and nothing else.

Example:
FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""
        messages = [{"role": "user", "content": ranking_prompt}]
        response = await query_model(role_def["model"], messages)
        return role_name, role_def["model"], response

    tasks = [rank_role(rn, rd) for rn, rd in ROLE_DEFINITIONS.items()]
    results = await asyncio.gather(*tasks)

    stage2 = []
    for role_name, model, response in results:
        if response is not None:
            full_text = response.get("content", "")
            stage2.append({
                "role": role_name,
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parse_ranking_from_text(full_text),
            })
    return stage2, label_to_model


async def stage3_roles(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    audit_report: str = "FABRICATION FLAGS: none",
) -> Dict[str, Any]:
    """Stage 3: role-aware Chairman synthesis with role-tagged inputs + the fabrication-audit report.
    Resilient: retries Opus once on a transient failure, then falls back to a strong model so a
    momentary chairman outage never returns a bare error."""
    stage1_text = "\n\n".join([
        f"[{r.get('role', '?').upper()}] ({r['model']}):\n{r['response']}"
        for r in stage1_results
    ])
    stage2_text = "\n\n".join([
        f"[{r.get('role', '?').upper()}] ranked:\n{r['ranking']}"
        for r in stage2_results
    ])

    chairman_prompt = (
        f"{CHAIRMAN_SYSTEM}\n\n"
        f"Original Question: {user_query}\n\n"
        f"FABRICATION AUDIT (an independent auditor flagged the following Stage-1 claims as likely "
        f"fabricated or uncheckable -- discard their evidential weight ENTIRELY and penalize the role "
        f"that made them; do NOT let any flagged source influence the outcome):\n{audit_report}\n\n"
        f"STAGE 1 - Role-tagged answers:\n{stage1_text}\n\n"
        f"STAGE 2 - Role-specific rankings:\n{stage2_text}"
    )
    messages = [{"role": "user", "content": chairman_prompt}]

    # Chairman = Opus 4.8 ULTRATHINK (effort=max). Extended thinking runs ~120-180s, so timeout=300
    # (past query_model's 120s default). Retry once on a transient None, then fall back to a strong
    # model so a momentary Opus outage never kills the synthesis. Safe under the async job/poll path.
    response = await query_model(CHAIRMAN_MODEL, messages, timeout=300)
    if response is None:
        response = await query_model(CHAIRMAN_MODEL, messages, timeout=300)
    if response is not None:
        return {"model": CHAIRMAN_MODEL, "response": response.get("content", "")}
    fallback = await query_model(CHAIRMAN_FALLBACK_MODEL, messages, timeout=180)
    if fallback is not None:
        return {"model": f"{CHAIRMAN_FALLBACK_MODEL} (chairman-fallback)",
                "response": fallback.get("content", "")}
    return {"model": CHAIRMAN_MODEL, "response": "Error: Unable to generate final synthesis."}


async def run_full_council_roles(
    user_query: str,
    role_betas: Dict[str, float] = None,
) -> Tuple[List, List, Dict, Dict]:
    """Run the complete role-specialized 3-stage council process."""
    stage1 = await stage1_roles(user_query, role_betas)
    if not stage1:
        return [], [], {
            "model": "error",
            "response": "All roles failed to respond. Please try again."
        }, {}

    # Stage 2 (rankings) and the fabrication audit both depend only on Stage 1 -> run concurrently,
    # so the audit adds ~0 wall-clock latency (it hides under the slower Stage-2 rankings).
    (stage2, label_to_model), audit_report = await asyncio.gather(
        stage2_roles(user_query, stage1, role_betas),
        audit_fabrication(user_query, stage1),
    )
    aggregate = calculate_aggregate_rankings(stage2, label_to_model)
    stage3 = await stage3_roles(user_query, stage1, stage2, audit_report)

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate,
        "fabrication_audit": audit_report,
    }
    return stage1, stage2, stage3, metadata
