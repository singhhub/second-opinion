"""Stage 1 response collection + claim-diff chairman synthesis."""

import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from . import llm_client
from .llm_client import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
from .claim_diff import Claim, ClaimPair, ClaimState, categorize_claim


class ObservationGroup(BaseModel):
    label: str
    items: List[str] = []


class ChairmanSummary(BaseModel):
    agreed_findings: str
    disagreement_summary: str
    observations: List[ObservationGroup] = []
    questions_for_doctor: List[str] = []


CHAIRMAN_NEVER_EMERGENCY_RULE = (
    "You must NEVER, in your own voice, state or imply whether this is or "
    "is not a medical emergency, and must NEVER tell the user to go to the "
    "ER or that it's safe not to - that call belongs to the user and their "
    "doctor, not you.\n\n"
    "This is different from reporting what a source model said. If one "
    "model explicitly recommended immediate ER/emergency care and the "
    "other did not make that recommendation, that IS the disagreement this "
    "summary exists to preserve - name it explicitly (e.g. 'Model A "
    "recommended immediate ER care; Model B did not make that "
    "recommendation') rather than omitting it because it mentions an "
    "emergency. Do not soften, hedge, or drop a source model's own "
    "recommendation out of caution about the never-emergency-verdict rule "
    "above - that rule binds your own voice only, never what you report "
    "about the source models. You may always name specific observations, "
    "symptoms, or red flags worth checking."
)


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Returns one entry per *configured* model (COUNCIL_MODELS), not only
    successes — a failed call must stay visible downstream with an
    explicit status, never silently vanish (see design doc's Degraded-run
    safety rule: the old `if response is not None` filter is the exact
    error-path-to-false-unanimity that rule exists to prevent).

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model', 'status' ('ok'/'failed'), and
        'response' (None if failed) keys.
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results - one entry per configured model, always
    stage1_results = []
    for model in COUNCIL_MODELS:
        response = responses.get(model)
        if response is not None:
            stage1_results.append({
                "model": model,
                "status": "ok",
                "response": response.get('content', '')
            })
        else:
            stage1_results.append({
                "model": model,
                "status": "failed",
                "response": None
            })

    return stage1_results


def build_ranked_diff_items(
    conflicting: List[ClaimPair],
    unconfirmed_a: List[Claim],
    unconfirmed_b: List[Claim],
) -> List[Dict[str, Any]]:
    """
    Merge CONFLICTING pairs and UNCONFIRMED claims (from both sides) into
    one list, ranked together by actionability category.

    The design doc's flooding-prevention fix ranks CONFLICTING and
    UNCONFIRMED together in one section, not each state separately -
    otherwise a lower-category conflict could still outrank a
    high-category unconfirmed finding like a drug interaction. Lossless:
    every input claim/pair appears exactly once in the output.
    """
    items: List[Dict[str, Any]] = []

    for pair in conflicting:
        items.append({
            "state": ClaimState.CONFLICTING,
            "category": categorize_claim(pair.claim_a),
            "claim_a": pair.claim_a,
            "claim_b": pair.claim_b,
        })

    for claim in unconfirmed_a:
        items.append({
            "state": ClaimState.UNCONFIRMED,
            "category": categorize_claim(claim),
            "side": "A",
            "claim": claim,
        })

    for claim in unconfirmed_b:
        items.append({
            "state": ClaimState.UNCONFIRMED,
            "category": categorize_claim(claim),
            "side": "B",
            "claim": claim,
        })

    items.sort(key=lambda item: item["category"].value)
    return items


def _format_agreed_section(agreed: List[ClaimPair]) -> str:
    if not agreed:
        return "No agreed findings - the two models did not converge on any claim."
    return "\n".join(f"- {pair.claim_a.text}" for pair in agreed)


def _format_diff_section(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No conflicting or unconfirmed claims - the two models fully agreed."

    lines = []
    for item in items:
        if item["state"] == ClaimState.CONFLICTING:
            lines.append(
                f'- CONFLICTING: Model A says "{item["claim_a"].text}"; '
                f'Model B says "{item["claim_b"].text}"'
            )
        else:
            lines.append(f'- UNCONFIRMED (Model {item["side"]} only): {item["claim"].text}')
    return "\n".join(lines)


CHAIRMAN_SYSTEM_PROMPT = f"""{CHAIRMAN_NEVER_EMERGENCY_RULE}

You are organizing a disagreement-preserving summary for two family caregivers who already saw two AI models' anonymized answers to their question about a family member's care. The chairman organizes and explains the mechanically-computed diff; it does not re-decide what counts as a conflict.

Respond with ONLY a JSON object matching this exact shape - no markdown fences, no extra text before or after:
{{
  "agreed_findings": "<a short paragraph summarizing what both models agree on - an empty string if there are none, never omit the field>",
  "disagreement_summary": "<a short paragraph explaining the disagreement or gap in plain language - an empty string if there are no conflicting/unconfirmed items>",
  "observations": [
    {{"label": "<short label for what this group of checks is for>", "items": ["<a specific observation, symptom, or check - not which claim to believe, but what to go look at>"]}}
  ],
  "questions_for_doctor": ["<a short, specific question the caregivers should bring to the next appointment>"]
}}

If there is no disagreement, set "disagreement_summary" to an empty string and "observations" to an empty list - never omit them or leave them implicit."""


def _parse_chairman_summary(raw: Optional[Dict[str, Any]]) -> Optional[ChairmanSummary]:
    if raw is None or not raw.get("content"):
        return None
    try:
        data = json.loads(llm_client.strip_json_fence(raw["content"]))
        return ChairmanSummary(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _flatten_chairman_summary(summary: ChairmanSummary) -> str:
    lines = ["Agreed Findings:", summary.agreed_findings or "None."]
    lines += ["", "Conflicting / Unconfirmed:", summary.disagreement_summary or "None."]
    for group in summary.observations:
        lines.append(f"- {group.label}:")
        for item in group.items:
            lines.append(f"  - {item}")
    lines += ["", "Questions for the Doctor:"]
    for question in summary.questions_for_doctor:
        lines.append(f"- {question}")
    return "\n".join(lines)


async def synthesize_claim_diff_chairman(
    user_query: str,
    agreed: List[ClaimPair],
    ranked_diff_items: List[Dict[str, Any]],
    chairman_model: str = CHAIRMAN_MODEL,
) -> Dict[str, Any]:
    """
    Chairman synthesis for the claim-diff mechanism.

    Returns structured JSON (agreed findings / disagreement + observations /
    questions for the doctor), Pydantic-validated with retry-then-fallback -
    the same pattern as claim_diff.extract_claims, so a malformed or failed
    LLM response never crashes the caller. "structured" is the ChairmanSummary
    object for callers that render it (the API endpoint); "response" is a
    flattened plain-text rendition for callers that only need to judge text
    (eval_harness's retention check).
    """
    agreed_text = _format_agreed_section(agreed)
    diff_text = _format_diff_section(ranked_diff_items)

    user_message = f"""Original Question: {user_query}

AGREED FINDINGS (both models agree):
{agreed_text}

CONFLICTING / UNCONFIRMED CLAIMS (ranked by actionability, most important first):
{diff_text}"""

    messages = [
        {"role": "system", "content": CHAIRMAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(2):
        raw = await query_model(chairman_model, messages)
        summary = _parse_chairman_summary(raw)
        if summary is not None:
            return {
                "model": chairman_model,
                "structured": summary,
                "response": _flatten_chairman_summary(summary),
            }

    fallback = ChairmanSummary(
        agreed_findings="",
        disagreement_summary="Unable to generate a summary for this case - please try again.",
        observations=[],
        questions_for_doctor=[],
    )
    return {
        "model": chairman_model,
        "structured": fallback,
        "response": _flatten_chairman_summary(fallback),
    }
