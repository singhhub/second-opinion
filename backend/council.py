"""Stage 1 response collection + claim-diff chairman synthesis."""

import asyncio
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from . import llm_client
from .llm_client import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
from .structured_diff import (
    STRUCTURED_BLOCK_INSTRUCTION,
    StructuredAssessment,
    StructuredDisagreement,
    StructuredDisagreementKind,
    split_structured_block,
)
from .claim_diff import (
    ActionabilityCategory,
    Claim,
    ClaimPair,
    ClaimState,
    categorize_claim,
)
from .challenge import ChallengeOutcome, SettledClaimPair


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


async def request_structured_block(
    model: str, question: str, answer: str
) -> Optional[StructuredAssessment]:
    """One follow-up call asking a model for only its structured block.

    Returns None on any failure (failed call, missing or malformed block);
    never raises, so a repair failure never blocks the run.
    """
    try:
        prompt = (
            f"Original question:\n{question}\n\n"
            f"Your earlier answer:\n{answer}\n\n"
            "You did not include the machine-readable summary. Reply with "
            "ONLY the <structured>...</structured> block for your answer "
            "above, and nothing else."
            f"{STRUCTURED_BLOCK_INSTRUCTION}"
        )
        response = await query_model(model, [{"role": "user", "content": prompt}])
        if response is None:
            return None
        _, block = split_structured_block(response.get("content", "") or "")
        return block
    except Exception:
        return None


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Returns one entry per *configured* model (COUNCIL_MODELS), not only
    successes — a failed call must stay visible downstream with an
    explicit status, never silently vanish (see design doc's Degraded-run
    safety rule: the old `if response is not None` filter is the exact
    error-path-to-false-unanimity that rule exists to prevent).

    Each model is also asked, in the same call, for a machine-readable
    structured block. A successful model whose block is missing or
    malformed gets one follow-up repair call; if that fails too,
    'structured' stays None (never read as agreement).

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model', 'status' ('ok'/'failed'), 'response'
        (free text with the block removed; None if failed) and
        'structured' (StructuredAssessment or None) keys.
    """
    messages = [
        {"role": "user", "content": user_query + "\n\n" + STRUCTURED_BLOCK_INSTRUCTION}
    ]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results - one entry per configured model, always
    stage1_results = []
    for model in COUNCIL_MODELS:
        response = responses.get(model)
        if response is not None:
            free_text, structured = split_structured_block(
                response.get('content', '') or ''
            )
            stage1_results.append({
                "model": model,
                "status": "ok",
                "response": free_text,
                "structured": structured,
            })
        else:
            stage1_results.append({
                "model": model,
                "status": "failed",
                "response": None,
                "structured": None,
            })

    # One repair attempt per ok model that produced no usable block
    to_repair = [
        r for r in stage1_results
        if r["status"] == "ok" and r["structured"] is None
    ]
    if to_repair:
        repaired = await asyncio.gather(*[
            request_structured_block(r["model"], user_query, r["response"])
            for r in to_repair
        ])
        for entry, block in zip(to_repair, repaired):
            entry["structured"] = block

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


def _quoted_position(text: Optional[str]) -> str:
    return "(no position)" if text is None else f'"{text}"'


def _format_agreed_line(pair: ClaimPair) -> str:
    if isinstance(pair, SettledClaimPair):
        return (
            f"- {pair.claim_a.text} (settled after challenge; originally "
            f"Model A: {_quoted_position(pair.original_a)}; "
            f"Model B: {_quoted_position(pair.original_b)})"
        )
    return f"- {pair.claim_a.text}"


def _format_agreed_section(agreed: List[ClaimPair]) -> str:
    if not agreed:
        return "No agreed findings - the two models did not converge on any claim."
    return "\n".join(_format_agreed_line(pair) for pair in agreed)


def _format_diff_section(
    items: List[Dict[str, Any]], has_structured_disagreements: bool = False
) -> str:
    if not items:
        if has_structured_disagreements:
            return (
                "No conflicting or unconfirmed claims in the claim comparison, but "
                "the models still disagree - see the STRUCTURED COMPARISON "
                "DISAGREEMENTS section below."
            )
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


CHALLENGE_SECTION_NOTE = """

The user message also contains a CHALLENGE ROUND RESULTS section. It lists disagreements the two models were asked to reconsider after seeing each other's position. For each entry you are given the status, any flags, the two original positions, and the final positions.
- For a RESOLVED item, say the disagreement was settled after the models were challenged, and name both original positions so the caregivers can see what changed.
- Flag meanings: "changed_without_evidence" means a model changed its position without quoting supporting text from the patient record; "mutual_concession" means each model adopted the other's position (a swap, not real agreement); "resolved_without_record_evidence" means no patient record was available, so the settlement could not be checked against any record. State these plainly when they apply.
- Never present any outcome as verified, confirmed, or clinician-checked. A settled item is still only what the models said after being challenged.
- Never soften or omit a source model's own emergency or urgent recommendation, even if that item is RESOLVED. Urgency disagreements are always reported."""


STRUCTURED_SECTION_NOTE = """

The user message also contains a STRUCTURED COMPARISON DISAGREEMENTS section. It lists differences found by comparing the two models' machine-readable summaries (possible causes, interaction flags, medication changes, urgency) field by field. These are real disagreements between the two models even when the claim list shows none, so include them in disagreement_summary - never describe the models as fully agreeing while this section has entries.
- A row tagged "urgency: always report" is an urgency-level disagreement: always name both models' urgency levels, and never soften or omit the more urgent one."""


def _format_structured_section(disagreements: List[StructuredDisagreement]) -> str:
    lines = []
    for d in disagreements:
        line = f"- {d.summary} (Model A: {d.detail_a}; Model B: {d.detail_b})"
        if d.kind == StructuredDisagreementKind.URGENCY:
            line += " [urgency: always report]"
        lines.append(line)
    return "\n".join(lines)


def _format_challenge_section(outcomes: List[ChallengeOutcome]) -> str:
    def pos(text: Optional[str]) -> str:
        return "(none)" if text is None else f'"{text}"'

    lines = []
    for o in outcomes:
        flags = ", ".join(o.flags) if o.flags else "none"
        tag = ""
        if (
            o.item.always_visible
            or o.item.category == ActionabilityCategory.RISK_OR_RED_FLAG
        ):
            tag = " [urgency/red flag: always report]"
        lines.append(
            f"- Category: {o.item.category.name}{tag}; "
            f"Status: {o.status.value}; flags: {flags}\n"
            f"  Original: Model A {pos(o.item.text_a)}; Model B {pos(o.item.text_b)}\n"
            f"  Final: Model A {pos(o.final_a)}; Model B {pos(o.final_b)}"
        )
    return "\n".join(lines)


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
    challenge_outcomes: Optional[List[ChallengeOutcome]] = None,
    structured_disagreements: Optional[List[StructuredDisagreement]] = None,
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

    structured_disagreements (from compare_structured) are given to the
    chairman as their own section, independent of challenge outcomes; when
    None or empty, the messages sent are unchanged.
    """
    agreed_text = _format_agreed_section(agreed)
    diff_text = _format_diff_section(
        ranked_diff_items, has_structured_disagreements=bool(structured_disagreements)
    )

    user_message = f"""Original Question: {user_query}

AGREED FINDINGS (both models agree):
{agreed_text}

CONFLICTING / UNCONFIRMED CLAIMS (ranked by actionability, most important first):
{diff_text}"""

    system_prompt = CHAIRMAN_SYSTEM_PROMPT
    if structured_disagreements:
        user_message += (
            "\n\nSTRUCTURED COMPARISON DISAGREEMENTS (from each model's "
            "machine-readable summary; reported independently of the claims "
            "above):\n" + _format_structured_section(structured_disagreements)
        )
        system_prompt += STRUCTURED_SECTION_NOTE
    if challenge_outcomes:
        user_message += (
            "\n\nCHALLENGE ROUND RESULTS (disagreements the two models were "
            "asked to reconsider):\n" + _format_challenge_section(challenge_outcomes)
        )
        system_prompt += CHALLENGE_SECTION_NOTE

    messages = [
        {"role": "system", "content": system_prompt},
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
