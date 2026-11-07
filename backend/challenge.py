"""Challenge round: choose which disagreements to challenge, then run it.

Selection (select_challenge_items) is pure code: no LLM calls, no I/O, and
lossless - every selectable input appears exactly once across selected +
deferred. Execution (run_challenge_round) shows each model the other's
position and asks it to maintain, revise, or concede, then re-classifies.
Fail-safe: anything malformed, failed, or unverifiable never resolves.
"""

import asyncio
import json
import re
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ValidationError

from . import llm_client
from .claim_diff import (
    ActionabilityCategory,
    Claim,
    ClaimPair,
    ClaimState,
    ClaimType,
    categorize_claim,
    classify_compatibility,
)
from .config import (
    CHALLENGE_FALLBACK_ITEMS,
    CHALLENGE_MAX_ITEMS,
    CHALLENGE_MAX_ROUNDS,
    CHALLENGE_MAX_ROUNDS_HARD_CAP,
)
from .llm_client import query_model
from .structured_diff import StructuredDisagreement, StructuredDisagreementKind


class ChallengeItem(BaseModel):
    item_id: str
    source: Literal["conflicting", "unconfirmed", "structured"]
    category: ActionabilityCategory
    text_a: Optional[str]
    text_b: Optional[str]
    pair: Optional[ClaimPair] = None
    claim: Optional[Claim] = None
    side: Optional[Literal["A", "B"]] = None
    always_visible: bool = False


class ChallengeSelection(BaseModel):
    selected: List[ChallengeItem]
    deferred: List[ChallengeItem]
    cap_hit: bool


_STRUCTURED_CATEGORY = {
    StructuredDisagreementKind.URGENCY: ActionabilityCategory.RISK_OR_RED_FLAG,
    StructuredDisagreementKind.TOP_CAUSE: ActionabilityCategory.DIAGNOSTIC_SUGGESTION,
    StructuredDisagreementKind.ONE_SIDED_CAUSE: ActionabilityCategory.DIAGNOSTIC_SUGGESTION,
    StructuredDisagreementKind.INTERACTION_FLAG: ActionabilityCategory.DRUG_INTERACTION,
    StructuredDisagreementKind.MEDICATION_CHANGE: ActionabilityCategory.DRUG_INTERACTION,
}


def _unconfirmed_item(index: int, claim: Claim, side: str) -> ChallengeItem:
    return ChallengeItem(
        item_id=f"u{index}",
        source="unconfirmed",
        category=categorize_claim(claim),
        text_a=claim.text if side == "A" else None,
        text_b=claim.text if side == "B" else None,
        claim=claim,
        side=side,
    )


def select_challenge_items(
    conflicting: List[ClaimPair],
    unconfirmed_a: List[Claim],
    unconfirmed_b: List[Claim],
    structured: List[StructuredDisagreement],
    max_items: int = CHALLENGE_MAX_ITEMS,
) -> ChallengeSelection:
    conflict_items = [
        ChallengeItem(
            item_id=f"c{i}",
            source="conflicting",
            category=categorize_claim(pair.claim_a),
            text_a=pair.claim_a.text,
            text_b=pair.claim_b.text,
            pair=pair,
        )
        for i, pair in enumerate(conflicting)
    ]
    structured_items = [
        ChallengeItem(
            item_id=f"s{i}",
            source="structured",
            category=_STRUCTURED_CATEGORY[sd.kind],
            text_a=sd.detail_a,
            text_b=sd.detail_b,
            always_visible=sd.kind == StructuredDisagreementKind.URGENCY,
        )
        for i, sd in enumerate(structured)
    ]
    all_unconfirmed = [_unconfirmed_item(i, c, "A") for i, c in enumerate(unconfirmed_a)]
    offset = len(unconfirmed_a)
    all_unconfirmed += [
        _unconfirmed_item(offset + i, c, "B") for i, c in enumerate(unconfirmed_b)
    ]
    unconfirmed_items = [
        i for i in all_unconfirmed
        if i.category != ActionabilityCategory.BACKGROUND_OR_CAVEAT
    ]

    pool = conflict_items + structured_items + unconfirmed_items
    if not pool and all_unconfirmed:
        pool = all_unconfirmed[:CHALLENGE_FALLBACK_ITEMS]

    ordered = sorted(pool, key=lambda item: item.category.value)  # stable
    limit = max(max_items, 0)
    selected = ordered[:limit]
    deferred = ordered[limit:]
    return ChallengeSelection(
        selected=selected, deferred=deferred, cap_hit=bool(deferred)
    )


# ---------------------------------------------------------------------------
# Challenge round execution
# ---------------------------------------------------------------------------

FLAG_CHANGED_WITHOUT_EVIDENCE = "changed_without_evidence"
FLAG_NO_RECORD_EVIDENCE = "resolved_without_record_evidence"

CHALLENGE_SYSTEM_PROMPT = """You are a careful medical assistant reviewing a \
second, independent answer to the same question. You will be shown your own \
position on one point and another assistant's position on the same point.

Reconsider the point on its merits. You may:
- "maintain": keep your position as it is.
- "revise": restate your position differently (give the new wording in \
"revised_claim").
- "concede": accept the other position as correct.

Only change your position if there is a genuine clinical reason to. Do not \
change it merely because the other assistant holds a different view. If the \
patient record is provided, base any change on what the record says, and \
copy the supporting passage into "evidence_quote" exactly as written in the \
record (do not paraphrase). Use null for "evidence_quote" if no passage \
supports your answer, and null for "revised_claim" unless you revise.

Respond with ONLY a JSON object with exactly these four keys: "stance" \
(one of "maintain", "revise", "concede"), "revised_claim" (string or null), \
"reasoning" (a short string), "evidence_quote" (string or null). No other \
text, no markdown fences."""


class ChallengeStance(str, Enum):
    MAINTAIN = "maintain"
    REVISE = "revise"
    CONCEDE = "concede"


class ChallengeResponse(BaseModel):
    stance: ChallengeStance
    revised_claim: Optional[str] = None
    reasoning: str = ""
    evidence_quote: Optional[str] = None


class ChallengeStatus(str, Enum):
    RESOLVED = "resolved"
    STILL_CONFLICTING = "still_conflicting"
    UNRESOLVED_ERROR = "unresolved_error"


class ChallengeOutcome(BaseModel):
    item: ChallengeItem
    final_a: Optional[str]
    final_b: Optional[str]
    status: ChallengeStatus
    flags: List[str] = []
    rounds_used: int = 0


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_in_record(quote: Optional[str], record_context: str) -> bool:
    """Case-sensitive substring check, whitespace runs collapsed on both sides."""
    if not quote:
        return False
    collapsed = _collapse_ws(quote)
    if not collapsed:
        return False
    return collapsed in _collapse_ws(record_context or "")


def _parse_challenge_response(raw) -> Optional[ChallengeResponse]:
    if raw is None or not raw.get("content"):
        return None
    try:
        data = json.loads(llm_client.strip_json_fence(raw["content"]))
        if isinstance(data, dict) and isinstance(data.get("stance"), str):
            data["stance"] = data["stance"].strip().lower()
        parsed = ChallengeResponse(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None
    if parsed.stance == ChallengeStance.REVISE and not (
        parsed.revised_claim and parsed.revised_claim.strip()
    ):
        return None
    return parsed


def _build_prompt(
    question: str,
    own: str,
    other: str,
    record_context: Optional[str],
) -> List[dict]:
    parts = [f"Original question:\n{question}\n"]
    if record_context is not None:
        parts.append(
            "Patient record (evidence_quote must be copied exactly from "
            f"this text):\n{record_context}\n"
        )
    parts.append(f"Your position:\n{own}\n")
    parts.append(f"The other assistant's position:\n{other}\n")
    parts.append("Respond with the JSON object only.")
    return [
        {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


async def _ask(
    model: str,
    question: str,
    own: str,
    other: str,
    record_context: Optional[str],
) -> Optional[ChallengeResponse]:
    """One model call, retried once on malformed output. None = failed."""
    messages = _build_prompt(question, own, other, record_context)
    for _ in range(2):
        parsed = _parse_challenge_response(await query_model(model, messages))
        if parsed is not None:
            return parsed
    return None


class _ItemState:
    def __init__(self, item: ChallengeItem):
        self.item = item
        self.pos_a = item.text_a
        self.pos_b = item.text_b
        self.status = ChallengeStatus.STILL_CONFLICTING
        self.flags: List[str] = []
        self.rounds_used = 0

    def outcome(self) -> ChallengeOutcome:
        return ChallengeOutcome(
            item=self.item,
            final_a=self.pos_a,
            final_b=self.pos_b,
            status=self.status,
            flags=list(self.flags),
            rounds_used=self.rounds_used,
        )


def _claim_type_for(item: ChallengeItem, side: str) -> ClaimType:
    if item.pair is not None:
        return (item.pair.claim_a if side == "A" else item.pair.claim_b).claim_type
    if item.claim is not None and item.side == side:
        return item.claim.claim_type
    return ClaimType.ASSERTION


def _final_position(
    resp: ChallengeResponse, own: Optional[str], other: Optional[str]
) -> Optional[str]:
    if resp.stance == ChallengeStance.REVISE:
        return resp.revised_claim
    if resp.stance == ChallengeStance.CONCEDE:
        return other
    return own


async def _challenge_once(
    state: _ItemState,
    question: str,
    model_a: str,
    model_b: str,
    record_context: Optional[str],
) -> None:
    """Run one round for one item, mutating state. May raise; caller isolates."""
    pa, pb = state.pos_a, state.pos_b
    ask_a = pb is not None
    ask_b = pa is not None
    calls = []
    if ask_a:
        calls.append(_ask(model_a, question, pa or "", pb, record_context))
    if ask_b:
        calls.append(_ask(model_b, question, pb or "", pa, record_context))
    if not calls:
        state.status = ChallengeStatus.UNRESOLVED_ERROR
        return
    results = list(await asyncio.gather(*calls))
    state.rounds_used += 1
    resp_a = results.pop(0) if ask_a else None
    resp_b = results.pop(0) if ask_b else None
    if (ask_a and resp_a is None) or (ask_b and resp_b is None):
        state.status = ChallengeStatus.UNRESOLVED_ERROR
        state.flags = []
        return

    final_a = _final_position(resp_a, pa, pb) if resp_a else pa
    final_b = _final_position(resp_b, pb, pa) if resp_b else pb
    responses = [r for r in (resp_a, resp_b) if r is not None]
    changed = [r for r in responses if r.stance != ChallengeStance.MAINTAIN]

    flags = list(state.flags)
    unsupported = False
    if record_context is not None:
        for r in changed:
            if not evidence_in_record(r.evidence_quote, record_context):
                unsupported = True
        if unsupported and FLAG_CHANGED_WITHOUT_EVIDENCE not in flags:
            flags.append(FLAG_CHANGED_WITHOUT_EVIDENCE)

    status = ChallengeStatus.STILL_CONFLICTING
    if changed and final_a is not None and final_b is not None:
        verdict = await classify_compatibility(
            ClaimPair(
                claim_a=Claim(text=final_a, claim_type=_claim_type_for(state.item, "A")),
                claim_b=Claim(text=final_b, claim_type=_claim_type_for(state.item, "B")),
            )
        )
        if verdict == ClaimState.AGREED and not unsupported:
            status = ChallengeStatus.RESOLVED
            if record_context is None and FLAG_NO_RECORD_EVIDENCE not in flags:
                flags.append(FLAG_NO_RECORD_EVIDENCE)

    state.pos_a, state.pos_b = final_a, final_b
    state.status = status
    state.flags = flags


async def run_challenge_round(
    items: List[ChallengeItem],
    question: str,
    model_a: str,
    model_b: str,
    record_context: Optional[str] = None,
    max_rounds: int = CHALLENGE_MAX_ROUNDS,
) -> List[ChallengeOutcome]:
    """Challenge each item; one outcome per input item, same order, never raises.

    max_rounds < 1 is treated as 1; the total is capped at
    CHALLENGE_MAX_ROUNDS_HARD_CAP. Later rounds only re-challenge items that
    are STILL_CONFLICTING (errors are not retried).
    """
    states = [_ItemState(item) for item in items]
    rounds = min(max(max_rounds, 1), CHALLENGE_MAX_ROUNDS_HARD_CAP)

    async def guarded(state: _ItemState) -> None:
        try:
            await _challenge_once(state, question, model_a, model_b, record_context)
        except Exception:
            state.status = ChallengeStatus.UNRESOLVED_ERROR
            state.flags = []
            state.pos_a, state.pos_b = state.item.text_a, state.item.text_b

    for _ in range(rounds):
        active = [s for s in states if s.status == ChallengeStatus.STILL_CONFLICTING]
        if not active:
            break
        await asyncio.gather(*(guarded(s) for s in active), return_exceptions=True)

    return [s.outcome() for s in states]
