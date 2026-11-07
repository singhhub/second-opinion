"""Challenge round: choose which disagreements to challenge (pure code).

No LLM calls, no I/O. Selection is lossless: every selectable input appears
exactly once across selected + deferred.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel

from .claim_diff import (
    ActionabilityCategory,
    Claim,
    ClaimPair,
    categorize_claim,
)
from .config import CHALLENGE_FALLBACK_ITEMS, CHALLENGE_MAX_ITEMS
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
