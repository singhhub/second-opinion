"""Claim-diff mechanism (validation-phase v1).

Replaces stage 2's ranking + stage 3's single-answer synthesis with a
disagreement-preserving mechanism: extract atomized claims from each
model's response, align them by aboutness, classify compatibility, and
rank by actionability. See the design doc's "Claim-Diff Mechanism"
section for the full definition; this module implements Step 1
(Extraction).

The validation-phase claim struct is deliberately smaller than the
build-phase one: {text, claim_type} only. evidence_span, source_doc_id,
extraction_confidence, and page provenance require real documents and
retrieval and aren't needed to test whether the diff mechanism itself
preserves a disagreement.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from . import llm_client

DEFAULT_EXTRACTOR_MODEL = "claude/claude-sonnet-4-5-20250929"

EXTRACTION_SYSTEM_PROMPT = """You extract atomized claims from a medical assistant's response.

Break the response into one subject-predicate claim per fact or \
instruction. Do not fold multiple facts into one claim (e.g. "drug A + \
drug B: contraindicated interaction" is one claim, not folded into a \
paragraph).

Classify each claim's claim_type:
- "assertion": a factual or clinical statement.
- "recommendation": an action to take (e.g. "seek emergency care").
- "refusal_or_hedge": the response declines to answer, or hedges so \
heavily it asserts nothing.

Respond with ONLY a JSON array of objects, each with exactly two keys: \
"text" and "claim_type". No other text, no markdown fences."""


class ClaimType(str, Enum):
    ASSERTION = "assertion"
    RECOMMENDATION = "recommendation"
    REFUSAL_OR_HEDGE = "refusal_or_hedge"


class Claim(BaseModel):
    text: str
    claim_type: ClaimType


class _ClaimList(BaseModel):
    claims: List[Claim]


def _parse_claims(raw: Optional[Dict[str, Any]]) -> Optional[List[Claim]]:
    if raw is None or not raw.get("content"):
        return None
    try:
        data = json.loads(raw["content"])
        parsed = _ClaimList(claims=data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None
    return parsed.claims


async def extract_claims(
    response_text: str,
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
) -> List[Claim]:
    """
    Atomize a model's response text into typed claims via an LLM extractor.

    Retries once on malformed/unparseable output, then falls back to a
    single raw-assertion claim wrapping the whole response text. Never
    raises and never crashes the caller on extractor failure.
    """
    for _ in range(2):
        raw = await llm_client.query_model(
            extractor_model,
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": response_text},
            ],
        )
        claims = _parse_claims(raw)
        if claims is not None:
            return claims

    return [Claim(text=response_text, claim_type=ClaimType.ASSERTION)]
