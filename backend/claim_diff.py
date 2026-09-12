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
import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from . import llm_client

DEFAULT_EXTRACTOR_MODEL = "claude/claude-sonnet-4-5-20250929"

# First-pass defaults (not tuned — see design doc Open Questions: exact
# claim-similarity threshold and embedding model are deliberately left as
# a first-pass default, to be tuned once real eval cases exist).
ALIGNMENT_HIGH_THRESHOLD = 0.85
ALIGNMENT_LOW_THRESHOLD = 0.70

ALIGNMENT_JUDGE_SYSTEM_PROMPT = """You judge whether two medical claims are about the same specific topic — aboutness only, not whether they agree or disagree. "Drug A+B is contraindicated" and "Drug A+B is safe together" ARE about the same topic (they disagree, but that's a separate question). Respond with exactly one word: YES or NO."""

COMPATIBILITY_SYSTEM_PROMPT = """You classify whether two aligned medical claims are compatible (agree) or incompatible (conflict). "Aboutness" has already been established — both claims are about the same topic. Judge only whether they assert the same thing or contradict each other. Respond with exactly one word: COMPATIBLE or INCOMPATIBLE."""

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


class ClaimPair(BaseModel):
    claim_a: Claim
    claim_b: Claim


class AlignmentResult(BaseModel):
    aligned: List[ClaimPair]
    unaligned_a: List[Claim]
    unaligned_b: List[Claim]


class ClaimState(str, Enum):
    """Output states of the Claim-Diff Mechanism (three, matching n=2)."""

    AGREED = "AGREED"
    CONFLICTING = "CONFLICTING"
    UNCONFIRMED = "UNCONFIRMED"


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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _judge_aboutness(
    claim_a: Claim, claim_b: Claim, judge_model: str
) -> bool:
    raw = await llm_client.query_model(
        judge_model,
        [
            {"role": "system", "content": ALIGNMENT_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim A: {claim_a.text}\nClaim B: {claim_b.text}"},
        ],
    )
    if raw is None or not raw.get("content"):
        return False
    return raw["content"].strip().upper().startswith("YES")


async def align_claims(
    claims_a: List[Claim],
    claims_b: List[Claim],
    judge_model: str = DEFAULT_EXTRACTOR_MODEL,
    high_threshold: float = ALIGNMENT_HIGH_THRESHOLD,
    low_threshold: float = ALIGNMENT_LOW_THRESHOLD,
) -> AlignmentResult:
    """
    Align claims from two models by aboutness only (Step 2 — not agreement).

    Embedding similarity >= high_threshold aligns directly. Similarity in
    [low_threshold, high_threshold) is borderline and goes to an LLM judge.
    Below low_threshold, or if either claim's embedding fails, no
    alignment (never crashes on an embedding failure — the claim simply
    stays unaligned).

    At n=2, alignment is greedy best-match: each claim links to at most
    one claim on the other side — the globally highest-similarity
    candidate pairs are matched first, so a claim already matched is never
    reconsidered for a second, weaker match.
    """
    if not claims_a or not claims_b:
        return AlignmentResult(aligned=[], unaligned_a=list(claims_a), unaligned_b=list(claims_b))

    embeddings_a = [await llm_client.embed_text(c.text) for c in claims_a]
    embeddings_b = [await llm_client.embed_text(c.text) for c in claims_b]

    candidates = []
    for i, emb_a in enumerate(embeddings_a):
        if emb_a is None:
            continue
        for j, emb_b in enumerate(embeddings_b):
            if emb_b is None:
                continue
            similarity = _cosine_similarity(emb_a, emb_b)
            if similarity >= low_threshold:
                candidates.append((similarity, i, j))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_a: set = set()
    matched_b: set = set()
    aligned: List[ClaimPair] = []

    for similarity, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue

        is_aligned = similarity >= high_threshold
        if not is_aligned:
            is_aligned = await _judge_aboutness(claims_a[i], claims_b[j], judge_model)

        if is_aligned:
            aligned.append(ClaimPair(claim_a=claims_a[i], claim_b=claims_b[j]))
            matched_a.add(i)
            matched_b.add(j)

    unaligned_a = [c for idx, c in enumerate(claims_a) if idx not in matched_a]
    unaligned_b = [c for idx, c in enumerate(claims_b) if idx not in matched_b]

    return AlignmentResult(aligned=aligned, unaligned_a=unaligned_a, unaligned_b=unaligned_b)


async def classify_compatibility(
    pair: ClaimPair,
    judge_model: str = DEFAULT_EXTRACTOR_MODEL,
) -> ClaimState:
    """
    Classify an aligned claim pair as AGREED or CONFLICTING (Step 3).

    "Aboutness" (alignment) does not imply compatibility — near-identical
    embeddings can be opposite in meaning (e.g. "contraindicated" vs.
    "safe together"). Fail-safe default: any response other than exactly
    "COMPATIBLE" — an ambiguous judgment, malformed output, or a failed
    call — resolves to CONFLICTING, never AGREED.
    """
    raw = await llm_client.query_model(
        judge_model,
        [
            {"role": "system", "content": COMPATIBILITY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Claim A: {pair.claim_a.text}\nClaim B: {pair.claim_b.text}",
            },
        ],
    )

    if raw is not None and raw.get("content", "").strip().upper() == "COMPATIBLE":
        return ClaimState.AGREED

    return ClaimState.CONFLICTING
