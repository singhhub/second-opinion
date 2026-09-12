"""Tests for claim extraction (backend/claim_diff.py::extract_claims).

Per the design doc's Claim-Diff Mechanism, Step 1 (Extraction): each
model response is atomized into one subject-predicate claim per fact,
typed as assertion/recommendation/refusal_or_hedge. Malformed extractor
output must retry once, then fall back to a single raw-assertion claim
wrapping the whole response text — never crash.
"""

import json
from unittest.mock import AsyncMock, patch

from backend import claim_diff
from backend.claim_diff import Claim, ClaimType


def _fake_response(content: str):
    return {"content": content, "reasoning_details": None}


async def test_extract_claims_parses_valid_json_on_first_try():
    valid_json = json.dumps(
        [
            {"text": "Drug A + Drug B: contraindicated interaction", "claim_type": "assertion"},
            {"text": "Seek emergency care", "claim_type": "recommendation"},
        ]
    )
    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response(valid_json)),
    ) as mock_query:
        claims = await claim_diff.extract_claims("Drug A and B interact. Seek ER care.")

    assert mock_query.await_count == 1
    assert claims == [
        Claim(text="Drug A + Drug B: contraindicated interaction", claim_type=ClaimType.ASSERTION),
        Claim(text="Seek emergency care", claim_type=ClaimType.RECOMMENDATION),
    ]


async def test_extract_claims_retries_once_then_succeeds():
    responses = [
        _fake_response("not valid json at all"),
        _fake_response(json.dumps([{"text": "ok claim", "claim_type": "assertion"}])),
    ]
    with patch.object(
        claim_diff.llm_client, "query_model", new=AsyncMock(side_effect=responses)
    ) as mock_query:
        claims = await claim_diff.extract_claims("some response text")

    assert mock_query.await_count == 2
    assert claims == [Claim(text="ok claim", claim_type=ClaimType.ASSERTION)]


async def test_extract_claims_malformed_json_falls_back_to_raw_assertion():
    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("this is not json")),
    ) as mock_query:
        claims = await claim_diff.extract_claims("the original response text")

    assert mock_query.await_count == 2
    assert claims == [Claim(text="the original response text", claim_type=ClaimType.ASSERTION)]


async def test_extract_claims_query_failure_falls_back_without_raising():
    with patch.object(
        claim_diff.llm_client, "query_model", new=AsyncMock(return_value=None)
    ) as mock_query:
        claims = await claim_diff.extract_claims("the original response text")

    assert mock_query.await_count == 2
    assert claims == [Claim(text="the original response text", claim_type=ClaimType.ASSERTION)]


async def test_extract_claims_refusal_or_hedge_type_preserved():
    valid_json = json.dumps(
        [{"text": "I can't provide medical advice", "claim_type": "refusal_or_hedge"}]
    )
    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response(valid_json)),
    ):
        claims = await claim_diff.extract_claims("I can't provide medical advice.")

    assert claims == [
        Claim(text="I can't provide medical advice", claim_type=ClaimType.REFUSAL_OR_HEDGE)
    ]
