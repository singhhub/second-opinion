"""Tests for the claim-diff mechanism (backend/claim_diff.py).

Covers Step 1 (Extraction, extract_claims) and Step 2 (Alignment,
align_claims) of the design doc's Claim-Diff Mechanism. Extraction:
malformed extractor output must retry once, then fall back to a single
raw-assertion claim — never crash. Alignment: embedding similarity above
a threshold means "aboutness," with an LLM judge for borderline pairs;
at n=2, alignment is greedy best-match — each claim links to at most one
claim on the other side, or to none.
"""

import json
import math
from unittest.mock import AsyncMock, patch

from backend import claim_diff
from backend.claim_diff import Claim, ClaimType


def _fake_response(content: str):
    return {"content": content, "reasoning_details": None}


def _fake_embed(vectors: dict):
    async def _embed(text, *args, **kwargs):
        return vectors.get(text)

    return AsyncMock(side_effect=_embed)


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


async def test_align_claims_high_similarity_aligns_without_llm_judge():
    claim_a = Claim(text="drug A and drug B interact", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="A and B should not be combined", claim_type=ClaimType.ASSERTION)
    vectors = {
        claim_a.text: [1.0, 0.0],
        claim_b.text: [0.95, math.sqrt(1 - 0.95**2)],
    }

    with patch.object(claim_diff.llm_client, "embed_text", new=_fake_embed(vectors)), patch.object(
        claim_diff.llm_client, "query_model", new=AsyncMock()
    ) as mock_judge:
        result = await claim_diff.align_claims([claim_a], [claim_b])

    assert mock_judge.await_count == 0
    assert len(result.aligned) == 1
    assert result.aligned[0].claim_a == claim_a
    assert result.aligned[0].claim_b == claim_b
    assert result.unaligned_a == []
    assert result.unaligned_b == []


async def test_align_claims_low_similarity_leaves_unaligned_without_llm_judge():
    claim_a = Claim(text="drug A and drug B interact", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="unrelated topic entirely", claim_type=ClaimType.ASSERTION)
    vectors = {
        claim_a.text: [1.0, 0.0],
        claim_b.text: [0.0, 1.0],
    }

    with patch.object(claim_diff.llm_client, "embed_text", new=_fake_embed(vectors)), patch.object(
        claim_diff.llm_client, "query_model", new=AsyncMock()
    ) as mock_judge:
        result = await claim_diff.align_claims([claim_a], [claim_b])

    assert mock_judge.await_count == 0
    assert result.aligned == []
    assert result.unaligned_a == [claim_a]
    assert result.unaligned_b == [claim_b]


async def test_align_claims_borderline_similarity_llm_judge_says_yes_aligns():
    claim_a = Claim(text="claim A", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="claim B", claim_type=ClaimType.ASSERTION)
    vectors = {
        claim_a.text: [1.0, 0.0],
        claim_b.text: [0.77, math.sqrt(1 - 0.77**2)],
    }

    with patch.object(claim_diff.llm_client, "embed_text", new=_fake_embed(vectors)), patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("YES")),
    ) as mock_judge:
        result = await claim_diff.align_claims([claim_a], [claim_b])

    assert mock_judge.await_count == 1
    assert len(result.aligned) == 1
    assert result.unaligned_a == []
    assert result.unaligned_b == []


async def test_align_claims_borderline_similarity_llm_judge_says_no_stays_unaligned():
    claim_a = Claim(text="claim A", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="claim B", claim_type=ClaimType.ASSERTION)
    vectors = {
        claim_a.text: [1.0, 0.0],
        claim_b.text: [0.77, math.sqrt(1 - 0.77**2)],
    }

    with patch.object(claim_diff.llm_client, "embed_text", new=_fake_embed(vectors)), patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("NO")),
    ) as mock_judge:
        result = await claim_diff.align_claims([claim_a], [claim_b])

    assert mock_judge.await_count == 1
    assert result.aligned == []
    assert result.unaligned_a == [claim_a]
    assert result.unaligned_b == [claim_b]


async def test_align_claims_embedding_failure_resolves_to_unaligned_never_crashes():
    claim_a = Claim(text="claim A", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="claim B", claim_type=ClaimType.ASSERTION)

    with patch.object(claim_diff.llm_client, "embed_text", new=_fake_embed({})):
        result = await claim_diff.align_claims([claim_a], [claim_b])

    assert result.aligned == []
    assert result.unaligned_a == [claim_a]
    assert result.unaligned_b == [claim_b]


async def test_align_claims_greedy_best_match_prevents_double_matching():
    a1 = Claim(text="a1", claim_type=ClaimType.ASSERTION)
    a2 = Claim(text="a2", claim_type=ClaimType.ASSERTION)
    b1 = Claim(text="b1", claim_type=ClaimType.ASSERTION)
    vectors = {
        "a1": [1.0, 0.0],
        "a2": [1.0, 0.5],
        "b1": [1.0, 0.0],
    }

    with patch.object(claim_diff.llm_client, "embed_text", new=_fake_embed(vectors)), patch.object(
        claim_diff.llm_client, "query_model", new=AsyncMock()
    ) as mock_judge:
        result = await claim_diff.align_claims([a1, a2], [b1])

    assert mock_judge.await_count == 0
    assert len(result.aligned) == 1
    assert result.aligned[0].claim_a == a1
    assert result.aligned[0].claim_b == b1
    assert result.unaligned_a == [a2]
    assert result.unaligned_b == []


async def test_align_claims_empty_input_returns_empty_result():
    claim_a = Claim(text="claim A", claim_type=ClaimType.ASSERTION)

    result = await claim_diff.align_claims([claim_a], [])

    assert result.aligned == []
    assert result.unaligned_a == [claim_a]
    assert result.unaligned_b == []


def _pair(text_a="drug A+B is contraindicated", text_b="drug A+B is safe together"):
    from backend.claim_diff import ClaimPair

    return ClaimPair(
        claim_a=Claim(text=text_a, claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text=text_b, claim_type=ClaimType.ASSERTION),
    )


async def test_classify_compatibility_compatible_returns_agreed():
    from backend.claim_diff import ClaimState

    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("COMPATIBLE")),
    ):
        result = await claim_diff.classify_compatibility(_pair())

    assert result == ClaimState.AGREED


async def test_classify_compatibility_incompatible_returns_conflicting():
    from backend.claim_diff import ClaimState

    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("INCOMPATIBLE")),
    ):
        result = await claim_diff.classify_compatibility(_pair())

    assert result == ClaimState.CONFLICTING


async def test_classify_compatibility_ambiguous_response_defaults_to_conflicting():
    from backend.claim_diff import ClaimState

    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("unclear, could go either way")),
    ):
        result = await claim_diff.classify_compatibility(_pair())

    assert result == ClaimState.CONFLICTING


async def test_classify_compatibility_query_failure_defaults_to_conflicting():
    from backend.claim_diff import ClaimState

    with patch.object(
        claim_diff.llm_client, "query_model", new=AsyncMock(return_value=None)
    ):
        result = await claim_diff.classify_compatibility(_pair())

    assert result == ClaimState.CONFLICTING


async def test_classify_compatibility_sends_both_claim_texts():
    with patch.object(
        claim_diff.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("COMPATIBLE")),
    ) as mock_query:
        await claim_diff.classify_compatibility(_pair("claim text A", "claim text B"))

    sent_messages = mock_query.call_args.args[1]
    user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
    assert "claim text A" in user_content
    assert "claim text B" in user_content


def test_categorize_claim_drug_interaction_is_highest_category():
    from backend.claim_diff import ActionabilityCategory, categorize_claim

    claim = Claim(
        text="Drug A + Drug B: contraindicated interaction",
        claim_type=ClaimType.ASSERTION,
    )
    assert categorize_claim(claim) == ActionabilityCategory.DRUG_INTERACTION


def test_categorize_claim_red_flag_risk():
    from backend.claim_diff import ActionabilityCategory, categorize_claim

    claim = Claim(
        text="Sudden face drooping is a red flag warning sign of stroke",
        claim_type=ClaimType.ASSERTION,
    )
    assert categorize_claim(claim) == ActionabilityCategory.RISK_OR_RED_FLAG


def test_categorize_claim_diagnostic_suggestion():
    from backend.claim_diff import ActionabilityCategory, categorize_claim

    claim = Claim(
        text="These symptoms are consistent with a diagnosis of TIA",
        claim_type=ClaimType.ASSERTION,
    )
    assert categorize_claim(claim) == ActionabilityCategory.DIAGNOSTIC_SUGGESTION


def test_categorize_claim_monitoring_followup():
    from backend.claim_diff import ActionabilityCategory, categorize_claim

    claim = Claim(
        text="Monitor blood pressure and follow up in two weeks",
        claim_type=ClaimType.ASSERTION,
    )
    assert categorize_claim(claim) == ActionabilityCategory.MONITORING_FOLLOWUP


def test_categorize_claim_ambiguous_defaults_to_lowest_category():
    from backend.claim_diff import ActionabilityCategory, categorize_claim

    claim = Claim(
        text="The patient has a long history of well-managed hypertension",
        claim_type=ClaimType.ASSERTION,
    )
    assert categorize_claim(claim) == ActionabilityCategory.BACKGROUND_OR_CAVEAT


def test_rank_by_actionability_orders_highest_category_first():
    background = Claim(
        text="The patient has a long history of well-managed hypertension",
        claim_type=ClaimType.ASSERTION,
    )
    monitoring = Claim(
        text="Monitor blood pressure and follow up in two weeks",
        claim_type=ClaimType.ASSERTION,
    )
    interaction = Claim(
        text="Drug A + Drug B: contraindicated interaction",
        claim_type=ClaimType.ASSERTION,
    )
    red_flag = Claim(
        text="Sudden face drooping is a red flag warning sign of stroke",
        claim_type=ClaimType.ASSERTION,
    )

    ranked = claim_diff.rank_by_actionability([background, monitoring, interaction, red_flag])

    assert ranked == [interaction, red_flag, monitoring, background]


def test_rank_by_actionability_is_lossless():
    claims = [
        Claim(text="claim one", claim_type=ClaimType.ASSERTION),
        Claim(text="claim two", claim_type=ClaimType.ASSERTION),
        Claim(text="Drug A + Drug B: contraindicated interaction", claim_type=ClaimType.ASSERTION),
    ]

    ranked = claim_diff.rank_by_actionability(claims)

    assert len(ranked) == len(claims)
    assert set(c.text for c in ranked) == set(c.text for c in claims)
