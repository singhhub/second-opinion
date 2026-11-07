"""Tests for the eval harness (backend/eval_harness.py).

Runs each fixed eval case through the claim-diff mechanism and reports a
retention count, not a percentage (Success Criterion 1) — the sample is
too small for a rate to mean anything. All the underlying pipeline pieces
(extract_claims, align_claims, classify_compatibility,
build_ranked_diff_items, synthesize_claim_diff_chairman) are already
unit-tested elsewhere, so these tests mock them at the boundary and check
the harness's own orchestration and aggregation.
"""

import json
from unittest.mock import AsyncMock, patch

from backend import council, eval_harness
from backend.challenge import ChallengeOutcome, ChallengeStatus, select_challenge_items
from backend.claim_diff import AlignmentResult, Claim, ClaimPair, ClaimState, ClaimType
from backend.config import COUNCIL_MODELS


def _fake_response(content: str):
    return {"content": content, "reasoning_details": None}


def _fake_stage_result(content: str):
    return {"model": "some-model", "response": content}


def _sample_case(**overrides):
    case = {
        "id": "sample-case",
        "question": "is this drug combination safe?",
        "model_a_response": "drug A and drug B interact",
        "model_b_response": "no mention of any interaction",
        "known_correct_claim": "drug A and drug B interact",
        "case_source": "synthetic",
    }
    case.update(overrides)
    return case


async def test_check_claim_retained_yes():
    with patch.object(
        eval_harness.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("YES")),
    ):
        result = await eval_harness.check_claim_retained("some claim", "some text")
    assert result is True


async def test_check_claim_retained_no():
    with patch.object(
        eval_harness.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("NO")),
    ):
        result = await eval_harness.check_claim_retained("some claim", "some text")
    assert result is False


async def test_check_claim_retained_query_failure_defaults_to_false():
    with patch.object(
        eval_harness.llm_client, "query_model", new=AsyncMock(return_value=None)
    ):
        result = await eval_harness.check_claim_retained("some claim", "some text")
    assert result is False


async def test_run_new_mechanism_splits_agreed_vs_conflicting():
    case = _sample_case()
    claim_a = Claim(text="drug A and drug B interact", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="no interaction found", claim_type=ClaimType.ASSERTION)
    pair = ClaimPair(claim_a=claim_a, claim_b=claim_b)

    with patch.object(
        eval_harness, "extract_claims", new=AsyncMock(side_effect=[[claim_a], [claim_b]])
    ), patch.object(
        eval_harness,
        "align_claims",
        new=AsyncMock(return_value=AlignmentResult(aligned=[pair], unaligned_a=[], unaligned_b=[])),
    ), patch.object(
        eval_harness, "classify_compatibility", new=AsyncMock(return_value=ClaimState.CONFLICTING)
    ), patch.object(
        council, "build_ranked_diff_items", wraps=council.build_ranked_diff_items
    ) as mock_build_items, patch.object(
        council,
        "synthesize_claim_diff_chairman",
        new=AsyncMock(return_value=_fake_stage_result("chairman output")),
    ), patch.object(
        eval_harness, "run_challenge_round", new=AsyncMock(return_value=[])
    ), patch.object(
        eval_harness, "check_claim_retained", new=AsyncMock(return_value=True)
    ):
        result = await eval_harness.run_new_mechanism(case)

    call_kwargs = mock_build_items.call_args.kwargs
    assert call_kwargs["conflicting"] == [pair]
    assert call_kwargs["unconfirmed_a"] == []
    assert call_kwargs["unconfirmed_b"] == []
    assert result["retained"] is True
    assert result["mechanism"] == "claim_diff"


async def test_run_new_mechanism_agreed_pair_not_passed_as_conflicting():
    case = _sample_case()
    claim_a = Claim(text="shared finding", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="shared finding", claim_type=ClaimType.ASSERTION)
    pair = ClaimPair(claim_a=claim_a, claim_b=claim_b)

    with patch.object(
        eval_harness, "extract_claims", new=AsyncMock(side_effect=[[claim_a], [claim_b]])
    ), patch.object(
        eval_harness,
        "align_claims",
        new=AsyncMock(return_value=AlignmentResult(aligned=[pair], unaligned_a=[], unaligned_b=[])),
    ), patch.object(
        eval_harness, "classify_compatibility", new=AsyncMock(return_value=ClaimState.AGREED)
    ), patch.object(
        council, "build_ranked_diff_items", wraps=council.build_ranked_diff_items
    ) as mock_build_items, patch.object(
        council,
        "synthesize_claim_diff_chairman",
        new=AsyncMock(return_value=_fake_stage_result("chairman output")),
    ) as mock_chairman, patch.object(
        eval_harness, "check_claim_retained", new=AsyncMock(return_value=False)
    ):
        await eval_harness.run_new_mechanism(case)

    assert mock_build_items.call_args.kwargs["conflicting"] == []
    assert mock_chairman.call_args.args[1] == [pair]


async def _run_with_one_conflict(challenge_mock):
    claim_a = Claim(text="drug A and drug B interact", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="no interaction found", claim_type=ClaimType.ASSERTION)
    pair = ClaimPair(claim_a=claim_a, claim_b=claim_b)
    with patch.object(
        eval_harness, "extract_claims", new=AsyncMock(side_effect=[[claim_a], [claim_b]])
    ), patch.object(
        eval_harness,
        "align_claims",
        new=AsyncMock(return_value=AlignmentResult(aligned=[pair], unaligned_a=[], unaligned_b=[])),
    ), patch.object(
        eval_harness, "classify_compatibility", new=AsyncMock(return_value=ClaimState.CONFLICTING)
    ), patch.object(
        council, "build_ranked_diff_items", wraps=council.build_ranked_diff_items
    ) as mock_build_items, patch.object(
        council,
        "synthesize_claim_diff_chairman",
        new=AsyncMock(return_value=_fake_stage_result("chairman output")),
    ) as mock_chairman, patch.object(
        eval_harness, "run_challenge_round", new=challenge_mock
    ), patch.object(
        eval_harness, "check_claim_retained", new=AsyncMock(return_value=True)
    ):
        await eval_harness.run_new_mechanism(_sample_case())
    return pair, mock_build_items, mock_chairman


async def test_run_new_mechanism_applies_challenge_outcomes():
    claim_a = Claim(text="drug A and drug B interact", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="no interaction found", claim_type=ClaimType.ASSERTION)
    probe = ClaimPair(claim_a=claim_a, claim_b=claim_b)
    item = select_challenge_items([probe], [], [], []).selected[0]

    async def fake_round(items, question, model_a, model_b, record_context=None):
        outcome = ChallengeOutcome(
            item=items[0],
            final_a="they interact; monitor",
            final_b="they interact; monitor",
            status=ChallengeStatus.RESOLVED,
        )
        return [outcome]

    mock_round = AsyncMock(side_effect=fake_round)
    pair, mock_build_items, mock_chairman = await _run_with_one_conflict(mock_round)

    assert item.source == "conflicting"
    args = mock_round.call_args
    assert args.args[1] == "is this drug combination safe?"
    assert args.args[2] == COUNCIL_MODELS[0]
    assert args.args[3] == COUNCIL_MODELS[1]
    assert args.kwargs["record_context"] is None
    assert mock_build_items.call_args.kwargs["conflicting"] == []
    agreed = mock_chairman.call_args.args[1]
    assert [p.claim_a.text for p in agreed] == ["they interact; monitor"]
    outcomes = mock_chairman.call_args.kwargs["challenge_outcomes"]
    assert outcomes and outcomes[0].status == ChallengeStatus.RESOLVED


async def test_run_new_mechanism_makes_no_challenge_call_when_nothing_selected():
    claim_a = Claim(text="shared finding", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="shared finding", claim_type=ClaimType.ASSERTION)
    pair = ClaimPair(claim_a=claim_a, claim_b=claim_b)
    mock_round = AsyncMock(return_value=[])
    with patch.object(
        eval_harness, "extract_claims", new=AsyncMock(side_effect=[[claim_a], [claim_b]])
    ), patch.object(
        eval_harness,
        "align_claims",
        new=AsyncMock(return_value=AlignmentResult(aligned=[pair], unaligned_a=[], unaligned_b=[])),
    ), patch.object(
        eval_harness, "classify_compatibility", new=AsyncMock(return_value=ClaimState.AGREED)
    ), patch.object(
        council,
        "synthesize_claim_diff_chairman",
        new=AsyncMock(return_value=_fake_stage_result("chairman output")),
    ) as mock_chairman, patch.object(
        eval_harness, "run_challenge_round", new=mock_round
    ), patch.object(
        eval_harness, "check_claim_retained", new=AsyncMock(return_value=True)
    ):
        await eval_harness.run_new_mechanism(_sample_case())

    mock_round.assert_not_called()
    assert mock_chairman.call_args.kwargs["challenge_outcomes"] is None


async def test_run_new_mechanism_challenge_failure_leaves_lists_untouched():
    mock_round = AsyncMock(side_effect=RuntimeError("boom"))
    pair, mock_build_items, mock_chairman = await _run_with_one_conflict(mock_round)

    assert mock_build_items.call_args.kwargs["conflicting"] == [pair]
    assert mock_chairman.call_args.args[1] == []
    assert mock_chairman.call_args.kwargs["challenge_outcomes"] is None


async def test_run_eval_harness_aggregates_counts_not_percentages():
    case1 = _sample_case(id="case-1")
    case2 = _sample_case(id="case-2")

    async def fake_new_mechanism(case):
        return {"mechanism": "claim_diff", "retained": case["id"] == "case-1", "chairman_output": "x"}

    with patch.object(
        eval_harness, "run_new_mechanism", new=AsyncMock(side_effect=fake_new_mechanism)
    ):
        report = await eval_harness.run_eval_harness([case1, case2])

    assert report["total_cases"] == 2
    assert report["new_mechanism_retained_count"] == 1
    assert len(report["cases"]) == 2


def test_load_eval_cases_reads_and_combines_json_files(tmp_path):
    file_a = tmp_path / "a.json"
    file_b = tmp_path / "b.json"
    file_a.write_text(json.dumps([_sample_case(id="a1")]))
    file_b.write_text(json.dumps([_sample_case(id="b1"), _sample_case(id="b2")]))

    cases = eval_harness.load_eval_cases(file_a, file_b)

    assert [c["id"] for c in cases] == ["a1", "b1", "b2"]
