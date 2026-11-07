"""Tests for the /api/second-opinion/analyze endpoint (backend/main.py)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.claim_diff import AlignmentResult, Claim, ClaimPair, ClaimState, ClaimType
from backend.council import ChairmanSummary, ObservationGroup
from backend.config import COUNCIL_MODELS
from backend.challenge import ChallengeItem, ChallengeOutcome, ChallengeStatus
from backend.claim_diff import ActionabilityCategory
from backend.structured_diff import (
    StructuredAssessment,
    StructuredDisagreement,
    StructuredDisagreementKind,
    Urgency,
    UrgencyLevel,
)

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _no_network_challenge():
    with patch.object(main, "run_challenge_round", new=AsyncMock(return_value=[])) as m:
        yield m


def _stage1_ok():
    return [
        {"model": COUNCIL_MODELS[0], "status": "ok", "response": "answer A"},
        {"model": COUNCIL_MODELS[1], "status": "ok", "response": "answer B"},
    ]


def _fake_chairman_result(summary):
    return {"model": "x", "structured": summary, "response": "flattened"}


def test_analyze_returns_structured_result_on_success():
    claim_a = Claim(text="claim a", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="claim b", claim_type=ClaimType.ASSERTION)
    summary = ChairmanSummary(
        agreed_findings="",
        disagreement_summary="they disagree",
        observations=[ObservationGroup(label="check this", items=["do x"])],
        questions_for_doctor=["ask this"],
    )

    with patch.object(
        main, "stage1_collect_responses", new=AsyncMock(return_value=_stage1_ok())
    ), patch.object(
        main, "extract_claims", new=AsyncMock(side_effect=[[claim_a], [claim_b]])
    ), patch.object(
        main,
        "align_claims",
        new=AsyncMock(
            return_value=AlignmentResult(aligned=[], unaligned_a=[claim_a], unaligned_b=[claim_b])
        ),
    ), patch.object(
        main, "synthesize_claim_diff_chairman", new=AsyncMock(return_value=_fake_chairman_result(summary))
    ):
        response = client.post("/api/second-opinion/analyze", json={"question": "is this ok?"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False
    assert body["counts"] == {"agreed": 0, "conflicting": 0, "unconfirmed": 2, "resolved_after_challenge": 0}
    assert body["summary"]["disagreement_summary"] == "they disagree"
    assert body["summary"]["questions_for_doctor"] == ["ask this"]
    assert body["models"][0]["name"] == COUNCIL_MODELS[0].partition("/")[0].capitalize()
    assert body["models"][0]["answer"] == "answer A"
    assert body["models"][0]["label"] == "Model A"
    assert body["models"][1]["name"] == COUNCIL_MODELS[1].partition("/")[0].capitalize()
    assert body["models"][1]["answer"] == "answer B"
    assert body["models"][1]["label"] == "Model B"


def test_analyze_returns_degraded_when_a_model_fails():
    stage1_degraded = [
        {"model": COUNCIL_MODELS[0], "status": "ok", "response": "answer A"},
        {"model": COUNCIL_MODELS[1], "status": "failed", "response": None},
    ]
    with patch.object(main, "stage1_collect_responses", new=AsyncMock(return_value=stage1_degraded)):
        response = client.post("/api/second-opinion/analyze", json={"question": "is this ok?"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["summary"] is None
    assert body["models"] == []


def test_analyze_requires_a_question_field():
    response = client.post("/api/second-opinion/analyze", json={})

    assert response.status_code == 422


def test_analyze_counts_reflect_classify_compatibility_result():
    """
    [Finding 4] The agreed/conflicting classification branch was never
    exercised by any test, and main.classify_compatibility was never
    patched - meaning a future test fixture that supplies an aligned pair
    without remembering to patch it would make a real, billed LLM call.
    This test supplies one aligned pair and patches classify_compatibility
    to return AGREED, asserting the counts reflect that classification.
    """
    claim_a = Claim(text="claim a", claim_type=ClaimType.ASSERTION)
    claim_b = Claim(text="claim b", claim_type=ClaimType.ASSERTION)
    pair = ClaimPair(claim_a=claim_a, claim_b=claim_b)
    summary = ChairmanSummary(agreed_findings="", disagreement_summary="")

    with patch.object(
        main, "stage1_collect_responses", new=AsyncMock(return_value=_stage1_ok())
    ), patch.object(
        main, "extract_claims", new=AsyncMock(side_effect=[[claim_a], [claim_b]])
    ), patch.object(
        main,
        "align_claims",
        new=AsyncMock(
            return_value=AlignmentResult(aligned=[pair], unaligned_a=[], unaligned_b=[])
        ),
    ), patch.object(
        main, "classify_compatibility", new=AsyncMock(side_effect=[ClaimState.AGREED])
    ), patch.object(
        main, "synthesize_claim_diff_chairman", new=AsyncMock(return_value=_fake_chairman_result(summary))
    ):
        response = client.post("/api/second-opinion/analyze", json={"question": "is this ok?"})

    assert response.status_code == 200
    body = response.json()
    assert body["counts"] == {"agreed": 1, "conflicting": 0, "unconfirmed": 0, "resolved_after_challenge": 0}


def test_analyze_returns_degraded_when_pipeline_raises_mid_stream():
    """
    [Finding 5] No error handling existed around the mid-pipeline calls
    (extract_claims/align_claims/classify_compatibility/
    synthesize_claim_diff_chairman) - if any of them raised after stage1
    already succeeded, the request would 500 with a raw error instead of
    the graceful degraded response the endpoint already has a shape for.
    """
    with patch.object(
        main, "stage1_collect_responses", new=AsyncMock(return_value=_stage1_ok())
    ), patch.object(
        main, "extract_claims", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        response = client.post("/api/second-opinion/analyze", json={"question": "is this ok?"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["error"] == "Something went wrong while comparing the two answers. Please try again."


def _assessment(level=UrgencyLevel.ROUTINE):
    return StructuredAssessment(
        possible_causes=[], interaction_flags=[], medication_changes=[],
        urgency=Urgency(level=level),
    )


def _run(aligned, ua, ub, stage1=None, patches=()):
    summary = ChairmanSummary(agreed_findings="", disagreement_summary="")
    chairman = AsyncMock(return_value=_fake_chairman_result(summary))
    with patch.object(
        main, "stage1_collect_responses", new=AsyncMock(return_value=stage1 or _stage1_ok())
    ), patch.object(
        main, "extract_claims", new=AsyncMock(side_effect=[[], []])
    ), patch.object(
        main, "align_claims",
        new=AsyncMock(return_value=AlignmentResult(aligned=aligned, unaligned_a=ua, unaligned_b=ub)),
    ), patch.object(
        main, "classify_compatibility", new=AsyncMock(return_value=ClaimState.CONFLICTING)
    ), patch.object(main, "synthesize_claim_diff_chairman", new=chairman):
        for p in patches:
            p.start()
        try:
            response = client.post("/api/second-opinion/analyze", json={"question": "q"})
        finally:
            for p in patches:
                p.stop()
    return response.json(), chairman


def _pair():
    return ClaimPair(
        claim_a=Claim(text="a says x", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="b says y", claim_type=ClaimType.ASSERTION),
    )


def _outcome(pair, status, final_a="a says x", final_b="a says x"):
    item = ChallengeItem(
        item_id="c0", source="conflicting", category=ActionabilityCategory.RISK_OR_RED_FLAG,
        text_a=pair.claim_a.text, text_b=pair.claim_b.text, pair=pair,
    )
    return ChallengeOutcome(item=item, final_a=final_a, final_b=final_b, status=status, flags=["f"], rounds_used=1)


def test_analyze_resolved_conflict_counts_as_agreed_and_keeps_original_in_challenge_block(
    _no_network_challenge,
):
    pair = _pair()
    _no_network_challenge.return_value = [_outcome(pair, ChallengeStatus.RESOLVED)]
    body, chairman = _run([pair], [], [])

    assert body["counts"] == {
        "agreed": 1, "conflicting": 0, "unconfirmed": 0, "resolved_after_challenge": 1,
    }
    ch = body["challenge"]
    assert ch["ran"] is True and ch["error"] is None
    assert ch["items"][0]["original_a"] == "a says x"
    assert ch["items"][0]["original_b"] == "b says y"
    assert ch["items"][0]["final_b"] == "a says x"
    assert ch["items"][0]["id"] == "c0"
    assert ch["items"][0]["category"] == "RISK_OR_RED_FLAG"
    assert ch["items"][0]["flags"] == ["f"]
    assert chairman.call_args.kwargs["challenge_outcomes"] is not None


def test_analyze_challenge_exception_keeps_all_items_and_sets_error(_no_network_challenge):
    _no_network_challenge.side_effect = RuntimeError("boom")
    body, _ = _run([_pair()], [], [])

    assert body["degraded"] is False
    assert body["counts"]["conflicting"] == 1
    assert body["counts"]["agreed"] == 0
    assert body["counts"]["resolved_after_challenge"] == 0
    assert body["challenge"]["ran"] is False
    assert body["challenge"]["items"] == []
    assert body["challenge"]["error"]


def test_analyze_failure_while_building_challenge_block_keeps_conflict_in_diff(
    _no_network_challenge,
):
    """If building the response items raises after apply, no list may have moved."""
    pair = _pair()
    real = _outcome(pair, ChallengeStatus.RESOLVED)

    class _Flaky:
        def __getattr__(self, name):
            if name == "flags":
                raise RuntimeError("serialization boom")
            return getattr(real, name)

    _no_network_challenge.return_value = [_Flaky()]
    body, _ = _run([pair], [], [])

    assert body["degraded"] is False
    assert body["counts"]["conflicting"] == 1
    assert body["counts"]["agreed"] == 0
    assert body["counts"]["resolved_after_challenge"] == 0
    assert body["challenge"]["error"]
    assert body["challenge"]["ran"] is False


def test_analyze_no_conflicts_makes_no_challenge_call(_no_network_challenge):
    body, chairman = _run([], [], [])

    assert body["challenge"]["ran"] is False
    assert body["challenge"]["error"] is None
    _no_network_challenge.assert_not_awaited()
    assert chairman.call_args.kwargs["challenge_outcomes"] is None


def test_analyze_structured_unavailable_when_one_block_missing():
    stage1 = _stage1_ok()
    stage1[0]["structured"] = _assessment()
    stage1[1]["structured"] = None
    body, _ = _run([], [], [], stage1=stage1)

    assert body["structured"]["available"] is False
    assert body["structured"]["disagreements"] == []
    assert body["structured"]["a"]["urgency"]["level"] == UrgencyLevel.ROUTINE.value
    assert body["structured"]["b"] is None


def test_analyze_urgency_disagreement_listed_and_challenged(_no_network_challenge):
    stage1 = _stage1_ok()
    stage1[0]["structured"] = _assessment(UrgencyLevel.ROUTINE)
    stage1[1]["structured"] = _assessment(UrgencyLevel.SAME_DAY)
    sd = StructuredDisagreement(
        kind=StructuredDisagreementKind.URGENCY, summary="urgency differs",
        detail_a="routine", detail_b="urgent", sort_key=0,
    )
    body, _ = _run(
        [], [], [], stage1=stage1,
        patches=[patch.object(main, "compare_structured", return_value=[sd])],
    )

    assert body["structured"]["available"] is True
    assert body["structured"]["disagreements"] == [
        {"kind": "urgency", "summary": "urgency differs", "detail_a": "routine", "detail_b": "urgent"}
    ]
    _no_network_challenge.assert_awaited_once()
    items = _no_network_challenge.call_args.args[0]
    assert [i.source for i in items] == ["structured"]
    assert items[0].always_visible is True


def test_analyze_cap_hit_is_reported(_no_network_challenge):
    pairs = [_pair() for _ in range(40)]
    body, _ = _run(pairs, [], [])

    assert body["challenge"]["cap_hit"] is True


def test_ui_route_serves_html_file():
    response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Second Opinion" in response.content
