"""Tests for the /api/second-opinion/analyze endpoint (backend/main.py)."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend import main
from backend.claim_diff import AlignmentResult, Claim, ClaimPair, ClaimState, ClaimType
from backend.council import ChairmanSummary, ObservationGroup
from backend.config import COUNCIL_MODELS

client = TestClient(main.app)


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
    assert body["counts"] == {"agreed": 0, "conflicting": 0, "unconfirmed": 2}
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
    assert body["counts"] == {"agreed": 1, "conflicting": 0, "unconfirmed": 0}


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


def test_ui_route_serves_html_file():
    response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Second Opinion" in response.content
