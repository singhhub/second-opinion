"""Tests for the /api/second-opinion/analyze endpoint (backend/main.py)."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend import main
from backend.claim_diff import AlignmentResult, Claim, ClaimType
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
    expected_names = {name.partition("/")[0].capitalize() for name in COUNCIL_MODELS}
    assert {m["name"] for m in body["models"]} == expected_names


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
