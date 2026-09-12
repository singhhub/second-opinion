"""Tests for backend/council.py's stage1 degraded-run handling (T8).

Design doc's Degraded-run safety rule: stage1_collect_responses used to
filter `if response is not None`, silently dropping a failed model call
from everything downstream — the exact error-path-to-false-unanimity the
rule exists to prevent. It must now return one entry per *configured*
model with an explicit ok/failed status, and stage2/stage3/run_full_council
must never treat a failed model as silent agreement.
"""

from unittest.mock import AsyncMock, patch

from backend import council
from backend.config import COUNCIL_MODELS


def _fake_response(content: str):
    return {"content": content, "reasoning_details": None}


async def test_stage1_returns_one_entry_per_configured_model_all_ok():
    responses = {model: _fake_response(f"answer from {model}") for model in COUNCIL_MODELS}
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ):
        results = await council.stage1_collect_responses("a question")

    assert len(results) == len(COUNCIL_MODELS)
    assert {r["model"] for r in results} == set(COUNCIL_MODELS)
    assert all(r["status"] == "ok" for r in results)


async def test_stage1_keeps_failed_model_as_explicit_failed_entry():
    failing_model = COUNCIL_MODELS[0]
    responses = {model: _fake_response("ok") for model in COUNCIL_MODELS}
    responses[failing_model] = None

    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ):
        results = await council.stage1_collect_responses("a question")

    assert len(results) == len(COUNCIL_MODELS)
    by_model = {r["model"]: r for r in results}
    assert by_model[failing_model]["status"] == "failed"
    assert by_model[failing_model]["response"] is None


async def test_stage1_all_models_fail_returns_all_failed_not_empty():
    responses = {model: None for model in COUNCIL_MODELS}
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ):
        results = await council.stage1_collect_responses("a question")

    assert len(results) == len(COUNCIL_MODELS)
    assert all(r["status"] == "failed" for r in results)


async def test_stage2_only_ranks_ok_models_not_failed_ones():
    stage1_results = [
        {"model": COUNCIL_MODELS[0], "status": "ok", "response": "good answer"},
        {"model": COUNCIL_MODELS[1], "status": "failed", "response": None},
    ]
    with patch.object(
        council,
        "query_models_parallel",
        new=AsyncMock(return_value={m: _fake_response("FINAL RANKING:\n1. Response A") for m in COUNCIL_MODELS}),
    ):
        stage2_results, label_to_model = await council.stage2_collect_rankings(
            "a question", stage1_results
        )

    assert list(label_to_model.values()) == [COUNCIL_MODELS[0]]
    assert "Response A" in label_to_model


async def test_run_full_council_flags_degraded_run_when_one_model_fails():
    ok_model, failing_model = COUNCIL_MODELS[0], COUNCIL_MODELS[1]
    stage1_responses = {ok_model: _fake_response("a real answer"), failing_model: None}

    async def fake_query_models_parallel(models, messages):
        if messages[0]["content"] == "a question":
            return stage1_responses
        return {m: _fake_response("FINAL RANKING:\n1. Response A") for m in models}

    with patch.object(
        council, "query_models_parallel", new=AsyncMock(side_effect=fake_query_models_parallel)
    ), patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ):
        stage1_results, stage2_results, stage3_result, metadata = await council.run_full_council(
            "a question"
        )

    assert metadata["degraded"] is True
    assert metadata["failed_models"] == [failing_model]
    by_model = {r["model"]: r for r in stage1_results}
    assert by_model[failing_model]["status"] == "failed"
    assert by_model[ok_model]["status"] == "ok"


async def test_run_full_council_not_degraded_when_all_models_succeed():
    stage1_responses = {model: _fake_response("answer") for model in COUNCIL_MODELS}

    async def fake_query_models_parallel(models, messages):
        if messages[0]["content"] == "a question":
            return stage1_responses
        return {m: _fake_response("FINAL RANKING:\n1. Response A") for m in models}

    with patch.object(
        council, "query_models_parallel", new=AsyncMock(side_effect=fake_query_models_parallel)
    ), patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ):
        _, _, _, metadata = await council.run_full_council("a question")

    assert metadata["degraded"] is False
    assert metadata["failed_models"] == []
