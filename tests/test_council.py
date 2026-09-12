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
from backend.claim_diff import Claim, ClaimPair, ClaimState, ClaimType
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


def test_build_ranked_diff_items_merges_and_ranks_by_actionability():
    background_pair = ClaimPair(
        claim_a=Claim(text="patient has a long history of hypertension", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="patient has managed hypertension for years", claim_type=ClaimType.ASSERTION),
    )
    drug_interaction_claim = Claim(
        text="Drug A + Drug B: contraindicated interaction", claim_type=ClaimType.ASSERTION
    )

    items = council.build_ranked_diff_items(
        conflicting=[background_pair],
        unconfirmed_a=[drug_interaction_claim],
        unconfirmed_b=[],
    )

    assert len(items) == 2
    assert items[0]["state"] == ClaimState.UNCONFIRMED
    assert items[0]["claim"] == drug_interaction_claim
    assert items[1]["state"] == ClaimState.CONFLICTING


def test_build_ranked_diff_items_is_lossless():
    pair = ClaimPair(
        claim_a=Claim(text="claim a", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="claim b", claim_type=ClaimType.ASSERTION),
    )
    unconfirmed_a = [Claim(text="only A said this", claim_type=ClaimType.ASSERTION)]
    unconfirmed_b = [
        Claim(text="only B said this", claim_type=ClaimType.ASSERTION),
        Claim(text="and this too", claim_type=ClaimType.ASSERTION),
    ]

    items = council.build_ranked_diff_items([pair], unconfirmed_a, unconfirmed_b)

    assert len(items) == 1 + len(unconfirmed_a) + len(unconfirmed_b)


async def test_chairman_prompt_includes_never_emergency_rule():
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("is this an emergency?", [], [])

    sent_prompt = mock_query.call_args.args[1][0]["content"]
    assert "NEVER" in sent_prompt
    assert "emergency" in sent_prompt.lower()


async def test_chairman_empty_agreed_section_says_so_explicitly():
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("a question", [], [])

    sent_prompt = mock_query.call_args.args[1][0]["content"]
    assert "No agreed findings" in sent_prompt


async def test_chairman_empty_diff_section_says_so_explicitly():
    agreed_pair = ClaimPair(
        claim_a=Claim(text="shared finding", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="shared finding", claim_type=ClaimType.ASSERTION),
    )
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("a question", [agreed_pair], [])

    sent_prompt = mock_query.call_args.args[1][0]["content"]
    assert "No conflicting or unconfirmed claims" in sent_prompt


async def test_chairman_prompt_never_leaks_real_model_names():
    agreed_pair = ClaimPair(
        claim_a=Claim(text="shared finding", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="shared finding", claim_type=ClaimType.ASSERTION),
    )
    items = council.build_ranked_diff_items(
        conflicting=[],
        unconfirmed_a=[Claim(text="drug interaction finding", claim_type=ClaimType.ASSERTION)],
        unconfirmed_b=[],
    )
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("a question", [agreed_pair], items)

    sent_prompt = mock_query.call_args.args[1][0]["content"]
    for real_model in COUNCIL_MODELS:
        assert real_model not in sent_prompt
    assert "Model A" in sent_prompt or "Model B" in sent_prompt


async def test_chairman_diff_section_includes_conflicting_and_unconfirmed_text():
    pair = ClaimPair(
        claim_a=Claim(text="claude claim text", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="gemini claim text", claim_type=ClaimType.ASSERTION),
    )
    items = council.build_ranked_diff_items(
        conflicting=[pair],
        unconfirmed_a=[Claim(text="unconfirmed claim text", claim_type=ClaimType.ASSERTION)],
        unconfirmed_b=[],
    )
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("synthesis"))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("a question", [], items)

    sent_prompt = mock_query.call_args.args[1][0]["content"]
    assert "claude claim text" in sent_prompt
    assert "gemini claim text" in sent_prompt
    assert "unconfirmed claim text" in sent_prompt


async def test_chairman_query_failure_returns_error_response_without_raising():
    with patch.object(council, "query_model", new=AsyncMock(return_value=None)):
        result = await council.synthesize_claim_diff_chairman("a question", [], [])

    assert result["model"] == council.CHAIRMAN_MODEL
    assert "response" in result
