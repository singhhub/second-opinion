"""Tests for backend/council.py: stage1 degraded-run handling (T8) and the
claim-diff chairman synthesis.

Design doc's Degraded-run safety rule: stage1_collect_responses used to
filter `if response is not None`, silently dropping a failed model call
from everything downstream — the exact error-path-to-false-unanimity the
rule exists to prevent. It must now return one entry per *configured*
model with an explicit ok/failed status.
"""

import json
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
    ), patch.object(council, "query_model", new=AsyncMock(return_value=None)):
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
    ), patch.object(council, "query_model", new=AsyncMock(return_value=None)):
        results = await council.stage1_collect_responses("a question")

    assert len(results) == len(COUNCIL_MODELS)
    by_model = {r["model"]: r for r in results}
    assert by_model[failing_model]["status"] == "failed"
    assert by_model[failing_model]["response"] is None


async def test_stage1_all_models_fail_returns_all_failed_not_empty():
    responses = {model: None for model in COUNCIL_MODELS}
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ), patch.object(council, "query_model", new=AsyncMock(return_value=None)):
        results = await council.stage1_collect_responses("a question")

    assert len(results) == len(COUNCIL_MODELS)
    assert all(r["status"] == "failed" for r in results)


_BLOCK_JSON = json.dumps({
    "possible_causes": [
        {"name": "viral URI", "system": "respiratory", "rank": 1, "confidence": "high"}
    ],
    "interaction_flags": [],
    "medication_changes": [],
    "urgency": {"level": "routine", "escalate_if": []},
})
_BLOCK = f"<structured>{_BLOCK_JSON}</structured>"


async def test_stage1_strips_block_from_response_and_parses_it():
    responses = {
        m: _fake_response(f"free text {m}\n{_BLOCK}") for m in COUNCIL_MODELS
    }
    follow = AsyncMock()
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ), patch.object(council, "query_model", new=follow):
        results = await council.stage1_collect_responses("a question")

    for r in results:
        assert r["status"] == "ok"
        assert r["response"] == f"free text {r['model']}"
        assert r["structured"] is not None
        assert r["structured"].possible_causes[0].name == "viral URI"
    follow.assert_not_called()


async def test_stage1_sends_instruction_in_user_message():
    responses = {m: _fake_response(f"a\n{_BLOCK}") for m in COUNCIL_MODELS}
    mock_parallel = AsyncMock(return_value=responses)
    with patch.object(council, "query_models_parallel", new=mock_parallel):
        await council.stage1_collect_responses("a question")

    messages = mock_parallel.call_args[0][1]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("a question")
    assert council.STRUCTURED_BLOCK_INSTRUCTION in messages[0]["content"]


async def test_stage1_repairs_missing_block_with_one_followup_call():
    repaired = COUNCIL_MODELS[0]
    responses = {m: _fake_response(f"a\n{_BLOCK}") for m in COUNCIL_MODELS}
    responses[repaired] = _fake_response("answer without block")
    follow = AsyncMock(return_value=_fake_response(_BLOCK))
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ), patch.object(council, "query_model", new=follow):
        results = await council.stage1_collect_responses("a question")

    assert follow.await_count == 1
    assert follow.call_args[0][0] == repaired
    by_model = {r["model"]: r for r in results}
    assert by_model[repaired]["structured"] is not None
    assert by_model[repaired]["response"] == "answer without block"


async def test_stage1_followup_failure_leaves_structured_none_and_run_continues():
    responses = {m: _fake_response("answer without block") for m in COUNCIL_MODELS}
    follow = AsyncMock(return_value=None)
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ), patch.object(council, "query_model", new=follow):
        results = await council.stage1_collect_responses("a question")

    assert len(results) == len(COUNCIL_MODELS)
    assert all(r["status"] == "ok" and r["structured"] is None for r in results)

    follow_bad = AsyncMock(return_value=_fake_response("still no block"))
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ), patch.object(council, "query_model", new=follow_bad):
        results = await council.stage1_collect_responses("a question")
    assert all(r["structured"] is None for r in results)


async def test_stage1_failed_model_still_listed_with_structured_none():
    failing = COUNCIL_MODELS[0]
    responses = {m: _fake_response(f"a\n{_BLOCK}") for m in COUNCIL_MODELS}
    responses[failing] = None
    follow = AsyncMock()
    with patch.object(
        council, "query_models_parallel", new=AsyncMock(return_value=responses)
    ), patch.object(council, "query_model", new=follow):
        results = await council.stage1_collect_responses("a question")

    by_model = {r["model"]: r for r in results}
    assert len(results) == len(COUNCIL_MODELS)
    assert by_model[failing] == {
        "model": failing, "status": "failed", "response": None, "structured": None,
    }
    follow.assert_not_called()


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


async def test_chairman_system_prompt_includes_never_emergency_rule():
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response('{"agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": []}'))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("is this an emergency?", [], [])

    sent_system_prompt = mock_query.call_args.args[1][0]["content"]
    assert "NEVER" in sent_system_prompt
    assert "emergency" in sent_system_prompt.lower()


async def test_chairman_system_prompt_distinguishes_own_verdict_from_reporting_a_source_recommendation():
    """
    Live run (2026-09-14) found the chairman omitting a real disagreement
    entirely: one model explicitly recommended immediate ER/911 care
    (correctly ranked near the top of its UNCONFIRMED input), the other
    didn't, and the chairman's summary never mentioned either fact - likely
    over-applying the never-emergency-verdict rule to reporting what a
    SOURCE model said, not just to issuing its own verdict. The rule must
    explicitly distinguish the two.
    """
    with patch.object(
        council,
        "query_model",
        new=AsyncMock(return_value=_fake_response('{"agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": []}')),
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("is this an emergency?", [], [])

    sent_system_prompt = mock_query.call_args.args[1][0]["content"]
    # The rule must clearly separate "your own voice" from "what a source model said".
    assert "own voice" in sent_system_prompt.lower() or "own verdict" in sent_system_prompt.lower()
    # It must instruct naming it, not omitting it, when only one model recommended emergency care.
    assert "did not make that recommendation" in sent_system_prompt.lower() or "name it explicitly" in sent_system_prompt.lower()


async def test_chairman_valid_json_parses_into_structured_summary():
    valid_json = json.dumps({
        "agreed_findings": "both models agree on X",
        "disagreement_summary": "they disagree on Y",
        "observations": [{"label": "check this", "items": ["do a", "do b"]}],
        "questions_for_doctor": ["ask the doctor this"],
    })
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        result = await council.synthesize_claim_diff_chairman("a question", [], [])

    assert mock_query.await_count == 1
    assert result["structured"].agreed_findings == "both models agree on X"
    assert result["structured"].observations[0].label == "check this"
    assert "ask the doctor this" in result["response"]


async def test_chairman_malformed_json_retries_then_falls_back_safely():
    valid_json = json.dumps({
        "agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": [],
    })
    responses = [_fake_response("not valid json"), _fake_response(valid_json)]
    with patch.object(
        council, "query_model", new=AsyncMock(side_effect=responses)
    ) as mock_query:
        result = await council.synthesize_claim_diff_chairman("a question", [], [])

    assert mock_query.await_count == 2
    assert result["structured"].agreed_findings == ""


async def test_chairman_permanently_malformed_json_falls_back_without_raising():
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response("still not json"))
    ) as mock_query:
        result = await council.synthesize_claim_diff_chairman("a question", [], [])

    assert mock_query.await_count == 2
    assert result["structured"].agreed_findings == ""
    assert isinstance(result["response"], str)


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
    valid_json = json.dumps({
        "agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": [],
    })
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("a question", [agreed_pair], items)

    system_prompt = mock_query.call_args.args[1][0]["content"]
    user_message = mock_query.call_args.args[1][1]["content"]
    for real_model in COUNCIL_MODELS:
        assert real_model not in system_prompt
        assert real_model not in user_message
    assert "Model A" in user_message or "Model B" in user_message


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
    valid_json = json.dumps({
        "agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": [],
    })
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("a question", [], items)

    user_message = mock_query.call_args.args[1][1]["content"]
    assert "claude claim text" in user_message
    assert "gemini claim text" in user_message
    assert "unconfirmed claim text" in user_message


async def test_chairman_query_failure_returns_safe_fallback_without_raising():
    with patch.object(council, "query_model", new=AsyncMock(return_value=None)) as mock_query:
        result = await council.synthesize_claim_diff_chairman("a question", [], [])

    assert mock_query.await_count == 2
    assert result["model"] == council.CHAIRMAN_MODEL
    assert result["structured"].agreed_findings == ""
    assert isinstance(result["response"], str)


def test_chairman_summary_models_accept_full_shape():
    from backend.council import ChairmanSummary, ObservationGroup

    summary = ChairmanSummary(
        agreed_findings="both agree on X",
        disagreement_summary="they disagree on Y",
        observations=[ObservationGroup(label="check this", items=["do a", "do b"])],
        questions_for_doctor=["ask this"],
    )

    assert summary.agreed_findings == "both agree on X"
    assert summary.observations[0].items == ["do a", "do b"]


def test_chairman_summary_models_default_empty_lists():
    from backend.council import ChairmanSummary

    summary = ChairmanSummary(agreed_findings="", disagreement_summary="")

    assert summary.observations == []
    assert summary.questions_for_doctor == []


def _challenge_outcome(status, flags=None):
    from backend.challenge import ChallengeItem, ChallengeOutcome
    from backend.claim_diff import ActionabilityCategory
    item = ChallengeItem(
        item_id="c0", source="conflicting",
        category=ActionabilityCategory.DIAGNOSTIC_SUGGESTION,
        text_a="ORIGINAL POSITION ALPHA", text_b="ORIGINAL POSITION BETA",
    )
    return ChallengeOutcome(
        item=item, final_a="FINAL ALPHA", final_b="FINAL BETA",
        status=status, flags=flags or [],
    )


_EMPTY_SUMMARY = json.dumps({
    "agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": [],
})


async def test_chairman_message_includes_challenge_section_when_outcomes_given():
    from backend.challenge import ChallengeStatus
    outcome = _challenge_outcome(ChallengeStatus.RESOLVED, ["mutual_concession"])
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response(_EMPTY_SUMMARY))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman(
            "q", [], [], challenge_outcomes=[outcome]
        )
    system_prompt = mock_query.call_args.args[1][0]["content"]
    user_message = mock_query.call_args.args[1][1]["content"]
    assert "CHALLENGE ROUND RESULTS" in user_message
    for text in ("ORIGINAL POSITION ALPHA", "ORIGINAL POSITION BETA", "FINAL ALPHA", "FINAL BETA"):
        assert text in user_message
    assert "resolved" in user_message and "mutual_concession" in user_message
    assert council.CHAIRMAN_NEVER_EMERGENCY_RULE in system_prompt
    assert council.CHALLENGE_SECTION_NOTE in system_prompt


async def test_chairman_message_unchanged_when_no_outcomes():
    captured = []
    for kwargs in ({}, {"challenge_outcomes": None}, {"challenge_outcomes": []}):
        with patch.object(
            council, "query_model", new=AsyncMock(return_value=_fake_response(_EMPTY_SUMMARY))
        ) as mock_query:
            await council.synthesize_claim_diff_chairman("q", [], [], **kwargs)
        captured.append(mock_query.call_args.args[1])
    assert captured[0] == captured[1] == captured[2]
    assert captured[0][0]["content"] == council.CHAIRMAN_SYSTEM_PROMPT
    assert "CHALLENGE" not in captured[0][1]["content"]


# ---- Final-review fixes: structured disagreements, settled tagging, categories ----

def _sd_urgency():
    from backend.structured_diff import StructuredDisagreement, StructuredDisagreementKind
    return StructuredDisagreement(
        kind=StructuredDisagreementKind.URGENCY,
        summary="Urgency differs: emergency vs within_days",
        detail_a="Urgency: emergency", detail_b="Urgency: within_days", sort_key=0,
    )


def _sd_top_cause():
    from backend.structured_diff import StructuredDisagreement, StructuredDisagreementKind
    return StructuredDisagreement(
        kind=StructuredDisagreementKind.TOP_CAUSE,
        summary="Top cause differs: cardiac vs neurological",
        detail_a="angina (cardiac)", detail_b="stroke (neurological)", sort_key=300,
    )


async def _chairman_messages(*args, **kwargs):
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response(_EMPTY_SUMMARY))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman(*args, **kwargs)
    return mock_query.call_args.args[1]


async def test_chairman_gets_structured_section_and_never_says_fully_agreed():
    msgs = await _chairman_messages(
        "q", [], [], structured_disagreements=[_sd_urgency(), _sd_top_cause()]
    )
    system_prompt, user_message = msgs[0]["content"], msgs[1]["content"]
    assert "STRUCTURED COMPARISON DISAGREEMENTS" in user_message
    assert "Urgency differs: emergency vs within_days" in user_message
    assert "Urgency: emergency" in user_message and "Urgency: within_days" in user_message
    assert "Top cause differs: cardiac vs neurological" in user_message
    assert "fully agreed" not in user_message
    # urgency row tagged, top-cause row not
    urgency_line = [l for l in user_message.splitlines() if "Urgency differs" in l][0]
    cause_line = [l for l in user_message.splitlines() if "Top cause differs" in l][0]
    assert "urgency: always report" in urgency_line
    assert "urgency: always report" not in cause_line
    assert system_prompt.startswith(council.CHAIRMAN_SYSTEM_PROMPT)
    assert council.STRUCTURED_SECTION_NOTE in system_prompt
    assert council.CHAIRMAN_NEVER_EMERGENCY_RULE in system_prompt


async def test_chairman_structured_section_independent_of_challenge_outcomes():
    from backend.challenge import ChallengeStatus
    msgs = await _chairman_messages(
        "q", [], [],
        challenge_outcomes=[_challenge_outcome(ChallengeStatus.STILL_CONFLICTING)],
        structured_disagreements=[_sd_urgency()],
    )
    user_message = msgs[1]["content"]
    assert "STRUCTURED COMPARISON DISAGREEMENTS" in user_message
    assert "CHALLENGE ROUND RESULTS" in user_message
    assert council.STRUCTURED_SECTION_NOTE in msgs[0]["content"]
    assert council.CHALLENGE_SECTION_NOTE in msgs[0]["content"]


async def test_chairman_message_unchanged_when_no_structured_disagreements():
    base = await _chairman_messages("q", [], [])
    for value in (None, []):
        msgs = await _chairman_messages("q", [], [], structured_disagreements=value)
        assert msgs == base
    assert "STRUCTURED" not in base[1]["content"]


async def test_chairman_agreed_section_tags_items_settled_after_challenge():
    from backend.challenge import SettledClaimPair
    plain = ClaimPair(
        claim_a=Claim(text="keep a medication list", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="keep a medication list", claim_type=ClaimType.ASSERTION),
    )
    settled = SettledClaimPair(
        claim_a=Claim(text="likely a UTI", claim_type=ClaimType.ASSERTION),
        claim_b=Claim(text="likely a UTI", claim_type=ClaimType.ASSERTION),
        original_a="likely a UTI", original_b="likely a drug side effect",
    )
    msgs = await _chairman_messages("q", [plain, settled], [])
    user_message = msgs[1]["content"]
    lines = user_message.splitlines()
    plain_line = [l for l in lines if "keep a medication list" in l][0]
    settled_line = [l for l in lines if l.startswith("- likely a UTI")][0]
    assert plain_line == "- keep a medication list"
    assert "settled after challenge" in settled_line
    assert 'originally Model A: "likely a UTI"' in settled_line
    assert 'Model B: "likely a drug side effect"' in settled_line


async def test_chairman_challenge_section_names_category_and_tags_red_flags():
    from backend.challenge import ChallengeItem, ChallengeOutcome, ChallengeStatus
    from backend.claim_diff import ActionabilityCategory

    def outcome(cat, always_visible=False, text="T"):
        item = ChallengeItem(
            item_id="x", source="structured" if always_visible else "conflicting",
            category=cat, text_a=text + " A", text_b=text + " B",
            always_visible=always_visible,
        )
        return ChallengeOutcome(item=item, final_a="fa", final_b="fb",
                                status=ChallengeStatus.STILL_CONFLICTING)

    msgs = await _chairman_messages("q", [], [], challenge_outcomes=[
        outcome(ActionabilityCategory.RISK_OR_RED_FLAG, always_visible=True, text="URG"),
        outcome(ActionabilityCategory.RISK_OR_RED_FLAG, text="RED"),
        outcome(ActionabilityCategory.DIAGNOSTIC_SUGGESTION, text="DIAG"),
    ])
    user_message = msgs[1]["content"]
    section = user_message.split("CHALLENGE ROUND RESULTS")[1]
    entries = section.split("\n- ")[1:]
    assert len(entries) == 3
    assert "RISK_OR_RED_FLAG" in entries[0] and "urgency/red flag: always report" in entries[0]
    assert "RISK_OR_RED_FLAG" in entries[1] and "urgency/red flag: always report" in entries[1]
    assert "DIAGNOSTIC_SUGGESTION" in entries[2]
    assert "always report" not in entries[2]
