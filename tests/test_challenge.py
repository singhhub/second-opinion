import asyncio
import json
from unittest.mock import AsyncMock, patch

from backend.challenge import (
    FLAG_CHANGED_WITHOUT_EVIDENCE,
    FLAG_MUTUAL_CONCESSION,
    FLAG_NO_RECORD_EVIDENCE,
    ChallengeStatus,
    evidence_in_record,
    run_challenge_round,
    select_challenge_items,
)
from backend.claim_diff import (
    ActionabilityCategory,
    Claim,
    ClaimPair,
    ClaimState,
    ClaimType,
)
from backend.structured_diff import StructuredDisagreement, StructuredDisagreementKind


def _claim(text):
    return Claim(text=text, claim_type=ClaimType.ASSERTION)


def _pair(a, b):
    return ClaimPair(claim_a=_claim(a), claim_b=_claim(b))


def _sd(kind, sort_key=0):
    return StructuredDisagreement(
        kind=kind, summary="s", detail_a="da", detail_b="db", sort_key=sort_key
    )


def _ids(items):
    return [i.item_id for i in items]


def test_all_conflicting_pairs_selected():
    pairs = [_pair("takes aspirin", "avoid aspirin"), _pair("rest well", "stay active")]
    sel = select_challenge_items(pairs, [], [], [])
    assert len(sel.selected) == 2
    assert all(i.source == "conflicting" and i.pair is not None for i in sel.selected)
    assert sel.deferred == []
    assert sel.cap_hit is False


def test_unconfirmed_diagnostic_claims_selected_when_unaligned():
    a = [_claim("Pain is consistent with cardiac ischemia")]
    b = [_claim("Pain is consistent with a neurological cause")]
    sel = select_challenge_items([], a, b, [])
    assert len(sel.selected) == 2
    assert {i.side for i in sel.selected} == {"A", "B"}
    assert all(i.category == ActionabilityCategory.DIAGNOSTIC_SUGGESTION for i in sel.selected)


def test_unconfirmed_background_claim_not_selected_when_others_selected():
    a = [_claim("Pain is consistent with cardiac ischemia"), _claim("Water is wet")]
    sel = select_challenge_items([], a, [], [])
    assert [i.text_a for i in sel.selected] == ["Pain is consistent with cardiac ischemia"]
    assert sel.deferred == []


def test_keywordless_diagnosis_selected_via_fallback():
    a = [_claim("this sounds like a heart problem")]
    sel = select_challenge_items([], a, [], [])
    assert len(sel.selected) == 1
    assert sel.selected[0].source == "unconfirmed"
    assert sel.selected[0].text_a == "this sounds like a heart problem"
    assert sel.selected[0].text_b is None


def test_no_items_when_nothing_differs():
    sel = select_challenge_items([], [], [], [])
    assert sel.selected == []
    assert sel.deferred == []
    assert sel.cap_hit is False


def test_urgency_structured_item_is_always_visible():
    sel = select_challenge_items(
        [], [], [],
        [
            _sd(StructuredDisagreementKind.URGENCY),
            _sd(StructuredDisagreementKind.TOP_CAUSE),
            _sd(StructuredDisagreementKind.MEDICATION_CHANGE),
        ],
    )
    by_kind = {i.item_id: i for i in sel.selected}
    urg = by_kind["s0"]
    assert urg.always_visible is True
    assert urg.category == ActionabilityCategory.RISK_OR_RED_FLAG
    assert by_kind["s1"].category == ActionabilityCategory.DIAGNOSTIC_SUGGESTION
    assert by_kind["s2"].category == ActionabilityCategory.DRUG_INTERACTION
    assert by_kind["s1"].always_visible is False
    assert urg.text_a == "da" and urg.text_b == "db"


def test_cap_applies_in_actionability_order_and_sets_cap_hit():
    pairs = [_pair(f"background note {i}", f"other note {i}") for i in range(10)]
    drug = [
        _sd(StructuredDisagreementKind.INTERACTION_FLAG),
        _sd(StructuredDisagreementKind.MEDICATION_CHANGE),
    ]
    sel = select_challenge_items(pairs, [], [], drug, max_items=10)
    assert len(sel.selected) == 10
    assert len(sel.deferred) == 2
    assert sel.cap_hit is True
    assert {"s0", "s1"} <= set(_ids(sel.selected))
    assert all(i.category == ActionabilityCategory.BACKGROUND_OR_CAVEAT for i in sel.deferred)


def test_selection_is_lossless():
    pairs = [_pair("dose too high", "dose fine"), _pair("x", "y")]
    ua = [_claim("consistent with A"), _claim("monitor weight"), _claim("trivia")]
    ub = [_claim("risk of stroke")]
    structured = [_sd(StructuredDisagreementKind.URGENCY)]
    sel = select_challenge_items(pairs, ua, ub, structured, max_items=3)
    ids = _ids(sel.selected) + _ids(sel.deferred)
    # trivia (u2) is background and not selectable
    assert sorted(ids) == sorted(["c0", "c1", "u0", "u1", "u3", "s0"])
    assert len(ids) == len(set(ids))


def test_fallback_items_count_toward_cap():
    ua = [_claim(f"vague {i}") for i in range(5)]
    sel = select_challenge_items([], ua, [], [], max_items=2)
    assert len(sel.selected) == 2
    assert len(sel.deferred) == 1  # fallback picks 3, cap keeps 2
    assert sel.cap_hit is True


# ---- Task 5: run_challenge_round ----

def _resp(stance, revised=None, quote=None):
    return {
        "content": json.dumps(
            {
                "stance": stance,
                "revised_claim": revised,
                "reasoning": "because",
                "evidence_quote": quote,
            }
        )
    }


def _items_conflict(n=1):
    sel = select_challenge_items(
        [_pair(f"takes aspirin {i}", f"avoid aspirin {i}") for i in range(n)], [], [], []
    )
    return sel.selected


def _run(items, qm, cls=ClaimState.AGREED, **kw):
    cm = AsyncMock(return_value=cls)
    with patch("backend.challenge.query_model", qm), patch(
        "backend.challenge.classify_compatibility", cm
    ):
        out = asyncio.run(run_challenge_round(items, "Q?", "mA", "mB", **kw))
    return out, cm


def test_both_concede_is_mutual_capitulation_not_resolved():
    items = _items_conflict()
    qm = AsyncMock(return_value=_resp("CONCEDE"))
    out, cm = _run(items, qm)
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert FLAG_MUTUAL_CONCESSION in out[0].flags
    assert cm.await_count == 0
    assert qm.await_count == 2
    assert out[0].item.text_a == items[0].text_a
    assert out[0].item.text_b == items[0].text_b


def test_one_sided_concede_still_resolves_and_keeps_item_texts():
    items = _items_conflict()

    def side(model, messages, *a, **k):
        return _resp("CONCEDE") if model == "mA" else _resp("MAINTAIN")

    out, cm = _run(items, AsyncMock(side_effect=side))
    assert out[0].status == ChallengeStatus.RESOLVED
    assert cm.await_count == 1
    assert out[0].final_a == items[0].text_b and out[0].final_b == items[0].text_b
    assert out[0].item.text_a == items[0].text_a
    assert out[0].item.text_b == items[0].text_b


def test_guard_flag_is_sticky_across_rounds():
    items = select_challenge_items([_pair("rest at home", "go to ER now")], [], [], []).selected
    calls = {"mA": 0}

    def side(model, messages, *a, **k):
        if model == "mB":
            return _resp("MAINTAIN")
        calls["mA"] += 1
        quote = "invented" if calls["mA"] == 1 else "Patient"
        return _resp("CONCEDE", quote=quote)

    async def honest(pair):
        return ClaimState.AGREED if pair.claim_a.text == pair.claim_b.text else ClaimState.CONFLICTING

    cm = AsyncMock(side_effect=honest)
    with patch("backend.challenge.query_model", AsyncMock(side_effect=side)), patch(
        "backend.challenge.classify_compatibility", cm
    ):
        out = asyncio.run(
            run_challenge_round(
                items, "Q?", "mA", "mB",
                record_context="Patient has chest pain.", max_rounds=2,
            )
        )
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert FLAG_CHANGED_WITHOUT_EVIDENCE in out[0].flags
    assert out[0].rounds_used == 2


def test_round_two_retries_unresolved_error_items():
    items = _items_conflict()
    state = {"n": 0}

    def side(model, messages, *a, **k):
        state["n"] += 1
        if state["n"] <= 4:  # round 1: both models, each tried twice
            return None
        return _resp("MAINTAIN")

    qm = AsyncMock(side_effect=side)
    out, _ = _run(items, qm, max_rounds=2)
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert out[0].rounds_used == 2
    assert qm.await_count == 6


def test_one_model_maintains_stays_conflicting_and_makes_no_classify_call():
    out, cm = _run(_items_conflict(), AsyncMock(return_value=_resp("MAINTAIN")))
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert cm.await_count == 0


def test_malformed_json_twice_gives_unresolved_error():
    items = _items_conflict()
    qm = AsyncMock(return_value={"content": "not json"})
    out, _ = _run(items, qm)
    assert out[0].status == ChallengeStatus.UNRESOLVED_ERROR
    assert out[0].final_a == items[0].text_a and out[0].final_b == items[0].text_b
    assert out[0].flags == []


def test_one_of_two_calls_failing_gives_unresolved_error():
    def side(model, messages, *a, **k):
        return None if model == "mB" else _resp("CONCEDE")

    out, cm = _run(_items_conflict(), AsyncMock(side_effect=side))
    assert out[0].status == ChallengeStatus.UNRESOLVED_ERROR
    assert cm.await_count == 0


def test_malformed_then_valid_retries_once():
    seq = {"mA": [{"content": "junk"}, _resp("MAINTAIN")], "mB": [_resp("MAINTAIN")]}

    def side(model, messages, *a, **k):
        return seq[model].pop(0)

    out, _ = _run(_items_conflict(), AsyncMock(side_effect=side))
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING


def test_revise_without_revised_claim_is_malformed():
    out, _ = _run(_items_conflict(), AsyncMock(return_value=_resp("REVISE")))
    assert out[0].status == ChallengeStatus.UNRESOLVED_ERROR


def test_unexpected_exception_isolated_to_item():
    items = _items_conflict(2)

    def side(model, messages, *a, **k):
        if "aspirin 0" in messages[-1]["content"]:
            raise RuntimeError("boom")
        return _resp("MAINTAIN")

    out, _ = _run(items, AsyncMock(side_effect=side))
    assert [o.status for o in out] == [
        ChallengeStatus.UNRESOLVED_ERROR,
        ChallengeStatus.STILL_CONFLICTING,
    ]


def test_changed_position_with_fabricated_quote_stays_conflicting_when_record_supplied():
    qm = AsyncMock(return_value=_resp("REVISE", revised="new view", quote="invented text"))
    out, cm = _run(_items_conflict(), qm, record_context="Patient takes warfarin daily.")
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert FLAG_CHANGED_WITHOUT_EVIDENCE in out[0].flags


def test_changed_position_with_real_quote_resolves_when_record_supplied():
    qm = AsyncMock(
        return_value=_resp("REVISE", revised="new view", quote="takes  warfarin\ndaily")
    )
    out, _ = _run(_items_conflict(), qm, record_context="Patient takes warfarin daily.")
    assert out[0].status == ChallengeStatus.RESOLVED
    assert out[0].flags == []


def test_changed_position_without_record_resolves_with_no_record_evidence_flag():
    qm = AsyncMock(return_value=_resp("REVISE", revised="new view"))
    out, _ = _run(_items_conflict(), qm)
    assert out[0].status == ChallengeStatus.RESOLVED
    assert FLAG_NO_RECORD_EVIDENCE in out[0].flags


def test_reclassified_conflicting_stays_conflicting():
    qm = AsyncMock(return_value=_resp("REVISE", revised="new view"))
    out, _ = _run(_items_conflict(), qm, cls=ClaimState.CONFLICTING, max_rounds=1)
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert out[0].flags == []


def test_evidence_in_record_accepts_whitespace_difference_rejects_paraphrase():
    rec = "Patient   takes\nwarfarin  daily."
    assert evidence_in_record("takes warfarin daily", rec)
    assert not evidence_in_record("Takes warfarin daily", rec)
    assert not evidence_in_record("is on warfarin every day", rec)
    assert not evidence_in_record(None, rec)
    assert not evidence_in_record("   ", rec)


def test_unconfirmed_item_only_asks_the_other_model():
    sel = select_challenge_items([], [_claim("risk of bleeding")], [], [])
    qm = AsyncMock(return_value=_resp("CONCEDE"))
    out, _ = _run(sel.selected, qm)
    assert qm.await_count == 1
    assert qm.await_args.args[0] == "mB"
    assert out[0].status == ChallengeStatus.RESOLVED
    assert out[0].final_a == "risk of bleeding" and out[0].final_b == "risk of bleeding"


def test_unconfirmed_maintain_is_still_conflicting_without_classify():
    sel = select_challenge_items([], [], [_claim("risk of bleeding")], [])
    qm = AsyncMock(return_value=_resp("MAINTAIN"))
    out, cm = _run(sel.selected, qm)
    assert qm.await_args.args[0] == "mA"
    assert out[0].status == ChallengeStatus.STILL_CONFLICTING
    assert out[0].final_a is None and out[0].final_b == "risk of bleeding"
    assert cm.await_count == 0


def test_rounds_default_one_and_never_exceed_two():
    qm = AsyncMock(return_value=_resp("MAINTAIN"))
    out, _ = _run(_items_conflict(), qm)
    assert qm.await_count == 2 and out[0].rounds_used == 1
    qm = AsyncMock(return_value=_resp("MAINTAIN"))
    out, _ = _run(_items_conflict(), qm, max_rounds=5)
    assert qm.await_count == 4 and out[0].rounds_used == 2


def test_round_two_skips_resolved_items():
    items = _items_conflict(2)

    def side(model, messages, *a, **k):
        if "aspirin 0" in messages[-1]["content"]:
            return _resp("CONCEDE") if model == "mA" else _resp("MAINTAIN")
        return _resp("MAINTAIN")

    qm = AsyncMock(side_effect=side)
    out, _ = _run(items, qm, max_rounds=2)
    assert out[0].status == ChallengeStatus.RESOLVED and out[0].rounds_used == 1
    assert out[1].status == ChallengeStatus.STILL_CONFLICTING and out[1].rounds_used == 2
    assert qm.await_count == 2 + 2 + 2


def test_outcomes_one_per_item_in_order():
    items = _items_conflict(3)
    out, _ = _run(items, AsyncMock(return_value=_resp("MAINTAIN")))
    assert [o.item.item_id for o in out] == [i.item_id for i in items]


def test_no_items_makes_no_calls():
    qm = AsyncMock()
    out, cm = _run([], qm)
    assert out == [] and qm.await_count == 0 and cm.await_count == 0


def test_prompt_includes_record_and_does_not_name_provider():
    qm = AsyncMock(return_value=_resp("MAINTAIN"))
    _run(_items_conflict(), qm, record_context="RECORD-XYZ")
    for call in qm.await_args_list:
        blob = json.dumps(call.args[1])
        assert "RECORD-XYZ" in blob and "Q?" in blob
        assert "mA" not in blob and "mB" not in blob


# ---------------------------------------------------------------------------
# apply_challenge_outcomes
# ---------------------------------------------------------------------------

from backend.challenge import ChallengeItem, ChallengeOutcome, apply_challenge_outcomes  # noqa: E402


def _conflict_outcome(pair, status, final_a="fa", final_b="fb"):
    item = ChallengeItem(
        item_id="c0", source="conflicting",
        category=ActionabilityCategory.DIAGNOSTIC_SUGGESTION,
        text_a=pair.claim_a.text, text_b=pair.claim_b.text, pair=pair,
    )
    return ChallengeOutcome(item=item, final_a=final_a, final_b=final_b, status=status)


def _unconfirmed_outcome(claim, side, status, final_a=None, final_b=None):
    item = ChallengeItem(
        item_id="u0", source="unconfirmed",
        category=ActionabilityCategory.DIAGNOSTIC_SUGGESTION,
        text_a=claim.text if side == "A" else None,
        text_b=claim.text if side == "B" else None,
        claim=claim, side=side,
    )
    return ChallengeOutcome(item=item, final_a=final_a, final_b=final_b, status=status)


def test_resolved_conflict_moves_to_agreed_with_final_texts():
    p1, p2 = _pair("a1", "b1"), _pair("a2", "b2")
    out = _conflict_outcome(p2, ChallengeStatus.RESOLVED, "same a", "same b")
    agreed, conflicting, ua, ub = apply_challenge_outcomes([p1, p2], [], [], [out])
    assert conflicting == [p1]
    assert len(agreed) == 1
    assert agreed[0].claim_a.text == "same a"
    assert agreed[0].claim_b.text == "same b"
    assert agreed[0].claim_a.claim_type == ClaimType.ASSERTION
    assert ua == [] and ub == []


def test_still_conflicting_item_stays_in_conflicting():
    p1 = _pair("a1", "b1")
    out = _conflict_outcome(p1, ChallengeStatus.STILL_CONFLICTING)
    agreed, conflicting, _, _ = apply_challenge_outcomes([p1], [], [], [out])
    assert agreed == [] and conflicting == [p1]


def test_resolved_urgency_structured_item_does_not_move():
    p1 = _pair("a1", "b1")
    item = ChallengeItem(
        item_id="s0", source="structured",
        category=ActionabilityCategory.RISK_OR_RED_FLAG,
        text_a="da", text_b="db", always_visible=True,
    )
    out = ChallengeOutcome(item=item, final_a="x", final_b="x", status=ChallengeStatus.RESOLVED)
    agreed, conflicting, ua, ub = apply_challenge_outcomes([p1], [], [], [out])
    assert agreed == [] and conflicting == [p1] and ua == [] and ub == []


def test_unconfirmed_resolved_leaves_unconfirmed_list_and_joins_agreed():
    c1, c2 = _claim("same text"), _claim("same text")
    out = _unconfirmed_outcome(c2, "A", ChallengeStatus.RESOLVED, "fa", "fb")
    agreed, conflicting, ua, ub = apply_challenge_outcomes([], [c1, c2], [], [out])
    assert len(ua) == 1 and ua[0] is c1  # identity, not text equality
    assert [p.claim_a.text for p in agreed] == ["fa"]
    assert [p.claim_b.text for p in agreed] == ["fb"]
    assert conflicting == [] and ub == []


def test_resolved_with_none_final_does_not_move():
    c1 = _claim("x")
    out = _unconfirmed_outcome(c1, "A", ChallengeStatus.RESOLVED, "fa", None)
    agreed, _, ua, _ = apply_challenge_outcomes([], [c1], [], [out])
    assert agreed == [] and ua == [c1]


def test_apply_is_lossless():
    p1, p2, p3 = _pair("a1", "b1"), _pair("a2", "b2"), _pair("a3", "b3")
    ca, cb = _claim("ua"), _claim("ub")
    outs = [
        _conflict_outcome(p2, ChallengeStatus.RESOLVED, "x", "y"),
        _conflict_outcome(p3, ChallengeStatus.UNRESOLVED_ERROR),
        _unconfirmed_outcome(cb, "B", ChallengeStatus.RESOLVED, "x", "y"),
    ]
    agreed, conflicting, ua, ub = apply_challenge_outcomes([p1, p2, p3], [ca], [cb], outs)
    assert len(agreed) + len(conflicting) + len(ua) + len(ub) == 5
    assert conflicting == [p1, p3] and ua == [ca] and ub == []
    assert len(agreed) == 2
