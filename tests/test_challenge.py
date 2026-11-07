from backend.challenge import select_challenge_items
from backend.claim_diff import ActionabilityCategory, Claim, ClaimPair, ClaimType
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
