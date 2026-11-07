"""Tests for backend/structured_diff.py."""

import json
from enum import Enum

from backend import structured_diff as sd
from backend.structured_diff import (
    CauseSystem,
    Confidence,
    MedAction,
    Severity,
    StructuredAssessment,
    StructuredDisagreementKind as K,
    UrgencyLevel,
    compare_structured,
    split_structured_block,
)

BLOCK = {
    "possible_causes": [
        {"name": "migraine", "system": "neurological", "rank": 1, "confidence": "medium"}
    ],
    "interaction_flags": [
        {"drugs": ["a", "b"], "concern": "bleeding", "severity": "high"}
    ],
    "medication_changes": [{"drug": "a", "action": "stop", "reason": "bleeding"}],
    "urgency": {"level": "same_day", "escalate_if": ["worsening"]},
}


def _wrap(obj, fence=False):
    body = json.dumps(obj)
    if fence:
        body = "```json\n" + body + "\n```"
    return "<structured>\n" + body + "\n</structured>"


def test_split_returns_free_text_without_block():
    text, _ = split_structured_block("Some prose.\n" + _wrap(BLOCK))
    assert "Some prose." in text
    assert "<structured>" not in text


def test_split_parses_valid_block():
    _, block = split_structured_block("Prose " + _wrap(BLOCK))
    assert block.urgency.level == UrgencyLevel.SAME_DAY
    assert len(block.possible_causes) == 1
    assert block.possible_causes[0].system == CauseSystem.NEUROLOGICAL


def test_split_tolerates_json_fence_inside_tags():
    _, block = split_structured_block("Prose " + _wrap(BLOCK, fence=True))
    assert block is not None
    assert block.medication_changes[0].action == MedAction.STOP


def test_split_missing_block_returns_none_and_full_text():
    text, block = split_structured_block("Just prose.")
    assert block is None
    assert text == "Just prose."


def test_split_malformed_json_returns_none_and_strips_block():
    text, block = split_structured_block("Prose <structured>{not json</structured>")
    assert block is None
    assert "<structured>" not in text
    assert "Prose" in text


def test_split_unknown_enum_value_returns_none():
    bad = json.loads(json.dumps(BLOCK))
    bad["urgency"]["level"] = "whenever"
    _, block = split_structured_block("Prose " + _wrap(bad))
    assert block is None


def test_split_uses_first_block_when_repeated():
    second = json.loads(json.dumps(BLOCK))
    second["urgency"]["level"] = "routine"
    _, block = split_structured_block("P " + _wrap(BLOCK) + " mid " + _wrap(second))
    assert block.urgency.level == UrgencyLevel.SAME_DAY


def test_split_missing_closing_tag_counts_as_no_block():
    text, block = split_structured_block("Prose <structured>{}")
    assert block is None
    assert "Prose" in text


def test_urgency_rank_orders_emergency_first():
    assert sd.urgency_rank(UrgencyLevel.EMERGENCY) == 0
    assert sd.urgency_rank(UrgencyLevel.ROUTINE) == 3


def test_instruction_names_every_enum_value():
    for enum_cls in (CauseSystem, Confidence, Severity, MedAction, UrgencyLevel):
        assert issubclass(enum_cls, Enum)
        for member in enum_cls:
            assert member.value in sd.STRUCTURED_BLOCK_INSTRUCTION


def _sa(causes=(), flags=(), meds=(), urgency="routine"):
    return StructuredAssessment.model_validate(
        {
            "possible_causes": [
                {"name": n, "system": sy, "rank": r, "confidence": c}
                for n, sy, r, c in causes
            ],
            "interaction_flags": [
                {"drugs": d, "concern": "x", "severity": sv} for d, sv in flags
            ],
            "medication_changes": [
                {"drug": d, "action": a, "reason": "r"} for d, a in meds
            ],
            "urgency": {"level": urgency},
        }
    )


def _kinds(result, kind):
    return [d for d in result if d.kind == kind]


def test_top_cause_differs():
    a = _sa(causes=[("migraine", "neurological", 1, "high")])
    b = _sa(causes=[("drug effect", "medication", 1, "high")])
    assert len(_kinds(compare_structured(a, b), K.TOP_CAUSE)) == 1


def test_same_top_cause_system_no_top_cause_disagreement():
    a = _sa(causes=[("migraine", "neurological", 1, "high")])
    b = _sa(causes=[("tension headache", "neurological", 1, "high")])
    assert _kinds(compare_structured(a, b), K.TOP_CAUSE) == []


def test_cause_system_raised_by_one_model_only():
    a = _sa(
        causes=[
            ("migraine", "neurological", 1, "high"),
            ("angina", "cardiac", 2, "low"),
        ]
    )
    b = _sa(causes=[("migraine", "neurological", 1, "high")])
    found = _kinds(compare_structured(a, b), K.ONE_SIDED_CAUSE)
    assert len(found) == 1
    assert "not raised" in found[0].detail_b


def test_urgency_mismatch_same_day_vs_within_days():
    a = _sa(urgency="same_day", causes=[("m", "neurological", 1, "high")])
    b = _sa(urgency="within_days", causes=[("m", "medication", 1, "high")])
    result = compare_structured(a, b)
    assert result[0].kind == K.URGENCY
    assert "same_day" in result[0].detail_a


def test_interaction_flag_severity_differs_shows_higher_first():
    a = _sa(flags=[(["Warfarin", "Aspirin"], "moderate")])
    b = _sa(flags=[(["aspirin", "warfarin"], "high")])
    found = _kinds(compare_structured(a, b), K.INTERACTION_FLAG)
    assert len(found) == 1
    assert "high" in found[0].summary
    assert found[0].summary.index("high") < found[0].summary.index("moderate")


def test_interaction_flag_in_only_one_block():
    a = _sa(flags=[(["a", "b"], "high")])
    b = _sa()
    assert len(_kinds(compare_structured(a, b), K.INTERACTION_FLAG)) == 1


def test_medication_change_differs():
    a = _sa(meds=[("Ibuprofen", "avoid")])
    b = _sa(meds=[("ibuprofen", "continue")])
    assert len(_kinds(compare_structured(a, b), K.MEDICATION_CHANGE)) == 1


def test_agreeing_blocks_produce_no_disagreements():
    a = _sa(
        causes=[("m", "neurological", 1, "high")],
        flags=[(["a", "b"], "high")],
        meds=[("a", "stop")],
        urgency="same_day",
    )
    assert compare_structured(a, a.model_copy(deep=True)) == []


def test_confidence_difference_alone_is_not_a_disagreement():
    a = _sa(causes=[("m", "neurological", 1, "high")])
    b = _sa(causes=[("m", "neurological", 1, "low")])
    assert compare_structured(a, b) == []


def test_none_side_returns_empty():
    a = _sa()
    assert compare_structured(None, a) == []
    assert compare_structured(a, None) == []
    assert compare_structured(None, None) == []


def test_results_sorted_urgency_flags_meds_causes():
    a = _sa(
        causes=[("m", "neurological", 1, "high")],
        flags=[(["a", "b"], "high")],
        meds=[("a", "avoid")],
        urgency="emergency",
    )
    b = _sa(
        causes=[("d", "medication", 1, "high")],
        meds=[("a", "continue")],
        urgency="routine",
    )
    kinds = [d.kind for d in compare_structured(a, b)]
    assert kinds == [
        K.URGENCY,
        K.INTERACTION_FLAG,
        K.MEDICATION_CHANGE,
        K.TOP_CAUSE,
        K.ONE_SIDED_CAUSE,
        K.ONE_SIDED_CAUSE,
    ]
