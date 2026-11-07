"""Tests for backend/structured_diff.py."""

import json
from enum import Enum

from backend import structured_diff as sd
from backend.structured_diff import (
    CauseSystem,
    Confidence,
    MedAction,
    Severity,
    UrgencyLevel,
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
