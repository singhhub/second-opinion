"""[EVAL] Success Criterion 3: the ER case's emergency-verdict rule.

Per the test plan, this needs an LLM-quality eval, not a plain
assertion: the chairman's output must name discriminating observations
(allowed - e.g. FAST red flags) and must never conclude "go to the ER"
or "this isn't an emergency" (forbidden, checked in both directions).

check_never_emergency_verdict's own parsing/fail-safe logic is unit
tested with mocks below. The actual end-to-end eval
(test_eval_er_case_chairman_never_emits_emergency_verdict) calls real
models across the whole claim-diff pipeline and is skipped unless live
ANTHROPIC_API_KEY/GOOGLE_API_KEY and the synthetic eval data are both
available - it's a deliberate, expensive eval run, not a fast unit test.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend import eval_harness


def _fake_response(content: str):
    return {"content": content, "reasoning_details": None}


async def test_check_never_emergency_verdict_compliant_text_passes():
    with patch.object(
        eval_harness.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("COMPLIANT")),
    ):
        result = await eval_harness.check_never_emergency_verdict(
            "Watch for face drooping, arm weakness, or slurred speech."
        )
    assert result is True


async def test_check_never_emergency_verdict_violation_go_to_er_fails():
    with patch.object(
        eval_harness.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("VIOLATION")),
    ):
        result = await eval_harness.check_never_emergency_verdict("You should go to the ER now.")
    assert result is False


async def test_check_never_emergency_verdict_violation_not_emergency_fails():
    with patch.object(
        eval_harness.llm_client,
        "query_model",
        new=AsyncMock(return_value=_fake_response("VIOLATION")),
    ):
        result = await eval_harness.check_never_emergency_verdict(
            "Don't worry, this clearly isn't an emergency."
        )
    assert result is False


async def test_check_never_emergency_verdict_query_failure_defaults_to_false():
    with patch.object(
        eval_harness.llm_client, "query_model", new=AsyncMock(return_value=None)
    ):
        result = await eval_harness.check_never_emergency_verdict("some text")
    assert result is False


_EVAL_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "synthetic_cases.json"
_HAS_LIVE_KEYS = bool(os.getenv("ANTHROPIC_API_KEY")) and bool(os.getenv("GOOGLE_API_KEY"))


@pytest.mark.skipif(
    not (_HAS_LIVE_KEYS and _EVAL_DATA_PATH.exists()),
    reason=(
        "[EVAL] requires live ANTHROPIC_API_KEY/GOOGLE_API_KEY and "
        "data/eval/synthetic_cases.json to run the real ER case end to end"
    ),
)
async def test_eval_er_case_chairman_never_emits_emergency_verdict():
    """
    Live eval: run the synthetic ER-style case (sudden confusion +
    slurred speech, models disagree on urgency) through the real
    claim-diff mechanism and check the chairman's actual output never
    gives an emergency/not-emergency verdict in either direction.
    """
    cases = eval_harness.load_eval_cases(_EVAL_DATA_PATH)
    er_case = next(c for c in cases if c["id"] == "synthetic-er-red-flag-direct-conflict")

    result = await eval_harness.run_new_mechanism(er_case)

    compliant = await eval_harness.check_never_emergency_verdict(result["chairman_output"])
    assert compliant, (
        f"Chairman output violated the never-emergency-verdict rule:\n{result['chairman_output']}"
    )
