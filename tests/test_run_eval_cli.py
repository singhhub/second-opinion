"""Tests for the eval harness CLI runner (backend/run_eval_cli.py).

Only the pure functions (filtering, formatting, the API-key preflight
check) are unit tested here - main()/main_async() are thin argparse/IO
glue over eval_harness, which is already tested in test_eval_harness.py.
"""

import io
import os

import pytest

from backend import run_eval_cli


def _case(case_id, source="synthetic"):
    return {
        "id": case_id,
        "question": "q",
        "model_a_response": "a",
        "model_b_response": "b",
        "known_correct_claim": "claim",
        "case_source": source,
    }


def test_filter_cases_no_filter_returns_all():
    cases = [_case("one"), _case("two")]
    assert run_eval_cli.filter_cases(cases, None) == cases


def test_filter_cases_selects_matching_ids_only():
    cases = [_case("one"), _case("two"), _case("three")]
    result = run_eval_cli.filter_cases(cases, ["two", "three"])
    assert [c["id"] for c in result] == ["two", "three"]


def test_filter_cases_unknown_id_yields_empty():
    cases = [_case("one")]
    assert run_eval_cli.filter_cases(cases, ["nonexistent"]) == []


def test_format_report_includes_aggregate_counts_not_percentage():
    report = {
        "total_cases": 2,
        "new_mechanism_retained_count": 2,
        "control_retained_count": 1,
        "cases": [
            {
                "case_id": "one",
                "case_source": "synthetic",
                "new_mechanism": {"retained": True},
                "control": {"retained": True},
            },
            {
                "case_id": "two",
                "case_source": "synthetic",
                "new_mechanism": {"retained": True},
                "control": {"retained": False},
            },
        ],
    }

    text = run_eval_cli.format_report(report)

    assert "2/2" in text
    assert "1/2" in text
    assert "%" not in text
    assert "one" in text
    assert "two" in text
    assert "RETAINED" in text
    assert "lost" in text


def test_check_api_keys_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        run_eval_cli.check_api_keys()


def test_check_api_keys_passes_when_both_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-test")

    run_eval_cli.check_api_keys()  # must not raise


def test_tee_writes_to_every_stream():
    stream_a = io.StringIO()
    stream_b = io.StringIO()
    tee = run_eval_cli.Tee(stream_a, stream_b)

    tee.write("hello\n")

    assert stream_a.getvalue() == "hello\n"
    assert stream_b.getvalue() == "hello\n"


def test_tee_flush_flushes_every_stream():
    flushed = []

    class _Recorder(io.StringIO):
        def flush(self):
            flushed.append(self)
            super().flush()

    a, b = _Recorder(), _Recorder()
    run_eval_cli.Tee(a, b).flush()

    assert flushed == [a, b]
