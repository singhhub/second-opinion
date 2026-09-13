# Live Data Result Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the published "Second Opinion Result" screen design to real data — a live backend endpoint that runs the claim-diff mechanism end to end on a real question, and a plain local HTML file (no build step, no React) that fetches from it and renders the same visual design with real content instead of hardcoded text.

**Architecture:** The chairman's output moves from free-text markdown to a Pydantic-validated structured JSON object (mirroring `extract_claims`'s retry-then-fallback pattern) so the frontend can render it reliably without fragile markdown parsing. A new stateless `POST /api/second-opinion/analyze` endpoint runs `stage1_collect_responses` → `extract_claims` ×2 → `align_claims` → `classify_compatibility` per pair → `build_ranked_diff_items` → `synthesize_claim_diff_chairman`, and returns the structured result as JSON. The static HTML file is served by the same FastAPI app at `GET /ui`, so the browser page and its API calls share one origin — no CORS configuration needed, no `file://` quirks.

**Tech Stack:** FastAPI (existing), Pydantic (existing), vanilla JS `fetch()` in a single static HTML file, `fastapi.testclient.TestClient` for endpoint tests.

**Spec:** No separate spec file — requirements were established in conversation with the user on 2026-09-13 (see Global Constraints below for the exact decisions made).

## Global Constraints

- Chairman output must remain Pydantic-validated with retry-then-fallback (2 attempts, then a safe empty-but-valid fallback) — never crash on malformed LLM output, matching `claim_diff.extract_claims`'s established pattern.
- The never-emergency-verdict rule (`CHAIRMAN_NEVER_EMERGENCY_RULE`) must stay in the chairman's system prompt unchanged in spirit.
- Model identity stays anonymized as "Model A"/"Model B" in everything sent *to* the chairman LLM (the existing anonymization in `_format_agreed_section`/`_format_diff_section` is unaffected by this plan) — real names are only attached afterward, in the endpoint response, for display.
- The new endpoint is stateless: no `conversation_id`, no `storage.py` involvement. Single question in, single result out.
- The `/ui` static file route serves same-origin with the API — do not add CORS origins for this; that need is what same-origin serving eliminates.
- No new automated test may call a real LLM API — every test mocks at the same boundaries already established in `tests/test_council.py` and `tests/test_eval_harness.py`.
- Backend stays bound to `127.0.0.1` (already true in `main.py`) — do not change this.

---

## File Structure

- **Modify:** `backend/council.py` — add `ObservationGroup`/`ChairmanSummary` Pydantic models; rewrite `synthesize_claim_diff_chairman()` to request and parse structured JSON with retry-then-fallback instead of free-text markdown.
- **Modify:** `tests/test_council.py` — rewrite the 6 existing chairman tests for the new structured contract; add 2 new tests (valid-JSON parse, malformed-JSON retry-then-fallback).
- **Modify:** `backend/main.py` — add `POST /api/second-opinion/analyze` (runs the live pipeline, returns structured JSON) and `GET /ui` (serves the static HTML file).
- **Create:** `tests/test_main.py` — `TestClient`-based tests for both new routes, mocking the pipeline functions.
- **Create:** `static/second-opinion-result.html` — the real, fetch-driven UI (adapted from the already-published mockup's CSS/structure).

---

### Task 1: Structured chairman output models

**Files:**
- Modify: `backend/council.py:1-14` (imports and top-of-file constants)
- Test: `tests/test_council.py` (new tests, appended)

**Interfaces:**
- Produces: `ObservationGroup(label: str, items: List[str])`, `ChairmanSummary(agreed_findings: str, disagreement_summary: str, observations: List[ObservationGroup], questions_for_doctor: List[str])` — both importable as `from backend.council import ObservationGroup, ChairmanSummary`, consumed by Task 2 and Task 4.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_council.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_council.py -k chairman_summary_models -v`
Expected: FAIL with `ImportError: cannot import name 'ChairmanSummary' from 'backend.council'`

- [ ] **Step 3: Write minimal implementation**

In `backend/council.py`, add after the existing imports (currently lines 1-6) and before `CHAIRMAN_NEVER_EMERGENCY_RULE`:

```python
"""3-stage LLM Council orchestration."""

import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, ValidationError
from .llm_client import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
from .claim_diff import Claim, ClaimPair, ClaimState, categorize_claim


class ObservationGroup(BaseModel):
    label: str
    items: List[str] = []


class ChairmanSummary(BaseModel):
    agreed_findings: str
    disagreement_summary: str
    observations: List[ObservationGroup] = []
    questions_for_doctor: List[str] = []


CHAIRMAN_NEVER_EMERGENCY_RULE = (
    "You must NEVER state or imply whether this is or is not a medical "
    "emergency, and must NEVER tell the user to go to the ER or that it's "
    "safe not to - that call belongs to the user and their doctor, not you. "
    "You may name specific observations, symptoms, or red flags worth "
    "checking - never the verdict itself."
)
```

This replaces the current lines 1-14 of `backend/council.py` (the module docstring, imports, and `CHAIRMAN_NEVER_EMERGENCY_RULE` constant) wholesale.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_council.py -k chairman_summary_models -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/council.py tests/test_council.py
git commit -m "Add structured ChairmanSummary/ObservationGroup models"
```

---

### Task 2: Structured-JSON chairman with retry-then-fallback

**Files:**
- Modify: `backend/council.py:436-516` (replaces `_format_agreed_section`, `_format_diff_section`, and `synthesize_claim_diff_chairman` entirely; `_format_agreed_section`/`_format_diff_section` are kept, reused for building the user message)
- Modify: `tests/test_council.py:160-242` (rewrite the 6 existing chairman tests — this range ends right before `test_regression_failed_model_call_never_silently_dropped_or_treated_as_agreement` at line 245, an unrelated pre-existing regression test that must NOT be touched)

**Interfaces:**
- Consumes: `ObservationGroup`, `ChairmanSummary` (Task 1); `ClaimPair`, `ClaimState`, `categorize_claim` (existing, from `claim_diff`); `query_model` (existing, from `llm_client`)
- Produces: `synthesize_claim_diff_chairman(user_query: str, agreed: List[ClaimPair], ranked_diff_items: List[Dict[str, Any]], chairman_model: str = CHAIRMAN_MODEL) -> Dict[str, Any]` returning `{"model": str, "structured": ChairmanSummary, "response": str}` — the `"structured"` key is new and is what Task 4's endpoint uses; `"response"` is a flattened plain-text rendition kept for `eval_harness.py`'s existing `check_claim_retained`/`check_never_emergency_verdict` calls, which only need a string to judge.

- [ ] **Step 1: Write the failing tests**

Replace the entire block from `async def test_chairman_prompt_includes_never_emergency_rule():` (line 160) through the end of `test_chairman_query_failure_returns_error_response_without_raising` (line 242) in `tests/test_council.py` with:

```python
async def test_chairman_system_prompt_includes_never_emergency_rule():
    with patch.object(
        council, "query_model", new=AsyncMock(return_value=_fake_response('{"agreed_findings": "", "disagreement_summary": "", "observations": [], "questions_for_doctor": []}'))
    ) as mock_query:
        await council.synthesize_claim_diff_chairman("is this an emergency?", [], [])

    sent_system_prompt = mock_query.call_args.args[1][0]["content"]
    assert "NEVER" in sent_system_prompt
    assert "emergency" in sent_system_prompt.lower()


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
```

Also add `import json` to the top of `tests/test_council.py` (it currently has no `json` import — check with `grep -n "^import\|^from" tests/test_council.py` first; add `import json` alongside the existing `from unittest.mock import AsyncMock, patch` line if not already present).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_council.py -k "test_chairman" -v`
Expected: FAIL — most with `AttributeError`/`KeyError: 'structured'` since `synthesize_claim_diff_chairman` doesn't return that key yet, and the old prompt-format assertions (`args[1][0]` as a single combined prompt vs. a system+user split) won't match.

- [ ] **Step 3: Write minimal implementation**

Replace lines 436-516 of `backend/council.py` (from `def _format_agreed_section` through the end of `synthesize_claim_diff_chairman`) with:

```python
def _format_agreed_section(agreed: List[ClaimPair]) -> str:
    if not agreed:
        return "No agreed findings - the two models did not converge on any claim."
    return "\n".join(f"- {pair.claim_a.text}" for pair in agreed)


def _format_diff_section(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No conflicting or unconfirmed claims - the two models fully agreed."

    lines = []
    for item in items:
        if item["state"] == ClaimState.CONFLICTING:
            lines.append(
                f'- CONFLICTING: Model A says "{item["claim_a"].text}"; '
                f'Model B says "{item["claim_b"].text}"'
            )
        else:
            lines.append(f'- UNCONFIRMED (Model {item["side"]} only): {item["claim"].text}')
    return "\n".join(lines)


CHAIRMAN_SYSTEM_PROMPT = f"""{CHAIRMAN_NEVER_EMERGENCY_RULE}

You are organizing a disagreement-preserving summary for two family caregivers who already saw two AI models' anonymized answers to their question about a family member's care. The chairman organizes and explains the mechanically-computed diff; it does not re-decide what counts as a conflict.

Respond with ONLY a JSON object matching this exact shape - no markdown fences, no extra text before or after:
{{
  "agreed_findings": "<a short paragraph summarizing what both models agree on - an empty string if there are none, never omit the field>",
  "disagreement_summary": "<a short paragraph explaining the disagreement or gap in plain language - an empty string if there are no conflicting/unconfirmed items>",
  "observations": [
    {{"label": "<short label for what this group of checks is for>", "items": ["<a specific observation, symptom, or check - not which claim to believe, but what to go look at>"]}}
  ],
  "questions_for_doctor": ["<a short, specific question the caregivers should bring to the next appointment>"]
}}

If there is no disagreement, set "disagreement_summary" to an empty string and "observations" to an empty list - never omit them or leave them implicit."""


def _parse_chairman_summary(raw: Optional[Dict[str, Any]]) -> Optional[ChairmanSummary]:
    if raw is None or not raw.get("content"):
        return None
    try:
        data = json.loads(raw["content"])
        return ChairmanSummary(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _flatten_chairman_summary(summary: ChairmanSummary) -> str:
    lines = ["Agreed Findings:", summary.agreed_findings or "None."]
    lines += ["", "Conflicting / Unconfirmed:", summary.disagreement_summary or "None."]
    for group in summary.observations:
        lines.append(f"- {group.label}:")
        for item in group.items:
            lines.append(f"  - {item}")
    lines += ["", "Questions for the Doctor:"]
    for question in summary.questions_for_doctor:
        lines.append(f"- {question}")
    return "\n".join(lines)


async def synthesize_claim_diff_chairman(
    user_query: str,
    agreed: List[ClaimPair],
    ranked_diff_items: List[Dict[str, Any]],
    chairman_model: str = CHAIRMAN_MODEL,
) -> Dict[str, Any]:
    """
    Chairman synthesis for the claim-diff mechanism.

    Returns structured JSON (agreed findings / disagreement + observations /
    questions for the doctor), Pydantic-validated with retry-then-fallback -
    the same pattern as claim_diff.extract_claims, so a malformed or failed
    LLM response never crashes the caller. "structured" is the ChairmanSummary
    object for callers that render it (the API endpoint); "response" is a
    flattened plain-text rendition for callers that only need to judge text
    (eval_harness's retention/emergency-verdict checks).
    """
    agreed_text = _format_agreed_section(agreed)
    diff_text = _format_diff_section(ranked_diff_items)

    user_message = f"""Original Question: {user_query}

AGREED FINDINGS (both models agree):
{agreed_text}

CONFLICTING / UNCONFIRMED CLAIMS (ranked by actionability, most important first):
{diff_text}"""

    messages = [
        {"role": "system", "content": CHAIRMAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(2):
        raw = await query_model(chairman_model, messages)
        summary = _parse_chairman_summary(raw)
        if summary is not None:
            return {
                "model": chairman_model,
                "structured": summary,
                "response": _flatten_chairman_summary(summary),
            }

    fallback = ChairmanSummary(
        agreed_findings="",
        disagreement_summary="Unable to generate a summary for this case - please try again.",
        observations=[],
        questions_for_doctor=[],
    )
    return {
        "model": chairman_model,
        "structured": fallback,
        "response": _flatten_chairman_summary(fallback),
    }
```

Add `Optional` to the `typing` import at the top of the file (Task 1's Step 3 left it as `from typing import List, Dict, Any, Tuple` — change to `from typing import List, Dict, Any, Tuple, Optional`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_council.py -v`
Expected: PASS (all tests in the file, including the unrelated T8/T12 tests already there)

- [ ] **Step 5: Commit**

```bash
git add backend/council.py tests/test_council.py
git commit -m "Rewrite chairman synthesis as structured JSON with retry-then-fallback"
```

---

### Task 3: Verify eval_harness still works against the new chairman shape

**Files:**
- Test: `tests/test_eval_harness.py` (no source changes expected — this task is a verification checkpoint, not new code)

**Interfaces:**
- Consumes: `synthesize_claim_diff_chairman`'s new return shape (Task 2) via `chairman_result["response"]`, already how `eval_harness.run_new_mechanism` uses it — no signature change on eval_harness's side.

- [ ] **Step 1: Run the existing eval_harness test suite**

Run: `uv run pytest tests/test_eval_harness.py -v`
Expected: PASS, unchanged. `run_new_mechanism`'s mock of `council.synthesize_claim_diff_chairman` returns a full dict it controls (see `tests/test_eval_harness.py`'s `_fake_stage_result` helper), so it isn't sensitive to the real function's internal prompt format - only to the `"response"` key still being present, which Task 2 preserves.

- [ ] **Step 2: If anything fails, fix the mock shape, not eval_harness.py**

If a mock in `tests/test_eval_harness.py` constructs a chairman result dict missing `"structured"`, that's fine - `run_new_mechanism` never reads `"structured"`, only `"response"`. A failure here would mean something else regressed; investigate before editing.

- [ ] **Step 3: Commit (only if any test file needed a fix)**

```bash
git add tests/test_eval_harness.py
git commit -m "Verify eval_harness compatibility with structured chairman output"
```

---

### Task 4: `POST /api/second-opinion/analyze` endpoint

**Files:**
- Modify: `backend/main.py:1-32` (imports and app setup)
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `stage1_collect_responses`, `build_ranked_diff_items`, `synthesize_claim_diff_chairman` (from `.council`); `extract_claims`, `align_claims`, `classify_compatibility`, `ClaimState` (from `.claim_diff`) - all imported by name into `main.py` so tests can `patch.object(main, "...")`, matching the existing pattern already used for `run_full_council` etc.
- Produces: `POST /api/second-opinion/analyze` accepting `{"question": str}`, returning `{"question": str, "degraded": bool, "error": Optional[str], "models": List[{"name": str, "answer": str}], "counts": Optional[{"agreed": int, "conflicting": int, "unconfirmed": int}], "summary": Optional[{"agreed_findings": str, "disagreement_summary": str, "observations": List[{"label": str, "items": List[str]}], "questions_for_doctor": List[str]}]}` — this is what Task 6's frontend fetches.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL with `AttributeError: <module 'backend.main' ...> does not have the attribute 'stage1_collect_responses'` or a 404 (route doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

Replace lines 1-13 of `backend/main.py` (the module docstring through the `council` import) with:

```python
"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import json
import asyncio

from . import storage
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    build_ranked_diff_items,
    synthesize_claim_diff_chairman,
)
from .claim_diff import align_claims, classify_compatibility, extract_claims, ClaimState

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
```

Then, immediately after the existing `CreateConversationRequest`/`SendMessageRequest`/etc. Pydantic model definitions (search for `class SendMessageRequest` in `main.py` and add the new request model right after its closing line), add:

```python
class AnalyzeRequest(BaseModel):
    """Request to run the claim-diff mechanism on a single question."""
    question: str
```

Then, after the existing `@app.post("/api/conversations/{conversation_id}/message")` route's full function body (search for `return {` followed by `"stage1": stage1_results,` to find its end, right before the `@app.post("/api/conversations/{conversation_id}/message/stream")` route), insert:

```python
def _model_display_name(model_id: str) -> str:
    provider, _, _ = model_id.partition("/")
    return provider.capitalize()


@app.post("/api/second-opinion/analyze")
async def analyze_question(request: AnalyzeRequest):
    """
    Run the claim-diff mechanism end to end on a single question: get both
    models' raw answers, extract claims, align, classify, rank, and
    synthesize the chairman's structured summary. Stateless - no
    conversation_id, no storage. Makes real, billed calls to Claude and
    Gemini.
    """
    stage1_results = await stage1_collect_responses(request.question)
    ok_results = [r for r in stage1_results if r["status"] == "ok"]

    if len(ok_results) < 2:
        return {
            "question": request.question,
            "degraded": True,
            "error": "One or both models failed to respond. Please try again.",
            "models": [],
            "counts": None,
            "summary": None,
        }

    model_a_result, model_b_result = ok_results[0], ok_results[1]

    claims_a = await extract_claims(model_a_result["response"])
    claims_b = await extract_claims(model_b_result["response"])

    alignment = await align_claims(claims_a, claims_b)

    agreed = []
    conflicting = []
    for pair in alignment.aligned:
        state = await classify_compatibility(pair)
        if state == ClaimState.AGREED:
            agreed.append(pair)
        else:
            conflicting.append(pair)

    diff_items = build_ranked_diff_items(
        conflicting=conflicting,
        unconfirmed_a=alignment.unaligned_a,
        unconfirmed_b=alignment.unaligned_b,
    )

    chairman_result = await synthesize_claim_diff_chairman(request.question, agreed, diff_items)
    summary = chairman_result["structured"]

    return {
        "question": request.question,
        "degraded": False,
        "error": None,
        "models": [
            {"name": _model_display_name(model_a_result["model"]), "answer": model_a_result["response"]},
            {"name": _model_display_name(model_b_result["model"]), "answer": model_b_result["response"]},
        ],
        "counts": {
            "agreed": len(agreed),
            "conflicting": len(conflicting),
            "unconfirmed": len(alignment.unaligned_a) + len(alignment.unaligned_b),
        },
        "summary": {
            "agreed_findings": summary.agreed_findings,
            "disagreement_summary": summary.disagreement_summary,
            "observations": [
                {"label": group.label, "items": group.items} for group in summary.observations
            ],
            "questions_for_doctor": summary.questions_for_doctor,
        },
    }


@app.get("/ui")
async def serve_ui():
    """Serve the standalone result screen, same-origin with the API - no CORS needed."""
    return FileResponse(STATIC_DIR / "second-opinion-result.html")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_main.py -v`
Expected: `test_analyze_returns_structured_result_on_success` and `test_analyze_returns_degraded_when_a_model_fails` PASS; `test_analyze_requires_a_question_field` PASSES too. `GET /ui` will 404/500 until Task 6 creates the file - that's expected at this point, no test for it yet.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_main.py
git commit -m "Add POST /api/second-opinion/analyze endpoint"
```

---

### Task 5: `GET /ui` route test

**Files:**
- Test: `tests/test_main.py` (append)

**Interfaces:**
- Consumes: `static/second-opinion-result.html` must exist on disk (Task 6) for this test to pass - this task's test is written now but will only go green after Task 6.

- [ ] **Step 1: Write the test**

Append to `tests/test_main.py`:

```python
def test_ui_route_serves_html_file():
    response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Second Opinion" in response.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main.py -k test_ui_route -v`
Expected: FAIL (404 or file-not-found) - `static/second-opinion-result.html` doesn't exist yet.

- [ ] **Step 3: No implementation yet - this is intentionally red until Task 6**

Leave it failing. Do not create a placeholder HTML file here; Task 6 creates the real one in one step so there's no throwaway intermediate file to delete later.

- [ ] **Step 4: Commit the test in its known-red state, noting why**

```bash
git add tests/test_main.py
git commit -m "Add GET /ui route test (red until Task 6 creates the static file)"
```

---

### Task 6: The real static HTML result screen

**Files:**
- Create: `static/second-opinion-result.html`

**Interfaces:**
- Consumes: `POST /api/second-opinion/analyze`'s response shape (Task 4) via `fetch()`.

- [ ] **Step 1: Create the file**

Create `static/second-opinion-result.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Second Opinion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --page-bg:#efe9db; --paper:#f8f5ee; --surface:#ffffff; --surface-2:#efe8d9;
    --ink:#211c16; --muted:#7a7161; --faint:#a49c8b; --line:#e3dbc7; --line-strong:#d3c8ac;
    --accent:#2c5d63; --accent-ink:#183c41; --accent-soft:#dde9e8;
    --agreed:#3e7d57; --agreed-soft:#e3efe4;
    --conflicting:#a9631f; --conflicting-soft:#f5e8d7;
    --unconfirmed:#6c5e86; --unconfirmed-soft:#eae4f2;
    --danger:#a6402e; --danger-soft:#f3ded9;
    --shadow: 0 1px 2px rgba(33,28,22,.06), 0 20px 44px -20px rgba(33,28,22,.28);
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --page-bg:#0f0d0a; --paper:#1a1712; --surface:#211d17; --surface-2:#282319;
      --ink:#ece5d6; --muted:#a89d89; --faint:#6f6656; --line:#3a3226; --line-strong:#4c4230;
      --accent:#7cbcbf; --accent-ink:#cfe7e7; --accent-soft:#22322f;
      --agreed:#7ecb98; --agreed-soft:#1e2c22;
      --conflicting:#e5a868; --conflicting-soft:#332619;
      --unconfirmed:#bba8de; --unconfirmed-soft:#282236;
      --danger:#e08574; --danger-soft:#3d2621;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 20px 44px -20px rgba(0,0,0,.7);
    }
  }
  :root[data-theme="dark"]{
    --page-bg:#0f0d0a; --paper:#1a1712; --surface:#211d17; --surface-2:#282319;
    --ink:#ece5d6; --muted:#a89d89; --faint:#6f6656; --line:#3a3226; --line-strong:#4c4230;
    --accent:#7cbcbf; --accent-ink:#cfe7e7; --accent-soft:#22322f;
    --agreed:#7ecb98; --agreed-soft:#1e2c22;
    --conflicting:#e5a868; --conflicting-soft:#332619;
    --unconfirmed:#bba8de; --unconfirmed-soft:#282236;
    --danger:#e08574; --danger-soft:#3d2621;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 20px 44px -20px rgba(0,0,0,.7);
  }
  *{box-sizing:border-box;}
  body{
    background:var(--page-bg); color:var(--ink);
    font-family:"Public Sans", ui-sans-serif, system-ui, sans-serif;
    margin:0; padding-block:28px 48px; padding-inline:16px;
  }
  .frame{
    max-width:440px; margin-inline:auto; background:var(--paper);
    border-radius:22px; box-shadow:var(--shadow); overflow:hidden; border:1px solid var(--line);
  }
  .topbar{display:flex; align-items:center; gap:10px; padding:16px 20px 14px; border-bottom:1px solid var(--line);}
  .topbar .titles{flex:1; min-width:0;}
  .topbar .app-name{font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600;}
  .topbar .screen-name{font-size:1.05rem; font-weight:700; margin-top:1px;}
  .scroll{padding:18px 20px 26px; display:flex; flex-direction:column; gap:20px;}

  #form-section{display:flex; flex-direction:column; gap:10px;}
  #form-section label{font-size:.82rem; font-weight:600; color:var(--muted);}
  #question-input{
    width:100%; font-family:inherit; font-size:.95rem; padding:12px 14px;
    border:1px solid var(--line-strong); border-radius:12px; background:var(--surface); color:var(--ink);
    resize:vertical; min-height:72px;
  }
  #question-input:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
  #submit-btn{
    font-family:inherit; font-size:.92rem; font-weight:700; color:var(--paper);
    background:var(--accent); border:none; border-radius:12px; padding:12px; cursor:pointer;
  }
  #submit-btn:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
  #submit-btn:disabled{opacity:.6; cursor:not-allowed;}

  #loading-section, #error-section{text-align:center; padding:24px 8px; font-size:.9rem; color:var(--muted);}
  #error-section{color:var(--danger); background:var(--danger-soft); border-radius:12px;}

  .question-label{font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); font-weight:600; margin-bottom:6px;}
  .question-card{
    background:var(--accent-soft); color:var(--accent-ink); border-radius:16px 16px 16px 4px;
    padding:14px 16px; font-size:.98rem; line-height:1.45; font-weight:500;
  }

  .summary-strip{display:flex; gap:8px;}
  .chip{flex:1; display:flex; flex-direction:column; align-items:center; gap:2px; padding:10px 4px; border-radius:12px; border:1px solid var(--line); background:var(--surface);}
  .chip .num{font-size:1.35rem; font-weight:800; font-variant-numeric:tabular-nums; line-height:1;}
  .chip .lbl{font-size:.68rem; color:var(--muted); font-weight:600;}
  .chip.agreed .num{color:var(--agreed);}
  .chip.conflicting .num{color:var(--conflicting);}
  .chip.unconfirmed .num{color:var(--unconfirmed);}

  .section-title{display:flex; align-items:baseline; gap:8px; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:700;}
  .section-title .hint{font-size:.72rem; letter-spacing:0; text-transform:none; color:var(--faint); font-weight:500;}

  .tabs{display:flex; gap:6px; background:var(--surface-2); padding:4px; border-radius:12px;}
  .tab{flex:1; border:none; background:transparent; color:var(--muted); font-family:inherit; font-size:.86rem; font-weight:600; padding:9px 4px; border-radius:9px; cursor:pointer;}
  .tab[aria-selected="true"]{background:var(--surface); color:var(--ink); box-shadow:var(--shadow);}
  .tab:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}

  .raw-panel{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:16px; display:none;}
  .raw-panel[data-active="true"]{display:block;}
  .raw-panel .disclaimer{font-size:.76rem; color:var(--faint); margin-bottom:10px; font-style:italic;}
  .raw-panel p{font-size:.92rem; line-height:1.55; margin:0;}

  .verdict-card{border:1px solid var(--line); border-radius:14px; overflow:hidden;}
  .verdict-block{padding:16px; border-left:4px solid var(--c); background:var(--bg);}
  .verdict-block + .verdict-block{border-top:1px solid var(--line);}
  .verdict-block .vt{display:flex; align-items:center; gap:8px; font-size:.78rem; font-weight:700; letter-spacing:.03em; text-transform:uppercase; color:var(--c); margin-bottom:8px;}
  .verdict-block p{font-size:.92rem; line-height:1.55; margin:0;}

  .observe-box{background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-top:10px;}
  .observe-box .ot{font-size:.74rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin-bottom:6px;}
  .observe-box ul{margin:0; padding-left:18px;}
  .observe-box li{font-size:.88rem; line-height:1.5; margin-bottom:6px;}
  .observe-box li:last-child{margin-bottom:0;}
  .observe-box b{color:var(--ink);}

  .qlist{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px;}
  .qitem-label{display:flex; align-items:flex-start; gap:10px; cursor:pointer;}
  .qitem input{
    appearance:none; width:19px; height:19px; border:1.5px solid var(--line-strong); border-radius:5px;
    margin-top:2px; flex-shrink:0; background:var(--surface); cursor:pointer; position:relative;
  }
  .qitem input:checked{background:var(--accent); border-color:var(--accent);}
  .qitem input:checked::after{content:""; position:absolute; left:5px; top:1px; width:5px; height:10px; border:solid var(--paper); border-width:0 2px 2px 0; transform:rotate(40deg);}
  .qitem input:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
  .qitem span{font-size:.92rem; line-height:1.5;}
  .qitem input:checked ~ span{color:var(--muted); text-decoration:line-through; text-decoration-color:var(--faint);}

  .footnote{font-size:.78rem; color:var(--muted); line-height:1.5; background:var(--surface-2); border-radius:12px; padding:12px 14px;}
  .footnote b{color:var(--ink);}

  [hidden]{display:none !important;}
</style>
</head>
<body>

<div class="frame">
  <div class="topbar">
    <div class="titles">
      <div class="app-name">Second Opinion</div>
      <div class="screen-name">Ask a Question</div>
    </div>
  </div>

  <div class="scroll">

    <form id="form-section">
      <label for="question-input">What would you like a second opinion on?</label>
      <textarea id="question-input" placeholder="e.g. My father, 78, suddenly seems confused..."></textarea>
      <button type="submit" id="submit-btn">Ask both models</button>
    </form>

    <div id="loading-section" hidden>Asking Claude and Gemini, then comparing their answers - this takes a little while...</div>

    <div id="error-section" hidden></div>

    <div id="result-section" hidden style="display:flex; flex-direction:column; gap:20px;">

      <div>
        <div class="question-label">You asked</div>
        <div class="question-card" id="question-text"></div>
      </div>

      <div class="summary-strip">
        <div class="chip agreed"><span class="num" id="chip-agreed-num">0</span><span class="lbl">Agreed</span></div>
        <div class="chip conflicting"><span class="num" id="chip-conflicting-num">0</span><span class="lbl">Conflicting</span></div>
        <div class="chip unconfirmed"><span class="num" id="chip-unconfirmed-num">0</span><span class="lbl">Unconfirmed</span></div>
      </div>

      <div>
        <div class="section-title" style="margin-bottom:10px;">What each model said <span class="hint">- tap to switch</span></div>
        <div class="tabs" id="model-tabs" role="tablist" aria-label="Model raw reads"></div>
        <div id="model-panels" style="margin-top:10px;"></div>
      </div>

      <div>
        <div class="section-title" style="margin-bottom:10px;">Chairman's summary</div>
        <div class="verdict-card">
          <div class="verdict-block" style="--c:var(--agreed); --bg:var(--agreed-soft);">
            <div class="vt">Agreed findings</div>
            <p id="agreed-text"></p>
          </div>
          <div class="verdict-block" style="--c:var(--unconfirmed); --bg:var(--unconfirmed-soft);">
            <div class="vt">Conflicting / unconfirmed</div>
            <p id="disagreement-text"></p>
            <div class="observe-box" id="observe-box" hidden>
              <div class="ot">What to check</div>
              <ul id="observe-list"></ul>
            </div>
          </div>
          <div class="verdict-block" style="--c:var(--accent); --bg:var(--surface);">
            <div class="vt">Questions for the doctor</div>
            <ul class="qlist" id="questions-list"></ul>
          </div>
        </div>
      </div>

      <div class="footnote">
        <b>This isn't a diagnosis.</b> Second Opinion never tells you whether something is or isn't an emergency - it shows you where two models agree, where they don't, and what to go check. The call is yours and your doctor's.
      </div>

    </div>
  </div>
</div>

<script>
  function showState(state) {
    document.getElementById('form-section').hidden = state === 'loading' || state === 'result';
    document.getElementById('loading-section').hidden = state !== 'loading';
    document.getElementById('error-section').hidden = state !== 'error';
    document.getElementById('result-section').hidden = state !== 'result';
    document.getElementById('submit-btn').disabled = state === 'loading';
  }

  function showError(message) {
    var el = document.getElementById('error-section');
    el.textContent = message;
    showState('error');
  }

  function selectModelTab(index) {
    var tabs = document.querySelectorAll('#model-tabs .tab');
    var panels = document.querySelectorAll('#model-panels .raw-panel');
    tabs.forEach(function (tab, i) { tab.setAttribute('aria-selected', i === index); });
    panels.forEach(function (panel, i) { panel.setAttribute('data-active', i === index); });
  }

  function renderResult(data) {
    document.getElementById('question-text').textContent = data.question;

    document.getElementById('chip-agreed-num').textContent = data.counts.agreed;
    document.getElementById('chip-conflicting-num').textContent = data.counts.conflicting;
    document.getElementById('chip-unconfirmed-num').textContent = data.counts.unconfirmed;

    var tabsEl = document.getElementById('model-tabs');
    var panelsEl = document.getElementById('model-panels');
    tabsEl.innerHTML = '';
    panelsEl.innerHTML = '';

    data.models.forEach(function (model, i) {
      var tab = document.createElement('button');
      tab.type = 'button';
      tab.className = 'tab';
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
      tab.textContent = model.name;
      tab.addEventListener('click', function () { selectModelTab(i); });
      tabsEl.appendChild(tab);

      var panel = document.createElement('div');
      panel.className = 'raw-panel';
      panel.setAttribute('data-active', i === 0 ? 'true' : 'false');

      var disclaimer = document.createElement('div');
      disclaimer.className = 'disclaimer';
      disclaimer.textContent = model.name + "'s raw read - not a final answer.";

      var p = document.createElement('p');
      p.textContent = model.answer;

      panel.appendChild(disclaimer);
      panel.appendChild(p);
      panelsEl.appendChild(panel);
    });

    document.getElementById('agreed-text').textContent =
      data.summary.agreed_findings || 'No agreed findings - the two models did not converge on any shared claim.';

    document.getElementById('disagreement-text').textContent =
      data.summary.disagreement_summary || 'No conflicting or unconfirmed claims - the two models fully agreed.';

    var observeBox = document.getElementById('observe-box');
    var observeList = document.getElementById('observe-list');
    observeList.innerHTML = '';
    if (data.summary.observations.length > 0) {
      observeBox.hidden = false;
      data.summary.observations.forEach(function (group) {
        var li = document.createElement('li');
        var b = document.createElement('b');
        b.textContent = group.label + ': ';
        li.appendChild(b);
        li.appendChild(document.createTextNode(group.items.join('; ')));
        observeList.appendChild(li);
      });
    } else {
      observeBox.hidden = true;
    }

    var qlist = document.getElementById('questions-list');
    qlist.innerHTML = '';
    data.summary.questions_for_doctor.forEach(function (question, i) {
      var li = document.createElement('li');
      li.className = 'qitem';
      var label = document.createElement('label');
      label.className = 'qitem-label';
      var input = document.createElement('input');
      input.type = 'checkbox';
      input.id = 'q' + i;
      var span = document.createElement('span');
      span.textContent = question;
      label.appendChild(input);
      label.appendChild(span);
      li.appendChild(label);
      qlist.appendChild(li);
    });

    selectModelTab(0);
  }

  async function runAnalysis(event) {
    event.preventDefault();
    var question = document.getElementById('question-input').value.trim();
    if (!question) return;

    showState('loading');

    var response;
    try {
      response = await fetch('/api/second-opinion/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
      });
    } catch (err) {
      showError('Could not reach the backend. Is it running? (python -m backend.main)');
      return;
    }

    var data;
    try {
      data = await response.json();
    } catch (err) {
      showError('The backend returned something unexpected. Please try again.');
      return;
    }

    if (!response.ok) {
      showError(data.detail ? JSON.stringify(data.detail) : 'Something went wrong. Please try again.');
      return;
    }

    if (data.degraded) {
      showError(data.error || 'One or both models failed to respond. Please try again.');
      return;
    }

    renderResult(data);
    showState('result');
  }

  document.getElementById('form-section').addEventListener('submit', runAnalysis);
</script>

</body>
</html>
```

- [ ] **Step 2: Run the route test to verify it now passes**

Run: `uv run pytest tests/test_main.py -v`
Expected: PASS (all tests, including `test_ui_route_serves_html_file` from Task 5)

- [ ] **Step 3: Commit**

```bash
git add static/second-opinion-result.html
git commit -m "Add real fetch-driven static HTML result screen"
```

---

### Task 7: Manual end-to-end verification (not automated - real billed API calls)

**Files:** none (verification only)

- [ ] **Step 1: Run the full automated suite once**

Run: `uv run pytest tests/ 2>&1 | tail -15`
Expected: all tests pass except the pre-existing live-only `test_eval_er_case_chairman_never_emits_emergency_verdict`, which only runs when `ANTHROPIC_API_KEY`/`GOOGLE_API_KEY` are set (unrelated to this plan).

- [ ] **Step 2: Start the backend**

Run: `uv run python -m backend.main`
Expected: server starts on `http://127.0.0.1:8001`

- [ ] **Step 3: Open the UI and ask a real question (user does this manually - real, billed API calls)**

Open `http://localhost:8001/ui` in a browser, type a question, click "Ask both models." Confirm: both raw answers render under the tabs, the chairman's three sections populate, and the doctor-question checkboxes are tappable.

- [ ] **Step 4: Confirm the degraded path is reachable (optional, no action needed unless a real failure occurs)**

No step to force this deliberately - noted so that if a live call ever fails, the "degraded" error message (not a crash or blank screen) is the expected, already-tested behavior.
