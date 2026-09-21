# Patient Wiki — Phase 2: Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingest pipeline — extract text from an uploaded source, ask an LLM to propose which wiki page it affects, and write a `pending_diffs` row — stopping before any approval or wiki-file write-back exists (that's Phase 3).

**Architecture:** `backend/wiki_ingest.py` performs a linear pipeline per uploaded source: (1) extract raw text — local PyMuPDF for a PDF's real text layer or a TXT file, otherwise a vision LLM call per image or per non-text PDF page; (2) persist the extracted text via `wiki_store.write_raw_source()` and a matching `wiki_db.raw_sources` row; (3) ask an LLM to propose which wiki page the source affects (or a new page), producing that page's full new content plus a contradiction flag, with a safe needs-review fallback if the LLM's output is unusable after one retry; (4) apply the project's fixed tiered-approval rule — never delegated to the LLM's own judgment — and write a `wiki_db.pending_diffs` row. No file under `wiki/` is ever written in this phase.

**Tech Stack:** PyMuPDF (`fitz`) for local PDF text/page-image extraction (new dependency). `llm_client`'s existing direct Anthropic/Google dispatch, extended with a new `query_vision()` for image+text calls. Pydantic for the diff-proposal response schema, matching `claim_diff.py`'s extractor pattern.

**Spec:** `docs/superpowers/specs/2026-09-17-patient-wiki-design.md` (amended 2026-09-20)

## Global Constraints

- No file under `data/patients/<id>/wiki/` is ever written by this phase — `pending_diffs.diff_content` stores the *proposed* new page content only. Applying it is Phase 3's `wiki_review.py`.
- Tiered approval is a fixed rule in code, never an LLM judgment call: `allergies.md`, `medications.md`, any `contradiction=true`, and the needs-review fallback page always set `requires_approval=1`.
- A contradiction is never silently overwritten — `contradiction_flag=1` must always imply `requires_approval=1` (spec Testing priority #2).
- A source is never silently dropped when the diff-proposal LLM fails twice — it always produces a `pending_diffs` row, flagged for manual handling (spec's Error handling section), never a swallowed exception.
- No PHI leaves this machine except to Anthropic/Google directly (already true via `llm_client`, no proxy) — extracted text is persisted only under `data/patients/` via `wiki_store.write_raw_source()`, never written by hand elsewhere.
- `actor_id` on `approvals`/`audit_log` remains unpopulated — this phase never writes to either table, matching Phase 1's deferral (that's Phase 3's concern once diffs are actually approved).
- Every test in this phase mocks `llm_client.query_model`/`query_vision` — never a real network call. Live LLM calls are billed and the user is cost-conscious; this also matches the codebase's existing test convention (`tests/test_claim_diff.py`).

---

### Task 1: `raw_sources` + `pending_diffs` CRUD

**Files:**
- Modify: `backend/wiki_db.py`
- Test: `tests/test_wiki_db.py` (append)

**Interfaces:**
- Consumes: `wiki_db.get_connection`, `wiki_db._now`, `wiki_db.DB_PATH` (existing)
- Produces: `wiki_db.create_raw_source(source_id: str, patient_id: str, filename: Optional[str], source_type: str, extraction_method: str, extracted_path: str, document_date: Optional[str] = None, db_path: Path = DB_PATH) -> None`, `wiki_db.get_raw_source(source_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]`, `wiki_db.update_raw_source_status(source_id: str, status: str, db_path: Path = DB_PATH) -> None`, `wiki_db.create_pending_diff(patient_id: str, source_id: Optional[str], page_path: str, is_new_page: bool, diff_content: str, contradiction_flag: bool, contradiction_note: Optional[str], requires_approval: bool, db_path: Path = DB_PATH) -> str`, `wiki_db.list_pending_diffs(patient_id: str, status: Optional[str] = None, db_path: Path = DB_PATH) -> List[Dict[str, Any]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wiki_db.py`:

```python
def test_create_raw_source_persists_and_get_returns_it(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    wiki_db.create_raw_source(
        "source-1", patient_id, "discharge.pdf", "pdf", "local",
        "data/patients/x/raw/source-1.txt", document_date="2026-03-12",
        db_path=db_path,
    )

    source = wiki_db.get_raw_source("source-1", db_path)
    assert source["id"] == "source-1"
    assert source["patient_id"] == patient_id
    assert source["filename"] == "discharge.pdf"
    assert source["source_type"] == "pdf"
    assert source["extraction_method"] == "local"
    assert source["document_date"] == "2026-03-12"
    assert source["status"] == "pending_ingest"


def test_get_raw_source_returns_none_when_not_found(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)

    assert wiki_db.get_raw_source("nonexistent", db_path) is None


def test_update_raw_source_status_changes_status(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)
    wiki_db.create_raw_source(
        "source-1", patient_id, None, "note", "manual", "raw/source-1.txt", db_path=db_path
    )

    wiki_db.update_raw_source_status("source-1", "ingested", db_path)

    assert wiki_db.get_raw_source("source-1", db_path)["status"] == "ingested"


def test_create_pending_diff_returns_id_and_persists(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)
    wiki_db.create_raw_source(
        "source-1", patient_id, None, "note", "manual", "raw/source-1.txt", db_path=db_path
    )

    diff_id = wiki_db.create_pending_diff(
        patient_id, "source-1", "allergies.md", is_new_page=False,
        diff_content="# Allergies\n- Penicillin", contradiction_flag=False,
        contradiction_note=None, requires_approval=True, db_path=db_path,
    )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert len(diffs) == 1
    assert diffs[0]["id"] == diff_id
    assert diffs[0]["page_path"] == "allergies.md"
    assert diffs[0]["is_new_page"] == 0
    assert diffs[0]["contradiction_flag"] == 0
    assert diffs[0]["requires_approval"] == 1
    assert diffs[0]["status"] == "pending"


def test_list_pending_diffs_filters_by_patient(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_a = wiki_db.create_patient("Patient A", db_path)
    patient_b = wiki_db.create_patient("Patient B", db_path)
    wiki_db.create_pending_diff(
        patient_a, None, "overview.md", is_new_page=True, diff_content="a",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=False, db_path=db_path,
    )
    wiki_db.create_pending_diff(
        patient_b, None, "overview.md", is_new_page=True, diff_content="b",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=False, db_path=db_path,
    )

    diffs = wiki_db.list_pending_diffs(patient_a, db_path=db_path)

    assert len(diffs) == 1
    assert diffs[0]["diff_content"] == "a"


def test_list_pending_diffs_filters_by_status(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)
    pending_id = wiki_db.create_pending_diff(
        patient_id, None, "overview.md", is_new_page=True, diff_content="a",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=False, db_path=db_path,
    )
    approved_id = wiki_db.create_pending_diff(
        patient_id, None, "conditions.md", is_new_page=True, diff_content="b",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=False, db_path=db_path,
    )
    with wiki_db.get_connection(db_path) as conn:
        conn.execute("UPDATE pending_diffs SET status = 'approved' WHERE id = ?", (approved_id,))

    diffs = wiki_db.list_pending_diffs(patient_id, status="pending", db_path=db_path)

    assert [d["id"] for d in diffs] == [pending_id]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_db.py -v`
Expected: FAIL with `AttributeError: module 'backend.wiki_db' has no attribute 'create_raw_source'`

- [ ] **Step 3: Write the implementation**

In `backend/wiki_db.py`, change the typing import line to add `List`:

```python
from typing import Any, Dict, List, Optional
```

Append to `backend/wiki_db.py`:

```python
def create_raw_source(
    source_id: str,
    patient_id: str,
    filename: Optional[str],
    source_type: str,
    extraction_method: str,
    extracted_path: str,
    document_date: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sources
                (id, patient_id, filename, source_type, extraction_method,
                 extracted_path, document_date, uploaded_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_ingest')
            """,
            (source_id, patient_id, filename, source_type, extraction_method,
             extracted_path, document_date, _now()),
        )


def get_raw_source(source_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM raw_sources WHERE id = ?", (source_id,)
        ).fetchone()
    return dict(row) if row else None


def update_raw_source_status(source_id: str, status: str, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE raw_sources SET status = ? WHERE id = ?", (status, source_id)
        )


def create_pending_diff(
    patient_id: str,
    source_id: Optional[str],
    page_path: str,
    is_new_page: bool,
    diff_content: str,
    contradiction_flag: bool,
    contradiction_note: Optional[str],
    requires_approval: bool,
    db_path: Path = DB_PATH,
) -> str:
    diff_id = uuid.uuid4().hex
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pending_diffs
                (id, patient_id, source_id, page_path, is_new_page, diff_content,
                 contradiction_flag, contradiction_note, requires_approval,
                 status, proposed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                diff_id, patient_id, source_id, page_path, int(is_new_page),
                diff_content, int(contradiction_flag), contradiction_note,
                int(requires_approval), _now(),
            ),
        )
    return diff_id


def list_pending_diffs(
    patient_id: str, status: Optional[str] = None, db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM pending_diffs WHERE patient_id = ?"
    params: List[Any] = [patient_id]
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY proposed_at"
    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_db.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/wiki_db.py tests/test_wiki_db.py
git commit -m "feat: add raw_sources + pending_diffs CRUD to wiki_db"
```

---

### Task 2: Vision LLM support (`llm_client.query_vision`)

**Files:**
- Modify: `backend/llm_client.py`
- Test: `tests/test_llm_client.py` (append)

**Interfaces:**
- Consumes: `llm_client._post_json`, `ANTHROPIC_API_URL`, `ANTHROPIC_VERSION`, `ANTHROPIC_MAX_TOKENS`, `ANTHROPIC_API_KEY`, `GOOGLE_API_URL`, `GOOGLE_API_KEY`, `llm_client._log` (all existing)
- Produces: `llm_client.query_vision(model: str, image_bytes: bytes, media_type: str, prompt: str, timeout: float = 120.0) -> Optional[Dict[str, Any]]`

- [ ] **Step 1: Write the failing tests**

At the top of `tests/test_llm_client.py`, add `import base64` alongside the existing `import json`. Append these tests:

```python
async def test_query_vision_claude_success():
    fake_response = _mock_response(
        {"content": [{"type": "text", "text": "extracted text"}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = await llm_client.query_vision(
            "claude/claude-sonnet-4-5-20250929",
            b"fake-image-bytes",
            "image/png",
            "Transcribe this document.",
        )

    assert result == {"content": "extracted text", "reasoning_details": None}
    sent_payload = mock_post.call_args.kwargs["json"]
    image_block = sent_payload["messages"][0]["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == base64.b64encode(b"fake-image-bytes").decode("ascii")


async def test_query_vision_gemini_success():
    fake_response = _mock_response(
        {"candidates": [{"content": {"parts": [{"text": "extracted text"}]}}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = await llm_client.query_vision(
            "gemini/gemini-3.1-pro-preview",
            b"fake-image-bytes",
            "image/jpeg",
            "Transcribe this document.",
        )

    assert result == {"content": "extracted text", "reasoning_details": None}
    sent_payload = mock_post.call_args.kwargs["json"]
    image_part = sent_payload["contents"][0]["parts"][0]
    assert image_part["inlineData"]["mimeType"] == "image/jpeg"
    assert image_part["inlineData"]["data"] == base64.b64encode(b"fake-image-bytes").decode("ascii")


async def test_query_vision_unknown_provider_returns_none():
    result = await llm_client.query_vision(
        "unknown-provider/some-model", b"bytes", "image/png", "prompt"
    )
    assert result is None


async def test_query_vision_network_error_returns_none():
    with patch.object(llm_client, "RETRY_BACKOFF_SECONDS", 0), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectTimeout("timed out")),
    ):
        result = await llm_client.query_vision(
            "claude/claude-sonnet-4-5-20250929", b"bytes", "image/png", "prompt"
        )

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_client.py -v -k query_vision`
Expected: FAIL with `AttributeError: module 'backend.llm_client' has no attribute 'query_vision'`

- [ ] **Step 3: Write the implementation**

In `backend/llm_client.py`, add `import base64` alongside the existing `import hashlib`. Append at the end of the file:

```python
async def _query_claude_vision(
    model: str, image_base64: str, media_type: str, prompt: str, timeout: float
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    data = await _post_json(ANTHROPIC_API_URL, json_payload=payload, timeout=timeout, headers=headers)

    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )
    return {"content": text, "reasoning_details": None}


async def _query_gemini_vision(
    model: str, image_base64: str, media_type: str, prompt: str, timeout: float
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": media_type, "data": image_base64}},
                    {"text": prompt},
                ],
            }
        ]
    }
    url = f"{GOOGLE_API_URL}/{model}:generateContent"

    data = await _post_json(url, json_payload=payload, timeout=timeout, params={"key": GOOGLE_API_KEY})

    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts)
    return {"content": text, "reasoning_details": None}


_VISION_PROVIDERS = {
    "claude": _query_claude_vision,
    "gemini": _query_gemini_vision,
}


async def query_vision(
    model: str,
    image_bytes: bytes,
    media_type: str,
    prompt: str,
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model with an image + text prompt, directly against its
    provider. Used by wiki_ingest.py's extraction step for scanned PDF
    pages, JPGs, and PNGs, where no text layer exists to extract locally.

    Args:
        model: Provider-prefixed model identifier, e.g. "claude/claude-sonnet-4-5-20250929".
        image_bytes: Raw image bytes (a rendered PNG for a PDF page, or the
                     original JPG/PNG upload).
        media_type: MIME type, e.g. "image/png" or "image/jpeg".
        prompt: Text instruction accompanying the image.
        timeout: Request timeout in seconds.

    Returns:
        Response dict with 'content' and 'reasoning_details', or None if failed.
    """
    provider, _, model_id = model.partition("/")
    handler = _VISION_PROVIDERS.get(provider)

    if handler is None:
        print(f"Error querying model {model}: unknown provider '{provider}'")
        return None

    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    _log("QUERY_VISION start", model=model, media_type=media_type, prompt_preview=prompt[:200])

    try:
        result = await handler(model_id, image_base64, media_type, prompt, timeout)
        _log("QUERY_VISION done", model=model, content_preview=(result.get("content") or "")[:300])
        return result
    except Exception as e:
        print(f"Error querying vision model {model}: {e}")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: PASS (all tests, including the 4 new `query_vision` ones)

- [ ] **Step 5: Commit**

```bash
git add backend/llm_client.py tests/test_llm_client.py
git commit -m "feat: add query_vision for image+text LLM calls"
```

---

### Task 3: Extraction dispatch (`wiki_ingest.extract_text`)

**Files:**
- Create: `backend/wiki_ingest.py`
- Test: `tests/test_wiki_ingest_extract.py`
- Modify: `pyproject.toml` (add `pymupdf` dependency)

**Interfaces:**
- Consumes: `llm_client.query_vision` (Task 2)
- Produces: `wiki_ingest.ExtractionError(RuntimeError)`, `wiki_ingest.DEFAULT_VISION_MODEL: str`, `wiki_ingest.extract_text(file_bytes: bytes, source_type: str, vision_model: str = DEFAULT_VISION_MODEL) -> Tuple[str, str]` (returns `(text, extraction_method)`)

- [ ] **Step 1: Add the PyMuPDF dependency**

```bash
uv add pymupdf
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_wiki_ingest_extract.py
"""Tests for the patient wiki ingest pipeline's extraction step
(backend/wiki_ingest.py's extract_text). Split from the diff-proposal and
orchestrator tests because this is a distinct concern (PyMuPDF + vision
LLM dispatch, no database), worth its own reviewable unit.
"""

from unittest.mock import AsyncMock, patch

import fitz
import pytest

from backend import wiki_ingest


def _pdf_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.write()
    doc.close()
    return data


def _pdf_blank_page() -> bytes:
    doc = fitz.open()
    doc.new_page()
    data = doc.write()
    doc.close()
    return data


async def test_extract_text_txt_decodes_directly():
    text, method = await wiki_ingest.extract_text(b"Patient reports mild headache.", "txt")

    assert text == "Patient reports mild headache."
    assert method == "local"


async def test_extract_text_pdf_with_real_text_layer_uses_local():
    pdf_bytes = _pdf_with_text("Lisinopril 10mg once daily")

    with patch.object(wiki_ingest.llm_client, "query_vision", new=AsyncMock()) as mock_vision:
        text, method = await wiki_ingest.extract_text(pdf_bytes, "pdf")

    assert "Lisinopril" in text
    assert method == "local"
    assert mock_vision.await_count == 0


async def test_extract_text_pdf_scanned_page_falls_back_to_vision():
    pdf_bytes = _pdf_blank_page()

    with patch.object(
        wiki_ingest.llm_client,
        "query_vision",
        new=AsyncMock(return_value={"content": "transcribed from image", "reasoning_details": None}),
    ) as mock_vision:
        text, method = await wiki_ingest.extract_text(pdf_bytes, "pdf")

    assert text == "transcribed from image"
    assert method == "vision_llm"
    assert mock_vision.await_count == 1
    assert mock_vision.call_args.args[2] == "image/png"


async def test_extract_text_jpg_uses_vision_with_jpeg_media_type():
    with patch.object(
        wiki_ingest.llm_client,
        "query_vision",
        new=AsyncMock(return_value={"content": "transcribed", "reasoning_details": None}),
    ) as mock_vision:
        text, method = await wiki_ingest.extract_text(b"fake-jpg-bytes", "jpg")

    assert text == "transcribed"
    assert method == "vision_llm"
    assert mock_vision.call_args.args[1] == b"fake-jpg-bytes"
    assert mock_vision.call_args.args[2] == "image/jpeg"


async def test_extract_text_png_uses_vision_with_png_media_type():
    with patch.object(
        wiki_ingest.llm_client,
        "query_vision",
        new=AsyncMock(return_value={"content": "transcribed", "reasoning_details": None}),
    ) as mock_vision:
        await wiki_ingest.extract_text(b"fake-png-bytes", "png")

    assert mock_vision.call_args.args[2] == "image/png"


async def test_extract_text_vision_failure_raises_extraction_error():
    with patch.object(wiki_ingest.llm_client, "query_vision", new=AsyncMock(return_value=None)):
        with pytest.raises(wiki_ingest.ExtractionError):
            await wiki_ingest.extract_text(b"fake-jpg-bytes", "jpg")


async def test_extract_text_unsupported_source_type_raises():
    with pytest.raises(ValueError):
        await wiki_ingest.extract_text(b"data", "docx")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_ingest_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.wiki_ingest'`

- [ ] **Step 4: Write the implementation**

```python
# backend/wiki_ingest.py
"""Ingest pipeline for the patient wiki: extract text from an uploaded
source, propose a wiki diff via LLM, and write a pending_diffs row for
human review. See the design doc's INGEST flow. Applying an approved
diff to an actual wiki/*.md file is wiki_review.py (Phase 3), not this
module - nothing here ever writes under a patient's wiki/ directory.
"""

from typing import List, Tuple

import fitz  # PyMuPDF

from . import llm_client

DEFAULT_VISION_MODEL = "claude/claude-sonnet-4-5-20250929"

VISION_EXTRACTION_PROMPT = (
    "Transcribe every word of medical record text visible in this image, "
    "verbatim, in reading order. Output plain text only - no commentary, "
    "no markdown, no summary."
)

# A text-PDF page with fewer real characters than this (after stripping
# whitespace) is treated as scanned/image-only rather than as "just a
# short page" - real medical record pages are never this sparse when a
# text layer actually exists.
MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER = 20


class ExtractionError(RuntimeError):
    """Raised when text extraction fails for a source (e.g. the vision
    LLM call failed and there is no text to fall back to)."""


async def _extract_image_bytes(image_bytes: bytes, media_type: str, vision_model: str) -> str:
    result = await llm_client.query_vision(
        vision_model, image_bytes, media_type, VISION_EXTRACTION_PROMPT
    )
    if result is None or not result.get("content"):
        raise ExtractionError("vision extraction failed to produce any text")
    return result["content"]


async def _extract_pdf(file_bytes: bytes, vision_model: str) -> Tuple[str, str]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page_texts: List[str] = []
        used_vision = False
        for page in doc:
            local_text = page.get_text().strip()
            if len(local_text) >= MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER:
                page_texts.append(local_text)
                continue
            pixmap = page.get_pixmap()
            image_bytes = pixmap.tobytes("png")
            page_texts.append(await _extract_image_bytes(image_bytes, "image/png", vision_model))
            used_vision = True
        return "\n\n".join(page_texts), ("vision_llm" if used_vision else "local")
    finally:
        doc.close()


async def extract_text(
    file_bytes: bytes,
    source_type: str,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> Tuple[str, str]:
    """
    Extract text from an uploaded source. Returns (text, extraction_method).

    source_type "txt": decode directly, extraction_method "local".
    source_type "pdf": PyMuPDF's text layer per page; any page with no
        real text layer is rendered to a PNG and sent through the vision
        LLM instead - a scanned PDF is just a PDF where every page needs
        this. extraction_method is "local" only if every page had a real
        text layer, else "vision_llm" (a mixed document still counts as
        needing vision, since a caller can't otherwise tell which pages
        to trust).
    source_type "jpg"/"png": vision LLM directly on the raw image bytes,
        extraction_method "vision_llm".

    Typed notes never reach this function - they have no file at all
    (see wiki_ingest.ingest_source).
    """
    if source_type == "txt":
        return file_bytes.decode("utf-8"), "local"

    if source_type in ("jpg", "png"):
        media_type = "image/jpeg" if source_type == "jpg" else "image/png"
        text = await _extract_image_bytes(file_bytes, media_type, vision_model)
        return text, "vision_llm"

    if source_type == "pdf":
        return await _extract_pdf(file_bytes, vision_model)

    raise ValueError(f"unsupported source_type {source_type!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_ingest_extract.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/wiki_ingest.py tests/test_wiki_ingest_extract.py pyproject.toml uv.lock
git commit -m "feat: add wiki ingest text extraction (PyMuPDF + vision LLM)"
```

---

### Task 4: Diff-proposal (`wiki_ingest.propose_diff`)

**Files:**
- Modify: `backend/wiki_ingest.py`
- Test: `tests/test_wiki_ingest_diff.py`

**Interfaces:**
- Consumes: `llm_client.query_model`, `llm_client.strip_json_fence` (existing)
- Produces: `wiki_ingest.ProposedDiff` (Pydantic model: `page_path: str`, `is_new_page: bool`, `new_page_content: str`, `contradiction: bool`, `contradiction_note: Optional[str]`), `wiki_ingest.NEEDS_REVIEW_PAGE_PATH: str`, `wiki_ingest.DEFAULT_DIFF_MODEL: str`, `wiki_ingest.propose_diff(source_text: str, existing_pages: Dict[str, str], model: str = DEFAULT_DIFF_MODEL) -> ProposedDiff`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wiki_ingest_diff.py
"""Tests for the patient wiki ingest pipeline's diff-proposal step
(backend/wiki_ingest.py's propose_diff).
"""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

from backend import wiki_ingest


def _fake_response(content: str) -> Dict[str, Any]:
    return {"content": content, "reasoning_details": None}


async def test_propose_diff_parses_valid_response():
    valid_json = json.dumps({
        "page_path": "medications.md",
        "is_new_page": False,
        "new_page_content": "# Medications\n- Lisinopril 10mg",
        "contradiction": False,
        "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        result = await wiki_ingest.propose_diff("Lisinopril 10mg once daily", {})

    assert result.page_path == "medications.md"
    assert result.is_new_page is False
    assert result.contradiction is False
    assert mock_query.await_count == 1


async def test_propose_diff_strips_markdown_fence():
    fenced = "```json\n" + json.dumps({
        "page_path": "allergies.md", "is_new_page": True,
        "new_page_content": "# Allergies\n- Penicillin",
        "contradiction": False, "contradiction_note": None,
    }) + "\n```"
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(fenced))
    ):
        result = await wiki_ingest.propose_diff("Penicillin allergy noted", {})

    assert result.page_path == "allergies.md"


async def test_propose_diff_retries_once_on_malformed_json():
    valid_json = json.dumps({
        "page_path": "conditions.md", "is_new_page": True,
        "new_page_content": "# Conditions\n- Hypertension",
        "contradiction": False, "contradiction_note": None,
    })
    responses = [_fake_response("not json"), _fake_response(valid_json)]
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(side_effect=responses)
    ) as mock_query:
        result = await wiki_ingest.propose_diff("Hypertension diagnosed", {})

    assert result.page_path == "conditions.md"
    assert mock_query.await_count == 2


async def test_propose_diff_falls_back_to_needs_review_after_two_failures():
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response("not json"))
    ) as mock_query:
        result = await wiki_ingest.propose_diff("some source text", {})

    assert result.page_path == wiki_ingest.NEEDS_REVIEW_PAGE_PATH
    assert result.is_new_page is True
    assert "some source text" in result.new_page_content
    assert mock_query.await_count == 2


async def test_propose_diff_includes_existing_wiki_pages_in_prompt():
    valid_json = json.dumps({
        "page_path": "medications.md", "is_new_page": False,
        "new_page_content": "updated", "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        await wiki_ingest.propose_diff(
            "new source", {"medications.md": "# Medications\n- Warfarin 5mg"}
        )

    sent_prompt = mock_query.call_args.args[1][1]["content"]
    assert "Warfarin 5mg" in sent_prompt
    assert "medications.md" in sent_prompt


async def test_propose_diff_empty_wiki_says_no_pages_yet():
    valid_json = json.dumps({
        "page_path": "overview.md", "is_new_page": True,
        "new_page_content": "updated", "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        await wiki_ingest.propose_diff("new source", {})

    sent_prompt = mock_query.call_args.args[1][1]["content"]
    assert "no pages yet" in sent_prompt


async def test_propose_diff_contradiction_flag_and_note_pass_through():
    valid_json = json.dumps({
        "page_path": "medications.md", "is_new_page": False,
        "new_page_content": "updated", "contradiction": True,
        "contradiction_note": "source omits warfarin, which the wiki lists as current",
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        result = await wiki_ingest.propose_diff("new source", {"medications.md": "old"})

    assert result.contradiction is True
    assert "warfarin" in result.contradiction_note
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_ingest_diff.py -v`
Expected: FAIL with `AttributeError: module 'backend.wiki_ingest' has no attribute 'propose_diff'`

- [ ] **Step 3: Write the implementation**

Add these imports to the top of `backend/wiki_ingest.py` (alongside the existing ones):

```python
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError
```

Append to `backend/wiki_ingest.py`:

```python
DEFAULT_DIFF_MODEL = "claude/claude-sonnet-4-5-20250929"

NEEDS_REVIEW_PAGE_PATH = "needs-review.md"

DIFF_PROPOSAL_SYSTEM_PROMPT = """You maintain a per-patient medical wiki. \
You will be shown the patient's CURRENT WIKI (every existing page, or \
"(no pages yet)" if this is the first entry) and a NEW SOURCE (a medical \
record, note, or transcription just added for this patient).

Decide which single wiki page this source should update, or propose a \
new page if no existing page fits. Produce the FULL new content of that \
page after incorporating the source - not just the changed lines. \
Preserve everything in the current page that the source doesn't affect; \
merge new information in rather than replacing the whole page. Never \
remove or soften an existing allergy or medication entry - if the source \
seems to update one, note the change in the page but keep the prior \
entry's history visible.

Set "contradiction" to true if the source conflicts with something \
already in the wiki (e.g. a medication list that omits a drug the wiki \
says is current, or a fact that directly contradicts an existing page) \
rather than simply adding new information. If true, contradiction_note \
must explain the conflict in one sentence; if false, contradiction_note \
must be null.

Respond with ONLY a JSON object with exactly these keys: "page_path" \
(e.g. "medications.md" or "by-system/cardiac.md"), "is_new_page" \
(boolean), "new_page_content" (string, the full page), "contradiction" \
(boolean), "contradiction_note" (string or null). No other text, no \
markdown fences."""


class ProposedDiff(BaseModel):
    page_path: str
    is_new_page: bool
    new_page_content: str
    contradiction: bool
    contradiction_note: Optional[str] = None


def _format_current_wiki(pages: Dict[str, str]) -> str:
    if not pages:
        return "(no pages yet)"
    return "\n\n".join(
        f"--- {path} ---\n{content}" for path, content in sorted(pages.items())
    )


def _parse_proposed_diff(raw: Optional[Dict[str, Any]]) -> Optional[ProposedDiff]:
    if raw is None or not raw.get("content"):
        return None
    try:
        data = json.loads(llm_client.strip_json_fence(raw["content"]))
        return ProposedDiff(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


async def propose_diff(
    source_text: str,
    existing_pages: Dict[str, str],
    model: str = DEFAULT_DIFF_MODEL,
) -> ProposedDiff:
    """
    Ask an LLM which wiki page a new source should update (or propose a
    new page), producing that page's full new content plus a
    contradiction flag. Retries once on malformed output, then falls
    back to a needs-review placeholder a human must resolve manually - a
    source is never silently dropped just because the LLM couldn't
    propose a diff for it (design doc, Error handling).
    """
    prompt = (
        f"CURRENT WIKI:\n{_format_current_wiki(existing_pages)}\n\n"
        f"NEW SOURCE:\n{source_text}"
    )

    for _ in range(2):
        raw = await llm_client.query_model(
            model,
            [
                {"role": "system", "content": DIFF_PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        proposed = _parse_proposed_diff(raw)
        if proposed is not None:
            return proposed

    return ProposedDiff(
        page_path=NEEDS_REVIEW_PAGE_PATH,
        is_new_page=True,
        new_page_content=(
            "(automatic diff proposal failed after retries - needs manual "
            "handling)\n\n--- raw source text follows ---\n\n" + source_text
        ),
        contradiction=False,
        contradiction_note=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_ingest_diff.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/wiki_ingest.py tests/test_wiki_ingest_diff.py
git commit -m "feat: add wiki ingest diff proposal via LLM"
```

---

### Task 5: Orchestrator (`wiki_ingest.ingest_source`)

**Files:**
- Modify: `backend/wiki_ingest.py`
- Test: `tests/test_wiki_ingest.py`

**Interfaces:**
- Consumes: `wiki_db.create_raw_source`, `wiki_db.update_raw_source_status`, `wiki_db.create_pending_diff`, `wiki_db.DB_PATH` (Task 1); `wiki_store.write_raw_source`, `wiki_store.list_wiki_pages`, `wiki_store.read_wiki_page`, `wiki_store.PATIENTS_ROOT` (Phase 1); `extract_text` (Task 3); `propose_diff`, `NEEDS_REVIEW_PAGE_PATH` (Task 4)
- Produces: `wiki_ingest.TIERED_APPROVAL_PAGES: set`, `wiki_ingest.ingest_source(patient_id: str, source_type: str, file_bytes: Optional[bytes] = None, note_text: Optional[str] = None, filename: Optional[str] = None, document_date: Optional[str] = None, vision_model: str = DEFAULT_VISION_MODEL, diff_model: str = DEFAULT_DIFF_MODEL, db_path: Path = wiki_db.DB_PATH, patients_root: Path = wiki_store.PATIENTS_ROOT) -> str` (returns the new `pending_diffs` row's id)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wiki_ingest.py
"""Integration tests for the full patient wiki ingest pipeline
(backend/wiki_ingest.py's ingest_source) - extraction through the
pending_diffs row, per the design doc's Testing priority #4 (minus the
approve step, which is Phase 3).
"""

import json
from unittest.mock import AsyncMock, patch

import fitz
import pytest

from backend import wiki_db, wiki_ingest, wiki_store


def _pdf_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.write()
    doc.close()
    return data


def _fake_response(content: str):
    return {"content": content, "reasoning_details": None}


async def test_ingest_source_note_writes_raw_source_and_pending_diff_rows(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "allergies.md", "is_new_page": True,
        "new_page_content": "# Allergies\n- Penicillin (confirmed 2024-03-12)",
        "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        diff_id = await wiki_ingest.ingest_source(
            patient_id, "note", note_text="Doctor confirmed penicillin allergy today.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert len(diffs) == 1
    assert diffs[0]["id"] == diff_id
    assert diffs[0]["page_path"] == "allergies.md"
    assert diffs[0]["requires_approval"] == 1

    source_row = wiki_db.get_raw_source(diffs[0]["source_id"], db_path=db_path)
    assert source_row["status"] == "ingested"
    assert source_row["extraction_method"] == "manual"

    raw_path = wiki_store.raw_dir(patient_id, patients_root) / f"{diffs[0]['source_id']}.txt"
    assert raw_path.read_text() == "Doctor confirmed penicillin allergy today."


async def test_ingest_source_non_tiered_page_does_not_require_approval(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "overview.md", "is_new_page": True,
        "new_page_content": "# Overview\nGeneral clinical picture.",
        "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        await wiki_ingest.ingest_source(
            patient_id, "note", note_text="General checkup notes.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["requires_approval"] == 0


async def test_ingest_source_contradiction_always_requires_approval(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "overview.md", "is_new_page": False,
        "new_page_content": "updated", "contradiction": True,
        "contradiction_note": "conflicts with existing note",
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        await wiki_ingest.ingest_source(
            patient_id, "note", note_text="Conflicting update.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["requires_approval"] == 1
    assert diffs[0]["contradiction_flag"] == 1


async def test_ingest_source_needs_review_fallback_requires_approval(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response("not json"))
    ):
        await wiki_ingest.ingest_source(
            patient_id, "note", note_text="Unparseable case.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["page_path"] == wiki_ingest.NEEDS_REVIEW_PAGE_PATH
    assert diffs[0]["requires_approval"] == 1


async def test_ingest_source_never_writes_to_wiki_directory(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "medications.md", "is_new_page": True,
        "new_page_content": "# Medications\n- Lisinopril 10mg",
        "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        await wiki_ingest.ingest_source(
            patient_id, "note", note_text="Started Lisinopril 10mg.",
            db_path=db_path, patients_root=patients_root,
        )

    assert wiki_store.list_wiki_pages(patient_id, root=patients_root) == []


async def test_ingest_source_pdf_uses_extract_text(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)
    pdf_bytes = _pdf_with_text("Metformin 500mg twice daily")

    valid_json = json.dumps({
        "page_path": "medications.md", "is_new_page": True,
        "new_page_content": "# Medications\n- Metformin 500mg",
        "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ), patch.object(wiki_ingest.llm_client, "query_vision", new=AsyncMock()) as mock_vision:
        await wiki_ingest.ingest_source(
            patient_id, "pdf", file_bytes=pdf_bytes, filename="labs.pdf",
            db_path=db_path, patients_root=patients_root,
        )

    assert mock_vision.await_count == 0
    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    source_row = wiki_db.get_raw_source(diffs[0]["source_id"], db_path=db_path)
    assert source_row["extraction_method"] == "local"
    assert source_row["filename"] == "labs.pdf"


async def test_ingest_source_missing_note_text_raises():
    with pytest.raises(ValueError):
        await wiki_ingest.ingest_source("patient-a", "note")


async def test_ingest_source_missing_file_bytes_raises():
    with pytest.raises(ValueError):
        await wiki_ingest.ingest_source("patient-a", "pdf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_ingest.py -v`
Expected: FAIL with `AttributeError: module 'backend.wiki_ingest' has no attribute 'ingest_source'`

- [ ] **Step 3: Write the implementation**

Add these imports to the top of `backend/wiki_ingest.py` (alongside the existing ones):

```python
import uuid
from pathlib import Path

from . import wiki_db, wiki_store
```

Append to `backend/wiki_ingest.py`:

```python
TIERED_APPROVAL_PAGES = {"allergies.md", "medications.md"}


def _requires_approval(page_path: str, contradiction: bool) -> bool:
    """
    Tiered approval (design doc, amended 2026-09-20): allergies,
    medications, any contradiction, and the needs-review fallback page
    always require a human to confirm the diff against its source before
    it can be applied. This is a fixed rule in code - safety-critical
    tiering must never be the model's own judgment call.
    """
    return (
        page_path in TIERED_APPROVAL_PAGES
        or page_path == NEEDS_REVIEW_PAGE_PATH
        or contradiction
    )


async def ingest_source(
    patient_id: str,
    source_type: str,
    file_bytes: Optional[bytes] = None,
    note_text: Optional[str] = None,
    filename: Optional[str] = None,
    document_date: Optional[str] = None,
    vision_model: str = DEFAULT_VISION_MODEL,
    diff_model: str = DEFAULT_DIFF_MODEL,
    db_path: Path = wiki_db.DB_PATH,
    patients_root: Path = wiki_store.PATIENTS_ROOT,
) -> str:
    """
    Full ingest pipeline for one source: extract text, write the
    raw_sources row + extracted text file, ask an LLM to propose a wiki
    diff, and write the pending_diffs row. Never writes to wiki/ - that
    only happens once a human approves the diff (wiki_review.py, Phase 3).

    Returns the new pending_diffs row's id.
    """
    if source_type == "note":
        if note_text is None:
            raise ValueError("note_text is required when source_type is 'note'")
        text, extraction_method = note_text, "manual"
    else:
        if file_bytes is None:
            raise ValueError(f"file_bytes is required when source_type is {source_type!r}")
        text, extraction_method = await extract_text(file_bytes, source_type, vision_model)

    source_id = uuid.uuid4().hex
    extracted_path = wiki_store.write_raw_source(patient_id, source_id, text, root=patients_root)
    wiki_db.create_raw_source(
        source_id, patient_id, filename, source_type, extraction_method,
        str(extracted_path), document_date, db_path=db_path,
    )

    existing_pages = {
        page_path: wiki_store.read_wiki_page(patient_id, page_path, root=patients_root) or ""
        for page_path in wiki_store.list_wiki_pages(patient_id, root=patients_root)
    }
    proposed = await propose_diff(text, existing_pages, model=diff_model)
    requires_approval = _requires_approval(proposed.page_path, proposed.contradiction)

    diff_id = wiki_db.create_pending_diff(
        patient_id, source_id, proposed.page_path, proposed.is_new_page,
        proposed.new_page_content, proposed.contradiction,
        proposed.contradiction_note, requires_approval, db_path=db_path,
    )

    wiki_db.update_raw_source_status(source_id, "ingested", db_path=db_path)

    return diff_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_ingest.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Run the full wiki test suite together**

Run: `uv run pytest tests/test_wiki_db.py tests/test_wiki_store.py tests/test_wiki_store_git.py tests/test_wiki_ingest_extract.py tests/test_wiki_ingest_diff.py tests/test_wiki_ingest.py tests/test_llm_client.py -v`
Expected: PASS (all)

- [ ] **Step 6: Run the entire project test suite to check for regressions**

Run: `uv run pytest -v`
Expected: PASS (no regressions in claim_diff.py/council.py/main.py/eval_harness.py tests)

- [ ] **Step 7: Commit**

```bash
git add backend/wiki_ingest.py tests/test_wiki_ingest.py
git commit -m "feat: add wiki ingest orchestrator (ingest_source)"
```

---

## Self-review notes

- **Spec coverage:** extraction dispatch for all 5 document formats (PDF text/scanned, JPG, PNG, TXT; typed notes bypass extraction) ✓ Task 3 + Task 5; `raw_sources` row + extracted text file persisted before any LLM diff call ✓ Task 5; LLM proposes a diff naming the target page (existing or new) with full new content ✓ Task 4; contradiction flagged explicitly, never silently overwritten ✓ Task 4 + Task 5's `_requires_approval`; `pending_diffs` row written with `status='pending'`, no file in `wiki/` touched ✓ Task 5 (explicit test); tiered approval (allergies/medications/contradiction always block) is a fixed code rule, not LLM-delegated ✓ Task 5; diff-proposal failure never silently drops a source — falls back to a flagged needs-review row ✓ Task 4 + Task 5. `actor_id`/`approvals`/`audit_log` remain untouched, per the spec's own phase sequencing (Phase 3's concern).
- **No placeholders:** every step has real, complete code — no TBD/TODO, no "similar to Task N."
- **Type consistency:** `db_path: Path = DB_PATH` / `root: Path = PATIENTS_ROOT` defaults match Phase 1's existing convention exactly; `ProposedDiff`'s field names (`page_path`, `is_new_page`, `new_page_content`, `contradiction`, `contradiction_note`) are used identically in the prompt's requested JSON keys (Task 4), the parser (Task 4), and every call site in `ingest_source` (Task 5) — `contradiction` (not `contradiction_flag`) is the `ProposedDiff` field name throughout, while `contradiction_flag` is only ever the SQLite column name, matched explicitly at the `wiki_db.create_pending_diff` call boundary.
