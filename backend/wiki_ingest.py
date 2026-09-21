"""Ingest pipeline for the patient wiki: extract text from an uploaded
source, propose a wiki diff via LLM, and write a pending_diffs row for
human review. See the design doc's INGEST flow. Applying an approved
diff to an actual wiki/*.md file is wiki_review.py (Phase 3), not this
module - nothing here ever writes under a patient's wiki/ directory.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

import fitz  # PyMuPDF

from . import llm_client, wiki_db, wiki_store

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
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except RuntimeError as e:
        raise ExtractionError(f"failed to open PDF: {e}")
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
        try:
            return file_bytes.decode("utf-8"), "local"
        except UnicodeDecodeError as e:
            raise ExtractionError(f"txt source is not valid UTF-8: {e}")

    if source_type in ("jpg", "png"):
        media_type = "image/jpeg" if source_type == "jpg" else "image/png"
        text = await _extract_image_bytes(file_bytes, media_type, vision_model)
        return text, "vision_llm"

    if source_type == "pdf":
        return await _extract_pdf(file_bytes, vision_model)

    raise ValueError(f"unsupported source_type {source_type!r}")


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
