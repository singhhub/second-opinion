"""Ingest pipeline for the patient wiki: extract text from an uploaded
source, propose a wiki diff via LLM, and write a pending_diffs row. See
the design doc's INGEST flow.

Turning a decided diff into an actual wiki/*.md file change is
wiki_review.apply_diff()'s job, not this module's. ingest_source() does
call it, on one branch only: a diff whose requires_approval is False
(no tiered page, no contradiction) is auto-applied here immediately, so
that branch does write under the patient's wiki/. A diff that requires
approval is left untouched at status='pending' - nothing is written for
it until a human confirms it in a later phase.
"""

import json
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError

import fitz  # PyMuPDF

from . import config, llm_client, wiki_db, wiki_review, wiki_store

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
        text = "\n\n".join(page_texts)
        # A PDF with no pages at all (or whose only text is whitespace)
        # yields "" here. Returning that would hand an empty source to the
        # diff-proposal LLM, whose invention from nothing then becomes a
        # proposed medical-wiki diff - so refuse it instead. The vision
        # path already guards this in _extract_image_bytes.
        if not used_vision and not text.strip():
            raise ExtractionError("PDF contained no extractable text (no pages or empty text layer)")
        return text, ("vision_llm" if used_vision else "local")
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

    Raises ExtractionError if a source yields no text at all - an empty
    extraction is a failure to be surfaced, never an empty source to feed
    downstream.

    Typed notes never reach this function - they have no file at all
    (see wiki_ingest.ingest_source).
    """
    if source_type == "txt":
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ExtractionError(f"txt source is not valid UTF-8: {e}")
        if not text.strip():
            raise ExtractionError("txt source is empty")
        return text, "local"

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


def _normalize_page_path(page_path: str) -> str:
    """
    Reduce an LLM-proposed page_path to a canonical form for tiering
    comparisons. page_path is free text the model wrote, so the same page
    arrives spelled many ways - "wiki/medications.md" (the form the design
    doc's own schema comment uses), "Medications.md", "./medications.md",
    a trailing space, a backslash separator. Comparing the raw string
    would let any of those slip past the tiering rule.

    Trailing dots and whitespace are stripped per path segment, not just
    off the whole string: "medications.md." is the same page as
    "medications.md" to Windows, which silently drops the trailing dot at
    write time - so the tiering rule must see them as the same page too.

    The result is also what gets recorded in pending_diffs and written to
    disk (see ingest_source): the string that is tiered must be exactly
    the string that is applied.
    """
    normalized = page_path.strip().replace("\\", "/").lower()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("wiki/"):
        normalized = normalized[len("wiki/"):]
    # An all-dots segment ("." or "..") is left alone: that's a traversal
    # attempt for wiki_store to reject, not a page name to tidy up.
    segments = [seg.lstrip(" \t").rstrip(" \t.") or seg for seg in normalized.split("/")]
    return "/".join(segments)


def _requires_approval(page_path: str, contradiction: bool) -> bool:
    """
    Tiered approval (design doc, amended 2026-09-20): allergies,
    medications, any contradiction, and the needs-review fallback page
    always require a human to confirm the diff against its source before
    it can be applied. This is a fixed rule in code - safety-critical
    tiering must never be the model's own judgment call.

    Because page_path is LLM-generated free text, matching is done on the
    normalized *basename*, so a nested proposal like
    "by-system/medications.md" is tiered too. Erring toward requiring
    approval is the safe direction: ingest_source auto-applies
    requires_approval=0 diffs with no human in the loop.
    """
    normalized = _normalize_page_path(page_path)
    basename = PurePosixPath(normalized).name
    return (
        contradiction
        or normalized == _normalize_page_path(NEEDS_REVIEW_PAGE_PATH)
        or basename in TIERED_APPROVAL_PAGES
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
    diff, and write the pending_diffs row.

    If the proposed diff doesn't require approval (not a tiered page, no
    contradiction), it is then auto-applied immediately via
    wiki_review.apply_diff() - so this function does write under the
    patient's wiki/ on that branch. A diff that requires approval is left
    at status='pending' and nothing is written for it; applying that one
    waits on a human confirming it in a later phase.

    An auto-apply failure is degraded-but-recorded, not fatal: the
    raw_sources/pending_diffs rows already written are kept, the diff stays
    'pending', and an audit_log row (action 'auto_apply_failed') records
    the attempt. The two exceptions are wiki_store.WikiRemoteConfiguredError
    and wiki_store.UnsafePagePathError, which propagate - see below.

    Raises ValueError for an unusable request (unknown patient, missing or
    empty note text, missing file bytes) and ExtractionError if the source
    yields no text - but an extraction failure still leaves a
    status='failed' raw_sources row behind, so a caregiver's upload is
    never lost without a trace (design doc, Error handling).

    Returns the new pending_diffs row's id.
    """
    # Argument validation first: it needs no I/O and touches nothing.
    if source_type == "note":
        if note_text is None or not note_text.strip():
            raise ValueError("note_text is required and must be non-empty when source_type is 'note'")
    elif file_bytes is None:
        raise ValueError(f"file_bytes is required when source_type is {source_type!r}")

    # Then the patient, BEFORE any extraction or disk write. create_raw_source's
    # foreign key would catch an unknown patient_id eventually, but only after
    # the extracted medical text had already been written to
    # <patients_root>/<bogus-id>/raw/, with nothing to clean it up.
    if wiki_db.get_patient(patient_id, db_path) is None:
        raise ValueError(f"no such patient: {patient_id!r}")

    source_id = uuid.uuid4().hex
    # Computed up front (not inside the except below) so a path-unsafe id
    # surfaces as UnsafePagePathError before extraction, rather than while
    # handling an extraction failure.
    extracted_path = wiki_store.raw_dir(patient_id, patients_root) / f"{source_id}.txt"

    if source_type == "note":
        text, extraction_method = note_text, "manual"
    else:
        try:
            text, extraction_method = await extract_text(file_bytes, source_type, vision_model)
        except ExtractionError:
            # Record the upload as failed before re-raising. Without a row
            # there is no database record the upload ever happened, nothing
            # for the review page to surface, and no way for the caregiver
            # to learn their document vanished. extracted_path records where
            # the text would have gone; no file is written, since there is
            # no text to write.
            wiki_db.create_raw_source(
                source_id, patient_id, filename, source_type, "failed",
                str(extracted_path), document_date, db_path=db_path,
            )
            wiki_db.update_raw_source_status(source_id, "failed", db_path=db_path)
            raise

    written_path = wiki_store.write_raw_source(patient_id, source_id, text, root=patients_root)
    wiki_db.create_raw_source(
        source_id, patient_id, filename, source_type, extraction_method,
        str(written_path), document_date, db_path=db_path,
    )

    existing_pages = {
        page_path: wiki_store.read_wiki_page(patient_id, page_path, root=patients_root) or ""
        for page_path in wiki_store.list_wiki_pages(patient_id, root=patients_root)
    }
    proposed = await propose_diff(text, existing_pages, model=diff_model)
    # Normalize ONCE, here, and use that single value for the tiering
    # decision, the pending_diffs row, and the auto-apply write below. The
    # tiering gate used to normalize privately and throw the result away,
    # which left three different strings in play for one diff: a critical
    # page spelled "allergies.md." tiered as non-critical and was then
    # auto-applied (Windows drops the trailing dot on write), and the
    # spec's own "wiki/overview.md" spelling tiered correctly but landed
    # at wiki/wiki/overview.md. What is tiered must be what is recorded
    # and what is written.
    page_path = _normalize_page_path(proposed.page_path)
    requires_approval = _requires_approval(page_path, proposed.contradiction)

    diff_id = wiki_db.create_pending_diff(
        patient_id, source_id, page_path, proposed.is_new_page,
        proposed.new_page_content, proposed.contradiction,
        proposed.contradiction_note, requires_approval, db_path=db_path,
    )

    wiki_db.update_raw_source_status(source_id, "ingested", db_path=db_path)

    if not requires_approval:
        try:
            wiki_review.apply_diff(
                patient_id, diff_id, page_path, proposed.new_page_content,
                # wiki_review's own constant, not a literal: its
                # defense-in-depth tiering check keys off this exact value,
                # so the two must never drift apart.
                actor_id=config.AUTO_APPLY_ACTOR_ID,
                decision=wiki_review.AUTO_APPLY_DECISION,
                source_id=source_id, db_path=db_path, root=patients_root,
            )
        except (wiki_store.WikiRemoteConfiguredError, wiki_store.UnsafePagePathError):
            # Not routine auto-apply hiccups to absorb into the degraded
            # path below. A configured git remote means PHI could leave
            # this machine, and an unsafe page path means a write could
            # land outside the patient's wiki - both are alarms the caller
            # has to see, not stdout noise behind a normal-looking return.
            raise
        except Exception as e:
            # Partial-apply failure: pending_diffs stays at its pre-apply
            # 'pending' status (apply_diff never reached the status update),
            # surfaced later by a lint routine rather than losing this
            # ingest's already-written raw_sources/pending_diffs rows.
            print(f"[wiki_ingest] auto-apply failed for diff {diff_id}: {e}")
            # ...and a durable record of it, because the design doc's
            # orphaned-source lint cannot see this case: by now BOTH the
            # raw_sources row ('ingested') and the pending_diffs row exist,
            # so nothing is orphaned by that check's definition. Without
            # this row the only trace of a failed auto-apply is the print
            # above.
            try:
                wiki_db.create_audit_log(
                    patient_id,
                    config.AUTO_APPLY_ACTOR_ID,
                    "auto_apply_failed",
                    target=diff_id,
                    detail=json.dumps({
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "page_path": page_path,
                    }),
                    db_path=db_path,
                )
            except Exception as audit_error:
                # The audit write is best-effort: if the failure that got
                # us here was the database itself, this one fails too, and
                # turning that into a raised exception would undo the
                # graceful degradation this whole branch exists for.
                print(
                    f"[wiki_ingest] failed to record auto_apply_failed audit row "
                    f"for diff {diff_id}: {audit_error}"
                )

    return diff_id
