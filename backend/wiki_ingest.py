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
