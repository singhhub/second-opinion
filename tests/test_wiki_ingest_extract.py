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


# PyMuPDF refuses to *save* a zero-page document ("cannot save with zero
# pages"), so a hand-built one is the only way to exercise the real
# degenerate case: a PDF that opens fine and yields no text at all.
_PDF_NO_PAGES = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [] /Count 0 >>
endobj
trailer
<< /Root 1 0 R >>
%%EOF
"""


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


async def test_extract_text_malformed_pdf_raises_extraction_error():
    with pytest.raises(wiki_ingest.ExtractionError):
        await wiki_ingest.extract_text(b"not a pdf", "pdf")


async def test_extract_text_txt_invalid_utf8_raises_extraction_error():
    with pytest.raises(wiki_ingest.ExtractionError):
        await wiki_ingest.extract_text(b"\xff\xfe not valid utf-8", "txt")


async def test_extract_text_pdf_with_no_pages_raises_extraction_error():
    """A zero-page PDF used to return ("", "local") - empty text silently
    became a proposed medical-wiki diff built from nothing."""
    with patch.object(wiki_ingest.llm_client, "query_vision", new=AsyncMock()) as mock_vision:
        with pytest.raises(wiki_ingest.ExtractionError):
            await wiki_ingest.extract_text(_PDF_NO_PAGES, "pdf")

    assert mock_vision.await_count == 0


async def test_extract_text_empty_txt_raises_extraction_error():
    with pytest.raises(wiki_ingest.ExtractionError):
        await wiki_ingest.extract_text(b"", "txt")


async def test_extract_text_whitespace_only_txt_raises_extraction_error():
    with pytest.raises(wiki_ingest.ExtractionError):
        await wiki_ingest.extract_text(b"   \n\t  \n", "txt")
