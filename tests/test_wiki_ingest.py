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
