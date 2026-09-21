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


@pytest.mark.parametrize("note_text", ["", "   ", "\n\t  \n"])
async def test_ingest_source_empty_note_text_raises(note_text):
    """An empty note would otherwise be written to raw/ and handed to the
    diff-proposal LLM as "NEW SOURCE:\\n", whose invention becomes a
    proposed medical-wiki diff."""
    with pytest.raises(ValueError):
        await wiki_ingest.ingest_source("patient-a", "note", note_text=note_text)


# --- Tiering rule (_requires_approval) ------------------------------------
#
# page_path arrives verbatim from LLM-generated JSON, so the fixed tiering
# rule must survive every plausible formatting variant of the same page -
# a miss here silently produces requires_approval=0 for an allergy or
# medication edit.

@pytest.mark.parametrize(
    "page_path, contradiction, expected",
    [
        ("medications.md", False, True),
        ("allergies.md", False, True),
        ("Medications.md", False, True),
        ("ALLERGIES.MD", False, True),
        ("wiki/medications.md", False, True),
        ("./medications.md", False, True),
        ("  medications.md  ", False, True),
        ("medications.md ", False, True),
        ("by-system/medications.md", False, True),
        ("wiki\\allergies.md", False, True),
        (wiki_ingest.NEEDS_REVIEW_PAGE_PATH, False, True),
        ("wiki/" + wiki_ingest.NEEDS_REVIEW_PAGE_PATH, False, True),
        ("overview.md", True, True),
        ("by-system/cardiac.md", True, True),
        ("overview.md", False, False),
        ("by-system/cardiac.md", False, False),
        ("conditions.md", False, False),
    ],
)
def test_requires_approval_page_path_variants(page_path, contradiction, expected):
    assert wiki_ingest._requires_approval(page_path, contradiction) is expected


async def test_ingest_source_tiered_page_with_prefix_still_requires_approval(tmp_path):
    """End-to-end proof of the tiering fix: the design spec's own schema
    comment writes this page as "wiki/medications.md"."""
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "wiki/Medications.md", "is_new_page": True,
        "new_page_content": "# Medications\n- Warfarin 5mg",
        "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        await wiki_ingest.ingest_source(
            patient_id, "note", note_text="Started Warfarin 5mg.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["requires_approval"] == 1


# --- Failure paths --------------------------------------------------------

async def test_ingest_source_extraction_failure_records_failed_raw_source(tmp_path):
    """A failed extraction must leave a status='failed' row so the review
    page can surface it - the upload must never just vanish."""
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    with patch.object(
        wiki_ingest.llm_client, "query_vision", new=AsyncMock(return_value=None)
    ), patch.object(wiki_ingest.llm_client, "query_model", new=AsyncMock()) as mock_model:
        with pytest.raises(wiki_ingest.ExtractionError):
            await wiki_ingest.ingest_source(
                patient_id, "jpg", file_bytes=b"fake-jpg-bytes", filename="scan.jpg",
                db_path=db_path, patients_root=patients_root,
            )

    with wiki_db.get_connection(db_path) as conn:
        rows = [
            dict(row) for row in conn.execute(
                "SELECT * FROM raw_sources WHERE patient_id = ?", (patient_id,)
            ).fetchall()
        ]

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["filename"] == "scan.jpg"
    assert rows[0]["source_type"] == "jpg"
    # No text exists, so nothing was written and no diff was proposed.
    assert not (wiki_store.raw_dir(patient_id, patients_root) / f"{rows[0]['id']}.txt").exists()
    assert wiki_db.list_pending_diffs(patient_id, db_path=db_path) == []
    assert mock_model.await_count == 0


async def test_ingest_source_unknown_patient_raises_before_writing_phi(tmp_path):
    """PHI must not hit disk under data/patients/<bogus-id>/ before the
    foreign key on raw_sources gets a chance to complain."""
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    bogus_patient_id = "0" * 32

    with patch.object(wiki_ingest.llm_client, "query_model", new=AsyncMock()) as mock_model:
        with pytest.raises(ValueError):
            await wiki_ingest.ingest_source(
                bogus_patient_id, "note",
                note_text="Patient reports chest pain radiating to left arm.",
                db_path=db_path, patients_root=patients_root,
            )

    assert not (patients_root / bogus_patient_id).exists()
    assert mock_model.await_count == 0
