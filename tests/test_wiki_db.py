"""Tests for the patient wiki SQLite schema and patient CRUD (backend/wiki_db.py)."""

from backend import wiki_db


def _table_names(db_path):
    with wiki_db.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


def test_init_schema_creates_all_tables(tmp_path):
    db_path = tmp_path / "patients.db"

    wiki_db.init_schema(db_path)

    assert _table_names(db_path) >= {
        "patients", "raw_sources", "pending_diffs",
        "approvals", "audit_log", "lint_findings",
    }


def test_init_schema_is_idempotent(tmp_path):
    db_path = tmp_path / "patients.db"

    wiki_db.init_schema(db_path)
    wiki_db.init_schema(db_path)  # must not raise

    assert "patients" in _table_names(db_path)


def test_create_patient_returns_id_and_persists(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)

    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    patient = wiki_db.get_patient(patient_id, db_path)
    assert patient is not None
    assert patient["name"] == "Eleanor Vance"
    assert patient["id"] == patient_id


def test_get_patient_returns_none_when_not_found(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)

    assert wiki_db.get_patient("nonexistent-id", db_path) is None


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


def test_create_approval_persists(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)
    diff_id = wiki_db.create_pending_diff(
        patient_id, None, "overview.md", is_new_page=True, diff_content="a",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=False, db_path=db_path,
    )

    approval_id = wiki_db.create_approval(
        diff_id, "system-auto", "auto_applied", note=None, db_path=db_path
    )

    with wiki_db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
    assert row["diff_id"] == diff_id
    assert row["actor_id"] == "system-auto"
    assert row["decision"] == "auto_applied"
    assert row["note"] is None
    assert row["decided_at"] is not None


def test_create_audit_log_persists(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    entry_id = wiki_db.create_audit_log(
        patient_id, "system-auto", "diff_approved", target="some-diff-id",
        detail='{"page_path": "overview.md"}', db_path=db_path,
    )

    with wiki_db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE id = ?", (entry_id,)
        ).fetchone()
    assert row["patient_id"] == patient_id
    assert row["actor_id"] == "system-auto"
    assert row["action"] == "diff_approved"
    assert row["target"] == "some-diff-id"
    assert row["detail"] == '{"page_path": "overview.md"}'
    assert row["at"] is not None


def test_create_audit_log_target_and_detail_are_optional(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    entry_id = wiki_db.create_audit_log(patient_id, "system-auto", "lint_run", db_path=db_path)

    with wiki_db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE id = ?", (entry_id,)
        ).fetchone()
    assert row["target"] is None
    assert row["detail"] is None


def test_update_pending_diff_status_changes_status(tmp_path):
    db_path = tmp_path / "patients.db"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)
    diff_id = wiki_db.create_pending_diff(
        patient_id, None, "overview.md", is_new_page=True, diff_content="a",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=False, db_path=db_path,
    )

    wiki_db.update_pending_diff_status(diff_id, "auto_applied", db_path)

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "auto_applied"
