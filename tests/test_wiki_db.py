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
