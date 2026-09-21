"""SQLite storage for the patient wiki: raw sources, pending diffs,
approvals, audit log, and lint findings.

Schema only - wiki page content itself lives in
data/patients/<id>/wiki/*.md via wiki_store.py, not here. See the design
doc's Data model section for the full schema rationale (patient_id on
every table, requires_approval on pending_diffs, distinct actor ids for
human vs. auto-applied writes).
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(os.getenv("WIKI_DB_PATH", "data/patients.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_sources (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    filename TEXT,
    source_type TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    extracted_path TEXT NOT NULL,
    document_date TEXT,
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_ingest'
);

CREATE TABLE IF NOT EXISTS pending_diffs (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    source_id TEXT REFERENCES raw_sources(id),
    page_path TEXT NOT NULL,
    is_new_page INTEGER NOT NULL DEFAULT 0,
    diff_content TEXT NOT NULL,
    contradiction_flag INTEGER NOT NULL DEFAULT 0,
    contradiction_note TEXT,
    requires_approval INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    proposed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    diff_id TEXT NOT NULL REFERENCES pending_diffs(id),
    actor_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    at TEXT NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS lint_findings (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    finding_type TEXT NOT NULL,
    page_path TEXT,
    detail TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    found_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def create_patient(name: str, db_path: Path = DB_PATH) -> str:
    patient_id = uuid.uuid4().hex
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO patients (id, name, created_at) VALUES (?, ?, ?)",
            (patient_id, name, _now()),
        )
    return patient_id


def get_patient(patient_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, created_at FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
    return dict(row) if row else None


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
