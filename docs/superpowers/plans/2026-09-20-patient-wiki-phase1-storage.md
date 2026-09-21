# Patient Wiki — Phase 1: Storage Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the storage foundation for the per-patient wiki — a SQLite schema for metadata/approvals/audit, and patient-scoped file I/O for wiki pages and raw sources — with zero LLM dependency, fully testable via fixtures.

**Architecture:** Two independent modules. `backend/wiki_db.py` owns the SQLite schema (`patients`, `raw_sources`, `pending_diffs`, `approvals`, `audit_log`, `lint_findings`) and a connection helper. `backend/wiki_store.py` owns patient-scoped file I/O for wiki pages and raw sources, plus a **separate, remote-less git repo rooted at `data/patients/`** — this is the mechanism behind the spec's "local git history gives the audit trail for free, no PHI in the remote repo": the project's own `.gitignore` already excludes `data/` entirely from the main repo and its GitHub remote, so patient wiki history needs its own nested repo that is never configured with a remote, ever. Every function takes an explicit `db_path`/`root` parameter (defaulting to the real location) so tests run fully isolated via `tmp_path`, matching this codebase's existing test-isolation pattern (`tests/conftest.py`'s `_isolate_llm_disk_cache`).

**Tech Stack:** Python stdlib only — `sqlite3`, `pathlib`, `subprocess` (for git). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-patient-wiki-design.md` (amended 2026-09-20)

## Global Constraints

- Multi-patient from day one; every file path and SQLite row is scoped by `patient_id` — this is the one property that gets explicit, non-optional test coverage (spec's Testing priorities #3), not just incidental.
- No PHI ever enters the main project git repo or its GitHub remote. `data/` is already gitignored at the project root; patient wiki data gets its own separate, local-only git repo instead.
- Single-operator, no real auth yet. `actor_id` columns exist on `approvals`/`audit_log` for a future real identity, but this phase doesn't populate or use them — that's Phase 2/3's concern once diffs actually get approved or auto-applied.
- No LLM calls anywhere in this phase. Everything is testable with `tmp_path` fixtures alone.

---

### Task 1: SQLite schema + patient CRUD

**Files:**
- Create: `backend/wiki_db.py`
- Test: `tests/test_wiki_db.py`

**Interfaces:**
- Produces: `wiki_db.DB_PATH: Path`, `wiki_db.get_connection(db_path: Path = DB_PATH) -> contextmanager[sqlite3.Connection]`, `wiki_db.init_schema(db_path: Path = DB_PATH) -> None`, `wiki_db.create_patient(name: str, db_path: Path = DB_PATH) -> str`, `wiki_db.get_patient(patient_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wiki_db.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.wiki_db'`

- [ ] **Step 3: Write the implementation**

```python
# backend/wiki_db.py
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
from typing import Any, Dict, Optional

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_db.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/wiki_db.py tests/test_wiki_db.py
git commit -m "feat: add patient wiki SQLite schema + patient CRUD"
```

---

### Task 2: Patient-scoped wiki page + raw source file storage

**Files:**
- Create: `backend/wiki_store.py`
- Test: `tests/test_wiki_store.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent module)
- Produces: `wiki_store.PATIENTS_ROOT: Path`, `wiki_store.UnsafePagePathError(ValueError)`, `wiki_store.patient_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path`, `wiki_store.wiki_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path`, `wiki_store.raw_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path`, `wiki_store.write_raw_source(patient_id: str, source_id: str, text: str, root: Path = PATIENTS_ROOT) -> Path`, `wiki_store.read_wiki_page(patient_id: str, page_path: str, root: Path = PATIENTS_ROOT) -> Optional[str]`, `wiki_store.write_wiki_page(patient_id: str, page_path: str, content: str, root: Path = PATIENTS_ROOT) -> Path`, `wiki_store.list_wiki_pages(patient_id: str, root: Path = PATIENTS_ROOT) -> List[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wiki_store.py
"""Tests for patient wiki file storage helpers (backend/wiki_store.py)."""

import pytest

from backend import wiki_store


def test_write_and_read_wiki_page_round_trip(tmp_path):
    patient_id = "patient-a"

    wiki_store.write_wiki_page(patient_id, "allergies.md", "# Allergies\n- Penicillin", root=tmp_path)
    content = wiki_store.read_wiki_page(patient_id, "allergies.md", root=tmp_path)

    assert content == "# Allergies\n- Penicillin"


def test_read_wiki_page_returns_none_when_missing(tmp_path):
    assert wiki_store.read_wiki_page("patient-a", "allergies.md", root=tmp_path) is None


def test_write_wiki_page_creates_parent_directories(tmp_path):
    wiki_store.write_wiki_page("patient-a", "by-system/cardiac.md", "# Cardiac", root=tmp_path)

    path = wiki_store.wiki_dir("patient-a", tmp_path) / "by-system" / "cardiac.md"
    assert path.exists()
    assert path.read_text() == "# Cardiac"


def test_page_path_traversal_is_rejected(tmp_path):
    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.write_wiki_page("patient-a", "../../etc/passwd", "malicious", root=tmp_path)


def test_page_path_cannot_reach_another_patients_files(tmp_path):
    wiki_store.write_wiki_page("patient-a", "allergies.md", "patient A data", root=tmp_path)

    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.read_wiki_page("patient-a", "../patient-b/allergies.md", root=tmp_path)


def test_list_wiki_pages_returns_relative_paths_sorted(tmp_path):
    wiki_store.write_wiki_page("patient-a", "medications.md", "meds", root=tmp_path)
    wiki_store.write_wiki_page("patient-a", "allergies.md", "allergies", root=tmp_path)
    wiki_store.write_wiki_page("patient-a", "by-system/cardiac.md", "cardiac", root=tmp_path)

    pages = wiki_store.list_wiki_pages("patient-a", root=tmp_path)

    assert pages == ["allergies.md", "by-system/cardiac.md", "medications.md"]


def test_write_raw_source_creates_file(tmp_path):
    path = wiki_store.write_raw_source("patient-a", "source-1", "extracted text", root=tmp_path)

    assert path.exists()
    assert path.read_text() == "extracted text"
    assert path.parent.name == "raw"


def test_patient_dirs_never_overlap_across_patients(tmp_path):
    assert wiki_store.patient_dir("patient-a", tmp_path) != wiki_store.patient_dir("patient-b", tmp_path)
    assert wiki_store.wiki_dir("patient-a", tmp_path) != wiki_store.wiki_dir("patient-b", tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.wiki_store'`

- [ ] **Step 3: Write the implementation**

```python
# backend/wiki_store.py
"""Filesystem storage for patient wiki pages and raw sources.

Each patient's wiki lives under PATIENTS_ROOT/<patient_id>/. See
ensure_repo()/commit_wiki_change() (added in the next task) for why this
directory gets its own separate, remote-less git repo rather than being
tracked by the project's own.
"""

from pathlib import Path
from typing import List, Optional

PATIENTS_ROOT = Path("data/patients")


class UnsafePagePathError(ValueError):
    """Raised when a page_path would resolve outside a patient's wiki dir."""


def patient_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path:
    return root / patient_id


def wiki_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path:
    return patient_dir(patient_id, root) / "wiki"


def raw_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path:
    return patient_dir(patient_id, root) / "raw"


def _resolve_safe_page_path(patient_id: str, page_path: str, root: Path = PATIENTS_ROOT) -> Path:
    """
    Resolve page_path (e.g. "allergies.md" or "by-system/cardiac.md")
    against this patient's wiki dir, refusing anything that would escape
    it - patient isolation applies to paths, not just SQLite queries.
    """
    base = wiki_dir(patient_id, root).resolve()
    candidate = (base / page_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise UnsafePagePathError(f"page_path {page_path!r} escapes the wiki directory")
    return candidate


def write_raw_source(patient_id: str, source_id: str, text: str, root: Path = PATIENTS_ROOT) -> Path:
    target_dir = raw_dir(patient_id, root)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{source_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def read_wiki_page(patient_id: str, page_path: str, root: Path = PATIENTS_ROOT) -> Optional[str]:
    path = _resolve_safe_page_path(patient_id, page_path, root)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_wiki_page(patient_id: str, page_path: str, content: str, root: Path = PATIENTS_ROOT) -> Path:
    path = _resolve_safe_page_path(patient_id, page_path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def list_wiki_pages(patient_id: str, root: Path = PATIENTS_ROOT) -> List[str]:
    base = wiki_dir(patient_id, root)
    if not base.exists():
        return []
    return sorted(
        str(p.relative_to(base)).replace("\\", "/")
        for p in base.rglob("*.md")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_store.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/wiki_store.py tests/test_wiki_store.py
git commit -m "feat: add patient-scoped wiki page + raw source file storage"
```

---

### Task 3: Local, remote-less git audit trail for the wiki

**Files:**
- Modify: `backend/wiki_store.py`
- Test: `tests/test_wiki_store_git.py`

**Interfaces:**
- Consumes: `wiki_store.PATIENTS_ROOT`, `wiki_store.patient_dir` (from Task 2)
- Produces: `wiki_store.ensure_repo(root: Path = PATIENTS_ROOT) -> None`, `wiki_store.commit_wiki_change(patient_id: str, message: str, root: Path = PATIENTS_ROOT) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wiki_store_git.py
"""Tests for the patient wiki's local, remote-less git audit trail
(backend/wiki_store.py). Split from test_wiki_store.py because this is a
distinct concern (subprocess/git, not pure file I/O) worth its own
reviewable unit - see the design doc's PHI-handling requirement for why
"never configures a remote" gets its own explicit test, not just
incidental coverage.
"""

import subprocess

from backend import wiki_store


def _git(args, root):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )


def test_commit_wiki_change_creates_a_commit(tmp_path):
    wiki_store.write_wiki_page("patient-a", "allergies.md", "# Allergies", root=tmp_path)

    wiki_store.commit_wiki_change("patient-a", "Add allergies page", root=tmp_path)

    log = _git(["log", "--oneline"], tmp_path).stdout
    assert "Add allergies page" in log


def test_ensure_repo_never_configures_a_remote(tmp_path):
    wiki_store.ensure_repo(tmp_path)

    remotes = _git(["remote"], tmp_path).stdout
    assert remotes.strip() == ""


def test_ensure_repo_is_idempotent(tmp_path):
    wiki_store.ensure_repo(tmp_path)
    wiki_store.ensure_repo(tmp_path)  # must not raise

    assert (tmp_path / ".git").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_store_git.py -v`
Expected: FAIL with `AttributeError: module 'backend.wiki_store' has no attribute 'ensure_repo'`

- [ ] **Step 3: Add the git functions to the implementation**

Append to `backend/wiki_store.py` (add `import subprocess` to the top of the file alongside the existing imports):

```python
def ensure_repo(root: Path = PATIENTS_ROOT) -> None:
    """
    Idempotently git-init a repo scoped to root, if one doesn't already
    exist. Never configures a remote - this repo's only job is a local
    audit trail for PHI that must never leave this machine. A fixed
    local identity avoids depending on global git config being set up.
    """
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "second-opinion-wiki"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "wiki@localhost"], cwd=root, check=True)


def commit_wiki_change(patient_id: str, message: str, root: Path = PATIENTS_ROOT) -> None:
    ensure_repo(root)
    subprocess.run(["git", "add", patient_id], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)
```

Also add `import subprocess` near the top of `backend/wiki_store.py`, alongside the existing `from pathlib import Path` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_store_git.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full wiki test suite together**

Run: `uv run pytest tests/test_wiki_db.py tests/test_wiki_store.py tests/test_wiki_store_git.py -v`
Expected: PASS (15 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/wiki_store.py tests/test_wiki_store_git.py
git commit -m "feat: add local, remote-less git audit trail for the patient wiki"
```

---

## Self-review notes

- **Spec coverage:** schema (all 6 tables, Task 1) ✓, patient-scoped file I/O (Task 2) ✓, patient isolation as explicit non-incidental tests (Task 2's traversal + cross-patient tests) ✓, local git audit trail with no PHI in the remote (Task 3) ✓, zero LLM calls anywhere ✓. `actor_id` population, `pending_diffs`/`approvals`/`audit_log` row-writing, and the ingest/query/lint logic itself are explicitly Phase 2+ (they need real diffs to write, which don't exist until ingestion is built) — not a gap in this phase, per the spec's own build sequencing.
- **No placeholders:** every step has real, complete code — no TBD/TODO, no "similar to Task N."
- **Type consistency:** `db_path: Path = DB_PATH` is consistent across every `wiki_db` function; `root: Path = PATIENTS_ROOT` is consistent across every `wiki_store` function; `UnsafePagePathError` is the one exception type raised by both the path-traversal test and the implementation.
