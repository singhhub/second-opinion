# Patient Wiki — Phase 3: Auto-Apply Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Naming note:** this is "Phase 3" in this project's own plan-file sequence (Phase 1 = storage foundation, Phase 2 = ingest/diff-proposal, both already merged to `master`). It implements the spec's own **"2. Ingest + auto-apply path"** section under "Suggested build sequencing" — the design doc's phase numbers and this project's plan-file numbers are offset by one; this note exists so that offset is never silently lost again.

**Goal:** Close the loop for the common case — a `pending_diffs` row with `requires_approval=False` gets applied automatically: the wiki page is written, `index.md`/`log.md` are updated, the change is git-committed, and `approvals`/`audit_log` rows record it. No UI yet — critical-page/contradiction diffs still sit in `pending_diffs` untouched, waiting for Phase 4's human-confirm path.

**Architecture:** `backend/wiki_review.py` (new) owns a single shared primitive, `apply_diff()`, that turns an already-decided diff (approved by a human, or auto-approved by the system) into an actual file change: write the page via `wiki_store.write_wiki_page`, deterministically rebuild `index.md` from what's actually on disk (no LLM summarization), append a `log.md` entry, commit via `wiki_store.commit_wiki_change`, then record an `approvals` row and an `audit_log` row and flip `pending_diffs.status`. `backend/wiki_ingest.py`'s `ingest_source()` (Phase 2, already built) is the only caller in this phase — it calls `apply_diff()` immediately after writing a `pending_diffs` row, but only when `requires_approval` is `False`, using a new `AUTO_APPLY_ACTOR_ID` constant. Phase 4 will add a second caller (the human-confirm endpoint) using `DEFAULT_ACTOR_ID` instead — `apply_diff()` doesn't know or care which caller invoked it, per the spec's Components table ("shared by both the auto-apply path and the human-confirm path; only the caller and `actor_id` differ").

**Tech Stack:** No new dependencies — reuses `wiki_store.py`'s existing file I/O and git-commit helpers (Phase 1) and `wiki_db.py`'s existing connection/schema helpers (Phase 1/2).

**Spec:** `docs/superpowers/specs/2026-09-17-patient-wiki-design.md` (amended 2026-09-20) — see its "Architecture" INGEST diagram's auto-apply branch, the "Components" table rows for `wiki_review.py`/`wiki_ingest.py`, and the "Error handling" section's auto-apply partial-failure requirement.

## Global Constraints

- **Auto-apply only ever fires for `requires_approval=False`.** A diff on `allergies.md`, `medications.md`, or with `contradiction_flag=1` (including the needs-review fallback page) must never be auto-applied — this is Phase 2's existing `_requires_approval` gate, unchanged here; this phase only adds what happens on the `False` branch.
- **Partial-apply failure must not corrupt state.** If the file/index/log writes and git commit succeed but something after that fails (or vice versa — e.g. the git commit itself fails), `pending_diffs.status` must be left at its pre-apply value (`'pending'`), never marked `'auto_applied'` on a partial success. The design doc explicitly relies on a later lint routine (not built in this phase) to surface this case — this phase's job is only to not paper over it by marking success prematurely.
- **`ingest_source()` must not crash when auto-apply fails.** A failed auto-apply is a degraded-but-recorded outcome (`pending_diffs` stays `'pending'`), not an exception that propagates out of `ingest_source()` and loses the caller's `raw_sources`/`pending_diffs` rows that were already correctly written moments earlier.
- **No LLM calls in this phase.** `apply_diff()` is pure file/git/DB mechanics — the diff's content was already decided by Phase 2's `propose_diff()`. `index.md` regeneration is mechanical (derived from `wiki_store.list_wiki_pages()`), never an LLM summarization call — consistent with the design doc's "critical fields are never LLM-summarized on the way out" principle, generalized here to avoid adding an LLM cost to every auto-apply for a summary line the design doc doesn't actually require this phase to produce well.
- Every test uses real SQLite (`tmp_path`) and a real filesystem (`tmp_path`) — no LLM calls exist in this phase's code to mock, so no mocking is needed for `wiki_review.py`'s own tests; `wiki_ingest.py`'s existing `query_model`/`query_vision` mocks continue to apply in its tests, unchanged from Phase 2.
- `actor_id` is populated for the first time in this phase: `AUTO_APPLY_ACTOR_ID = "system-auto"` (new, added to `backend/config.py`). `DEFAULT_ACTOR_ID = "local-operator"` (the human-confirm actor) is **not** added in this phase — it belongs to Phase 4, which is the only phase that uses it; adding an unused constant now would be premature.

---

### Task 1: `approvals` + `audit_log` CRUD, and a `pending_diffs` status setter

**Files:**
- Modify: `backend/wiki_db.py`
- Test: `tests/test_wiki_db.py` (append)

**Interfaces:**
- Consumes: `wiki_db.get_connection`, `wiki_db._now`, `wiki_db.DB_PATH` (existing)
- Produces: `wiki_db.create_approval(diff_id: str, actor_id: str, decision: str, note: Optional[str] = None, db_path: Path = DB_PATH) -> str`, `wiki_db.create_audit_log(patient_id: str, actor_id: str, action: str, target: Optional[str] = None, detail: Optional[str] = None, db_path: Path = DB_PATH) -> str`, `wiki_db.update_pending_diff_status(diff_id: str, status: str, db_path: Path = DB_PATH) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_wiki_db.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_db.py -v`
Expected: FAIL with `AttributeError: module 'backend.wiki_db' has no attribute 'create_approval'`

- [ ] **Step 3: Write the implementation**

Append to `backend/wiki_db.py`:

```python
def create_approval(
    diff_id: str,
    actor_id: str,
    decision: str,
    note: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> str:
    approval_id = uuid.uuid4().hex
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO approvals (id, diff_id, actor_id, decision, decided_at, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (approval_id, diff_id, actor_id, decision, _now(), note),
        )
    return approval_id


def create_audit_log(
    patient_id: str,
    actor_id: str,
    action: str,
    target: Optional[str] = None,
    detail: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> str:
    entry_id = uuid.uuid4().hex
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (id, patient_id, actor_id, action, target, at, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_id, patient_id, actor_id, action, target, _now(), detail),
        )
    return entry_id


def update_pending_diff_status(diff_id: str, status: str, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE pending_diffs SET status = ? WHERE id = ?", (status, diff_id)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_db.py -v`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/wiki_db.py tests/test_wiki_db.py
git commit -m "feat: add approvals/audit_log CRUD and pending_diffs status setter"
```

---

### Task 2: `AUTO_APPLY_ACTOR_ID` config constant

**Files:**
- Modify: `backend/config.py`

**Interfaces:**
- Produces: `config.AUTO_APPLY_ACTOR_ID: str`

This is a one-line, non-independently-testable addition (a constant), folded into its own tiny task rather than a full TDD task since there is no behavior to test — Task 4 exercises it indirectly by asserting the actor_id recorded in `approvals` rows.

- [ ] **Step 1: Add the constant**

Append to `backend/config.py`:

```python

# Actor id for the wiki's auto-apply path (backend/wiki_review.py's shared
# apply_diff(), called from wiki_ingest.py for non-critical/no-contradiction
# diffs) - distinct from a future human reviewer's actor id so the audit
# log can always tell the two apart. See the design doc's Scope section.
AUTO_APPLY_ACTOR_ID = "system-auto"
```

- [ ] **Step 2: Commit**

```bash
git add backend/config.py
git commit -m "feat: add AUTO_APPLY_ACTOR_ID config constant"
```

---

### Task 3: `wiki_review.apply_diff()` — the shared apply-a-diff primitive

**Files:**
- Create: `backend/wiki_review.py`
- Test: `tests/test_wiki_review.py`

**Interfaces:**
- Consumes: `wiki_store.write_wiki_page`, `wiki_store.read_wiki_page`, `wiki_store.list_wiki_pages`, `wiki_store.commit_wiki_change`, `wiki_store.PATIENTS_ROOT` (Phase 1); `wiki_db.create_approval`, `wiki_db.create_audit_log`, `wiki_db.update_pending_diff_status`, `wiki_db.DB_PATH` (Task 1)
- Produces: `wiki_review.apply_diff(patient_id: str, diff_id: str, page_path: str, new_content: str, actor_id: str, decision: str, source_id: Optional[str] = None, note: Optional[str] = None, db_path: Path = wiki_db.DB_PATH, root: Path = wiki_store.PATIENTS_ROOT) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wiki_review.py
"""Tests for backend/wiki_review.py's shared apply-a-diff primitive.

apply_diff() is called from two places (only one exists yet):
wiki_ingest.py's auto-apply path (this plan's Task 4) and a future
Phase 4 human-confirm endpoint. These tests exercise apply_diff()
directly, independent of either caller.
"""

from unittest.mock import patch

import pytest

from backend import wiki_db, wiki_review, wiki_store


def _setup_patient(db_path):
    wiki_db.init_schema(db_path)
    return wiki_db.create_patient("Eleanor Vance", db_path)


def _create_diff(patient_id, db_path, page_path="overview.md", requires_approval=False):
    # source_id=None here deliberately: pending_diffs.source_id is a real
    # foreign key against raw_sources(id), and these tests exercise
    # apply_diff() in isolation from ingestion, so there's no raw_sources
    # row to reference. apply_diff() itself takes its OWN source_id
    # argument separately below (used only for the log/commit message
    # text, never FK-checked) - the two are unrelated parameters that
    # happen to share a name.
    return wiki_db.create_pending_diff(
        patient_id, None, page_path, is_new_page=True,
        diff_content="# Overview\nGeneral clinical picture.",
        contradiction_flag=False, contradiction_note=None,
        requires_approval=requires_approval, db_path=db_path,
    )


async def test_apply_diff_writes_page_content(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    wiki_review.apply_diff(
        patient_id, diff_id, "overview.md", "# Overview\nGeneral clinical picture.",
        actor_id="system-auto", decision="auto_applied", source_id="source-1",
        db_path=db_path, root=patients_root,
    )

    content = wiki_store.read_wiki_page(patient_id, "overview.md", root=patients_root)
    assert content == "# Overview\nGeneral clinical picture."


async def test_apply_diff_rebuilds_index_md_with_new_page(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path, page_path="medications.md")

    wiki_review.apply_diff(
        patient_id, diff_id, "medications.md", "# Medications\n- Lisinopril",
        actor_id="system-auto", decision="auto_applied", source_id="source-1",
        db_path=db_path, root=patients_root,
    )

    index_content = wiki_store.read_wiki_page(patient_id, "index.md", root=patients_root)
    assert "[Medications](medications.md)" in index_content


async def test_apply_diff_rebuild_index_excludes_index_itself(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)

    diff_1 = _create_diff(patient_id, db_path, page_path="overview.md")
    wiki_review.apply_diff(
        patient_id, diff_1, "overview.md", "# Overview", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )
    diff_2 = _create_diff(patient_id, db_path, page_path="conditions.md")
    wiki_review.apply_diff(
        patient_id, diff_2, "conditions.md", "# Conditions", actor_id="system-auto",
        decision="auto_applied", source_id="source-2", db_path=db_path, root=patients_root,
    )

    index_content = wiki_store.read_wiki_page(patient_id, "index.md", root=patients_root)
    assert "[Overview](overview.md)" in index_content
    assert "[Conditions](conditions.md)" in index_content
    assert "(index.md)" not in index_content  # index.md must never link to itself
    assert index_content.count("- [") == 2


async def test_apply_diff_appends_log_md_entry(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    wiki_review.apply_diff(
        patient_id, diff_id, "overview.md", "# Overview", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )

    log_content = wiki_store.read_wiki_page(patient_id, "log.md", root=patients_root)
    assert "source-1" in log_content
    assert "overview.md" in log_content


async def test_apply_diff_second_call_appends_not_overwrites_log(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)

    diff_1 = _create_diff(patient_id, db_path, page_path="overview.md")
    wiki_review.apply_diff(
        patient_id, diff_1, "overview.md", "# Overview v1", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )
    diff_2 = _create_diff(patient_id, db_path, page_path="conditions.md")
    wiki_review.apply_diff(
        patient_id, diff_2, "conditions.md", "# Conditions", actor_id="system-auto",
        decision="auto_applied", source_id="source-2", db_path=db_path, root=patients_root,
    )

    log_content = wiki_store.read_wiki_page(patient_id, "log.md", root=patients_root)
    assert "source-1" in log_content
    assert "source-2" in log_content


async def test_apply_diff_commits_to_git(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    wiki_review.apply_diff(
        patient_id, diff_id, "overview.md", "# Overview", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )

    import subprocess
    result = subprocess.run(
        ["git", "log", "--oneline"], cwd=patients_root, capture_output=True, text=True, check=True
    )
    assert diff_id in result.stdout
    assert "overview.md" in result.stdout


async def test_apply_diff_writes_approval_row(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    wiki_review.apply_diff(
        patient_id, diff_id, "overview.md", "# Overview", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )

    with wiki_db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM approvals WHERE diff_id = ?", (diff_id,)
        ).fetchone()
    assert row["actor_id"] == "system-auto"
    assert row["decision"] == "auto_applied"


async def test_apply_diff_writes_audit_log_row(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    wiki_review.apply_diff(
        patient_id, diff_id, "overview.md", "# Overview", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )

    with wiki_db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE patient_id = ? AND target = ?",
            (patient_id, diff_id),
        ).fetchone()
    assert row["actor_id"] == "system-auto"
    assert row["action"] == "diff_approved"


async def test_apply_diff_updates_pending_diff_status(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    wiki_review.apply_diff(
        patient_id, diff_id, "overview.md", "# Overview", actor_id="system-auto",
        decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
    )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "auto_applied"


async def test_apply_diff_partial_failure_leaves_status_untouched(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path)

    with patch.object(
        wiki_review.wiki_store, "commit_wiki_change", side_effect=RuntimeError("git commit failed")
    ):
        with pytest.raises(RuntimeError):
            wiki_review.apply_diff(
                patient_id, diff_id, "overview.md", "# Overview", actor_id="system-auto",
                decision="auto_applied", source_id="source-1", db_path=db_path, root=patients_root,
            )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "pending"
    with wiki_db.get_connection(db_path) as conn:
        approval_count = conn.execute(
            "SELECT COUNT(*) AS n FROM approvals WHERE diff_id = ?", (diff_id,)
        ).fetchone()["n"]
    assert approval_count == 0
    # The page write itself already happened before the commit failed -
    # that's the accepted partial state per the design doc's Error
    # handling section (surfaced later by lint, not rolled back here).
    content = wiki_store.read_wiki_page(patient_id, "overview.md", root=patients_root)
    assert content == "# Overview"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.wiki_review'`

- [ ] **Step 3: Write the implementation**

```python
# backend/wiki_review.py
"""Applying an already-decided wiki diff to disk.

apply_diff() is the shared primitive behind both the auto-apply path
(backend/wiki_ingest.py, for non-critical/no-contradiction diffs) and a
future human-confirm path (Phase 4) - only the caller and actor_id
differ, per the design doc's Components table. This module never decides
WHETHER to apply a diff (that's wiki_ingest.py's _requires_approval gate,
or Phase 4's human confirmation) - it only turns an already-decided diff
into an actual file change, git commit, and audit trail.
"""

from datetime import date
from pathlib import Path
from typing import Optional

from . import wiki_db, wiki_store


def _rebuild_index(patient_id: str, root: Path = wiki_store.PATIENTS_ROOT) -> str:
    """
    Deterministically rebuild index.md's page list from what's actually on
    disk - no LLM call. The design doc forbids LLM-summarizing critical
    pages on the way out; generating a real per-page clinical summary for
    every page is out of scope for this phase, so each entry is just a
    title-cased link (e.g. "- [Medications](medications.md)"), not the
    one-line clinical summary shown in the design doc's illustrative
    sample. index.md itself is always excluded from its own listing.
    """
    pages = [p for p in wiki_store.list_wiki_pages(patient_id, root) if p != "index.md"]
    lines = ["# Wiki Index", ""]
    for page_path in sorted(pages):
        title = Path(page_path).stem.replace("-", " ").replace("_", " ").title()
        lines.append(f"- [{title}]({page_path})")
    return "\n".join(lines) + "\n"


def apply_diff(
    patient_id: str,
    diff_id: str,
    page_path: str,
    new_content: str,
    actor_id: str,
    decision: str,
    source_id: Optional[str] = None,
    note: Optional[str] = None,
    db_path: Path = wiki_db.DB_PATH,
    root: Path = wiki_store.PATIENTS_ROOT,
) -> None:
    """
    Apply a decided diff: write the page, rebuild index.md, append a
    log.md entry, git commit, then record approvals + audit_log rows and
    flip pending_diffs.status to `decision`.

    If anything from the git commit onward raises, pending_diffs.status
    is deliberately left at its pre-apply value - never marked `decision`
    on a partial success. The design doc's Error handling section relies
    on a later lint routine (not built in this phase) to surface that
    case; this function's job is only to not paper over it by recording
    success prematurely. The exception propagates to the caller.
    """
    wiki_store.write_wiki_page(patient_id, page_path, new_content, root=root)

    index_content = _rebuild_index(patient_id, root)
    wiki_store.write_wiki_page(patient_id, "index.md", index_content, root=root)

    log_entry = (
        f"## [{date.today().isoformat()}] ingest | {source_id or 'n/a'}\n"
        f"Updated {page_path}. (diff {diff_id})\n\n"
    )
    existing_log = wiki_store.read_wiki_page(patient_id, "log.md", root=root) or ""
    wiki_store.write_wiki_page(patient_id, "log.md", existing_log + log_entry, root=root)

    wiki_store.commit_wiki_change(
        patient_id,
        f"Apply diff {diff_id}: update {page_path} (source {source_id or 'n/a'})",
        root=root,
    )

    wiki_db.create_approval(diff_id, actor_id, decision, note=note, db_path=db_path)
    wiki_db.create_audit_log(patient_id, actor_id, "diff_approved", target=diff_id, db_path=db_path)
    wiki_db.update_pending_diff_status(diff_id, decision, db_path=db_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_review.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/wiki_review.py tests/test_wiki_review.py
git commit -m "feat: add wiki_review.apply_diff, the shared apply-a-diff primitive"
```

---

### Task 4: Wire `ingest_source()` to auto-apply non-tiered diffs

**Files:**
- Modify: `backend/wiki_ingest.py`
- Test: `tests/test_wiki_ingest.py` (append)

**Interfaces:**
- Consumes: `wiki_review.apply_diff` (Task 3), `config.AUTO_APPLY_ACTOR_ID` (Task 2)
- Produces: no new public interface — `ingest_source()`'s signature and return type are unchanged; this task only changes its internal behavior.

- [ ] **Step 1: Write the failing tests**

In `tests/test_wiki_ingest.py`, add `config, wiki_review` to the existing `from backend import wiki_db, wiki_ingest, wiki_store` import line at the top of the file, so it reads `from backend import config, wiki_db, wiki_ingest, wiki_review, wiki_store`. Then append these tests:

```python
async def test_ingest_source_auto_applies_non_tiered_diff(tmp_path):
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
        diff_id = await wiki_ingest.ingest_source(
            patient_id, "note", note_text="General checkup notes.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "auto_applied"
    content = wiki_store.read_wiki_page(patient_id, "overview.md", root=patients_root)
    assert content == "# Overview\nGeneral clinical picture."
    with wiki_db.get_connection(db_path) as conn:
        approval = conn.execute(
            "SELECT * FROM approvals WHERE diff_id = ?", (diff_id,)
        ).fetchone()
    assert approval["actor_id"] == config.AUTO_APPLY_ACTOR_ID
    assert approval["decision"] == "auto_applied"


async def test_ingest_source_does_not_auto_apply_tiered_diff(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "allergies.md", "is_new_page": True,
        "new_page_content": "# Allergies\n- Penicillin",
        "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        await wiki_ingest.ingest_source(
            patient_id, "note", note_text="Penicillin allergy noted.",
            db_path=db_path, patients_root=patients_root,
        )

    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "pending"
    assert wiki_store.list_wiki_pages(patient_id, root=patients_root) == []


async def test_ingest_source_does_not_auto_apply_contradiction_diff(tmp_path):
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
    assert diffs[0]["status"] == "pending"
    assert wiki_store.list_wiki_pages(patient_id, root=patients_root) == []


async def test_ingest_source_auto_apply_failure_does_not_crash_ingest(tmp_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    wiki_db.init_schema(db_path)
    patient_id = wiki_db.create_patient("Eleanor Vance", db_path)

    valid_json = json.dumps({
        "page_path": "overview.md", "is_new_page": True,
        "new_page_content": "# Overview", "contradiction": False,
        "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ), patch.object(
        wiki_ingest.wiki_review, "apply_diff", side_effect=RuntimeError("git commit failed")
    ):
        diff_id = await wiki_ingest.ingest_source(
            patient_id, "note", note_text="General checkup notes.",
            db_path=db_path, patients_root=patients_root,
        )

    assert diff_id is not None
    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wiki_ingest.py -v -k auto_appl`
Expected: FAIL — `test_ingest_source_auto_applies_non_tiered_diff` and the failure-containment test fail because `pending_diffs.status` stays `"pending"` (no auto-apply wired yet); `test_ingest_source_does_not_auto_apply_tiered_diff`/`_contradiction_diff` already pass today (nothing auto-applies yet) and should be treated as already-green regression guards, not new red tests.

- [ ] **Step 3: Write the implementation**

Add these imports to the top of `backend/wiki_ingest.py` (alongside the existing ones):

```python
from . import config, wiki_review
```

In `ingest_source`, locate the existing block that writes the `pending_diffs` row and updates the raw source status:

```python
    diff_id = wiki_db.create_pending_diff(
        patient_id, source_id, proposed.page_path, proposed.is_new_page,
        proposed.new_page_content, proposed.contradiction,
        proposed.contradiction_note, requires_approval, db_path=db_path,
    )

    wiki_db.update_raw_source_status(source_id, "ingested", db_path=db_path)

    return diff_id
```

Replace it with:

```python
    diff_id = wiki_db.create_pending_diff(
        patient_id, source_id, proposed.page_path, proposed.is_new_page,
        proposed.new_page_content, proposed.contradiction,
        proposed.contradiction_note, requires_approval, db_path=db_path,
    )

    wiki_db.update_raw_source_status(source_id, "ingested", db_path=db_path)

    if not requires_approval:
        try:
            wiki_review.apply_diff(
                patient_id, diff_id, proposed.page_path, proposed.new_page_content,
                actor_id=config.AUTO_APPLY_ACTOR_ID, decision="auto_applied",
                source_id=source_id, db_path=db_path, root=patients_root,
            )
        except Exception as e:
            # Partial-apply failure: pending_diffs stays at its pre-apply
            # 'pending' status (apply_diff never reached the status update),
            # surfaced later by a lint routine rather than losing this
            # ingest's already-written raw_sources/pending_diffs rows.
            print(f"[wiki_ingest] auto-apply failed for diff {diff_id}: {e}")

    return diff_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wiki_ingest.py -v`
Expected: PASS (all tests in the file, including the 4 new ones)

- [ ] **Step 5: Run the full wiki test suite together**

Run: `uv run pytest tests/test_wiki_db.py tests/test_wiki_review.py tests/test_wiki_ingest.py tests/test_wiki_ingest_diff.py tests/test_wiki_ingest_extract.py tests/test_wiki_store.py tests/test_wiki_store_git.py -v`
Expected: PASS (all)

- [ ] **Step 6: Run the entire project test suite to check for regressions**

Run: `uv run pytest -v --deselect tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict`
Expected: PASS (no regressions in claim_diff.py/council.py/main.py/eval_harness.py tests). The `--deselect` avoids the one pre-existing test in this suite that makes real, billed LLM API calls when live credentials are present — never run it without explicit confirmation first (this project's established convention).

- [ ] **Step 7: Commit**

```bash
git add backend/wiki_ingest.py tests/test_wiki_ingest.py
git commit -m "feat: wire ingest_source to auto-apply non-tiered diffs"
```

---

## Self-review notes

- **Spec coverage:** the auto-apply branch of the design doc's INGEST architecture diagram (apply diff → update index.md → append log.md → git commit → audit_log row, actor_id=AUTO_APPLY_ACTOR_ID) ✓ Task 3 + Task 4; the Components table's `wiki_review.py` row (apply a diff to file, update index.md + log.md, git commit, write approvals + audit_log — shared by both paths) ✓ Task 3; the Components table's `wiki_ingest.py` row's auto-apply clause ✓ Task 4; the tiering gate itself (`allergies.md`/`medications.md`/contradiction never auto-apply) ✓ unchanged from Phase 2, re-verified by Task 4's two negative tests (spec Testing priority 2a); the Error handling section's partial-failure requirement (pending_diffs stays at pre-apply status, never marked applied on a partial success) ✓ Task 3's dedicated test + Task 4's failure-containment test. **Explicitly out of scope, left to Phase 4:** `questions_for_doctor.md` wiring, the human-confirm UI/endpoints, `DEFAULT_ACTOR_ID` — per the spec's own "Suggested build sequencing," item 3 assigns all three to the next phase, not this one.
- **No placeholders:** every step has real, complete code — no TBD/TODO, no "similar to Task N."
- **Type consistency:** `db_path: Path = DB_PATH` / `root: Path = PATIENTS_ROOT` defaults match Phase 1/2's existing convention exactly. `decision` is reused as both `approvals.decision` and the literal value written to `pending_diffs.status` — this is intentional and matches the spec's schema comments precisely (both columns share the same `'approved' | 'rejected' | 'auto_applied'` value space for a successful outcome), not an accidental type-punning.
