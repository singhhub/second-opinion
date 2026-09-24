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

    # decision="approved" (a human), not "auto_applied": medications.md is a
    # tiered page, which only ever reaches apply_diff() through the
    # human-confirm path - the auto-apply path is blocked from it.
    wiki_review.apply_diff(
        patient_id, diff_id, "medications.md", "# Medications\n- Lisinopril",
        actor_id="local-operator", decision="approved", source_id="source-1",
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


# --- Defense in depth: a tiered page can never be auto-applied -------------
#
# wiki_ingest's _requires_approval gate is the first line of defense. This
# is the second, independent one: even if a caller hands apply_diff() a
# critical page with decision="auto_applied" (a regression in that gate, or
# a future caller with different assumptions), nothing is written.

@pytest.mark.parametrize(
    "page_path",
    [
        "allergies.md",
        "medications.md",
        "wiki/medications.md",
        "medications.md.",
        "Medications.MD",
        "by-system/allergies.md",
        "wiki\\allergies.md",
    ],
)
async def test_apply_diff_refuses_to_auto_apply_a_tiered_page(tmp_path, page_path):
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path, page_path=page_path)

    with pytest.raises(wiki_review.TieringViolationError):
        wiki_review.apply_diff(
            patient_id, diff_id, page_path, "# Medications\n- Warfarin 5mg",
            actor_id="system-auto", decision="auto_applied", source_id="source-1",
            db_path=db_path, root=patients_root,
        )

    # Nothing at all happened: no file, no approval row, no status change.
    assert wiki_store.list_wiki_pages(patient_id, root=patients_root) == []
    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "pending"
    with wiki_db.get_connection(db_path) as conn:
        approval_count = conn.execute(
            "SELECT COUNT(*) AS n FROM approvals WHERE diff_id = ?", (diff_id,)
        ).fetchone()["n"]
    assert approval_count == 0


async def test_apply_diff_allows_a_human_to_apply_a_tiered_page(tmp_path):
    """The guard is scoped to the system applying a diff by itself - a
    human confirmation (decision="approved") is exactly how a tiered page
    is meant to be applied, and must still work."""
    db_path = tmp_path / "patients.db"
    patients_root = tmp_path / "patients"
    patient_id = _setup_patient(db_path)
    diff_id = _create_diff(patient_id, db_path, page_path="allergies.md")

    wiki_review.apply_diff(
        patient_id, diff_id, "allergies.md", "# Allergies\n- Penicillin",
        actor_id="local-operator", decision="approved", source_id="source-1",
        db_path=db_path, root=patients_root,
    )

    content = wiki_store.read_wiki_page(patient_id, "allergies.md", root=patients_root)
    assert content == "# Allergies\n- Penicillin"
    diffs = wiki_db.list_pending_diffs(patient_id, db_path=db_path)
    assert diffs[0]["status"] == "approved"
