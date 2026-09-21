"""Tests for the patient wiki's local, remote-less git audit trail
(backend/wiki_store.py). Split from test_wiki_store.py because this is a
distinct concern (subprocess/git, not pure file I/O) worth its own
reviewable unit - see the design doc's PHI-handling requirement for why
"never configures a remote" gets its own explicit test, not just
incidental coverage.
"""

import subprocess

import pytest

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


def test_commit_wiki_change_rejects_traversal_patient_id(tmp_path):
    """Path traversal in patient_id must raise UnsafePagePathError before git runs."""
    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.commit_wiki_change("..", "malicious", root=tmp_path)


def test_commit_wiki_change_respects_patient_isolation(tmp_path):
    """Write files for two patients, commit only one, verify the other's changes aren't staged."""
    wiki_store.write_wiki_page("patient-a", "notes.md", "Patient A notes", root=tmp_path)
    wiki_store.write_wiki_page("patient-b", "notes.md", "Patient B notes", root=tmp_path)

    wiki_store.commit_wiki_change("patient-a", "Add patient-a notes", root=tmp_path)

    # Check that only patient-a is in the commit
    log = _git(["log", "--name-only", "-1"], tmp_path).stdout
    assert "patient-a" in log
    assert "patient-b" not in log


def test_commit_wiki_change_treats_dash_prefixed_id_as_pathspec(tmp_path):
    """Verify dash-prefixed patient_id like '-A' is treated as pathspec, not git flag.

    If '-A' were interpreted as git's -A/--all flag instead of a directory name,
    the entire repo would be staged. By verifying only the dash-prefixed directory
    appears in the commit (and not another modified patient directory), we confirm
    the -- separator correctly prevents flag injection.
    """
    # Create files for two patients
    wiki_store.write_wiki_page("patient-a", "notes.md", "Patient A notes", root=tmp_path)
    wiki_store.write_wiki_page("patient-b", "notes.md", "Patient B notes", root=tmp_path)

    # Create a directory with literal dash-prefixed name
    dash_dir = tmp_path / "-A"
    dash_dir.mkdir(parents=True, exist_ok=True)
    (dash_dir / "file.txt").write_text("test file")

    # Initialize repo and commit initial state
    wiki_store.ensure_repo(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True)

    # Modify both the dash-dir and patient-a
    (dash_dir / "file.txt").write_text("modified")
    wiki_store.write_wiki_page("patient-a", "notes.md", "Updated notes", root=tmp_path)

    # Commit only the dash directory using commit_wiki_change with patient_id="-A"
    wiki_store.commit_wiki_change("-A", "Update dash dir", root=tmp_path)

    # Verify only -A is in the commit, not patient-a (which would be if -A was a flag)
    log = _git(["log", "--name-only", "-1"], tmp_path).stdout
    assert "-A" in log
    assert "patient-a" not in log
