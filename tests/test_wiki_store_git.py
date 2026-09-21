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
