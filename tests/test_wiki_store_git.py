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

    # Check what actually got staged/committed. --format= suppresses the commit
    # message, leaving only the changed file paths, so this can't be satisfied
    # by the commit message ("Add patient-a notes") incidentally mentioning
    # "patient-a" - it has to be the real staged files.
    files = _git(["log", "--name-only", "--format=", "-1"], tmp_path).stdout
    assert "patient-a" in files
    assert "patient-b" not in files


def test_commit_wiki_change_rejects_dash_prefixed_id(tmp_path):
    """A dash-prefixed patient_id like '-A' must be rejected outright before any
    git command runs, not treated as a literal pathspec via the '--' separator.

    Earlier rounds of this fix made '-A' work by forcing it through '--' as a
    literal pathspec. That's a narrower, riskier contract than simply refusing
    IDs shaped like git flags/pathspec-magic in the first place (see the
    pathspec-magic-prefix test below for why '--' alone isn't sufficient
    protection). Rejecting the whole shape up front is the safer contract.
    """
    wiki_store.write_wiki_page("patient-a", "notes.md", "Patient A notes", root=tmp_path)

    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.commit_wiki_change("-A", "Update dash dir", root=tmp_path)

    # Validation happens before ensure_repo()/git are invoked at all.
    assert not (tmp_path / ".git").exists()


@pytest.mark.parametrize(
    "magic_patient_id",
    [":/", ":(glob)**", ":(top)", ":!patient-b"],
)
def test_commit_wiki_change_rejects_git_pathspec_magic_prefixes(tmp_path, magic_patient_id):
    """Git pathspec "magic" prefixes like ':/' or ':(glob)' are not path traversal
    - they resolve to a normal single-segment child of root, so they'd pass a
    pure containment check - but git interprets a leading ':' as pathspec magic
    rather than a literal directory name once it reaches `git add`. ':/' in
    particular means "match everything from the repo root", which would stage
    every patient's files into one commit - a cross-patient PHI leak into the
    audit trail. This must be rejected before any git command runs, and the '--'
    separator (which only stops flag injection, a different mechanism) does not
    stop it.
    """
    wiki_store.write_wiki_page("patient-a", "notes.md", "Patient A notes", root=tmp_path)
    wiki_store.write_wiki_page("patient-b", "notes.md", "Patient B notes", root=tmp_path)

    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.commit_wiki_change(magic_patient_id, "malicious", root=tmp_path)

    # No repo, no staging, no commit - validation happens before any git call.
    assert not (tmp_path / ".git").exists()


def test_ensure_repo_rejects_a_remote_added_after_creation(tmp_path):
    """The "no remote" invariant must hold on every call, not just at creation.

    Something (a bug, a misconfigured tool, a future change) could add a
    remote to this repo after it's first created. Since no PHI may ever leave
    this machine, ensure_repo() must catch that on the very next call, not
    only check for it once up front.
    """
    wiki_store.ensure_repo(tmp_path)

    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/not-a-real-remote.git"],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(wiki_store.WikiRemoteConfiguredError):
        wiki_store.ensure_repo(tmp_path)
