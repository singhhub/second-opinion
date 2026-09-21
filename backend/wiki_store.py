"""Filesystem storage for patient wiki pages and raw sources.

Each patient's wiki lives under PATIENTS_ROOT/<patient_id>/. See
ensure_repo()/commit_wiki_change() (added in the next task) for why this
directory gets its own separate, remote-less git repo rather than being
tracked by the project's own.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Optional

PATIENTS_ROOT = Path("data/patients")

# Strict allowlist for patient_id/source_id: must start with an alphanumeric
# character, followed by alphanumerics, underscore, dot, or hyphen. This is
# checked before any filesystem resolution, and it exists for more than
# ordinary path traversal (which the containment check below already catches):
# strings like ":/" or ":(glob)**" resolve to a normal single-segment child of
# root, so they pass containment, but git interprets a leading ":" as pathspec
# magic rather than a literal path once the ID reaches `git add`. Rejecting
# the whole shape up front is simpler and safer than trying to make git accept
# it literally.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class UnsafePagePathError(ValueError):
    """Raised when a path component would resolve outside its containment directory."""


class WikiRemoteConfiguredError(RuntimeError):
    """Raised if the patient wiki's local audit-trail repo ever has a remote configured.

    No PHI may ever leave this machine, so this is checked on every call to
    ensure_repo(), not just at repo creation.
    """


def _validate_safe_id(root: Path, id_value: str, id_name: str) -> None:
    """
    Validate that an ID (patient_id, source_id) doesn't escape root when joined,
    and that its character set is safe. Raises UnsafePagePathError if the ID
    contains path traversal sequences, or doesn't match the strict character
    allowlist (checked first, before any filesystem resolution, to fail fast on
    an obviously-unsafe shape).
    """
    if not _SAFE_ID.match(id_value):
        raise UnsafePagePathError(f"{id_name} {id_value!r} contains disallowed characters")
    root_resolved = root.resolve()
    resolved = (root / id_value).resolve()
    if resolved.parent != root_resolved:
        raise UnsafePagePathError(f"{id_name} {id_value!r} escapes the root directory")


def patient_dir(patient_id: str, root: Path = PATIENTS_ROOT) -> Path:
    _validate_safe_id(root, patient_id, "patient_id")
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
    # Validate the exact filename we're about to write, not source_id alone -
    # the ".txt" suffix changes what gets resolved, so validating source_id
    # by itself would check a different path than the one actually written.
    filename = f"{source_id}.txt"
    _validate_safe_id(target_dir, filename, "source_id")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
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


def ensure_repo(root: Path = PATIENTS_ROOT) -> None:
    """
    Idempotently git-init a repo scoped to root, if one doesn't already
    exist. Never configures a remote - this repo's only job is a local
    audit trail for PHI that must never leave this machine. A fixed
    local identity avoids depending on global git config being set up.

    Also asserts, on EVERY call (not just when the repo is first created),
    that no remote has been configured. "No remote" is the single most
    safety-critical property in this module - it must hold continuously,
    not just at creation time, in case something added one in between calls.
    """
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "second-opinion-wiki"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "wiki@localhost"], cwd=root, check=True)

    remotes = subprocess.run(
        ["git", "remote"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    if remotes.strip():
        raise WikiRemoteConfiguredError(
            f"patient wiki repo at {root} has a remote configured "
            f"({remotes.strip()!r}) - PHI must never leave this machine"
        )


def commit_wiki_change(patient_id: str, message: str, root: Path = PATIENTS_ROOT) -> None:
    # Validate patient_id (path traversal, and disallowed characters such as a
    # leading "-" or ":" that could be misread as a git flag or pathspec magic)
    # before any git command runs. Also use '--' below so patient_id is always
    # treated as a pathspec argument, never as a flag, as defense in depth.
    patient_dir(patient_id, root)
    ensure_repo(root)
    subprocess.run(["git", "add", "--", patient_id], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)
