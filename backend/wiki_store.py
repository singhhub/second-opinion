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
