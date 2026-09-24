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
    sample. index.md and log.md (system pages) are always excluded from
    the listing.
    """
    pages = [p for p in wiki_store.list_wiki_pages(patient_id, root) if p not in ("index.md", "log.md")]
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
