# Per-Patient Wiki + Grounded Query System — Design

**Date:** 2026-09-17
**Status:** Approved (design), not yet built

**Amendment (2026-09-20):** the original approval gate ("every wiki write
requires human approval, no exceptions") assumed the reviewer could judge
medical correctness. The actual reviewer is a non-clinician sibling, who
can verify "does this match the source document" but not "is this
medically right." Revised to a **tiered gate**: `allergies.md`,
`medications.md`, and any contradiction always block on a human
confirming the diff matches its source (never a clinical judgment call);
everything else auto-applies with full logging, relying on the lint
routine as the after-the-fact check for the lower-stakes pages. A
contradiction is queued as a `questions_for_doctor.md` entry rather than
something the sibling resolves themselves — the same pattern the
claim-diff chairman already uses. See "Non-negotiable safety/compliance
requirements" and the updated architecture diagram below.

**Relationship to the existing medical-RAG spec** (`2026-08-23-medical-rag-design.md`,
amended 2026-09-11): this document **replaces that spec's retrieval
strategy entirely**, and supersedes it as the design for the build phase.
The wiki approach needs no chunking, no embeddings, and no pgvector — there
is no retrieval step at query time at all; the whole per-patient wiki is
small enough to load in full. Concretely dropped: the `document_chunks`
table, the HNSW index, the Gemini `embedContent` call, and the "Prior art:
Dating App's pgvector pattern" reuse. **Kept unchanged:** multi-patient
scoping from day one, the PHI/deferred-auth posture, and the file-type
extraction dispatch (PDF text vs. scanned/image → vision LLM) for turning
a raw upload into text — ingestion still needs that step regardless of
what happens to the text afterward. The old spec's "fixed spine" idea
(always-inject current meds/problem list regardless of ranking) is what
this design generalizes into `allergies.md`/`medications.md` being
injected verbatim on every query.

## Purpose

A caregiver (currently: one of the two siblings, acting as the single
local operator) selects a family member and either uploads a new medical
record or describes a new health issue. The system maintains a small,
structured, human-reviewed "wiki" per patient — not a store of raw
documents to search over — and grounds every query in the *full* current
wiki rather than a retrieved subset of it.

This is a personal project (one family member's records today, designed
multi-patient because retrofitting that later is expensive). It is
explicitly not attempting HIPAA compliance in this build — no BAA-covered
provider is configured, no encryption at rest, no real authentication.
See "Deferred" below; this is carried over unchanged from the old spec.

## Scope for this build

1. **Ingestion** — raw source in, LLM-proposed diff out. **Tiered
   approval, not a universal gate** (revised 2026-09-20 — see decision
   note below): a diff touching `allergies.md`, `medications.md`, or
   flagged as a contradiction always blocks on a human confirmation
   before applying; every other diff auto-applies with full logging.
2. **Query** — full wiki (verbatim critical pages + everything else) fed
   into the existing claim-diff mechanism (`backend/council.py`,
   `backend/claim_diff.py`), patient-scoped. The query and its answer
   become a candidate wiki entry through the same approval path as any
   other ingest.
3. **Lint** — a periodic check that surfaces contradictions, staleness,
   and orphaned data as a review queue. Never auto-fixes.

**Deferred, unchanged from the old spec:** real authentication, encryption
at rest, a BAA-covered provider. Every table below gets an `actor_id`
column so a real identity can slot in later without reshaping anything —
until then, human-confirmed writes use one constant
(`backend.config.DEFAULT_ACTOR_ID = "local-operator"`) and auto-applied
writes use a distinct one (`backend.config.AUTO_APPLY_ACTOR_ID =
"system-auto"`), so the audit log can always tell the two apart.

## Constraints

- Document formats: PDF (text and scanned), JPG, PNG, TXT, and a plain
  typed/pasted note (no file at all — the sibling just describes what a
  doctor said).
- Multi-patient from day one; every query, every file path, every SQLite
  row is scoped by `patient_id`. This is the one property that gets an
  explicit, non-optional test (see Testing priorities).
- No local embedding model, and none needed — this design has no
  embedding step.
- LLM calls go directly to Anthropic and Google, same as the rest of this
  codebase (`backend/llm_client.py`) — no proxy in the path of real
  medical text.
- Single-operator, no login. The human-review step (ingest diffs, lint
  findings) happens through the app's UI, not a CLI — this repo already
  has one static-page-served-same-origin pattern (`GET /ui`) and this
  design reuses it rather than inventing a new one.

## Architecture

```
INGEST
Raw upload (PDF/JPG/PNG/TXT) or typed note
    │
    ▼
Extraction (local PyMuPDF for text PDF/TXT; Claude/Gemini vision for
scanned PDF page/JPG/PNG; typed notes need no extraction)
    │
    ▼
raw_sources row written (status=pending_ingest) + extracted text saved to
data/patients/<id>/raw/<source_id>.txt
    │
    ▼
LLM proposes a diff: which existing page(s) it affects, or a new page
needed. Flags contradiction with existing wiki content explicitly rather
than silently overwriting.
    │
    ▼
pending_diffs row written — NO file in wiki/ touched yet.
requires_approval = page is allergies.md/medications.md OR contradiction_flag
    │
    ├─ requires_approval = false ─────────────────────────┐
    │                                                       ▼
    │                              Auto-apply immediately: apply diff →
    │                              update index.md → append log.md →
    │                              git commit → audit_log row
    │                              (actor_id = AUTO_APPLY_ACTOR_ID)
    │
    └─ requires_approval = true
            │
            ▼
       If contradiction_flag: also append an entry to
       wiki/questions_for_doctor.md immediately (additive, not gated —
       it's a note to raise at the next appointment, not a fact change)
            │
            ▼
       Human reviews on the wiki-review page, asked only "does this
       match the source document?" — never a medical judgment
            │
            ▼ (confirmed)
       Apply diff to the .md file → update index.md → append log.md →
       git commit → approvals row (actor_id = DEFAULT_ACTOR_ID) +
       audit_log row written


QUERY
Caregiver selects Patient X, describes a new issue
    │
    ▼
Load every page under data/patients/X/wiki/ (target: well under 15-20K
tokens combined) + load allergies.md/medications.md AGAIN, separately,
verbatim — never re-summarized
    │
    ▼
Assemble one prompt: system instructions + full wiki + the new complaint
    │
    ▼
Existing claim-diff mechanism: stage1 (both models, wiki-grounded context)
→ extract → align → classify → rank → chairman synthesis
    │
    ▼
Answer returned to caregiver AND written as a candidate pending_diffs
entry (routes through the same approval path as any other ingest)


LINT (periodic, triggered manually from the wiki-review page for now)
For each patient: contradiction check across pages, staleness check
(source newer than what a page cites), index.md referencing a missing
page, raw_sources rows with no corresponding pending_diffs ever proposed
    │
    ▼
lint_findings rows written (status=open) → surfaced on wiki-review page
→ human dismisses or acts on each, never auto-resolved
```

## Data model

### File layout (per patient, git-tracked, under the existing `data/` gitignore rule)

```
data/patients/<patient_id>/
  raw/<source_id>.txt              # extracted text, immutable
  wiki/
    index.md                       # one-line summary + link per page
    overview.md                    # one-paragraph current clinical picture
    allergies.md                   # structured list, always injected verbatim
    medications.md                 # current + recent, structured
    conditions.md                  # active + resolved, with dates
    by-system/
      cardiac.md, respiratory.md, ...   # created on demand
      archive.md                   # compacted resolved/inactive detail
    log.md                         # append-only: ## [YYYY-MM-DD] ingest | source-id
    questions_for_doctor.md        # append-only: unresolved contradictions,
                                    # queued to raise at the next appointment —
                                    # never resolved by the sibling themselves
```

### SQLite (`data/patients.db`) — metadata, approvals, audit, lint

```sql
CREATE TABLE patients (
    id TEXT PRIMARY KEY,              -- uuid4 hex
    name TEXT NOT NULL,
    created_at TEXT NOT NULL          -- ISO8601
);

CREATE TABLE raw_sources (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    filename TEXT,                    -- null for a typed/pasted note
    source_type TEXT NOT NULL,        -- pdf | jpg | png | txt | note
    extraction_method TEXT NOT NULL,  -- local | vision_llm | manual
    extracted_path TEXT NOT NULL,     -- data/patients/<id>/raw/<source_id>.txt
    document_date TEXT,               -- date the record is ABOUT, not upload date
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_ingest'   -- pending_ingest | ingested | failed
);

CREATE TABLE pending_diffs (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    source_id TEXT REFERENCES raw_sources(id),      -- null for query-derived entries
    page_path TEXT NOT NULL,          -- e.g. "wiki/medications.md"
    is_new_page INTEGER NOT NULL DEFAULT 0,
    diff_content TEXT NOT NULL,       -- unified diff / before-after text
    contradiction_flag INTEGER NOT NULL DEFAULT 0,
    contradiction_note TEXT,
    requires_approval INTEGER NOT NULL,  -- page_path in {allergies.md, medications.md} OR contradiction_flag
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected | auto_applied
    proposed_at TEXT NOT NULL
);

CREATE TABLE approvals (
    id TEXT PRIMARY KEY,
    diff_id TEXT NOT NULL REFERENCES pending_diffs(id),
    actor_id TEXT NOT NULL,           -- DEFAULT_ACTOR_ID (human) or AUTO_APPLY_ACTOR_ID (system)
    decision TEXT NOT NULL,           -- approved | rejected | auto_applied
    decided_at TEXT NOT NULL,
    note TEXT
);

CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,             -- ingest_proposed | diff_approved | diff_rejected | query | lint_run | compaction
    target TEXT,                      -- page_path / source_id / diff_id
    at TEXT NOT NULL,
    detail TEXT                       -- freeform JSON
);

CREATE TABLE lint_findings (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    finding_type TEXT NOT NULL,       -- contradiction | stale_claim | missing_page | orphaned_source
    page_path TEXT,
    detail TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open | dismissed | resolved
    found_at TEXT NOT NULL
);
```

No Alembic/migration framework — at this scale a single idempotent
`CREATE TABLE IF NOT EXISTS` script (`backend/wiki_db.py`) is enough; the
old spec's Postgres migration tooling isn't needed since Postgres itself
isn't needed.

## Components

| Module | Responsibility |
|---|---|
| `backend/wiki_db.py` | SQLite connection + idempotent schema setup |
| `backend/wiki_store.py` | Read/write wiki `.md` files and raw source files on disk; patient-scoped path helpers; git commit on approved writes |
| `backend/wiki_ingest.py` | extract → LLM proposes diff → determine `requires_approval` (critical page or contradiction) → write `pending_diffs` row; if a contradiction, also append to `questions_for_doctor.md` immediately; if `requires_approval` is false, calls into `wiki_review.py`'s apply step itself, auto-applying with `AUTO_APPLY_ACTOR_ID` |
| `backend/wiki_review.py` | apply a diff to file, update `index.md` + `log.md`, git commit, write `approvals` + `audit_log` — shared by both the auto-apply path and the human-confirm path; only the caller and `actor_id` differ |
| `backend/wiki_query.py` | load full wiki for a patient (verbatim critical pages injected separately), call the existing claim-diff pipeline, write the result as a candidate `pending_diffs` entry |
| `backend/wiki_lint.py` | the four lint checks, writes `lint_findings` |
| `backend/main.py` | modified: `/api/second-opinion/analyze` gains `patient_id`; new routes for pending-diffs list/approve/reject, lint-findings list/dismiss, `GET /wiki/review` |
| `static/wiki-review.html` | new: pending diffs (critical-page/contradiction diffs only — auto-applied ones never appear here) + lint findings, confirm/reject/dismiss. Confirm is framed as "does this match the source document?", never a medical judgment |
| `static/second-opinion-result.html` | modified: add patient selector |

## Non-negotiable safety/compliance requirements

- **Structured critical fields are never LLM-summarized on the way out.**
  `allergies.md` and `medications.md` are loaded and injected as raw text,
  a second time, separately from the general wiki load — never
  re-paraphrased by an intermediate step.
- **`allergies.md`, `medications.md`, and any contradiction always
  require explicit human confirmation** before being committed — no
  exceptions, not a configurable setting. That confirmation is scoped to
  "does this match the source document," never a clinical judgment call,
  because the reviewer is a non-clinician (see 2026-09-20 amendment
  above). Everything else auto-applies, on the reasoning that a routine,
  non-critical page being briefly wrong-then-caught-by-lint is a much
  smaller risk than every update depending on a human who isn't
  qualified to catch a clinically wrong one.
- **A contradiction is never resolved by the reviewer.** It's queued in
  `questions_for_doctor.md` and blocks the affected page from updating
  until a human confirms the diff matches its source — the actual
  clinical question (which claim is true) goes to the doctor, same
  pattern as the claim-diff chairman's `questions_for_doctor` field.
- **Provenance on every claim.** Every wiki statement should be traceable
  to the `source_id`/date it came from (`pending_diffs.source_id` →
  `raw_sources`, preserved permanently even after the diff is applied, via
  the git commit message referencing the diff/source id).
- **PHI handling.** Assume all of this is PHI. Encryption at rest, real
  per-patient access control tied to authenticated identity, and a
  BAA-covered provider are all explicitly deferred (see Scope) — this
  build is a local, single-operator tool, not yet compliant, and should
  not be the sole basis for a medical decision.
- **Size ceiling.** When a patient's full wiki crosses the target token
  budget, a compaction step proposes moving resolved/inactive detail into
  `by-system/archive.md` — itself an ingest-shaped `pending_diffs` entry
  requiring approval, never an automatic deletion.

## Error handling

- Extraction fails for one raw source → `status='failed'`, surfaced on
  the review page; other sources unaffected.
- Diff-proposal LLM call fails or returns malformed output → retry once,
  then fall back to a `pending_diffs` row with no proposed diff content
  and a note that it needs manual handling — a raw source is never
  silently dropped just because the LLM couldn't propose a diff for it.
- Contradiction detected → `contradiction_flag=1` always requires
  explicit human confirmation before the page updates; there is no
  auto-apply path for a contradicting diff. The `questions_for_doctor.md`
  entry itself is written immediately regardless (additive, not a fact
  change), so the question is never lost even if the diff sits pending.
- Diff proposed for a non-critical page, no contradiction → auto-applies
  immediately; if the auto-apply step itself fails partway (file write
  succeeds, git commit fails, or vice versa), the diff's `pending_diffs`
  row stays at its pre-apply status rather than being marked
  `auto_applied` on a partial success — surfaced by the lint routine's
  orphaned-source check rather than silently lost.
- Query for a patient with no wiki pages yet → a clear "no records for
  this patient yet" response, same pattern as the old spec's empty-corpus
  rule — never an empty/silent council run.
- Wiki crosses the token budget mid-query → the query still runs against
  the current (over-budget) wiki this one time, and a compaction proposal
  is queued for review rather than silently truncating context out from
  under an in-flight answer.
- One council model failing during a wiki-grounded query → existing
  graceful degradation applies unchanged (`stage1_collect_responses`'s
  explicit ok/failed status, never silently dropped).

## Sample patient wiki (fabricated, illustrative only)

```markdown
<!-- data/patients/demo-eleanor/wiki/index.md -->
# Eleanor Vance — Wiki Index
- [Overview](overview.md) — 84F, managed hypertension + early cognitive changes
- [Allergies](allergies.md) — penicillin (confirmed 2024-03-12)
- [Medications](medications.md) — 3 active, last updated 2026-09-02
- [Conditions](conditions.md) — hypertension (active), UTI (resolved 2026-08)

<!-- allergies.md -->
# Allergies
- **Penicillin** — confirmed reaction (rash), source: discharge summary
  2024-03-12. [source: raw/a1b2c3]

<!-- medications.md -->
# Medications (current)
- Lisinopril 10mg, once daily — started 2025-06-01, for hypertension.
  [source: raw/d4e5f6]
- Warfarin 5mg, once daily — started 2026-01-15, for atrial fibrillation.
  [source: raw/g7h8i9]

~~Ibuprofen 200mg PRN~~ — superseded 2026-09-02: discontinued due to
warfarin interaction risk. [source: raw/j0k1l2]

<!-- log.md -->
## [2026-09-02] ingest | source-j0k1l2
Updated medications.md: discontinued ibuprofen (warfarin interaction),
per pharmacist note.
```

## Testing priorities

1. **Verbatim injection** (safety-critical): a regression test asserting
   the exact prompt sent to the LLM contains the literal text of
   `allergies.md`/`medications.md`, not a paraphrase or summary of it.
2. **Contradiction never silently overwritten**: a fixture source that
   conflicts with existing wiki content must produce
   `contradiction_flag=1`, `requires_approval=1`, a `questions_for_doctor.md`
   entry, and must never reach `status='approved'`/`'auto_applied'`
   without an explicit confirm call.
2a. **Tiering is correct and can't be bypassed**: a fixture diff on
    `allergies.md` or `medications.md` always produces
    `requires_approval=1` regardless of contradiction status; a fixture
    diff on any other page with no contradiction always auto-applies
    without waiting on a confirm call.
3. **Patient isolation**: every SQLite query and file path scoped by
   `patient_id` — a query for patient A must never load or return
   patient B's wiki content, raw sources, or pending diffs, under any
   input.
4. **Ingest round trip**: fixture source → proposed diff → approve →
   correct file, `index.md`, and `log.md` state → a subsequent query
   sees the new content.
5. **Lint**: one fixture per finding type (contradiction, stale claim,
   missing page, orphaned source) that reliably triggers it.
6. **Compaction trigger**: a wiki fixture near the token budget produces
   a compaction proposal at the right threshold, not before or after.

## Suggested build sequencing

This is too much for one implementation pass. Suggested phases, each
independently testable:

1. **Storage foundation** — `wiki_db.py` schema, `wiki_store.py` file
   helpers, no LLM calls yet. Testable with fixture files alone.
2. **Ingest + auto-apply path** — `wiki_ingest.py` (extraction, diff
   proposal, tiering) and `wiki_review.py`'s shared apply-a-diff
   primitive, exercised only via the non-critical/no-contradiction path.
   Closes the loop for the common case: fixture source on a non-critical
   page → auto-applied → file/`index.md`/`log.md` actually change, no UI
   needed yet.
3. **Human-confirm path** — the `wiki-review.html` page and the
   confirm/reject/list endpoints, for the critical-page/contradiction
   diffs phase 2 deliberately left un-applied. Also wires up
   `questions_for_doctor.md`.
4. **Query** — `wiki_query.py`, the `patient_id` addition to
   `/api/second-opinion/analyze`, the patient selector on the existing
   `/ui` page. Depends on phase 3 existing (needs real wiki content to
   query against).
5. **Lint** — `wiki_lint.py` + its section of the review page. Fully
   independent of phase 4; could run in parallel with it instead.

Each phase is a candidate for its own `/gsd:plan-phase`-style plan and
PR, rather than one large branch.

## Open questions / left for implementation time

- Exact token budget threshold for the size ceiling — pick empirically
  once real data exists, same open question as the earlier full-context-
  vs-RAG discussion.
- Whether the SQLite write (approvals/audit rows) and the git commit
  (file write) need to be made atomic/retryable against each other, or
  whether "retry the whole approval action if either half fails" is
  sufficient at this scale — not fully resolved here, flagged for
  implementation time rather than blocking this design.
- `document_date` capture UX: does the human confirm it during review, or
  does the extraction step attempt to infer it? Either is compatible with
  this schema; not decided here.
- Whether new-page creation (`is_new_page=1`) should also require
  confirmation even on a non-critical page — a brand-new condition/system
  page is a bigger structural change than an edit to an existing one, but
  wasn't part of the 2026-09-20 tiering decision. Currently falls under
  "auto-applies if not critical/contradicting," same as any other
  non-critical diff; revisit if that turns out too permissive in
  practice.
- Explicit page-to-page cross-references (e.g. a new condition page
  linking to the existing `by-system` page it relates to) — raised during
  a comparison against a similar "compiled wiki" pattern seen elsewhere,
  not yet adopted into this design. Independent of the approval-tiering
  question above; still open.
