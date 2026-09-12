# Second Opinion — Context Handoff

Read this first in a fresh session. This project was renamed/moved from
`LLM Council\llm-council` on 2026-09-12; gstack's project-slug tooling still
files artifacts under the old slug (`LLMCouncil`), so `/context-restore`
won't find them automatically here — hence this file.

## What this is

An extension of the open-source `karpathy/llm-council` app (a 3-stage
multi-LLM deliberation tool) into **Second Opinion**: a tool for two
siblings acting as family caregivers to cross-check medical decisions for
an elderly family member, given two real incidents where single-model AI
answers were wrong or dangerously disagreed with each other, in a
healthcare system where geriatric-specific medical training is thin.

## Read these, in order

1. **`TODOS.md`** (repo root) — 5 deferred/tracked items with full context,
   effort, priority, and dependencies. Start here for "what's next."
2. **Design doc:**
   `~/.gstack/projects/LLMCouncil/vijay-master-design-20260911-162903.md`
   — the full approved design: problem statement, demand evidence, the
   claim-diff mechanism (replaces stage 2/3's ranking+synthesis with a
   disagreement-preserving mechanism), architecture, data model, success
   criteria. This is the source of truth for *why* things are built the
   way they are.
3. **Test plan:**
   `~/.gstack/projects/LLMCouncil/vijay-master-eng-review-test-plan-20260912-032601.md`
   — affected areas, edge cases, and critical paths from the eng review.
4. **Implementation tasks (JSONL):**
   `~/.gstack/projects/LLMCouncil/tasks-eng-review-20260912-121010.jsonl`
   — 14 tasks (T1-T14) synthesized from the eng review, with priority,
   effort estimates, and which finding each one came from.
5. **Amended upstream spec:**
   `docs/superpowers/specs/2026-08-23-medical-rag-design.md` (+ its
   companion `2026-08-23-embeddings-comparison.html`) — the original
   Postgres+pgvector/RAG spec, now amended to reflect the claim-diff
   mechanism and schema additions (`document_date`, page provenance).

## Current state (as of 2026-09-12)

- **Security fixes already applied and committed-ready:** path-traversal
  fix (UUID validation on `conversation_id`), backend bound to
  `127.0.0.1` instead of `0.0.0.0`, `npm audit fix` applied to the
  frontend. These are live in this folder's working code.
- **Council model config corrected:** `backend/config.py` uses
  `gemini/gemini-3.1-pro-preview` and `claude/claude-sonnet-4-5-20250929`
  (matching what this machine's OmniRoute instance actually exposes),
  not the original `google/...`/`anthropic/...` IDs.
- **Git identity is still unset in this repo** (`git config user.name`/
  `user.email` return nothing) — nothing has been committed yet; the
  working tree has the fixes above plus `TODOS.md` and the two spec docs,
  uncommitted.
- **`origin` still points at `https://github.com/karpathy/llm-council.git`**
  (the upstream repo) — a separate empty repo at
  `https://github.com/singhhub/second-opinion.git` was set up as the
  intended destination but never pushed to (blocked on the git identity
  issue above, then superseded by this folder move).
- **Design and eng review are both complete and approved.** Nothing has
  been implemented yet — the next work is T1-T14 in the tasks JSONL /
  TODOS.md, starting with test infra bootstrap, the direct-call LLM
  client, and eval-case reconstruction (independent, can start in
  parallel per the worktree strategy in the eng review).

## Environment notes

- No test framework installed yet (`pytest`/`pytest-asyncio` need adding —
  this is T1).
- `bun` is not installed in this environment — any gstack tool that shells
  out to it (`gstack-review-log`, `gstack-review-read`,
  `gstack-builder-profile`) will silently fail; doesn't affect the
  artifacts listed above.
- OmniRoute (the local LLM proxy used by the base LLM Council app) runs at
  `http://localhost:20128` — separate from the direct-call client this
  project's design calls for building (T2), which is what actually needs
  to run for the medical-RAG features.
