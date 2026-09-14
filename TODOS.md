# TODOS

## Second Opinion (medical RAG extension)

### Extract a shared claim-diff pipeline function (dedupe main.py / eval_harness.py)

**What:** `backend/main.py`'s `analyze_question` endpoint and
`backend/eval_harness.py`'s `run_new_mechanism` now both inline the same
6-stage pipeline (extract_claims x2 -> align_claims -> classify_compatibility
per pair -> build_ranked_diff_items -> synthesize_claim_diff_chairman).
Extract a shared `async def run_claim_diff_pipeline(question, response_a,
response_b) -> (agreed, conflicting, alignment, diff_items)` into
`claim_diff.py` or `council.py` and call it from both.

**Why:** Flagged by the final whole-branch review on the live-data-result-
screen plan (2026-09-13). The eval harness exists specifically to measure
the mechanism users actually get — a future change to one copy (a filter, a
threshold, a re-ranking) would silently diverge eval results from
production behavior, with no test that would notice.

**Context:** Explicitly deferred out of that plan's fix wave (a refactor
touching two files is out of scope for a single fix pass) — reviewer called
it "acceptable as an immediate fast-follow rather than a merge blocker, but
it should not sit."

**Effort:** S
**Priority:** P2
**Depends on:** None — can happen anytime.

### Build phase — full Postgres+pgvector data model and retrieval

**What:** Build the full spec (`docs/superpowers/specs/2026-08-23-medical-rag-design.md`),
including retrieval (fixed spine, hybrid keyword search), multi-patient
scoping, and a real persistence layer + richer UI for the claim-diff result
(the legacy `storage.py`/React frontend were removed once the claim-diff
mechanism became the only mechanism — this phase needs its own, built for
the claim-diff format from the start, not an update to the old ones).

**Why:** This is the actual product, not just the mechanism proof — the
validation phase only checks whether claim-diff preserves real disagreements.

**Context:** Only starts once the validation phase (see
`~/.gstack/projects/LLMCouncil/vijay-master-design-20260911-162903.md`)
proves the mechanism against the fixed eval set. The two schema additions
from the amended spec (`document_date` on `documents`, page provenance on
`document_chunks`) land here too, along with re-running ingestion against
the real schema (not migrating flat-file embeddings — re-embedding is
cheap at this scale).

**Effort:** L
**Priority:** P1
**Depends on:** Validation phase (this plan) passing its success criteria.

### Verify BAA availability for individual, non-enterprise accounts

**What:** Confirm whether Anthropic/Google's BAA (Business Associate
Agreement) coverage actually applies at an individual account tier, not
just enterprise.

**Why:** The direct-call design decision is partly justified by BAA
availability, which is currently asserted, not confirmed — a real
unverified assumption under a security-relevant decision.

**Context:** Gates real ingestion/query of the actual record backlog, not
the validation phase (which touches no new patient data). Zero-code
research task — just needs checking.

**Effort:** S
**Priority:** P1
**Depends on:** None — can happen anytime.

### Capture appointment notes as a document type

**What:** A 30-second voice/text note per doctor appointment, ingested like
any other document.

**Why:** Without it, nothing in the pipeline captures the actual thing being
cross-checked — the doctor's own stated plan and reasoning. Retrieved
records alone (labs, discharge summaries) never show what the doctor
concluded from them.

**Context:** Flagged during the office-hours session as high-value and
currently uncaptured. Not needed for the validation phase since that phase
doesn't touch retrieval at all.

**Effort:** S
**Priority:** P2
**Depends on:** The build phase (ingestion pipeline existing).

### Consider a third BAA-eligible model for majority/minority detection

**What:** Add a third model (e.g. an OpenAI enterprise account) to the
council, restoring genuine majority-vs-minority claim detection instead of
today's binary AGREED/CONFLICTING/UNCONFIRMED states.

**Why:** With 3 models, a 2-vs-1 split is a real majority signal — richer
than what 2 models can express.

**Context:** Not needed to validate the mechanism. Worth revisiting only
once the 2-model version is proven — adding a new provider integration
before that would be premature.

**Effort:** M
**Priority:** P3
**Depends on:** BAA verification (above) and the build phase.

### Revisit lean validation-phase choices if they prove insufficient

**What:** Two simplifications made during eng review — (a) LLM-judge-only
compatibility classification instead of a dedicated NLI model, (b)
retry-then-fallback claim parsing instead of provider structured-output/
function-calling — may need upgrading if they turn out too slow, costly,
or unreliable in practice.

**Why:** Both were chosen for "boring by default, prove the mechanism
cheaply" reasons — correct as a starting point, not necessarily correct
forever.

**Context:** Only actionable once real validation runs actually surface a
problem with either choice — not something to preemptively build.

**Effort:** M each
**Priority:** P3
**Depends on:** Running the validation phase first.
