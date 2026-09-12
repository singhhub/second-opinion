# Medical Records RAG — Design

**Date:** 2026-08-23
**Status:** Approved; partially superseded — see note below

**Amendment (2026-09-11):** A follow-up office-hours session (design doc at
`~/.gstack/projects/LLMCouncil/vijay-master-design-20260911-162903.md`)
revised two decisions in this spec after a cross-model review found the
original stage 2/3 reuse would likely bury a real disagreement between
models (the exact failure this project exists to prevent):

- **Stage 2/3 mechanism** (anonymized peer ranking → single synthesized
  answer) is replaced by a mechanical claim-diff (AGREED/CONFLICTING/
  UNCONFIRMED) plus a chairman that preserves disagreement instead of
  resolving it into one answer. See that doc's "Claim-Diff Mechanism"
  section for the actual definition.
- **Build sequencing**: rather than building the full Postgres+pgvector
  schema below immediately, a validation phase first proves the new stage
  2/3 mechanism against real historical Q&A plus synthetic adversarial
  cases, before the data model and ingestion pipeline in this document get
  built. The data model's core shape (patients/documents/document_chunks,
  pgvector, multi-patient from day one) is unchanged and still the target
  for that later phase — but see the two schema additions below, caught in
  a later review as non-deferrable.
- **Schema additions** (not yet reflected in the `CREATE TABLE` statements
  below, needed before real ingestion — not needed for the validation
  phase): a `document_date` column on `documents` (drug-interaction
  reasoning and "current med list" are meaningless without knowing which
  record is newest), and page-number provenance on `document_chunks` (every
  extracted fact touching a number needs a source-page citation, given OCR
  is the dominant quality risk in this pipeline).
- **Corrected BAA justification**: this document's original wording ("both
  offer BAAs on their direct enterprise APIs") assumed BAA availability on
  an individual/personal account, which is unverified, not confirmed. The
  real justification for direct calls is data hygiene by default and
  future-productization readiness, not a currently-applicable legal
  requirement — two siblings handling their own relative's records aren't a
  HIPAA-covered entity today. This gates real ingestion/query of the record
  backlog, not the validation phase (which touches no new patient data).
- **The "fixed spine"** (always-injected current med list, problem list,
  latest labs, regardless of retrieval ranking) is a real build-phase
  feature raised during review with no home yet in this spec — it belongs
  in the retrieval design once the validation phase is complete, as a
  constraint on top of top-K similarity search, not a replacement for it.
- **Graceful degradation is also revised.** The "Error handling" section
  below still says a failing council model just continues with whatever
  responses succeeded, unchanged. The design doc's Claim-Diff Mechanism
  overrides this for the redesigned stage 2/3: stage 1 must return one
  entry per *configured* model with an explicit `ok`/`failed` status, not
  silently drop failures — otherwise a single failed call turns a real
  disagreement into a false `AGREED` result through the error path itself.

## Purpose

Extend LLM Council so it can answer questions about a family member's medical
history. A caregiver (or the patient/doctor themselves) uploads medical
records over time; the system retrieves the relevant pieces of that history
for a given question and runs them through the existing multi-LLM council
(stage 1 responses → stage 2 anonymized peer ranking → stage 3 chairman
synthesis), same as it does today for a plain question.

This is a personal project to start (one family member's records), built
with real data hygiene from day one because it may become a paid tool for
other families later. It is explicitly **not** attempting full HIPAA
compliance in this first build — no BAAs are in place, this is not a
certified clinical system, and it should not be the sole basis for a medical
decision. See "Deferred" below.

## Scope for this build

This spec covers two of three sub-projects identified during design:

1. **Ingestion pipeline** — upload, extract, chunk, embed, store
2. **RAG-augmented council query** — retrieve relevant chunks, feed them into
   the existing 3-stage council, patient-scoped

**Deferred (sub-project 3):** real authentication, encryption at rest,
audit logging, retention/deletion policy. This app is currently a
single-operator local tool with no auth on any endpoint (by design, for now
— see the security review from earlier this session). Multi-patient support
IS included in this build (see Data model) because retrofitting it later is
expensive; access control is not, because it can be added later without
reshaping the schema.

## Constraints

- Document formats: PDF (text-based and scanned), JPG, PNG, TXT
- Volume: ~300MB across multiple years, for one family member so far
- Multi-patient from day one — documents are scoped to a patient, queries
  are answered about one patient at a time
- No local embedding model available — embeddings go through a cloud API
- LLM calls (council + embeddings) go **directly** to Anthropic and Google,
  not through the OmniRoute proxy used elsewhere in this app. Reasoning:
  Anthropic and Google both offer BAAs on their direct enterprise APIs;
  OmniRoute is a third piece of local software with no BAA that would sit
  in the path of every request once it's handling PHI. Direct calls means
  fewer parties touching patient data, not more infrastructure to build.

## Prior art: the Dating App's pgvector pattern

The Dating App project (same machine) already runs Postgres + pgvector for
embedding-based matching. We are reusing its embedding call, index, and
change-detection mechanism as-is, and reshaping only what needs to differ
for retrieval-over-documents instead of entity-to-entity matching. Full
comparison: `docs/superpowers/specs/2026-08-23-embeddings-comparison.html`
(published artifact: https://claude.ai/code/artifact/764a0609-db5f-4557-9813-48b90299322b).

**Carried over unchanged:**
- Postgres + `pgvector` extension, `<=>` cosine distance operator
- Embedding call: Gemini `embedContent`, model `gemini-embedding-001`
  (fallback `gemini-embedding-2`), `taskType: SEMANTIC_SIMILARITY`,
  `outputDimensionality: 768`
- HNSW index: `USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`
- Lazy re-embedding guard: MD5 hash of source text, skip the API call when
  unchanged

**Reshaped for retrieval:**
- One embedding per *chunk* in a dedicated table, not one embedding per row
  on an entity table — a patient has many documents, each document has many
  chunks
- No similarity floor. The dating app excludes candidates below 0.75
  cosine similarity because a weak match should never be shown. Here,
  retrieval always returns its best-available top-K regardless of score —
  silently returning nothing to a caregiver's real question about a real
  patient is a worse failure mode than the chairman synthesis seeing a weak
  record and saying so. If nothing is truly relevant, that's the chairman's
  judgment call to surface, not the SQL layer's job to hide.
- Result goes into the existing 3-stage council prompt, not straight to the
  user.

## Architecture

```
Upload (PDF/JPG/PNG/TXT)
    │
    ▼
Ingestion
    ├─ Text-based PDF/TXT → extract locally (PyMuPDF)
    └─ Scanned PDF page / JPG / PNG → Claude or Gemini vision → extracted text
    │
    ▼
Chunk text (~500-800 tokens, ~15% overlap)
    │
    ▼
Embed each chunk (Gemini embedContent, 768-dim) → store in Postgres/pgvector
    tagged with patient_id, document_id

Query time:
Caregiver asks a question about Patient X
    │
    ▼
Embed the question (same API) → pgvector search WHERE patient_id = X, top-K, no floor
    │
    ▼
Stage 1: Claude + Gemini each answer using question + retrieved chunks as context
    │
    ▼
Stage 2: anonymized peer ranking (existing, unchanged)
    │
    ▼
Stage 3: chairman synthesizes final answer, citing which records it drew on
    │
    ▼
Presented to caregiver/doctor
```

## Data model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,           -- pdf | jpg | png | txt
    extraction_method TEXT NOT NULL,   -- local | vision_llm
    status TEXT NOT NULL DEFAULT 'processing',  -- processing | ready | failed
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,  -- denormalized for fast scoping
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX document_chunks_embedding_hnsw
  ON document_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX document_chunks_patient_id ON document_chunks (patient_id);
```

`patient_id` is denormalized onto `document_chunks` (not just reachable via
`documents`) specifically so the retrieval query can filter with a single
indexed `WHERE patient_id = $1` instead of a join — this is the query that
must never leak another patient's data, so it should be the simplest
possible correct query.

## Components

| Module | Responsibility |
|---|---|
| `backend/db.py` | Postgres connection pool (new — existing conversation storage stays JSON, untouched) |
| `backend/ingestion.py` | Dispatch extraction by file type (local vs vision-LLM fallback), chunk text |
| `backend/embeddings.py` | Gemini `embedContent` call, ported from the Dating App's `embeddingService.mjs` pattern |
| `backend/retrieval.py` | Patient-scoped pgvector similarity query |
| `backend/council.py` | Modified: stage 1 prompt includes retrieved chunks when a patient is specified |
| `backend/main.py` | New endpoints: patients CRUD, document upload, patient-scoped ask |
| `frontend/` | Patient selector + document upload UI |

Migrations via Alembic (Python equivalent of the Dating App's
`node-pg-migrate`), scoped to the new tables only.

## Error handling

- Unsupported file type → 400 at upload, before any processing starts
- Extraction fails for one document (corrupt file, vision-LLM error) →
  that document's `status` is set to `failed`, surfaced in the UI; other
  documents are unaffected
- Embedding API failure for a chunk → retry once, then leave that chunk
  unembedded (excluded from retrieval, logged) rather than failing the
  whole upload
- No chunks exist yet for a patient → the ask endpoint returns a clear
  "no records uploaded for this patient yet" message instead of running an
  empty council query
- One council model failing during a patient query → existing graceful
  degradation applies unchanged (continue with whatever responses succeeded)

## Testing priorities

1. **Patient isolation** (safety-critical, not optional): a retrieval query
   for patient A must never return patient B's chunks, under any input.
   This gets an explicit test, not just incidental coverage.
2. Chunking: boundaries and overlap behave correctly on real-shaped text
3. Extraction: each file type (PDF text, PDF scanned, JPG, PNG, TXT) via a
   fixture per format, local path and vision-LLM fallback path both covered
4. One real end-to-end test: upload a fixture document → ask a question
   that should retrieve it → confirm it appears in the council's context
