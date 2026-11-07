# Challenge Round (Debate Before Diff) - Design

**Date:** 2026-11-07
**Status:** Draft, awaiting approval. Nothing built yet.

## Intent and why

Today the pipeline is a single pass: both models answer, claims are
extracted, aligned, classified AGREED/CONFLICTING/UNCONFIRMED, ranked, and
summarized by the chairman. Every disagreement goes straight to the human.

We want the orchestrator to make the two models *think again* about their
disagreements first - each sees the other's claim and reasoning and must
rebut, revise, or concede - and only surface to the human what is still
unresolved after that. The human review list (the diff) stays the product;
the challenge round just shrinks it to the disagreements that survive
scrutiny, without hiding that they existed.

## Goals

1. After classification, run a bounded challenge round on the claims that
   matter most, then re-classify.
2. Claims still CONFLICTING after the round go to the human diff showing
   both models' original and final positions side by side.
3. Claims that resolve are marked "resolved after challenge" with the
   original disagreement still visible - never silently dropped.
4. Every existing safety rule keeps holding (see Constraints).

## Non-goals

- Building the patient-wiki query path (phase 4 of the wiki spec). This
  design only leaves a hook for it.
- More than two rounds, free-form multi-turn debate, or a third model as
  tiebreaker.
- Redesigning the result page beyond rendering the new fields.

## Anti-goals (failure even if it "works")

- A conflict disappearing because one model caved to pushback with no
  support. Agreement reached by capitulation must not look like consensus.
- A failed or malformed model call during the round reading as agreement.
- Consensus presented as verified truth. Two models agreeing is still
  AI-generated, not clinician-verified.

## Constraints

- Direct provider calls only via `llm_client.py` (no proxy for medical text).
- Fail-safe defaults: any ambiguous, malformed, or failed response in the
  round resolves to "still conflicting", never "resolved".
- Chairman keeps `CHAIRMAN_NEVER_EMERGENCY_RULE`; it still reports a source
  model's emergency recommendation even if the round "resolved" it.
- Tests mock `llm_client.query_model`; only one live-gated eval hits APIs.
- Relative imports in `backend/`; ASCII-only debug output (Windows cp1252).

## Design

New module `backend/challenge.py`; `main.py` calls it between the
classify step and `build_ranked_diff_items`.

**Which claims are challenged.** All CONFLICTING pairs, plus UNCONFIRMED
claims in categories DRUG_INTERACTION, RISK_OR_RED_FLAG,
DIAGNOSTIC_SUGGESTION, and MONITORING_FOLLOWUP (from `categorize_claim`).
Only BACKGROUND_OR_CAVEAT unconfirmed claims skip the round; they still
appear in the diff exactly as today.

Why diagnoses are included: if one model says cardiac and the other
neurological, `align_claims` may not pair them (different topics can fall
below the similarity threshold), so they arrive as two UNCONFIRMED claims,
not a CONFLICTING pair. They must still be challenged - a differing
diagnosis is the disagreement the human most needs resolved or flagged.

Why BACKGROUND is still a risk: the category is a keyword match, and an
unmatched claim (e.g. "this sounds like a heart problem") defaults to
BACKGROUND_OR_CAVEAT. To avoid a real diagnosis skipping the round,
claims with `claim_type` of `assertion` or `recommendation` that name a
body system or condition are not assumed safe to skip: when the challenged
set would otherwise be empty while the two answers differ, the builder
challenges the top-N unconfirmed claims regardless of category.

**Cost cap.** `CHALLENGE_MAX_ITEMS` in `config.py` (default 10). If more
items qualify, challenge in actionability order and leave the rest in the
diff unchallenged, with a visible note that the cap was hit - never a
silent drop.

**The round.** For each challenged item, each model is shown the other
model's claim and given the original question. It returns structured JSON
(Pydantic-validated, retry-once-then-fallback, same pattern as
`extract_claims`): `stance` (`maintain` | `revise` | `concede`),
`revised_claim` (text, if revised), `reasoning`, and `evidence_quote`
(optional). For an UNCONFIRMED claim, only the model that did not make it
is asked, whether it agrees, disagrees, or cannot say.

**Re-classification.** Revised/conceded claim pairs go back through the
existing `classify_compatibility` (fail-safe to CONFLICTING).

**Capitulation guard.** If a model changes position (`revise`/`concede`)
and, when a `record_context` is supplied, its `evidence_quote` is not a
verbatim substring of that context, the item is flagged
`changed_without_evidence` and stays in the human list even if the
re-classification says compatible. With no `record_context` (today), a
position change is allowed to resolve but is labelled
`resolved_without_record_evidence` so the page can show it distinctly.

**Bounds.** `CHALLENGE_MAX_ROUNDS` in `config.py`, default 1, hard cap 2.
Rounds stop early when nothing is left conflicting.

**Outcome per item** (`ChallengeOutcome`): `original_a`, `original_b`,
`final_a`, `final_b`, `status` (`RESOLVED` | `STILL_CONFLICTING` |
`UNRESOLVED_ERROR`), and flags above. `UNRESOLVED_ERROR` is treated as
STILL_CONFLICTING downstream.

**Pipeline and API changes.** `main.py` partitions outcomes: RESOLVED items
join the agreed list (tagged); the rest flow into `build_ranked_diff_items`
unchanged. The response gains a `challenge` block (items with original and
final positions) and `counts.resolved_after_challenge`. The chairman prompt
receives resolved-after-challenge items as a separate labelled section so
it can mention that a disagreement was settled and how.

**Record hook.** `challenge_conflicts(..., record_context: str | None =
None)`. When the wiki query path lands, it passes the verbatim wiki text
here and the evidence rule becomes strict.

## Structured comparison block (added 2026-11-07)

Stage 1 stays free text. Each model also returns a small JSON block in the
same call, appended after its answer (for example inside a delimited
`<structured>...</structured>` section), used only for direct comparison.
It never replaces claim extraction and never gates the run.

**Schema** (Pydantic-validated, new `StructuredAssessment` model):
- `possible_causes`: list of `{name, system, rank, confidence}`. `system`
  is a fixed enum (cardiac, neurological, medication, hematologic,
  respiratory, gastrointestinal, infectious, other); `confidence` is
  low | medium | high.
- `interaction_flags`: list of `{drugs, concern, severity}`; `severity` is
  low | moderate | high.
- `medication_changes`: list of `{drug, action, reason}`; `action` is
  avoid | stop | reduce | continue | monitor.
- `urgency`: `{level, escalate_if}`; `level` is emergency | same_day |
  within_days | routine.

**Comparison** (`compare_structured(a, b)` in `backend/structured_diff.py`,
pure code, no LLM): produces disagreements on top-ranked cause, causes
raised by one model only (by `system`), interaction flags present in only
one block or differing in severity, medication changes that differ, and
urgency level mismatch. Each disagreement is fed into the challenge set
alongside the claim-diff items and carried into the human diff.

**Rules:**
- Urgency mismatch is always challenged and always stays visible to the
  human, even if the round resolves it. This extends the chairman's
  existing ER rule: a source model's higher urgency is never softened away.
- Confidence is displayed to the human but never used to decide which
  model is right, and never used to resolve a conflict.
- A missing, malformed, or unparseable block (retry once, then fallback)
  yields `None` for that model: the run proceeds on the free-text path
  only, and the result notes that structured comparison was unavailable.
  It never reads as agreement and never raises.
- Severity and urgency never get downgraded by merging: when the two
  blocks differ, the higher severity/urgency is the one shown first.
- The free text remains the backstop for anything the schema cannot
  express (hedges, conditions); the chairman still sees it.

**API/UI:** the response gains a `structured` block (both models' blocks
and the list of structured disagreements); the result page renders it as a
comparison table beside the claim diff.

## Testing

1. Both models concede/agree after the round -> item RESOLVED, original
   disagreement still present in the output.
2. One model maintains -> STILL_CONFLICTING, both original and final shown.
3. Model call fails or returns malformed JSON twice -> UNRESOLVED_ERROR,
   item remains in the human diff (never resolved).
4. Position change with a fabricated evidence quote and a supplied
   `record_context` -> `changed_without_evidence`, stays in the diff.
5. Round cap respected (1 by default, never above 2).
6. Lossless: every challenged item appears exactly once in the output.
6a. Model A claims a cardiac cause, Model B a neurological one, and the
   claims do not align (two UNCONFIRMED diagnostic claims) -> both are
   challenged, not skipped.
6b. A diagnosis phrased with no rubric keyword still gets challenged via
   the fallback, not silently skipped.
6c. More than `CHALLENGE_MAX_ITEMS` qualifying items -> the cap is applied
   in actionability order and the response notes that it was hit.
6d. Structured blocks: top-ranked cause differs (neurological vs
   medication) -> disagreement produced and challenged.
6e. Urgency `same_day` vs `within_days` -> always challenged and still
   shown to the human even if the round resolves it.
6f. A cause system raised by only one model -> flagged as one-sided.
6g. Malformed or missing block (after one retry) -> that model's block is
   `None`, the run still completes on the free-text path, and the result
   says structured comparison was unavailable. Never treated as agreement.
6h. Severity differs (high vs moderate) -> the higher one is shown first;
   confidence never changes a resolution.
7. Regression: the ER red-flag case (`synthetic-er-red-flag-direct-conflict`)
   still ends with the ER disagreement visible to the human.
8. Eval harness: `run_new_mechanism` gains the challenge step (and the
   known duplication with `main.analyze_question` is noted, not fixed here).

## My calls (not yet confirmed by you)

- Build on the current question-only pipeline now, not after the wiki.
- One round by default, hard cap two.
- All UNCONFIRMED claims except background/caveat are challenged,
  including diagnostic suggestions (revised 2026-11-07 after review: the
  original drug-interaction/red-flag-only rule would have let a
  cardiac-vs-neurological disagreement through unchallenged). A per-run
  cap of 10 items bounds cost.
- Resolved-by-concession without record evidence is allowed but labelled.
- A small structured block rides alongside the free text for direct
  comparison (causes, interaction flags, medication changes, urgency). It
  complements claim extraction, does not replace it, and a bad block never
  blocks the run. Enum values (systems, urgency levels, actions) are my
  first-pass lists and can be adjusted.

## Left to the builder

Exact prompt wording, JSON field names beyond those listed, how the result
page lays out original-vs-final positions, and caching of challenge calls
(follow the existing disk cache pattern if cheap).
