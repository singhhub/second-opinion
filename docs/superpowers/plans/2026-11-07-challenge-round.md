# Challenge Round + Structured Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the two models answer, compare a small structured block directly, run a bounded challenge round on the disagreements that matter, and show the human only what is still unresolved (with the original disagreement always visible).

**Architecture:** Stage 1 stays free text; each model also appends a delimited JSON block that pure code compares (`structured_diff.py`). A new `challenge.py` selects claim-diff and structured disagreements, asks the models to rebut/revise/concede, re-classifies with the existing fail-safe judge, and `main.py` folds resolved items into the agreed list while recording every original position.

**Tech Stack:** Python, FastAPI, Pydantic, pytest + pytest-asyncio (`asyncio_mode = "auto"`), `llm_client.query_model` (mocked in tests), static HTML page.

**Spec:** `docs/superpowers/specs/2026-11-07-challenge-round-design.md`

## Global Constraints

- Direct provider calls only via `backend/llm_client.py` (no proxy for medical text).
- Fail-safe defaults: any ambiguous, malformed, or failed response in the round resolves to "still conflicting", never "resolved".
- Chairman keeps `CHAIRMAN_NEVER_EMERGENCY_RULE`; it still reports a source model's emergency recommendation even if the round "resolved" it.
- Tests mock `llm_client.query_model`; only the one live-gated eval test hits APIs. Run the suite with `uv run pytest --deselect tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict`.
- Relative imports in `backend/`; ASCII-only debug output (Windows cp1252) - route logging through `llm_client._log()`.
- `CHALLENGE_MAX_ROUNDS` default 1, hard cap 2. `CHALLENGE_MAX_ITEMS` default 10.
- Confidence is displayed, never used to decide or resolve anything. Higher severity/urgency is always shown first.

## Review Focus

- Model wraps the block in a ```json fence, repeats it, or omits the closing tag -> parsed tolerantly or treated as missing; the free text shown to the user never contains block JSON (Task 1, 3).
- Only one model returns a usable block -> no structured disagreements are invented, and the result says structured comparison was unavailable (Task 2, 7).
- One of the two challenge calls fails or returns junk -> `UNRESOLVED_ERROR`, item stays in the human diff (Task 5).
- Evidence quote differs from the record only by line wrapping/extra whitespace -> accepted; a paraphrase -> rejected (Task 5).
- No conflicts at all (models fully agree) -> zero challenge calls and `challenge.ran == false` (Task 4, 7).

---

### Task 1: Config constants + structured schema + block splitting

**Files:**
- Modify: `backend/config.py`
- Create: `backend/structured_diff.py`
- Test: `tests/test_structured_diff.py`

**Interfaces:**
- Produces (config): `CHALLENGE_MAX_ROUNDS = 1`, `CHALLENGE_MAX_ROUNDS_HARD_CAP = 2`, `CHALLENGE_MAX_ITEMS = 10`, `CHALLENGE_FALLBACK_ITEMS = 3` (the last is this plan's call: how many unconfirmed claims the "answers differ but nothing qualified" fallback challenges).
- Produces: enums `CauseSystem` (cardiac, neurological, medication, hematologic, respiratory, gastrointestinal, infectious, other), `Confidence` (low, medium, high), `Severity` (low, moderate, high), `MedAction` (avoid, stop, reduce, continue, monitor), `UrgencyLevel` (emergency, same_day, within_days, routine) with a module function `urgency_rank(level) -> int` (emergency = 0 ... routine = 3).
- Produces: Pydantic models `PossibleCause{name, system, rank:int, confidence}`, `InteractionFlag{drugs:List[str], concern, severity}`, `MedicationChange{drug, action, reason}`, `Urgency{level, escalate_if:List[str]=[]}`, `StructuredAssessment{possible_causes, interaction_flags, medication_changes, urgency}`.
- Produces: `STRUCTURED_BLOCK_INSTRUCTION: str` (prompt text appended to the user question, naming the `<structured>...</structured>` delimiters and the exact schema/enum values) and `split_structured_block(response_text: str) -> Tuple[str, Optional[StructuredAssessment]]` returning (free text with the block removed, parsed block or None).

- [ ] **Step 1: Write the failing tests** in `tests/test_structured_diff.py`: `test_split_returns_free_text_without_block` (assert the returned text contains the prose and no `<structured>`), `test_split_parses_valid_block` (assert `.urgency.level == UrgencyLevel.SAME_DAY` and one cause with `system == CauseSystem.NEUROLOGICAL`), `test_split_tolerates_json_fence_inside_tags`, `test_split_missing_block_returns_none_and_full_text`, `test_split_malformed_json_returns_none_and_strips_block`, `test_split_unknown_enum_value_returns_none`, `test_split_uses_first_block_when_repeated`, `test_instruction_names_every_enum_value` (every enum value string appears in `STRUCTURED_BLOCK_INSTRUCTION`).
- [ ] **Step 2: Run** `uv run pytest tests/test_structured_diff.py -v` -> FAIL (module missing).
- [ ] **Step 3: Implement** the constants and `structured_diff.py` per the Interfaces block. Reuse `llm_client.strip_json_fence`; a block that fails Pydantic validation (including an unknown enum value) yields `None`, never raises. Malformed or missing closing tag counts as no block.
- [ ] **Step 4: Run** the same command -> PASS.
- [ ] **Step 5: Commit** `git add backend/config.py backend/structured_diff.py tests/test_structured_diff.py docs/superpowers/specs/2026-11-07-challenge-round-design.md docs/superpowers/plans/2026-11-07-challenge-round.md && git commit -m "feat: structured assessment schema and block splitting"` (end the message with the Co-Authored-By line from the session's attribution reminder).

### Task 2: `compare_structured`

**Files:**
- Modify: `backend/structured_diff.py`
- Test: `tests/test_structured_diff.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces: `StructuredDisagreementKind` (str Enum: `TOP_CAUSE`, `ONE_SIDED_CAUSE`, `INTERACTION_FLAG`, `MEDICATION_CHANGE`, `URGENCY`); `StructuredDisagreement{kind, summary:str, detail_a:str, detail_b:str, sort_key:int}`; `compare_structured(a: Optional[StructuredAssessment], b: Optional[StructuredAssessment]) -> List[StructuredDisagreement]` - pure code, no LLM, sorted by `sort_key` ascending (urgency first, then interaction severity high-to-low, then medication, then causes). Returns `[]` if either side is `None`.

- [ ] **Step 1: Write the failing tests** (one assertion target each): `test_top_cause_differs` (A rank-1 neurological vs B rank-1 medication -> one `TOP_CAUSE`), `test_same_top_cause_system_no_top_cause_disagreement`, `test_cause_system_raised_by_one_model_only` (A has cardiac, B does not -> `ONE_SIDED_CAUSE` whose `detail_b` says it was not raised), `test_urgency_mismatch_same_day_vs_within_days` (kind `URGENCY`, sorted first, `detail_a` contains "same_day"), `test_interaction_flag_severity_differs_shows_higher_first` (high vs moderate -> one `INTERACTION_FLAG`, summary names "high"), `test_interaction_flag_in_only_one_block`, `test_medication_change_differs` (avoid vs continue), `test_agreeing_blocks_produce_no_disagreements`, `test_confidence_difference_alone_is_not_a_disagreement`, `test_none_side_returns_empty`. Drug names compare case-insensitively and order-insensitively on `drugs`.
- [ ] **Step 2: Run** `uv run pytest tests/test_structured_diff.py -v` -> new tests FAIL.
- [ ] **Step 3: Implement** `compare_structured` per the Interfaces block.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat: compare structured assessments`.

### Task 3: Stage 1 requests the block and repairs it once

**Files:**
- Modify: `backend/council.py` (`stage1_collect_responses`, new `request_structured_block`)
- Modify: `tests/test_council.py`

**Interfaces:**
- Consumes: `STRUCTURED_BLOCK_INSTRUCTION`, `split_structured_block`.
- Produces: each stage-1 entry becomes `{"model", "status", "response", "structured"}` where `response` is the free text with the block removed and `structured` is `Optional[StructuredAssessment]` (`None` for a failed call too). New `async def request_structured_block(model: str, question: str, answer: str) -> Optional[StructuredAssessment]` makes one follow-up `query_model` call asking for only the block, parsed with `split_structured_block`; returns `None` on any failure.

- [ ] **Step 1: Write the failing tests** (mock `council.query_models_parallel` and `council.query_model`): `test_stage1_strips_block_from_response_and_parses_it`, `test_stage1_sends_instruction_in_user_message` (assert the message passed contains `STRUCTURED_BLOCK_INSTRUCTION`), `test_stage1_repairs_missing_block_with_one_followup_call` (follow-up returns a valid block -> `structured` is set, `query_model` called once), `test_stage1_followup_failure_leaves_structured_none_and_run_continues`, `test_stage1_failed_model_still_listed_with_structured_none`. Update any existing stage-1 test whose expected entries lack the new `structured` key.
- [ ] **Step 2: Run** `uv run pytest tests/test_council.py -v` -> new tests FAIL.
- [ ] **Step 3: Implement** the changes. The follow-up is only made for models with `status == "ok"` and no parsed block; follow-up calls for the two models may run concurrently.
- [ ] **Step 4: Run** the same command -> all PASS.
- [ ] **Step 5: Commit** `feat: stage 1 collects a structured block alongside the free text`.

### Task 4: Challenge item selection and cap

**Files:**
- Create: `backend/challenge.py`
- Test: `tests/test_challenge.py`

**Interfaces:**
- Consumes: `ClaimPair`, `Claim`, `ClaimType`, `ActionabilityCategory`, `categorize_claim` from `claim_diff`; `StructuredDisagreement` from `structured_diff`; config caps.
- Produces: `ChallengeItem{item_id:str, source: Literal["conflicting","unconfirmed","structured"], category: ActionabilityCategory, text_a: Optional[str], text_b: Optional[str], pair: Optional[ClaimPair]=None, claim: Optional[Claim]=None, side: Optional[Literal["A","B"]]=None, always_visible: bool=False}`; `ChallengeSelection{selected: List[ChallengeItem], deferred: List[ChallengeItem], cap_hit: bool}`; `select_challenge_items(conflicting: List[ClaimPair], unconfirmed_a: List[Claim], unconfirmed_b: List[Claim], structured: List[StructuredDisagreement], max_items: int = CHALLENGE_MAX_ITEMS) -> ChallengeSelection`.
- Rules: every conflicting pair is selected; unconfirmed claims in categories 1-4 are selected, category 5 (background) is not; structured disagreements map URGENCY -> `RISK_OR_RED_FLAG` with `always_visible=True`, TOP_CAUSE/ONE_SIDED_CAUSE -> `DIAGNOSTIC_SUGGESTION`, INTERACTION_FLAG/MEDICATION_CHANGE -> `DRUG_INTERACTION`; fallback: if nothing was selected and any unconfirmed claim exists, select the first `CHALLENGE_FALLBACK_ITEMS` unconfirmed claims regardless of category; order by category value then source order; items beyond `max_items` go to `deferred` with `cap_hit=True`.

- [ ] **Step 1: Write the failing tests:** `test_all_conflicting_pairs_selected`, `test_unconfirmed_diagnostic_claims_selected_when_unaligned` (A cardiac vs B neurological as two unconfirmed `consistent with` claims -> both selected), `test_unconfirmed_background_claim_not_selected_when_others_selected`, `test_keywordless_diagnosis_selected_via_fallback` ("this sounds like a heart problem" as the only unconfirmed claim -> selected), `test_no_items_when_nothing_differs` (all empty -> `selected == []`, `cap_hit is False`), `test_urgency_structured_item_is_always_visible`, `test_cap_applies_in_actionability_order_and_sets_cap_hit` (12 items -> 10 selected, 2 deferred, drug-interaction items kept before background), `test_selection_is_lossless` (selected + deferred accounts for every selectable input exactly once).
- [ ] **Step 2: Run** `uv run pytest tests/test_challenge.py -v` -> FAIL.
- [ ] **Step 3: Implement** `ChallengeItem`, `ChallengeSelection`, `select_challenge_items` per the rules above.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat: select disagreements to challenge`.

### Task 5: Challenge round execution

**Files:**
- Modify: `backend/challenge.py`
- Test: `tests/test_challenge.py`

**Interfaces:**
- Consumes: Task 4 types; `llm_client.query_model`, `strip_json_fence`; `classify_compatibility`, `ClaimState` from `claim_diff`.
- Produces: `ChallengeStance` (str Enum: `MAINTAIN`, `REVISE`, `CONCEDE`); `ChallengeResponse{stance, revised_claim: Optional[str], reasoning: str, evidence_quote: Optional[str]}`; `ChallengeStatus` (str Enum: `RESOLVED`, `STILL_CONFLICTING`, `UNRESOLVED_ERROR`); `ChallengeOutcome{item: ChallengeItem, final_a: Optional[str], final_b: Optional[str], status: ChallengeStatus, flags: List[str], rounds_used: int}`; flag constants `FLAG_CHANGED_WITHOUT_EVIDENCE = "changed_without_evidence"` and `FLAG_NO_RECORD_EVIDENCE = "resolved_without_record_evidence"`; `evidence_in_record(quote: Optional[str], record_context: str) -> bool`; `async def run_challenge_round(items: List[ChallengeItem], question: str, model_a: str, model_b: str, record_context: Optional[str] = None, max_rounds: int = CHALLENGE_MAX_ROUNDS) -> List[ChallengeOutcome]` - returns exactly one outcome per input item, same order, never raises.
- Semantics: for `conflicting`/`structured` items both models are shown the other's position; for `unconfirmed` items only the model that did not make the claim is asked (`CONCEDE` = it agrees, `REVISE` = it would state it differently, `MAINTAIN` = it does not endorse it). Each call retries once on malformed JSON then yields `UNRESOLVED_ERROR` for that item. Both maintain -> `STILL_CONFLICTING` with no re-classification call. Any revise/concede -> re-classify `ClaimPair(final_a, final_b)` via `classify_compatibility`; `AGREED` -> `RESOLVED`, anything else `STILL_CONFLICTING`. Capitulation guard: with `record_context` set, a revise/concede whose `evidence_quote` fails `evidence_in_record` adds `FLAG_CHANGED_WITHOUT_EVIDENCE` and forces `STILL_CONFLICTING`; without `record_context`, a resolved item that involved a position change gets `FLAG_NO_RECORD_EVIDENCE`. `evidence_in_record` is a case-sensitive substring check after collapsing runs of whitespace in both strings. Rounds = `min(max_rounds, CHALLENGE_MAX_ROUNDS_HARD_CAP)`; round 2 only re-challenges items not `RESOLVED`, using their final texts; stop early when none remain.

- [ ] **Step 1: Write the failing tests** (mock `challenge.query_model` and `challenge.classify_compatibility`): `test_both_concede_resolves_and_keeps_original_texts`, `test_one_model_maintains_stays_conflicting_and_makes_no_classify_call`, `test_malformed_json_twice_gives_unresolved_error`, `test_one_of_two_calls_failing_gives_unresolved_error`, `test_changed_position_with_fabricated_quote_stays_conflicting_when_record_supplied` (flag asserted), `test_changed_position_without_record_resolves_with_no_record_evidence_flag`, `test_evidence_in_record_accepts_whitespace_difference_rejects_paraphrase`, `test_unconfirmed_item_only_asks_the_other_model` (exactly one `query_model` call), `test_rounds_default_one_and_never_exceed_two` (pass `max_rounds=5`, assert at most 2 rounds' calls), `test_round_two_skips_resolved_items`, `test_outcomes_one_per_item_in_order`, `test_no_items_makes_no_calls`.
- [ ] **Step 2: Run** `uv run pytest tests/test_challenge.py -v` -> new tests FAIL.
- [ ] **Step 3: Implement** the types and `run_challenge_round` per the semantics above. Calls for different items run concurrently with `asyncio.gather`; use one parse helper `_parse_challenge_response(raw) -> Optional[ChallengeResponse]` in the style of `claim_diff._parse_claims`. Prompt text: ask for JSON only with the three fields; instruct the model that `evidence_quote` must be copied exactly from the patient record when one is provided.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat: run the challenge round`.

### Task 6: Apply outcomes + chairman section

**Files:**
- Modify: `backend/challenge.py`, `backend/council.py`
- Test: `tests/test_challenge.py`, `tests/test_council.py`

**Interfaces:**
- Produces: `apply_challenge_outcomes(conflicting: List[ClaimPair], unconfirmed_a: List[Claim], unconfirmed_b: List[Claim], outcomes: List[ChallengeOutcome]) -> Tuple[List[ClaimPair], List[ClaimPair], List[Claim], List[Claim]]` returning `(newly_agreed, conflicting, unconfirmed_a, unconfirmed_b)`. An outcome moves its item to `newly_agreed` (as a `ClaimPair` of claims built from `final_a`/`final_b`, `claim_type` taken from the original) only when `status == RESOLVED`, `item.source != "structured"`, and `not item.always_visible`; everything else stays where it was. Lossless: each input claim/pair ends up in exactly one of the four outputs.
- Modifies: `synthesize_claim_diff_chairman(user_query, agreed, ranked_diff_items, chairman_model=CHAIRMAN_MODEL, challenge_outcomes: Optional[List[ChallengeOutcome]] = None)` - when outcomes are given, the user message gets a `CHALLENGE ROUND RESULTS` section listing every outcome (status, flags, original and final positions) so the chairman can say a disagreement was settled and how; `CHAIRMAN_NEVER_EMERGENCY_RULE` is unchanged. Add `_format_challenge_section(outcomes) -> str` in `council.py`.

- [ ] **Step 1: Write the failing tests:** `test_resolved_conflict_moves_to_agreed_with_final_texts`, `test_still_conflicting_item_stays_in_conflicting`, `test_resolved_urgency_structured_item_does_not_move`, `test_unconfirmed_resolved_leaves_unconfirmed_list_and_joins_agreed`, `test_apply_is_lossless`; in `tests/test_council.py`: `test_chairman_message_includes_challenge_section_when_outcomes_given` (assert the text of both original positions appears in the `query_model` messages), `test_chairman_message_unchanged_when_no_outcomes`.
- [ ] **Step 2: Run** `uv run pytest tests/test_challenge.py tests/test_council.py -v` -> new tests FAIL.
- [ ] **Step 3: Implement** both changes per the Interfaces block.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat: fold challenge outcomes into agreed list and chairman input`.

### Task 7: Wire into `/api/second-opinion/analyze`

**Files:**
- Modify: `backend/main.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `compare_structured`, `select_challenge_items`, `run_challenge_round`, `apply_challenge_outcomes`, the new chairman parameter.
- Produces response additions (existing keys unchanged): `counts.resolved_after_challenge: int` (counts of agreed/conflicting/unconfirmed reflect the state after the challenge); `challenge: {"ran": bool, "cap_hit": bool, "error": Optional[str], "items": [{"id", "source", "category", "original_a", "original_b", "final_a", "final_b", "status", "flags", "always_visible"}]}`; `structured: {"available": bool, "a": Optional[dict], "b": Optional[dict], "disagreements": [{"kind", "summary", "detail_a", "detail_b"}]}` where `available` is true only when both blocks parsed. Degraded responses carry `challenge: None` and `structured: None`.
- Behaviour: stage-1 entries are read with `.get("structured")`. The challenge step runs inside its own `try`; on any exception the challenge is skipped, `challenge.error` is set, and every item stays in the human diff (the endpoint does not go degraded). `run_challenge_round` is called with `model_a`/`model_b` from the two ok results and `record_context=None` (the wiki hook for later). Zero selected items -> no call and `ran: false`.

- [ ] **Step 1: Update and add tests** in `tests/test_main.py`: add an autouse fixture patching `main.run_challenge_round` to `AsyncMock(return_value=[])` so existing tests never reach the network; update `test_analyze_returns_structured_result_on_success` and the counts test to include `resolved_after_challenge: 0`. New tests: `test_analyze_resolved_conflict_counts_as_agreed_and_keeps_original_in_challenge_block`, `test_analyze_challenge_exception_keeps_all_items_and_sets_error`, `test_analyze_no_conflicts_makes_no_challenge_call` (assert `ran is False`, the patched `run_challenge_round` not awaited), `test_analyze_structured_unavailable_when_one_block_missing` (`available is False`, `disagreements == []`), `test_analyze_urgency_disagreement_listed_and_challenged`, `test_analyze_cap_hit_is_reported`.
- [ ] **Step 2: Run** `uv run pytest tests/test_main.py -v` -> new tests FAIL, updated ones FAIL.
- [ ] **Step 3: Implement** the wiring in `analyze_question` after classification and before `build_ranked_diff_items`, per the Behaviour above.
- [ ] **Step 4: Run** `uv run pytest tests/test_main.py -v` -> all PASS.
- [ ] **Step 5: Commit** `feat: challenge round and structured comparison in analyze endpoint`.

### Task 8: Eval harness runs the challenge step

**Files:**
- Modify: `backend/eval_harness.py` (`run_new_mechanism`)
- Modify: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: Task 4-6 functions. Eval cases carry only raw answer text, so there are no structured blocks (`structured=[]`); `record_context=None`; models come from `COUNCIL_MODELS`.
- Behaviour: insert the select -> run -> apply steps between classification and `build_ranked_diff_items`, pass `challenge_outcomes` to the chairman, update the stage log labels. The existing duplication with `main.analyze_question` is noted in CLAUDE.md, not fixed here.

- [ ] **Step 1: Write the failing tests:** `test_run_new_mechanism_applies_challenge_outcomes` (patch `eval_harness.run_challenge_round` to resolve one conflict; assert the chairman call received it via `challenge_outcomes` and that the resolved claim is in `agreed`), `test_run_new_mechanism_makes_no_challenge_call_when_nothing_selected`. Check existing tests in the file still pass unchanged apart from patching the new function.
- [ ] **Step 2: Run** `uv run pytest tests/test_eval_harness.py -v` -> new tests FAIL.
- [ ] **Step 3: Implement** the step.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat: eval harness runs the challenge step`.

### Task 9: Result page

**Files:**
- Modify: `static/second-opinion-result.html` (add to `renderResult`, new containers beside `#result-section`'s existing cards)

**Interfaces:**
- Consumes: the `challenge`, `structured`, and `counts.resolved_after_challenge` fields from Task 7.

- [ ] **Step 1: Add** to `renderResult`: a fourth chip "Resolved after challenge" bound to `counts.resolved_after_challenge`; a "Challenge round" card listing each item with original A / original B / final A / final B, status, and flag text (`changed_without_evidence` -> "changed position without record evidence", `resolved_without_record_evidence` -> "agreed after challenge, no patient record to check against"); urgency items always shown with a prominent badge even when resolved; a "Structured comparison" table of `structured.disagreements` (kind, Model A, Model B); notes when `structured.available` is false ("structured comparison unavailable") or `challenge.cap_hit` ("some disagreements were not challenged - cost cap reached") or `challenge.error`; and a fixed line under the card "AI-generated comparison - not clinician-verified". All dynamic text goes through `textContent`, never `innerHTML`.
- [ ] **Step 2: Verify visually** with a scratchpad copy of the page whose `fetch` is stubbed to return a fixture response (one resolved item, one still-conflicting item, one urgency item, `structured.available` true, then repeat with `available: false` and `cap_hit: true`); open it in the browser and confirm each element renders and nothing overflows at phone width. Real-API verification costs money - ask before running it.
- [ ] **Step 3: Commit** `feat: render challenge round and structured comparison`.

### Task 10: Docs and full verification

**Files:**
- Modify: `CLAUDE.md` (architecture, data-flow diagram, Common Gotchas: `eval_harness.run_new_mechanism` now also duplicates the challenge step)
- Modify: `TODOS.md` (add: wire `record_context` once the wiki query path exists)

- [ ] **Step 1: Update** the docs to describe `structured_diff.py`, `challenge.py`, the new stage-1 entry shape, the response additions, and the "Chairman never softens urgency" interaction with always-visible urgency items.
- [ ] **Step 2: Run** the full suite `uv run pytest --deselect tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict -q` -> all PASS.
- [ ] **Step 3: Ask the user** before running the billed ER regression (`tests/test_eval_er_case.py`, case `synthetic-er-red-flag-direct-conflict`); if approved, run it and confirm the ER disagreement is still visible to the human after the challenge round. Report the result either way.
- [ ] **Step 4: Commit** `docs: describe challenge round and structured comparison`.
