# CLAUDE.md - Technical Notes for Second Opinion

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

Second Opinion is a claim-diff (disagreement-preserving) tool for two siblings cross-checking AI medical answers for an elderly family member. A question goes to two models (Claude + Gemini) directly, their answers are atomized into claims, the claims are aligned and classified as agreed/conflicting/unconfirmed, a challenge round lets the two models respond to each other's position on the contested items, and a chairman model synthesizes a structured summary that never issues its own emergency verdict but always names one if a source model gave one. Each model also emits a machine-readable `<structured>` block (possible causes, interaction flags, medication changes, urgency) that is compared field-by-field and shown as a comparison table.

This started as a fork of `karpathy/llm-council` (a 3-stage rank-and-synthesize chat). That original chat flow, its React frontend, and the local OmniRoute proxy it depended on have all been removed — the claim-diff mechanism is the only mechanism now, served behind a single stateless endpoint and a static result page.

## Architecture

### Backend Structure (`backend/`)

**`config.py`**
- `COUNCIL_MODELS`: the two models queried directly (provider/model-id format, e.g. `"claude/claude-sonnet-4-5-20250929"`)
- `CHAIRMAN_MODEL`: the model that synthesizes the claim-diff summary
- Backend runs on **port 8001** (NOT 8000 - user had another app on 8000)

**`llm_client.py`** - Direct-call LLM client
- No proxy in front of the providers — `query_model()`/`query_models_parallel()` call Anthropic's Messages API and Google's Generative Language API directly. Real medical text must never route through a third-party proxy (data-hygiene constraint).
- `embed_text()`: Gemini embeddings, used for claim alignment similarity
- Retry with exponential backoff (4 attempts) on 429/5xx and transport errors — tuned from a real observation that live runs saw multi-second network blips that only showed up on long sequential runs, not quick diagnostic calls
- Disk cache (`data/cache/*.json`, keyed by `sha256(model+input)`) for `extract_claims`/`embed_text` results, so eval-tuning iterations don't re-pay API cost
- `LLM_CLIENT_VERBOSE=1` enables full request/response tracing via `_log()`, which has a Windows cp1252-console ASCII fallback (see Common Gotchas)
- `strip_json_fence()`: strips a markdown code fence models sometimes wrap bare-JSON output in, despite being told not to — shared by `claim_diff.py`'s extractor and `council.py`'s chairman

**`claim_diff.py`** - The claim-diff mechanism (Steps 1-4)
- `extract_claims()`: atomizes a model's response into `{text, claim_type}` claims via an LLM extractor, retry-then-fallback (never raises)
- `align_claims()`: aligns claims by aboutness (not agreement) using embedding similarity, with an LLM judge for borderline scores
- `classify_compatibility()`: classifies an aligned pair as AGREED or CONFLICTING — fail-safe defaults to CONFLICTING, never AGREED
- `categorize_claim()`/`rank_by_actionability()`: fixed keyword rubric (drug interaction > risk/red flag > diagnostic suggestion > monitoring > background), so ranking is cheap and deterministic rather than another LLM call per claim

**`structured_diff.py`** - Structured comparison block
- Pydantic `StructuredAssessment` (possible causes, interaction flags, medication changes, `Urgency`), `split_structured_block()` (strips the `<structured>` block from free text and parses it; an unclosed/truncated block is cut from the free text, so raw JSON never reaches the model tab or claim extraction), and `compare_structured()`, which returns `StructuredDisagreement`s (top-cause mismatch, one-sided cause/interaction flag, medication-action mismatch, urgency disagreement). Pure code, no LLM call.

**`challenge.py`** - Challenge round
- `select_challenge_items()`: picks conflicting pairs, structured disagreements and (fallback) unconfirmed claims, ordered `always_visible` (urgency) items first and then by category, capped at `CHALLENGE_MAX_ITEMS` (reports `cap_hit`) - so the cap can never defer the urgency item
- `run_challenge_round()`: each model sees the other's position and answers MAINTAIN/CONCEDE/REVISE; the final positions are re-judged with `classify_compatibility`. One outcome per item, never raises. `apply_challenge_outcomes()` folds only RESOLVED, non-structured, non-`always_visible`, non-`RISK_OR_RED_FLAG` outcomes into a newly-agreed list, as `SettledClaimPair`s that carry both original positions. A RESOLVED red-flag item stays in the human diff (an ER-side concession must not read as consensus); its outcome still appears in the challenge block.
- Constants in `config.py`: `CHALLENGE_MAX_ROUNDS` (1), `CHALLENGE_MAX_ROUNDS_HARD_CAP` (2), `CHALLENGE_MAX_ITEMS` (10), `CHALLENGE_FALLBACK_ITEMS` (3)

**`council.py`** - Stage 1 + chairman synthesis
- `stage1_collect_responses()`: parallel queries to both configured models. Returns one entry per *configured* model, always — a failed call stays visible downstream with an explicit `status: "failed"`, never silently dropped (see Degraded-run safety rule below). Entries are now `{model, status, response, structured}`: the `<structured>` block is requested in the prompt (`request_structured_block()` is a follow-up repair call) and stripped from `response`, so claim extraction never sees it; `structured` is `None` if missing or unparseable
- `build_ranked_diff_items()`: merges CONFLICTING pairs and UNCONFIRMED claims (both sides) into one ranked list — ranked together, not filtered per-state, so a low-category conflict can't outrank a high-category unconfirmed drug interaction
- `synthesize_claim_diff_chairman()`: chairman synthesis, Pydantic-validated structured JSON (`ChairmanSummary`) with retry-then-fallback. `CHAIRMAN_NEVER_EMERGENCY_RULE` explicitly distinguishes the chairman's own voice (must never issue an emergency/not-emergency verdict) from reporting what a source model recommended (must always name it) — see Common Gotchas for why this distinction exists. It also takes optional `challenge_outcomes`, rendered by `_format_challenge_section()` with `CHALLENGE_SECTION_NOTE` (each entry names its category; `always_visible`/`RISK_OR_RED_FLAG` entries are tagged "urgency/red flag: always report"), and optional `structured_disagreements`, rendered as their own STRUCTURED COMPARISON DISAGREEMENTS section with `STRUCTURED_SECTION_NOTE` (urgency rows tagged "urgency: always report"), independent of the challenge outcomes. With neither, the messages are byte-for-byte the pre-challenge ones; with structured disagreements the diff section never says "fully agreed". In AGREED FINDINGS, `SettledClaimPair`s are tagged "settled after challenge" with both original positions

**`main.py`**
- `POST /api/second-opinion/analyze`: stateless end-to-end run of the mechanism on one question. No conversation_id, no persistence, no CORS (the UI it serves is same-origin)
- The analyze response gains `challenge` (`{ran, cap_hit, error, items[]}`; item `status` is the lowercase enum value, e.g. `"resolved"`), `structured` (`{available, a, b, disagreements}`) and `counts` keys `agreed`, `conflicting`, `unconfirmed`, `resolved_after_challenge`, `structured_disagreements`. `counts.agreed` EXCLUDES items settled after challenge (they count only in `resolved_after_challenge`). The structured disagreements are passed to the chairman on every path, including when the challenge step failed. Between classify and rank, `analyze_question` runs the structured comparison and challenge round; on any failure `challenge.error` is set and the original comparison is shown untouched
- `GET /ui`: serves `static/second-opinion-result.html`

**`eval_harness.py` / `run_eval_cli.py`** - Offline regression eval
- Runs the fixed cases in `data/eval/*.json` through the claim-diff mechanism and an LLM judge (`check_claim_retained`) that checks whether a case's known-correct claim survives into the chairman's output. Reports a **count**, never a percentage — the eval set is too small for a rate to mean anything.
- CLI-only, never called from the live API. Run with `uv run python -m backend.run_eval_cli [--cases id1,id2] [--output path]`; needs `ANTHROPIC_API_KEY`/`GOOGLE_API_KEY` and makes real, billed calls — always confirm with the user before running it, and scope to `--cases` rather than the full set when just verifying a fix.
- `LLM_CLIENT_VERBOSE=1 uv run python -m backend.run_eval_cli` saves both a `.json` report and a `.log` full trace to `data/eval/results/` (gitignored)

### No frontend

There is no React app. `static/second-opinion-result.html` is a single static page served same-origin by `GET /ui` — it's the only UI.

## Key Design Decisions

### Degraded-run safety rule
`stage1_collect_responses` must never filter out a failed model call — that's the exact error-path-to-false-unanimity bug this rule exists to prevent (a failed call silently vanishing would make a single surviving response look like agreement). `analyze_question` checks `len(ok_results) < 2` and returns a `degraded: true` response rather than proceeding with one model's answer.

### Chairman never issues its own emergency verdict, but always reports a source model's
A real live run surfaced this distinction the hard way: the chairman was dropping a genuine high-stakes disagreement (one model recommended immediate ER care, the other didn't) by over-applying the never-emergency-verdict rule to *reporting* what a source model said, not just to its own voice. `CHAIRMAN_NEVER_EMERGENCY_RULE` in `council.py` now says this explicitly. If you touch that prompt, re-run the eval case that caught it (`data/eval/synthetic_cases.json`, id `synthetic-er-red-flag-direct-conflict`) via the live-gated test in `tests/test_eval_er_case.py` — deliberately, on purpose, not as a side effect of an unrelated change.

### Challenge round fail-safe rules
- Never resolve on error or malformed output: a failed/unparseable challenge call is `UNRESOLVED_ERROR`, and the item stays in its original list.
- `changed_without_evidence` is sticky across rounds: once any round sets it, the item can never become RESOLVED. It is only set when a `record_context` is supplied (see below); without one, RESOLVED items carry `resolved_without_record_evidence` so the UI shows they were not checked against a record.
- Both models CONCEDE in the same round is a swap, not agreement: status `STILL_CONFLICTING` with the `mutual_concession` flag, positions unchanged.
- `UNRESOLVED_ERROR` items are retried in round 2 (every non-RESOLVED item is re-challenged). Rounds default to 1; the hard cap is 2 regardless of the argument.
- `CHALLENGE_MAX_ITEMS` caps how many items are challenged; the rest are deferred and `cap_hit` is surfaced visibly in the response/UI.

### Urgency disagreements are always visible
Urgency disagreements from the structured block are `always_visible`: they are ordered first so the cost cap never defers them, they are challenged (so the outcome is shown) but never folded into the agreed list (`apply_challenge_outcomes` skips `source == "structured"`, `always_visible` and `RISK_OR_RED_FLAG` items), and the chairman always gets them in the structured section. The result page never shows its "fully agreed" fallback while `structured.disagreements` is non-empty. The chairman still reports a source model's emergency recommendation even if an item was resolved.

### Challenge step commits state last
In `main.analyze_question`, the challenge/structured step builds `new_agreed`, `new_conflicting`, etc. locally and reassigns `agreed`/`conflicting`/`unconfirmed_*` only as the LAST statements inside the `try`, so an exception anywhere earlier leaves the original comparison intact. Keep it that way when editing.

### Optional `record_context` hook
`run_challenge_round(..., record_context=None)` is a hook for the future patient-wiki query path. It is `None` everywhere today. The strict verbatim-evidence rule (`evidence_in_record`) only activates when it is supplied. Wiring is tracked in TODOS.md.

### Fail-safe defaults, every judge call
Every LLM judge in this codebase (`classify_compatibility`, `check_claim_retained`, `check_never_emergency_verdict`) defaults to the *safer* outcome on an ambiguous or failed response: CONFLICTING rather than AGREED, not-retained rather than retained, not-compliant rather than compliant. Never assume a safety property is satisfied when unverified.

## Important Implementation Details

### Relative Imports
All backend modules use relative imports (e.g., `from .config import ...`) not absolute imports. This is critical for Python's module system to work correctly when running as `python -m backend.main`.

### Model Configuration
Models are hardcoded in `backend/config.py` as `provider/model-id` strings (`claude/...`, `gemini/...`) — `llm_client.py` dispatches on the `provider` prefix.

## Common Gotchas

1. **Module Import Errors**: Always run backend as `python -m backend.main` from project root, not from backend directory.
2. **Windows console + non-ASCII**: printing arrows/em-dashes to this machine's terminal can raise `UnicodeEncodeError` (cp1252 codec). `llm_client._log()` has a try/except ASCII-fallback guard — keep new debug-print code ASCII-only or route through `_log()`.
3. **`eval_harness.run_new_mechanism` duplicates `main.analyze_question`'s 7-stage sequence** (extract x2 → align → classify → challenge → rank → synthesize) — the underlying functions are shared, but the orchestration itself is hand-written twice. Tracked in TODOS.md (P2, not urgent) as a fast-follow, not resolved yet.
4. **Live eval calls cost real money**: `tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict` and anything run via `run_eval_cli.py` hit real Claude/Gemini APIs whenever `ANTHROPIC_API_KEY`/`GOOGLE_API_KEY` are set in `.env` (they are, by default, in this repo). Everything else is mocked. When running the test suite, either accept that one test will make live calls, or run `pytest --deselect tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict` — clearing the env vars in your shell does **not** help, since `llm_client.py` calls `load_dotenv()` at import time and reads `.env` directly regardless of the parent shell's environment.
5. **The live-gated ER test now runs the real challenge round**: `tests/test_eval_er_case.py` goes through `run_new_mechanism`, which runs the challenge round on conflicting and unconfirmed claim-diff items only (it does not call `stage1_collect_responses`, request a structured block, or run `compare_structured`; it passes `[]` as the structured disagreements). Running it live therefore makes extra billed challenge calls, but it does NOT exercise urgency/structured items - do not assume the ER regression covers them.

## Testing Notes

`tests/conftest.py` gives every test its own throwaway `llm_client.CACHE_DIR` via `tmp_path`, so tests never read or write the real `data/cache/`. Everything except the one live-gated ER case test in `test_eval_er_case.py` mocks `llm_client.query_model`/`query_models_parallel` — no test run should hit real APIs unless you're deliberately running the live eval.

## Data Flow Summary

```
User Question
    ↓
Stage 1: query both models directly (Anthropic + Google) → [answer A, answer B]
         each answer carries a <structured> block, stripped from the text and parsed
    ↓
Extract claims from each answer (LLM extractor, retry-then-fallback)
    ↓
Align claims by aboutness (embedding similarity + LLM judge for borderline scores)
    ↓
Classify each aligned pair: AGREED or CONFLICTING (fail-safe → CONFLICTING)
    ↓
Compare structured blocks; challenge round on contested items (models respond to
each other; RESOLVED only if re-judged AGREED, never on error; urgency stays visible)
    ↓
Rank CONFLICTING + UNCONFIRMED together by actionability (drug interaction highest)
    ↓
Chairman synthesizes structured JSON: agreed findings, disagreement summary,
observations, questions for the doctor — never its own emergency verdict
    ↓
Return: {models, counts, summary, challenge, structured}  →  rendered by static/second-opinion-result.html
```

The entire flow is async where possible to minimize latency.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke superpowers:systematic-debugging first (root-cause before any fix), or /investigate for a live-site QA angle
- Multi-step implementation plans → invoke superpowers:writing-plans, then superpowers:subagent-driven-development (or superpowers:executing-plans) to execute
- New features/behavior changes → invoke superpowers:brainstorming before any implementation
- Any code change → follow superpowers:test-driven-development (failing test first, watch it fail, then implement)
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review, or superpowers:requesting-code-review for a plan/branch-scoped review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Finishing a feature branch → invoke superpowers:finishing-a-development-branch
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
