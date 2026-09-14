# CLAUDE.md - Technical Notes for Second Opinion

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

Second Opinion is a claim-diff (disagreement-preserving) tool for two siblings cross-checking AI medical answers for an elderly family member. A question goes to two models (Claude + Gemini) directly, their answers are atomized into claims, the claims are aligned and classified as agreed/conflicting/unconfirmed, and a chairman model synthesizes a structured summary that never issues its own emergency verdict but always names one if a source model gave one.

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

**`council.py`** - Stage 1 + chairman synthesis
- `stage1_collect_responses()`: parallel queries to both configured models. Returns one entry per *configured* model, always — a failed call stays visible downstream with an explicit `status: "failed"`, never silently dropped (see Degraded-run safety rule below)
- `build_ranked_diff_items()`: merges CONFLICTING pairs and UNCONFIRMED claims (both sides) into one ranked list — ranked together, not filtered per-state, so a low-category conflict can't outrank a high-category unconfirmed drug interaction
- `synthesize_claim_diff_chairman()`: chairman synthesis, Pydantic-validated structured JSON (`ChairmanSummary`) with retry-then-fallback. `CHAIRMAN_NEVER_EMERGENCY_RULE` explicitly distinguishes the chairman's own voice (must never issue an emergency/not-emergency verdict) from reporting what a source model recommended (must always name it) — see Common Gotchas for why this distinction exists

**`main.py`**
- `POST /api/second-opinion/analyze`: stateless end-to-end run of the mechanism on one question. No conversation_id, no persistence, no CORS (the UI it serves is same-origin)
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
3. **`eval_harness.run_new_mechanism` duplicates `main.analyze_question`'s 6-stage sequence** (extract x2 → align → classify → rank → synthesize) — the underlying functions are shared, but the orchestration itself is hand-written twice. Tracked in TODOS.md (P2, not urgent) as a fast-follow, not resolved yet.
4. **Live eval calls cost real money**: `tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict` and anything run via `run_eval_cli.py` hit real Claude/Gemini APIs whenever `ANTHROPIC_API_KEY`/`GOOGLE_API_KEY` are set in `.env` (they are, by default, in this repo). Everything else is mocked. When running the test suite, either accept that one test will make live calls, or run `pytest --deselect tests/test_eval_er_case.py::test_eval_er_case_chairman_never_emits_emergency_verdict` — clearing the env vars in your shell does **not** help, since `llm_client.py` calls `load_dotenv()` at import time and reads `.env` directly regardless of the parent shell's environment.

## Testing Notes

`tests/conftest.py` gives every test its own throwaway `llm_client.CACHE_DIR` via `tmp_path`, so tests never read or write the real `data/cache/`. Everything except the one live-gated ER case test in `test_eval_er_case.py` mocks `llm_client.query_model`/`query_models_parallel` — no test run should hit real APIs unless you're deliberately running the live eval.

## Data Flow Summary

```
User Question
    ↓
Stage 1: query both models directly (Anthropic + Google) → [answer A, answer B]
    ↓
Extract claims from each answer (LLM extractor, retry-then-fallback)
    ↓
Align claims by aboutness (embedding similarity + LLM judge for borderline scores)
    ↓
Classify each aligned pair: AGREED or CONFLICTING (fail-safe → CONFLICTING)
    ↓
Rank CONFLICTING + UNCONFIRMED together by actionability (drug interaction highest)
    ↓
Chairman synthesizes structured JSON: agreed findings, disagreement summary,
observations, questions for the doctor — never its own emergency verdict
    ↓
Return: {models, counts, summary}  →  rendered by static/second-opinion-result.html
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
