# Second Opinion

![header](header.jpg)

Two siblings, cross-checking AI medical answers for an elderly family member, is where this started. Instead of asking one LLM and trusting it, Second Opinion sends the question to two models directly (Claude + Gemini), and instead of just showing both answers side by side or picking a "winner," it computes a **claim-diff**: what the two models actually agree on, what they conflict on, and what only one of them said — ranked by how actionable it is (a drug interaction outranks background context), and summarized by a chairman model that is not allowed to issue its own emergency verdict, only to name it when a source model gave one.

In a bit more detail, here's what happens when you submit a question:

1. **Collect both answers.** The question goes to Claude and Gemini directly (no proxy — medical text stays between you and the two providers).
2. **Extract and align claims.** Each answer is atomized into individual claims, then claims from the two answers are aligned by what they're *about* (embedding similarity + an LLM judge for borderline cases) — alignment is separate from agreement.
3. **Classify and rank.** Each aligned pair is classified AGREED or CONFLICTING; unaligned claims are UNCONFIRMED. Conflicting and unconfirmed claims are ranked together by actionability, most important first.
4. **Chairman summary.** A chairman model organizes the mechanically-computed diff into agreed findings, a disagreement summary, specific observations to check, and questions to bring to the doctor — never its own "is this an emergency" verdict.

## Origin

This started as a fork of [karpathy/llm-council](https://github.com/karpathy/llm-council), a fun weekend hack for comparing multiple LLMs' answers side by side with anonymized peer ranking. That original 3-stage rank-and-synthesize chat (and its React frontend, and the local OmniRoute proxy it depended on) has since been fully replaced by the claim-diff mechanism above — there's no chat UI anymore, just the one-question analyze flow.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

```bash
uv sync
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
```

Both models are called directly against their providers' APIs (Anthropic's Messages API, Google's Generative Language API) — no local proxy required.

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize which models are used:

```python
COUNCIL_MODELS = [
    "gemini/gemini-3.1-pro-preview",
    "claude/claude-sonnet-4-5-20250929",
]

CHAIRMAN_MODEL = "gemini/gemini-3.1-pro-preview"
```

## Running the Application

```bash
./start.sh
```

or manually:

```bash
uv run python -m backend.main
```

Then open http://localhost:8001/ui in your browser.

## Running the Eval Harness

`data/eval/*.json` holds fixed test cases (synthetic + real historical) used to check that the mechanism preserves known-correct claims. This makes real, billed calls to both providers:

```bash
uv run python -m backend.run_eval_cli --cases <case-id>   # a single case
uv run python -m backend.run_eval_cli                      # all cases
```

Prefix with `LLM_CLIENT_VERBOSE=1` for a full request/response trace; every run saves a `.json` report and `.log` trace to `data/eval/results/` (gitignored).

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, direct Anthropic + Google API calls
- **UI:** One static HTML/JS page (`static/second-opinion-result.html`), served same-origin — no separate frontend build
- **Caching:** Disk cache for claim extraction / embeddings in `data/cache/` (gitignored)
- **Package Management:** uv
