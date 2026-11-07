"""Configuration for Second Opinion."""

# Council members - queried directly against their providers by
# llm_client.py (provider/model-id format).
COUNCIL_MODELS = [
    "gemini/gemini-3.1-pro-preview",
    "claude/claude-sonnet-4-5-20250929",
]

# Chairman model - synthesizes the claim-diff summary
CHAIRMAN_MODEL = "gemini/gemini-3.1-pro-preview"

# Actor id for the wiki's auto-apply path (backend/wiki_review.py's shared
# apply_diff(), called from wiki_ingest.py for non-critical/no-contradiction
# diffs) - distinct from a future human reviewer's actor id so the audit
# log can always tell the two apart. See the design doc's Scope section.
AUTO_APPLY_ACTOR_ID = "system-auto"

# Challenge round (see docs/superpowers/specs/2026-11-07-challenge-round-design.md)
CHALLENGE_MAX_ROUNDS = 1
CHALLENGE_MAX_ROUNDS_HARD_CAP = 2
CHALLENGE_MAX_ITEMS = 10
# How many unconfirmed claims the "answers differ but nothing qualified"
# fallback challenges.
CHALLENGE_FALLBACK_ITEMS = 3
