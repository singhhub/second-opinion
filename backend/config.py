"""Configuration for Second Opinion."""

# Council members - queried directly against their providers by
# llm_client.py (provider/model-id format).
COUNCIL_MODELS = [
    "gemini/gemini-3.1-pro-preview",
    "claude/claude-sonnet-4-5-20250929",
]

# Chairman model - synthesizes the claim-diff summary
CHAIRMAN_MODEL = "gemini/gemini-3.1-pro-preview"
