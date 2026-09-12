"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OmniRoute API key (from http://localhost:20128/dashboard/api-manager)
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY")

# Council members - model identifiers as exposed by your OmniRoute instance
# (see http://localhost:20128/dashboard or GET /v1/models for what's available)
COUNCIL_MODELS = [
    # "openai/gpt-5.1",
    "gemini/gemini-3.1-pro-preview",
    "claude/claude-sonnet-4-5-20250929",
    # "x-ai/grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "gemini/gemini-3.1-pro-preview"

# OmniRoute API endpoint (local instance)
OMNIROUTE_API_URL = os.getenv("OMNIROUTE_API_URL", "http://localhost:20128/v1/chat/completions")

# Data directory for conversation storage
DATA_DIR = "data/conversations"
