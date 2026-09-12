"""Direct-call LLM client for Anthropic and Google (no OmniRoute proxy).

Mirrors omniroute.py's query_model/query_models_parallel interface so
council.py can switch between the two clients without changing call sites.
Real medical text must never route through the third-party OmniRoute
proxy — see the design doc's data-hygiene constraint.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 4096

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GOOGLE_EMBEDDING_MODEL = "text-embedding-004"


async def _query_claude(
    model: str, messages: List[Dict[str, str]], timeout: float
) -> Dict[str, Any]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": conversation,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )
    return {"content": text, "reasoning_details": None}


async def _query_gemini(
    model: str, messages: List[Dict[str, str]], timeout: float
) -> Dict[str, Any]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
        if m["role"] != "system"
    ]

    payload: Dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }

    url = f"{GOOGLE_API_URL}/{model}:generateContent"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, params={"key": GOOGLE_API_KEY}, json=payload)
        response.raise_for_status()
        data = response.json()

    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts)
    return {"content": text, "reasoning_details": None}


_PROVIDERS = {
    "claude": _query_claude,
    "gemini": _query_gemini,
}


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model directly against its provider (Anthropic or Google).

    Args:
        model: Provider-prefixed model identifier, e.g. "claude/claude-sonnet-4-5-20250929"
               or "gemini/gemini-3.1-pro-preview" (same format as config.COUNCIL_MODELS).
        messages: List of message dicts with 'role' and 'content'.
        timeout: Request timeout in seconds.

    Returns:
        Response dict with 'content' and 'reasoning_details', or None if failed.
    """
    provider, _, model_id = model.partition("/")
    handler = _PROVIDERS.get(provider)

    if handler is None:
        print(f"Error querying model {model}: unknown provider '{provider}'")
        return None

    try:
        return await handler(model_id, messages, timeout)
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def embed_text(
    text: str,
    model: str = GOOGLE_EMBEDDING_MODEL,
    timeout: float = 60.0,
) -> Optional[List[float]]:
    """
    Embed a piece of text via Google's embedding API.

    No local embedding model is available, so embeddings go through Gemini —
    already a trusted party since it's also a council model (see design doc
    Constraints). Returns None on failure rather than raising, matching the
    graceful-degradation philosophy of query_model.
    """
    url = f"{GOOGLE_API_URL}/{model}:embedContent"
    payload = {"content": {"parts": [{"text": text}]}}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, params={"key": GOOGLE_API_KEY}, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["embedding"]["values"]
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel, directly against their providers.

    Args:
        models: List of provider-prefixed model identifiers.
        messages: List of message dicts to send to each model.

    Returns:
        Dict mapping model identifier to response dict (or None if failed).
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
