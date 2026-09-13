"""Direct-call LLM client for Anthropic and Google (no OmniRoute proxy).

Mirrors omniroute.py's query_model/query_models_parallel interface so
council.py can switch between the two clients without changing call sites.
Real medical text must never route through the third-party OmniRoute
proxy — see the design doc's data-hygiene constraint.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Disk cache for extraction/embedding results, keyed by input+prompt hash -
# avoids re-paying API cost on every eval-tuning iteration (Finding 8).
# Lives under data/ so it's covered by the existing data/ gitignore rule.
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", "data/cache"))

# Opt-in request/response logging for debugging live runs - off by default
# so normal usage and tests stay quiet. Enable with LLM_CLIENT_VERBOSE=1.
VERBOSE = os.getenv("LLM_CLIENT_VERBOSE", "").lower() in ("1", "true", "yes")


def _log(label: str, **fields: Any) -> None:
    if not VERBOSE:
        return
    parts = "  ".join(f"{k}={v!r}" for k, v in fields.items())
    print(f"[llm_client] {label}  {parts}")


def cache_key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def cache_read(key: str) -> Optional[Any]:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def cache_write(key: str, value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(value, f)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 4096

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"

# Real-world observation, not theoretical: live eval runs saw intermittent
# "all connection attempts failed" errors that never reproduced in dozens of
# isolated single/rapid-sequential/alternating-host diagnostic calls - only
# in the longer, slower real run. That points to a several-second network
# blip rather than a code bug, so retries need real wall-clock time to ride
# it out, not just a couple of quick attempts. Exponential backoff, capped
# retry count - never for a deterministic 4xx like bad auth or a bad
# request, where retrying just wastes another call for the same answer.
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def _post_json(
    url: str,
    *,
    json_payload: Dict[str, Any],
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(MAX_RETRIES + 1):
        _log("→ POST", url=url, attempt=f"{attempt + 1}/{MAX_RETRIES + 1}", payload_preview=str(json_payload)[:400])
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, params=params, json=json_payload)
                response.raise_for_status()
                data = response.json()
                _log("← 200", url=url, attempt=attempt + 1, body_preview=str(data)[:400])
                return data
        except httpx.HTTPStatusError as e:
            _log("← HTTP ERROR", url=url, attempt=attempt + 1, status=e.response.status_code, body=e.response.text[:400])
            if e.response.status_code not in _RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise
            last_error = e
        except httpx.TransportError as e:
            _log("← TRANSPORT ERROR", url=url, attempt=attempt + 1, error=f"{type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                raise
            last_error = e
        delay = RETRY_BACKOFF_SECONDS * (2**attempt)
        _log("  retrying after", seconds=delay)
        await asyncio.sleep(delay)
    raise last_error


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

    data = await _post_json(ANTHROPIC_API_URL, json_payload=payload, timeout=timeout, headers=headers)

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

    data = await _post_json(url, json_payload=payload, timeout=timeout, params={"key": GOOGLE_API_KEY})

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

    last_msg = messages[-1]["content"] if messages else ""
    _log("QUERY_MODEL start", model=model, num_messages=len(messages), last_message_preview=last_msg[:200])

    try:
        result = await handler(model_id, messages, timeout)
        _log("QUERY_MODEL done", model=model, content_preview=(result.get("content") or "")[:300])
        return result
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def embed_text(
    text: str,
    model: str = GOOGLE_EMBEDDING_MODEL,
    timeout: float = 60.0,
    use_cache: bool = True,
) -> Optional[List[float]]:
    """
    Embed a piece of text via Google's embedding API.

    No local embedding model is available, so embeddings go through Gemini —
    already a trusted party since it's also a council model (see design doc
    Constraints). Returns None on failure rather than raising, matching the
    graceful-degradation philosophy of query_model.

    Cached on disk by (model, text) - a failed call is never cached, so a
    transient failure doesn't permanently stick as a cached None.
    """
    key = cache_key("embed", model, text)
    if use_cache:
        cached = cache_read(key)
        if cached is not None:
            _log("EMBED cache hit", model=model, text_preview=text[:100], dims=len(cached))
            return cached

    url = f"{GOOGLE_API_URL}/{model}:embedContent"
    payload = {"content": {"parts": [{"text": text}]}}

    _log("EMBED start", model=model, text_preview=text[:100])

    try:
        data = await _post_json(url, json_payload=payload, timeout=timeout, params={"key": GOOGLE_API_KEY})
        values = data["embedding"]["values"]
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

    _log("EMBED done", model=model, dims=len(values))

    if use_cache:
        cache_write(key, values)
    return values


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
