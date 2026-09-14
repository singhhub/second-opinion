"""Tests for the direct-call LLM client (backend/llm_client.py).

llm_client mirrors omniroute.py's query_model/query_models_parallel
interface but calls Anthropic and Google directly instead of routing
through the local OmniRoute proxy.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend import llm_client


def _mock_response(json_data):
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


async def test_query_model_claude_success():
    fake_response = _mock_response(
        {"content": [{"type": "text", "text": "Hello from Claude"}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result == {"content": "Hello from Claude", "reasoning_details": None}
    assert mock_post.call_args.args[0] == llm_client.ANTHROPIC_API_URL


async def test_query_model_gemini_success():
    fake_response = _mock_response(
        {"candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = await llm_client.query_model(
            "gemini/gemini-3.1-pro-preview",
            [{"role": "user", "content": "hi"}],
        )

    assert result == {"content": "Hello from Gemini", "reasoning_details": None}
    assert mock_post.call_args.args[0].startswith(llm_client.GOOGLE_API_URL)


async def test_query_model_unknown_provider_returns_none():
    result = await llm_client.query_model(
        "unknown-provider/some-model",
        [{"role": "user", "content": "hi"}],
    )

    assert result is None


async def test_query_model_network_error_returns_none():
    with patch.object(llm_client, "RETRY_BACKOFF_SECONDS", 0), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectTimeout("timed out")),
    ):
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result is None


async def test_query_model_retries_transient_connection_error_then_succeeds():
    fake_response = _mock_response({"content": [{"type": "text", "text": "ok"}]})
    with patch.object(llm_client, "RETRY_BACKOFF_SECONDS", 0), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=[httpx.ConnectError("all connection attempts failed"), fake_response]),
    ) as mock_post:
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result == {"content": "ok", "reasoning_details": None}
    assert mock_post.await_count == 2


async def test_query_model_retries_503_then_succeeds():
    error_response = MagicMock(spec=httpx.Response)
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "service unavailable", request=MagicMock(), response=MagicMock(status_code=503)
    )
    fake_response = _mock_response({"content": [{"type": "text", "text": "ok"}]})
    with patch.object(llm_client, "RETRY_BACKOFF_SECONDS", 0), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=[error_response, fake_response]),
    ) as mock_post:
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result == {"content": "ok", "reasoning_details": None}
    assert mock_post.await_count == 2


async def test_query_model_does_not_retry_a_400_bad_request():
    error_response = MagicMock(spec=httpx.Response)
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad request", request=MagicMock(), response=MagicMock(status_code=400)
    )
    with patch.object(llm_client, "RETRY_BACKOFF_SECONDS", 0), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=error_response)
    ) as mock_post:
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result is None
    assert mock_post.await_count == 1


async def test_query_model_gives_up_after_max_retries():
    with patch.object(llm_client, "RETRY_BACKOFF_SECONDS", 0), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectError("all connection attempts failed")),
    ) as mock_post:
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result is None
    assert mock_post.await_count == llm_client.MAX_RETRIES + 1


async def test_embed_text_retries_transient_connection_error_then_succeeds(tmp_path):
    fake_response = _mock_response({"embedding": {"values": [1.0, 2.0]}})
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        llm_client, "RETRY_BACKOFF_SECONDS", 0
    ), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=[httpx.ConnectError("all connection attempts failed"), fake_response]),
    ) as mock_post:
        result = await llm_client.embed_text("some text")

    assert result == [1.0, 2.0]
    assert mock_post.await_count == 2


async def test_query_models_parallel_maps_models_to_responses():
    fake_claude = _mock_response({"content": [{"type": "text", "text": "A"}]})
    fake_gemini = _mock_response(
        {"candidates": [{"content": {"parts": [{"text": "B"}]}}]}
    )

    async def fake_post(url, **kwargs):
        if url == llm_client.ANTHROPIC_API_URL:
            return fake_claude
        return fake_gemini

    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(side_effect=fake_post)):
        result = await llm_client.query_models_parallel(
            ["claude/claude-sonnet-4-5-20250929", "gemini/gemini-3.1-pro-preview"],
            [{"role": "user", "content": "hi"}],
        )

    assert result["claude/claude-sonnet-4-5-20250929"]["content"] == "A"
    assert result["gemini/gemini-3.1-pro-preview"]["content"] == "B"


async def test_query_model_claude_separates_system_message():
    fake_response = _mock_response(
        {"content": [{"type": "text", "text": "ok"}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
            ],
        )

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["system"] == "You are a helpful assistant."
    assert sent_payload["messages"] == [{"role": "user", "content": "hi"}]


async def test_embed_text_success_returns_vector(tmp_path):
    fake_response = _mock_response({"embedding": {"values": [0.1, 0.2, 0.3]}})
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = await llm_client.embed_text("drug A and drug B interact")

    assert result == [0.1, 0.2, 0.3]
    called_url = mock_post.call_args.args[0]
    assert called_url.startswith(llm_client.GOOGLE_API_URL)
    assert called_url.endswith(":embedContent")


async def test_embed_text_failure_returns_none(tmp_path):
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectTimeout("timed out")),
    ):
        result = await llm_client.embed_text("drug A and drug B interact")

    assert result is None


async def test_embed_text_second_call_same_text_uses_cache(tmp_path):
    fake_response = _mock_response({"embedding": {"values": [1.0, 2.0]}})
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        first = await llm_client.embed_text("same text")
        second = await llm_client.embed_text("same text")

    assert first == second == [1.0, 2.0]
    assert mock_post.await_count == 1


async def test_embed_text_different_text_is_a_cache_miss(tmp_path):
    fake_response = _mock_response({"embedding": {"values": [1.0, 2.0]}})
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.embed_text("text one")
        await llm_client.embed_text("text two")

    assert mock_post.await_count == 2


async def test_embed_text_does_not_cache_failures(tmp_path):
    # First call exhausts every retry attempt (all transient failures) and
    # must give up with None; second call is a fresh attempt that succeeds -
    # a failure that exhausted retries must still not poison the cache.
    responses = [httpx.ConnectTimeout("timed out")] * (llm_client.MAX_RETRIES + 1) + [
        _mock_response({"embedding": {"values": [3.0, 4.0]}})
    ]
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        llm_client, "RETRY_BACKOFF_SECONDS", 0
    ), patch.object(httpx.AsyncClient, "post", new=AsyncMock(side_effect=responses)) as mock_post:
        first = await llm_client.embed_text("retry me")
        second = await llm_client.embed_text("retry me")

    assert first is None
    assert second == [3.0, 4.0]
    assert mock_post.await_count == llm_client.MAX_RETRIES + 1 + 1


async def test_embed_text_use_cache_false_bypasses_cache(tmp_path):
    fake_response = _mock_response({"embedding": {"values": [1.0, 2.0]}})
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.embed_text("same text", use_cache=False)
        await llm_client.embed_text("same text", use_cache=False)

    assert mock_post.await_count == 2


async def test_query_model_gemini_separates_system_message():
    fake_response = _mock_response(
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.query_model(
            "gemini/gemini-3.1-pro-preview",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"},
            ],
        )

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["systemInstruction"] == {
        "parts": [{"text": "You are a helpful assistant."}]
    }
    assert sent_payload["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


async def test_query_model_gemini_maps_assistant_role_to_model():
    fake_response = _mock_response(
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.query_model(
            "gemini/gemini-3.1-pro-preview",
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
            ],
        )

    sent_contents = mock_post.call_args.kwargs["json"]["contents"]
    roles = [c["role"] for c in sent_contents]
    assert roles == ["user", "model"]


async def test_query_model_claude_no_system_message_omits_system_key():
    fake_response = _mock_response({"content": [{"type": "text", "text": "ok"}]})
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    sent_payload = mock_post.call_args.kwargs["json"]
    assert "system" not in sent_payload


async def test_query_model_http_error_status_returns_none():
    error_response = MagicMock(spec=httpx.Response)
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=MagicMock(), response=error_response
    )
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=error_response)
    ):
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result is None


async def test_query_model_claude_concatenates_multiple_text_blocks():
    fake_response = _mock_response(
        {
            "content": [
                {"type": "text", "text": "first part. "},
                {"type": "text", "text": "second part."},
            ]
        }
    )
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)):
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result["content"] == "first part. second part."


async def test_query_model_gemini_concatenates_multiple_parts():
    fake_response = _mock_response(
        {"candidates": [{"content": {"parts": [{"text": "first. "}, {"text": "second."}]}}]}
    )
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)):
        result = await llm_client.query_model(
            "gemini/gemini-3.1-pro-preview",
            [{"role": "user", "content": "hi"}],
        )

    assert result["content"] == "first. second."


async def test_query_models_parallel_mixed_success_and_failure():
    async def fake_post(url, **kwargs):
        if url == llm_client.ANTHROPIC_API_URL:
            raise httpx.ConnectTimeout("timed out")
        return _mock_response({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(side_effect=fake_post)):
        result = await llm_client.query_models_parallel(
            ["claude/claude-sonnet-4-5-20250929", "gemini/gemini-3.1-pro-preview"],
            [{"role": "user", "content": "hi"}],
        )

    assert result["claude/claude-sonnet-4-5-20250929"] is None
    assert result["gemini/gemini-3.1-pro-preview"]["content"] == "ok"


async def test_embed_text_custom_model_used_in_url_and_cache_key(tmp_path):
    fake_response = _mock_response({"embedding": {"values": [9.0]}})
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        await llm_client.embed_text("some text", model="custom-embedding-model")

    called_url = mock_post.call_args.args[0]
    assert called_url == f"{llm_client.GOOGLE_API_URL}/custom-embedding-model:embedContent"

    # A different model for the same text must be a cache miss, not
    # accidentally share the default model's cache entry.
    with patch.object(llm_client, "CACHE_DIR", tmp_path), patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post_again:
        await llm_client.embed_text("some text")

    assert mock_post_again.await_count == 1


def test_strip_json_fence_no_fence_returns_unchanged():
    text = '{"a": 1}'
    result = llm_client.strip_json_fence(text)
    assert result == '{"a": 1}'
    assert json.loads(result) == {"a": 1}


def test_strip_json_fence_strips_json_language_tagged_fence():
    text = '```json\n{"a": 1}\n```'
    result = llm_client.strip_json_fence(text)
    assert json.loads(result) == {"a": 1}


def test_strip_json_fence_strips_bare_fence_without_language_tag():
    text = '```\n{"a": 1}\n```'
    result = llm_client.strip_json_fence(text)
    assert json.loads(result) == {"a": 1}


def test_strip_json_fence_handles_surrounding_whitespace():
    text = '  \n```json\n{"a": 1}\n```\n  '
    result = llm_client.strip_json_fence(text)
    assert json.loads(result) == {"a": 1}
