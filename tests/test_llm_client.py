"""Tests for the direct-call LLM client (backend/llm_client.py).

llm_client mirrors omniroute.py's query_model/query_models_parallel
interface but calls Anthropic and Google directly instead of routing
through the local OmniRoute proxy.
"""

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
    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectTimeout("timed out")),
    ):
        result = await llm_client.query_model(
            "claude/claude-sonnet-4-5-20250929",
            [{"role": "user", "content": "hi"}],
        )

    assert result is None


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


async def test_embed_text_success_returns_vector():
    fake_response = _mock_response({"embedding": {"values": [0.1, 0.2, 0.3]}})
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = await llm_client.embed_text("drug A and drug B interact")

    assert result == [0.1, 0.2, 0.3]
    called_url = mock_post.call_args.args[0]
    assert called_url.startswith(llm_client.GOOGLE_API_URL)
    assert called_url.endswith(":embedContent")


async def test_embed_text_failure_returns_none():
    with patch.object(
        httpx.AsyncClient,
        "post",
        new=AsyncMock(side_effect=httpx.ConnectTimeout("timed out")),
    ):
        result = await llm_client.embed_text("drug A and drug B interact")

    assert result is None
