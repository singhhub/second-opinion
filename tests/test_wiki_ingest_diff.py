"""Tests for the patient wiki ingest pipeline's diff-proposal step
(backend/wiki_ingest.py's propose_diff).
"""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

from backend import wiki_ingest


def _fake_response(content: str) -> Dict[str, Any]:
    return {"content": content, "reasoning_details": None}


async def test_propose_diff_parses_valid_response():
    valid_json = json.dumps({
        "page_path": "medications.md",
        "is_new_page": False,
        "new_page_content": "# Medications\n- Lisinopril 10mg",
        "contradiction": False,
        "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        result = await wiki_ingest.propose_diff("Lisinopril 10mg once daily", {})

    assert result.page_path == "medications.md"
    assert result.is_new_page is False
    assert result.contradiction is False
    assert mock_query.await_count == 1


async def test_propose_diff_strips_markdown_fence():
    fenced = "```json\n" + json.dumps({
        "page_path": "allergies.md", "is_new_page": True,
        "new_page_content": "# Allergies\n- Penicillin",
        "contradiction": False, "contradiction_note": None,
    }) + "\n```"
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(fenced))
    ):
        result = await wiki_ingest.propose_diff("Penicillin allergy noted", {})

    assert result.page_path == "allergies.md"


async def test_propose_diff_retries_once_on_malformed_json():
    valid_json = json.dumps({
        "page_path": "conditions.md", "is_new_page": True,
        "new_page_content": "# Conditions\n- Hypertension",
        "contradiction": False, "contradiction_note": None,
    })
    responses = [_fake_response("not json"), _fake_response(valid_json)]
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(side_effect=responses)
    ) as mock_query:
        result = await wiki_ingest.propose_diff("Hypertension diagnosed", {})

    assert result.page_path == "conditions.md"
    assert mock_query.await_count == 2


async def test_propose_diff_falls_back_to_needs_review_after_two_failures():
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response("not json"))
    ) as mock_query:
        result = await wiki_ingest.propose_diff("some source text", {})

    assert result.page_path == wiki_ingest.NEEDS_REVIEW_PAGE_PATH
    assert result.is_new_page is True
    assert "some source text" in result.new_page_content
    assert mock_query.await_count == 2


async def test_propose_diff_includes_existing_wiki_pages_in_prompt():
    valid_json = json.dumps({
        "page_path": "medications.md", "is_new_page": False,
        "new_page_content": "updated", "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        await wiki_ingest.propose_diff(
            "new source", {"medications.md": "# Medications\n- Warfarin 5mg"}
        )

    sent_prompt = mock_query.call_args.args[1][1]["content"]
    assert "Warfarin 5mg" in sent_prompt
    assert "medications.md" in sent_prompt


async def test_propose_diff_empty_wiki_says_no_pages_yet():
    valid_json = json.dumps({
        "page_path": "overview.md", "is_new_page": True,
        "new_page_content": "updated", "contradiction": False, "contradiction_note": None,
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ) as mock_query:
        await wiki_ingest.propose_diff("new source", {})

    sent_prompt = mock_query.call_args.args[1][1]["content"]
    assert "no pages yet" in sent_prompt


async def test_propose_diff_contradiction_flag_and_note_pass_through():
    valid_json = json.dumps({
        "page_path": "medications.md", "is_new_page": False,
        "new_page_content": "updated", "contradiction": True,
        "contradiction_note": "source omits warfarin, which the wiki lists as current",
    })
    with patch.object(
        wiki_ingest.llm_client, "query_model", new=AsyncMock(return_value=_fake_response(valid_json))
    ):
        result = await wiki_ingest.propose_diff("new source", {"medications.md": "old"})

    assert result.contradiction is True
    assert "warfarin" in result.contradiction_note
