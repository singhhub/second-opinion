"""Offline end-to-end test of /api/second-opinion/analyze.

Only the LLM transport is mocked: every binding of llm_client.query_model /
query_models_parallel / embed_text that the real code paths call. The real
stage-1 split, claim extraction parsing, alignment, classification,
compare_structured, select_challenge_items, run_challenge_round,
apply_challenge_outcomes, chairman prompt and response building all run.
"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.challenge import CHALLENGE_SYSTEM_PROMPT
from backend.claim_diff import (
    ALIGNMENT_JUDGE_SYSTEM_PROMPT,
    COMPATIBILITY_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
)
from backend.config import COUNCIL_MODELS

client = TestClient(main.app)

DIAG_A = "The confusion is consistent with a urinary tract infection."
DIAG_B = "The confusion is consistent with a medication side effect."
SHARED = "Keep an up-to-date list of every medication he takes."

ANSWER_A = "Model A prose about a likely infection."
ANSWER_B = "Model B prose about a likely side effect."


def _block(level):
    return (
        "<structured>"
        + json.dumps({
            "possible_causes": [],
            "interaction_flags": [],
            "medication_changes": [],
            "urgency": {"level": level, "escalate_if": []},
        })
        + "</structured>"
    )


_EMBED = {
    DIAG_A: [1.0, 0.0, 0.0],
    DIAG_B: [1.0, 0.0, 0.0],
    SHARED: [0.0, 1.0, 0.0],
}


class FakeLLM:
    """Routes each call by its system prompt; records chairman messages."""

    def __init__(self, claims_a, claims_b, concede_texts=()):
        self.claims = {ANSWER_A: claims_a, ANSWER_B: claims_b}
        self.concede_texts = set(concede_texts)
        self.chairman_messages = []
        self.challenge_calls = 0

    async def query_models_parallel(self, models, messages):
        prompt = messages[0]["content"]
        assert "<structured>" in prompt  # stage 1 asks for the block
        return {
            COUNCIL_MODELS[0]: {"content": ANSWER_A + "\n\n" + _block("emergency")},
            COUNCIL_MODELS[1]: {"content": ANSWER_B + "\n\n" + _block("within_days")},
        }

    async def query_model(self, model, messages, *args, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if system == EXTRACTION_SYSTEM_PROMPT:
            claims = self.claims[user.strip()]
            return {"content": json.dumps(
                [{"text": t, "claim_type": "assertion"} for t in claims]
            )}
        if system == ALIGNMENT_JUDGE_SYSTEM_PROMPT:
            return {"content": "NO"}
        if system == COMPATIBILITY_SYSTEM_PROMPT:
            a = user.split("Claim A: ")[1].split("\nClaim B: ")[0]
            b = user.split("\nClaim B: ")[1]
            return {"content": "COMPATIBLE" if a == b else "INCOMPATIBLE"}
        if system == CHALLENGE_SYSTEM_PROMPT:
            self.challenge_calls += 1
            own = user.split("Your position:\n")[1].split("\n")[0]
            stance = "concede" if own in self.concede_texts else "maintain"
            return {"content": json.dumps({
                "stance": stance, "revised_claim": None,
                "reasoning": "r", "evidence_quote": None,
            })}
        if "disagreement-preserving summary" in system:
            self.chairman_messages.append(messages)
            return {"content": json.dumps({
                "agreed_findings": "", "disagreement_summary": "",
                "observations": [], "questions_for_doctor": [],
            })}
        raise AssertionError("unexpected LLM call: " + system[:80])

    async def embed_text(self, text, *args, **kwargs):
        return _EMBED[text]


def _analyze(fake):
    with patch("backend.llm_client.query_model", new=fake.query_model), patch(
        "backend.llm_client.query_models_parallel", new=fake.query_models_parallel
    ), patch("backend.llm_client.embed_text", new=fake.embed_text), patch(
        "backend.council.query_model", new=fake.query_model
    ), patch(
        "backend.council.query_models_parallel", new=fake.query_models_parallel
    ), patch("backend.challenge.query_model", new=fake.query_model):
        response = client.post("/api/second-opinion/analyze", json={"question": "Dad is confused"})
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False, body
    return body


def _urgency_item(body):
    items = [i for i in body["challenge"]["items"] if i["source"] == "structured"]
    assert len(items) == 1
    return items[0]


def test_e2e_claims_agree_but_urgency_differs_is_never_fully_agreed():
    fake = FakeLLM([SHARED], [SHARED])
    body = _analyze(fake)

    assert body["models"][0]["answer"] == ANSWER_A
    assert body["counts"] == {
        "agreed": 1, "conflicting": 0, "unconfirmed": 0,
        "resolved_after_challenge": 0, "structured_disagreements": 1,
    }
    assert body["structured"]["available"] is True
    assert [d["kind"] for d in body["structured"]["disagreements"]] == ["urgency"]
    urg = _urgency_item(body)
    assert urg["always_visible"] is True
    assert urg["status"] == "still_conflicting"

    user_message = fake.chairman_messages[-1][1]["content"]
    assert "fully agreed" not in user_message
    assert "STRUCTURED COMPARISON DISAGREEMENTS" in user_message
    assert "Urgency differs: emergency vs within_days" in user_message
    assert "urgency: always report" in user_message


def test_e2e_both_models_maintain_nothing_resolves():
    fake = FakeLLM([DIAG_A, SHARED], [DIAG_B, SHARED])
    body = _analyze(fake)

    assert body["counts"] == {
        "agreed": 1, "conflicting": 1, "unconfirmed": 0,
        "resolved_after_challenge": 0, "structured_disagreements": 1,
    }
    statuses = {i["source"]: i["status"] for i in body["challenge"]["items"]}
    assert statuses == {"structured": "still_conflicting", "conflicting": "still_conflicting"}
    # urgency item challenged first, and always visible
    assert body["challenge"]["items"][0]["source"] == "structured"
    assert _urgency_item(body)["always_visible"] is True
    assert fake.challenge_calls == 4

    user_message = fake.chairman_messages[-1][1]["content"]
    assert "STRUCTURED COMPARISON DISAGREEMENTS" in user_message
    assert "fully agreed" not in user_message
    assert DIAG_A in user_message and DIAG_B in user_message
    assert "settled after challenge" not in user_message


def test_e2e_one_side_concedes_claim_pair_resolves_and_is_tagged():
    fake = FakeLLM([DIAG_A, SHARED], [DIAG_B, SHARED], concede_texts={DIAG_B})
    body = _analyze(fake)

    assert body["counts"] == {
        "agreed": 1, "conflicting": 0, "unconfirmed": 0,
        "resolved_after_challenge": 1, "structured_disagreements": 1,
    }
    by_source = {i["source"]: i for i in body["challenge"]["items"]}
    claim_item = by_source["conflicting"]
    assert claim_item["status"] == "resolved"
    assert claim_item["original_a"] == DIAG_A and claim_item["original_b"] == DIAG_B
    assert claim_item["final_b"] == DIAG_A
    assert "resolved_without_record_evidence" in claim_item["flags"]
    assert by_source["structured"]["status"] == "still_conflicting"
    assert by_source["structured"]["always_visible"] is True

    user_message = fake.chairman_messages[-1][1]["content"]
    agreed_section = user_message.split("AGREED FINDINGS")[1].split("CONFLICTING / UNCONFIRMED")[0]
    settled_line = [l for l in agreed_section.splitlines() if DIAG_A in l][0]
    assert "settled after challenge" in settled_line
    assert DIAG_B in settled_line
    assert "STRUCTURED COMPARISON DISAGREEMENTS" in user_message
    assert "fully agreed" not in user_message
    assert "DIAGNOSTIC_SUGGESTION" in user_message
