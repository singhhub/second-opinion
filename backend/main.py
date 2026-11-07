"""FastAPI backend for Second Opinion."""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from .council import (
    stage1_collect_responses,
    build_ranked_diff_items,
    synthesize_claim_diff_chairman,
)
from .challenge import (
    select_challenge_items,
    run_challenge_round,
    apply_challenge_outcomes,
)
from .structured_diff import compare_structured
from .claim_diff import align_claims, classify_compatibility, extract_claims, ClaimState

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Second Opinion API")


class AnalyzeRequest(BaseModel):
    """Request to run the claim-diff mechanism on a single question."""
    question: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Second Opinion API"}


def _model_display_name(model_id: str) -> str:
    provider, _, _ = model_id.partition("/")
    return provider.capitalize()


@app.post("/api/second-opinion/analyze")
async def analyze_question(request: AnalyzeRequest):
    """
    Run the claim-diff mechanism end to end on a single question: get both
    models' raw answers, extract claims, align, classify, rank, and
    synthesize the chairman's structured summary. Stateless - no
    conversation_id, no storage. Makes real, billed calls to Claude and
    Gemini.
    """
    stage1_results = await stage1_collect_responses(request.question)
    ok_results = [r for r in stage1_results if r["status"] == "ok"]

    if len(ok_results) < 2:
        return {
            "question": request.question,
            "degraded": True,
            "error": "One or both models failed to respond. Please try again.",
            "models": [],
            "counts": None,
            "summary": None,
            "challenge": None,
            "structured": None,
        }

    model_a_result, model_b_result = ok_results[0], ok_results[1]

    try:
        claims_a = await extract_claims(model_a_result["response"])
        claims_b = await extract_claims(model_b_result["response"])

        alignment = await align_claims(claims_a, claims_b)

        agreed = []
        conflicting = []
        for pair in alignment.aligned:
            state = await classify_compatibility(pair)
            if state == ClaimState.AGREED:
                agreed.append(pair)
            else:
                conflicting.append(pair)

        unconfirmed_a = alignment.unaligned_a
        unconfirmed_b = alignment.unaligned_b
        block_a = model_a_result.get("structured")
        block_b = model_b_result.get("structured")
        disagreements = []
        structured_ok = True
        outcomes = None
        newly_agreed = []
        challenge = {"ran": False, "cap_hit": False, "error": None, "items": []}

        # Challenge step: any failure leaves every list untouched (fail-safe).
        try:
            if block_a is not None and block_b is not None:
                try:
                    disagreements = compare_structured(block_a, block_b)
                except Exception:
                    structured_ok = False
                    disagreements = []
            selection = select_challenge_items(
                conflicting, unconfirmed_a, unconfirmed_b, disagreements
            )
            if selection.selected:
                # record_context=None for now; the wiki/record hook plugs in here.
                outcomes = await run_challenge_round(
                    selection.selected,
                    request.question,
                    model_a_result["model"],
                    model_b_result["model"],
                    record_context=None,
                )
                new_agreed, new_conflicting, new_ua, new_ub = apply_challenge_outcomes(
                    conflicting, unconfirmed_a, unconfirmed_b, outcomes
                )
                new_challenge = {
                    "ran": True,
                    "cap_hit": selection.cap_hit,
                    "error": None,
                    "items": [
                        {
                            "id": o.item.item_id,
                            "source": o.item.source,
                            "category": o.item.category.name,
                            "original_a": o.item.text_a,
                            "original_b": o.item.text_b,
                            "final_a": o.final_a,
                            "final_b": o.final_b,
                            "status": o.status.value,
                            "flags": o.flags,
                            "always_visible": o.item.always_visible,
                        }
                        for o in outcomes
                    ],
                }
                # Commit state last: nothing above may raise after this point.
                agreed = agreed + new_agreed
                conflicting, unconfirmed_a, unconfirmed_b = new_conflicting, new_ua, new_ub
                newly_agreed = new_agreed
                challenge = new_challenge
            else:
                challenge["cap_hit"] = selection.cap_hit
        except Exception:
            outcomes = None
            newly_agreed = []
            challenge = {
                "ran": False,
                "cap_hit": False,
                "error": "challenge round failed; showing the original comparison",
                "items": [],
            }

        diff_items = build_ranked_diff_items(
            conflicting=conflicting,
            unconfirmed_a=unconfirmed_a,
            unconfirmed_b=unconfirmed_b,
        )

        chairman_result = await synthesize_claim_diff_chairman(
            request.question, agreed, diff_items, challenge_outcomes=outcomes
        )
        summary = chairman_result["structured"]
    except Exception:
        return {
            "question": request.question,
            "degraded": True,
            "error": "Something went wrong while comparing the two answers. Please try again.",
            "models": [],
            "counts": None,
            "summary": None,
            "challenge": None,
            "structured": None,
        }

    return {
        "question": request.question,
        "degraded": False,
        "error": None,
        "models": [
            {
                "label": "Model A",
                "name": _model_display_name(model_a_result["model"]),
                "answer": model_a_result["response"],
            },
            {
                "label": "Model B",
                "name": _model_display_name(model_b_result["model"]),
                "answer": model_b_result["response"],
            },
        ],
        "counts": {
            "agreed": len(agreed),
            "conflicting": len(conflicting),
            "unconfirmed": len(unconfirmed_a) + len(unconfirmed_b),
            "resolved_after_challenge": len(newly_agreed),
        },
        "challenge": challenge,
        "structured": {
            "available": structured_ok and block_a is not None and block_b is not None,
            "a": block_a.model_dump(mode="json") if block_a is not None else None,
            "b": block_b.model_dump(mode="json") if block_b is not None else None,
            "disagreements": [
                {
                    "kind": d.kind.value,
                    "summary": d.summary,
                    "detail_a": d.detail_a,
                    "detail_b": d.detail_b,
                }
                for d in disagreements
            ],
        },
        "summary": {
            "agreed_findings": summary.agreed_findings,
            "disagreement_summary": summary.disagreement_summary,
            "observations": [
                {"label": group.label, "items": group.items} for group in summary.observations
            ],
            "questions_for_doctor": summary.questions_for_doctor,
        },
    }


@app.get("/ui")
async def serve_ui():
    """Serve the standalone result screen, same-origin with the API - no CORS needed."""
    return FileResponse(STATIC_DIR / "second-opinion-result.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
