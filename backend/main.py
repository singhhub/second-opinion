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

        diff_items = build_ranked_diff_items(
            conflicting=conflicting,
            unconfirmed_a=alignment.unaligned_a,
            unconfirmed_b=alignment.unaligned_b,
        )

        chairman_result = await synthesize_claim_diff_chairman(request.question, agreed, diff_items)
        summary = chairman_result["structured"]
    except Exception:
        return {
            "question": request.question,
            "degraded": True,
            "error": "Something went wrong while comparing the two answers. Please try again.",
            "models": [],
            "counts": None,
            "summary": None,
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
            "unconfirmed": len(alignment.unaligned_a) + len(alignment.unaligned_b),
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
