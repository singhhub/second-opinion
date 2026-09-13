"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import uuid
import json
import asyncio

from . import storage
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings,
    build_ranked_diff_items,
    synthesize_claim_diff_chairman,
)
from .claim_diff import align_claims, classify_compatibility, extract_claims, ClaimState

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="LLM Council API")


def validate_conversation_id(conversation_id: str) -> None:
    """Reject anything that isn't a well-formed UUID before it touches the filesystem."""
    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class AnalyzeRequest(BaseModel):
    """Request to run the claim-diff mechanism on a single question."""
    question: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    validate_conversation_id(conversation_id)
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    validate_conversation_id(conversation_id)
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


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


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    validate_conversation_id(conversation_id)
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(request.content)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2: Collect rankings
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results)
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
