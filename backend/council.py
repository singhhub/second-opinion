"""3-stage LLM Council orchestration."""

import json
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, ValidationError
from .llm_client import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
from .claim_diff import Claim, ClaimPair, ClaimState, categorize_claim


class ObservationGroup(BaseModel):
    label: str
    items: List[str] = []


class ChairmanSummary(BaseModel):
    agreed_findings: str
    disagreement_summary: str
    observations: List[ObservationGroup] = []
    questions_for_doctor: List[str] = []


CHAIRMAN_NEVER_EMERGENCY_RULE = (
    "You must NEVER state or imply whether this is or is not a medical "
    "emergency, and must NEVER tell the user to go to the ER or that it's "
    "safe not to - that call belongs to the user and their doctor, not you. "
    "You may name specific observations, symptoms, or red flags worth "
    "checking - never the verdict itself."
)


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Returns one entry per *configured* model (COUNCIL_MODELS), not only
    successes — a failed call must stay visible downstream with an
    explicit status, never silently vanish (see design doc's Degraded-run
    safety rule: the old `if response is not None` filter is the exact
    error-path-to-false-unanimity that rule exists to prevent).

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model', 'status' ('ok'/'failed'), and
        'response' (None if failed) keys.
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results - one entry per configured model, always
    stage1_results = []
    for model in COUNCIL_MODELS:
        response = responses.get(model)
        if response is not None:
            stage1_results.append({
                "model": model,
                "status": "ok",
                "response": response.get('content', '')
            })
        else:
            stage1_results.append({
                "model": model,
                "status": "failed",
                "response": None
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Only rank models that actually responded - a failed call has no
    # content to anonymize or evaluate, and must not be silently treated
    # as agreement by including it as an empty/placeholder response.
    ok_results = [r for r in stage1_results if r.get("status") == "ok"]

    if not ok_results:
        return [], {}

    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(ok_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, ok_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, ok_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    ok_results = [r for r in stage1_results if r.get("status") == "ok"]
    failed_models = [r["model"] for r in stage1_results if r.get("status") == "failed"]

    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in ok_results
    ])
    if failed_models:
        stage1_text += (
            "\n\n[DEGRADED RUN] The following models failed to respond and are "
            f"NOT reflected above: {', '.join(failed_models)}"
        )

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("gemini/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses (one entry per configured model,
    # including failed ones - see stage1_collect_responses)
    stage1_results = await stage1_collect_responses(user_query)

    ok_results = [r for r in stage1_results if r["status"] == "ok"]
    failed_models = [r["model"] for r in stage1_results if r["status"] == "failed"]

    # If no models responded successfully, return error
    if not ok_results:
        return stage1_results, [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {
            "label_to_model": {},
            "aggregate_rankings": [],
            "degraded": True,
            "failed_models": failed_models,
        }

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )

    # Prepare metadata - a run is degraded if any configured model failed,
    # and this must stay visible, never silently reported as if every
    # model agreed (see design doc's Degraded-run safety rule).
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "degraded": len(failed_models) > 0,
        "failed_models": failed_models,
    }

    return stage1_results, stage2_results, stage3_result, metadata


def build_ranked_diff_items(
    conflicting: List[ClaimPair],
    unconfirmed_a: List[Claim],
    unconfirmed_b: List[Claim],
) -> List[Dict[str, Any]]:
    """
    Merge CONFLICTING pairs and UNCONFIRMED claims (from both sides) into
    one list, ranked together by actionability category.

    The design doc's flooding-prevention fix ranks CONFLICTING and
    UNCONFIRMED together in one section, not each state separately -
    otherwise a lower-category conflict could still outrank a
    high-category unconfirmed finding like a drug interaction. Lossless:
    every input claim/pair appears exactly once in the output.
    """
    items: List[Dict[str, Any]] = []

    for pair in conflicting:
        items.append({
            "state": ClaimState.CONFLICTING,
            "category": categorize_claim(pair.claim_a),
            "claim_a": pair.claim_a,
            "claim_b": pair.claim_b,
        })

    for claim in unconfirmed_a:
        items.append({
            "state": ClaimState.UNCONFIRMED,
            "category": categorize_claim(claim),
            "side": "A",
            "claim": claim,
        })

    for claim in unconfirmed_b:
        items.append({
            "state": ClaimState.UNCONFIRMED,
            "category": categorize_claim(claim),
            "side": "B",
            "claim": claim,
        })

    items.sort(key=lambda item: item["category"].value)
    return items


def _format_agreed_section(agreed: List[ClaimPair]) -> str:
    if not agreed:
        return "No agreed findings - the two models did not converge on any claim."
    return "\n".join(f"- {pair.claim_a.text}" for pair in agreed)


def _format_diff_section(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No conflicting or unconfirmed claims - the two models fully agreed."

    lines = []
    for item in items:
        if item["state"] == ClaimState.CONFLICTING:
            lines.append(
                f'- CONFLICTING: Model A says "{item["claim_a"].text}"; '
                f'Model B says "{item["claim_b"].text}"'
            )
        else:
            lines.append(f'- UNCONFIRMED (Model {item["side"]} only): {item["claim"].text}')
    return "\n".join(lines)


CHAIRMAN_SYSTEM_PROMPT = f"""{CHAIRMAN_NEVER_EMERGENCY_RULE}

You are organizing a disagreement-preserving summary for two family caregivers who already saw two AI models' anonymized answers to their question about a family member's care. The chairman organizes and explains the mechanically-computed diff; it does not re-decide what counts as a conflict.

Respond with ONLY a JSON object matching this exact shape - no markdown fences, no extra text before or after:
{{
  "agreed_findings": "<a short paragraph summarizing what both models agree on - an empty string if there are none, never omit the field>",
  "disagreement_summary": "<a short paragraph explaining the disagreement or gap in plain language - an empty string if there are no conflicting/unconfirmed items>",
  "observations": [
    {{"label": "<short label for what this group of checks is for>", "items": ["<a specific observation, symptom, or check - not which claim to believe, but what to go look at>"]}}
  ],
  "questions_for_doctor": ["<a short, specific question the caregivers should bring to the next appointment>"]
}}

If there is no disagreement, set "disagreement_summary" to an empty string and "observations" to an empty list - never omit them or leave them implicit."""


def _parse_chairman_summary(raw: Optional[Dict[str, Any]]) -> Optional[ChairmanSummary]:
    if raw is None or not raw.get("content"):
        return None
    try:
        data = json.loads(raw["content"])
        return ChairmanSummary(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _flatten_chairman_summary(summary: ChairmanSummary) -> str:
    lines = ["Agreed Findings:", summary.agreed_findings or "None."]
    lines += ["", "Conflicting / Unconfirmed:", summary.disagreement_summary or "None."]
    for group in summary.observations:
        lines.append(f"- {group.label}:")
        for item in group.items:
            lines.append(f"  - {item}")
    lines += ["", "Questions for the Doctor:"]
    for question in summary.questions_for_doctor:
        lines.append(f"- {question}")
    return "\n".join(lines)


async def synthesize_claim_diff_chairman(
    user_query: str,
    agreed: List[ClaimPair],
    ranked_diff_items: List[Dict[str, Any]],
    chairman_model: str = CHAIRMAN_MODEL,
) -> Dict[str, Any]:
    """
    Chairman synthesis for the claim-diff mechanism.

    Returns structured JSON (agreed findings / disagreement + observations /
    questions for the doctor), Pydantic-validated with retry-then-fallback -
    the same pattern as claim_diff.extract_claims, so a malformed or failed
    LLM response never crashes the caller. "structured" is the ChairmanSummary
    object for callers that render it (the API endpoint); "response" is a
    flattened plain-text rendition for callers that only need to judge text
    (eval_harness's retention/emergency-verdict checks).
    """
    agreed_text = _format_agreed_section(agreed)
    diff_text = _format_diff_section(ranked_diff_items)

    user_message = f"""Original Question: {user_query}

AGREED FINDINGS (both models agree):
{agreed_text}

CONFLICTING / UNCONFIRMED CLAIMS (ranked by actionability, most important first):
{diff_text}"""

    messages = [
        {"role": "system", "content": CHAIRMAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(2):
        raw = await query_model(chairman_model, messages)
        summary = _parse_chairman_summary(raw)
        if summary is not None:
            return {
                "model": chairman_model,
                "structured": summary,
                "response": _flatten_chairman_summary(summary),
            }

    fallback = ChairmanSummary(
        agreed_findings="",
        disagreement_summary="Unable to generate a summary for this case - please try again.",
        observations=[],
        questions_for_doctor=[],
    )
    return {
        "model": chairman_model,
        "structured": fallback,
        "response": _flatten_chairman_summary(fallback),
    }
