"""Structured assessment schema and block splitting (challenge round).

Each model answers with free text plus an optional machine-readable block
between literal <structured> and </structured> tags. This module defines
the schema and splits a response into (free text, parsed block). Pure code:
a missing or malformed block yields None and never raises.
"""

import re
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ValidationError

from . import llm_client


class CauseSystem(str, Enum):
    CARDIAC = "cardiac"
    NEUROLOGICAL = "neurological"
    MEDICATION = "medication"
    HEMATOLOGIC = "hematologic"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    INFECTIOUS = "infectious"
    OTHER = "other"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class MedAction(str, Enum):
    AVOID = "avoid"
    STOP = "stop"
    REDUCE = "reduce"
    CONTINUE = "continue"
    MONITOR = "monitor"


class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"
    SAME_DAY = "same_day"
    WITHIN_DAYS = "within_days"
    ROUTINE = "routine"


_URGENCY_ORDER = [
    UrgencyLevel.EMERGENCY,
    UrgencyLevel.SAME_DAY,
    UrgencyLevel.WITHIN_DAYS,
    UrgencyLevel.ROUTINE,
]


def urgency_rank(level: UrgencyLevel) -> int:
    """Lower is more urgent: emergency = 0 ... routine = 3."""
    return _URGENCY_ORDER.index(level)


class PossibleCause(BaseModel):
    name: str
    system: CauseSystem
    rank: int
    confidence: Confidence


class InteractionFlag(BaseModel):
    drugs: List[str]
    concern: str
    severity: Severity


class MedicationChange(BaseModel):
    drug: str
    action: MedAction
    reason: str


class Urgency(BaseModel):
    level: UrgencyLevel
    escalate_if: List[str] = []


class StructuredAssessment(BaseModel):
    possible_causes: List[PossibleCause]
    interaction_flags: List[InteractionFlag]
    medication_changes: List[MedicationChange]
    urgency: Urgency


def _values(enum_cls) -> str:
    return " | ".join(m.value for m in enum_cls)


STRUCTURED_BLOCK_INSTRUCTION = f"""

After your answer, append a machine-readable summary of your assessment between the literal tags <structured> and </structured>, containing a single JSON object with exactly these keys:
- "possible_causes": list of {{"name": string, "system": one of [{_values(CauseSystem)}], "rank": integer (1 = most likely), "confidence": one of [{_values(Confidence)}]}}
- "interaction_flags": list of {{"drugs": list of strings, "concern": string, "severity": one of [{_values(Severity)}]}}
- "medication_changes": list of {{"drug": string, "action": one of [{_values(MedAction)}], "reason": string}}
- "urgency": {{"level": one of [{_values(UrgencyLevel)}], "escalate_if": list of strings}}
Use empty lists where nothing applies. Use only the enum values listed above. Do not mention the tags anywhere else in your answer."""

_OPEN = "<structured>"
_CLOSE = "</structured>"


def split_structured_block(
    response_text: str,
) -> Tuple[str, Optional[StructuredAssessment]]:
    """Split a response into (free text without the block, parsed block or None).

    Uses the first complete <structured>...</structured> block. A missing
    closing tag counts as no block (text returned unchanged). Never raises.
    """
    try:
        start = response_text.find(_OPEN)
        if start == -1:
            return response_text, None
        end = response_text.find(_CLOSE, start + len(_OPEN))
        if end == -1:
            return response_text, None
        raw = response_text[start + len(_OPEN):end]
        free = response_text[:start] + response_text[end + len(_CLOSE):]
        # Drop any further blocks from the free text as well.
        free = re.sub(
            re.escape(_OPEN) + r".*?" + re.escape(_CLOSE), "", free, flags=re.DOTALL
        ).strip()
        try:
            block = StructuredAssessment.model_validate_json(
                llm_client.strip_json_fence(raw)
            )
        except (ValidationError, ValueError):
            block = None
        return free, block
    except Exception:
        return response_text, None
