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


class StructuredDisagreementKind(str, Enum):
    TOP_CAUSE = "top_cause"
    ONE_SIDED_CAUSE = "one_sided_cause"
    INTERACTION_FLAG = "interaction_flag"
    MEDICATION_CHANGE = "medication_change"
    URGENCY = "urgency"


class StructuredDisagreement(BaseModel):
    kind: StructuredDisagreementKind
    summary: str
    detail_a: str
    detail_b: str
    sort_key: int


# Lower sort_key is shown first. Bands leave room for in-band ordering.
_SORT_URGENCY = 0
_SORT_FLAG = 100  # + 0 (high) .. 2 (low)
_SORT_MED = 200
_SORT_TOP_CAUSE = 300
_SORT_ONE_SIDED = 310

_SEVERITY_ORDER = [Severity.HIGH, Severity.MODERATE, Severity.LOW]
# Most severe first: restrictive actions outrank continue/monitor.
_MED_ACTION_ORDER = [
    MedAction.AVOID,
    MedAction.STOP,
    MedAction.REDUCE,
    MedAction.MONITOR,
    MedAction.CONTINUE,
]


def _top_cause(a: StructuredAssessment) -> Optional[PossibleCause]:
    if not a.possible_causes:
        return None
    return min(a.possible_causes, key=lambda c: c.rank)


def _flag_map(a: StructuredAssessment) -> dict:
    out = {}
    for f in a.interaction_flags:
        key = frozenset(d.strip().lower() for d in f.drugs)
        prev = out.get(key)
        if prev is None or _SEVERITY_ORDER.index(f.severity) < _SEVERITY_ORDER.index(
            prev.severity
        ):
            out[key] = f
    return out


def _med_map(a: StructuredAssessment) -> dict:
    out = {}
    for m in a.medication_changes:
        key = m.drug.strip().lower()
        prev = out.get(key)
        if prev is None or _MED_ACTION_ORDER.index(m.action) < _MED_ACTION_ORDER.index(
            prev.action
        ):
            out[key] = m
    return out


def _flag_label(f: InteractionFlag) -> str:
    return " + ".join(sorted(d.strip() for d in f.drugs))


def compare_structured(
    a: Optional[StructuredAssessment], b: Optional[StructuredAssessment]
) -> List[StructuredDisagreement]:
    """Compare two structured blocks; pure code, sorted by sort_key ascending.

    Returns [] if either side is None. Confidence is never compared.
    """
    if a is None or b is None:
        return []
    out: List[StructuredDisagreement] = []

    # Urgency
    if a.urgency.level != b.urgency.level:
        hi_u, lo_u = sorted((a.urgency.level, b.urgency.level), key=urgency_rank)
        out.append(
            StructuredDisagreement(
                kind=StructuredDisagreementKind.URGENCY,
                summary=f"Urgency differs: {hi_u.value} vs {lo_u.value}",
                detail_a=f"Urgency: {a.urgency.level.value}",
                detail_b=f"Urgency: {b.urgency.level.value}",
                sort_key=_SORT_URGENCY,
            )
        )

    # Interaction flags
    fa, fb = _flag_map(a), _flag_map(b)
    for key in sorted(set(fa) | set(fb), key=lambda k: sorted(k)):
        x, y = fa.get(key), fb.get(key)
        if x is not None and y is not None:
            if x.severity == y.severity:
                continue
            hi, lo = sorted(
                (x.severity, y.severity), key=lambda s: _SEVERITY_ORDER.index(s)
            )
            summary = (
                f"Interaction {_flag_label(x)}: severity {hi.value} vs {lo.value}"
            )
            detail_a = f"Severity {x.severity.value}: {x.concern}"
            detail_b = f"Severity {y.severity.value}: {y.concern}"
            sev = hi
        else:
            present = x if x is not None else y
            who = "A" if x is not None else "B"
            summary = (
                f"Interaction {_flag_label(present)} ({present.severity.value}) "
                f"flagged only by {who}"
            )
            detail_a = (
                f"Severity {x.severity.value}: {x.concern}"
                if x is not None
                else "Not flagged"
            )
            detail_b = (
                f"Severity {y.severity.value}: {y.concern}"
                if y is not None
                else "Not flagged"
            )
            sev = present.severity
        out.append(
            StructuredDisagreement(
                kind=StructuredDisagreementKind.INTERACTION_FLAG,
                summary=summary,
                detail_a=detail_a,
                detail_b=detail_b,
                sort_key=_SORT_FLAG + _SEVERITY_ORDER.index(sev),
            )
        )

    # Medication changes
    ma, mb = _med_map(a), _med_map(b)
    for drug in sorted(set(ma) | set(mb)):
        x, y = ma.get(drug), mb.get(drug)
        if x is not None and y is not None:
            if x.action == y.action:
                continue
            summary = (
                f"Medication {x.drug}: {x.action.value} vs {y.action.value}"
            )
        else:
            present = x if x is not None else y
            # Silence on a drug is consistent with continue/monitor, so only
            # restrictive one-sided actions are real disagreements.
            if present.action in (MedAction.CONTINUE, MedAction.MONITOR):
                continue
            who = "A" if x is not None else "B"
            summary = (
                f"Medication {present.drug}: {present.action.value} "
                f"proposed only by {who}"
            )
        out.append(
            StructuredDisagreement(
                kind=StructuredDisagreementKind.MEDICATION_CHANGE,
                summary=summary,
                detail_a=(
                    f"{x.action.value}: {x.reason}" if x is not None else "Not addressed"
                ),
                detail_b=(
                    f"{y.action.value}: {y.reason}" if y is not None else "Not addressed"
                ),
                sort_key=_SORT_MED,
            )
        )

    # Causes
    ta, tb = _top_cause(a), _top_cause(b)
    if ta is not None and tb is not None and ta.system != tb.system:
        out.append(
            StructuredDisagreement(
                kind=StructuredDisagreementKind.TOP_CAUSE,
                summary=(
                    f"Top cause differs: {ta.system.value} vs {tb.system.value}"
                ),
                detail_a=f"{ta.name} ({ta.system.value})",
                detail_b=f"{tb.name} ({tb.system.value})",
                sort_key=_SORT_TOP_CAUSE,
            )
        )
    sa = {c.system: c for c in sorted(a.possible_causes, key=lambda c: c.rank)}
    sb = {c.system: c for c in sorted(b.possible_causes, key=lambda c: c.rank)}
    for system in CauseSystem:
        x, y = sa.get(system), sb.get(system)
        if (x is None) == (y is None):
            continue
        who = "A" if x is not None else "B"
        out.append(
            StructuredDisagreement(
                kind=StructuredDisagreementKind.ONE_SIDED_CAUSE,
                summary=f"{system.value} cause raised only by {who}",
                detail_a=(
                    f"{x.name} (rank {x.rank})" if x is not None else "Was not raised"
                ),
                detail_b=(
                    f"{y.name} (rank {y.rank})" if y is not None else "Was not raised"
                ),
                sort_key=_SORT_ONE_SIDED,
            )
        )

    out.sort(key=lambda d: d.sort_key)
    return out
