
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

from config import INFERENCE_LOG_FILE, GOVERNANCE_LOG_FILE


@dataclass
class InferenceLogRecord:
    timestamp: str
    user_id: str
    is_child: bool
    consent_granted: bool
    model_name: str
    model_version: str
    input_hash: str
    prediction: Any
    risk_level: str
    explanation: Dict[str, float]


@dataclass
class GovernanceEvent:
    timestamp: str
    event_type: str  # e.g. "CONSENT_DENIED", "CHILD_INFERENCE_BLOCKED", "ENERGY_USAGE"
    details: Dict[str, Any]


def _append_jsonl(path, obj: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def log_inference(record: InferenceLogRecord) -> None:
    _append_jsonl(INFERENCE_LOG_FILE, asdict(record))


def log_governance_event(event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
    event = GovernanceEvent(
        timestamp=datetime.utcnow().isoformat(),
        event_type=event_type,
        details=details or {},
    )
    _append_jsonl(GOVERNANCE_LOG_FILE, asdict(event))
