
import json
from pathlib import Path
from typing import Dict, Any, List

from config import (
    INFERENCE_LOG_FILE,
    GOVERNANCE_LOG_FILE,
    SYSTEM_NAME,
    SYSTEM_VERSION,
    RISK_LEVEL,
)
from .registry import AlgorithmRegistry, verify_dataset_licensing
from .energy import compute_energy_summary


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def generate_aia_report(policy_mode: str = "Research") -> Dict[str, Any]:
    """Automated Algorithmic Impact Assessment skeleton."""
    inf_logs = _load_jsonl(Path(INFERENCE_LOG_FILE))
    gov_logs = _load_jsonl(Path(GOVERNANCE_LOG_FILE))
    registry = AlgorithmRegistry()
    licence_status = verify_dataset_licensing(registry)
    energy_summary = compute_energy_summary()

    n_inf = len(inf_logs)
    users = {r.get("user_id") for r in inf_logs}
    n_users = len(users)

    n_high = 0
    for r in inf_logs:
        pred = r.get("prediction", {})
        if isinstance(pred, dict) and int(pred.get("label", 0)) == 1:
            n_high += 1

    consent_denied = sum(1 for g in gov_logs if g.get("event_type") == "CONSENT_DENIED")
    child_flags = sum(1 for g in gov_logs if g.get("event_type") == "CHILD_PROFILE_FLAGGED")
    child_blocks = sum(1 for g in gov_logs if g.get("event_type") == "CHILD_INFERENCE_BLOCKED")

    system_overview = {
        "system_name": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
        "risk_level": RISK_LEVEL,
        "operational_mode": policy_mode,
    }

    usage_section = {
        "total_inferences": n_inf,
        "unique_users": n_users,
        "share_high_risk_predictions": (n_high / n_inf) if n_inf else 0.0,
    }

    consent_section = {
        "consent_denied_events": consent_denied,
        "child_profiles_flagged": child_flags,
        "child_inference_blocks": child_blocks,
    }

    data_governance_section = {
        "licensing_valid": licence_status["valid"],
        "licensing_issues": licence_status["issues"],
        "algorithms_registered": registry.list_algorithms(),
    }

    energy_section = energy_summary

    qualitative_risks = []
    if not licence_status["valid"]:
        qualitative_risks.append("Unresolved dataset licensing issues.")
    if consent_denied > 0 and policy_mode == "Public Service":
        qualitative_risks.append("Attempted operation without consent in Public Service mode.")
    if child_blocks > 0:
        qualitative_risks.append("Children attempted to use system; safeguards working but risk remains.")

    overall_risk_level = "medium"
    if qualitative_risks:
        overall_risk_level = "high"

    mitigations = [
        "Formalise DPIA (Data Protection Impact Assessment) with the DPO.",
        "Regular fairness audits by protected group with documented thresholds.",
        "Set maximum retention period for raw logs and strictly pseudonymise IDs.",
    ]
    if not licence_status["valid"]:
        mitigations.insert(0, "Resolve all dataset licensing gaps before any Public Service deployment.")

    return {
        "system_overview": system_overview,
        "usage_summary": usage_section,
        "consent_and_children": consent_section,
        "data_governance": data_governance_section,
        "energy_and_carbon": energy_section,
        "qualitative_risk_assessment": {
            "overall_level": overall_risk_level,
            "identified_issues": qualitative_risks,
        },
        "recommended_mitigations": mitigations,
    }
