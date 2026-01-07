
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from config import (
    REGISTRY_FILE,
    STRESS_MODEL_CARD_FILE,
    SYSTEM_NAME,
    SYSTEM_VERSION,
    PROVIDER,
    RISK_LEVEL,
)


@dataclass
class AlgorithmEntry:
    name: str
    version: str
    provider: str
    risk_level: str
    description: str
    developer_contact: str
    data_sources: List[Dict[str, Any]]
    last_updated: str
    governance_refs: Dict[str, str]


class AlgorithmRegistry:
    def __init__(self, path: Path = REGISTRY_FILE):
        self.path = path
        self._registry = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"algorithms": []}

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register_or_update(self, entry: AlgorithmEntry) -> None:
        updated = False
        for idx, algo in enumerate(self._registry["algorithms"]):
            if algo["name"] == entry.name and algo["version"] == entry.version:
                self._registry["algorithms"][idx] = asdict(entry)
                updated = True
                break
        if not updated:
            self._registry["algorithms"].append(asdict(entry))
        self.save()

    def list_algorithms(self) -> List[Dict[str, Any]]:
        return self._registry.get("algorithms", [])


def init_stress_risk_model_card() -> None:
    card = {
        "model_name": f"{SYSTEM_NAME} – Stress Risk Model",
        "version": SYSTEM_VERSION,
        "intended_use": "Early-warning stress risk scoring for dairy farms to guide advisory interventions.",
        "out_of_scope_use": [
            "Direct eligibility decisions for payments/grants without human review",
            "Individual worker performance evaluation",
        ],
        "risk_level": RISK_LEVEL,
        "developer": PROVIDER,
        "input_features": [
            "milk_yield",
            "soil_moisture",
            "grass_growth_index",
            "financial_stress_index",
            "herd_size",
            "age_of_farmer",
            "gender",
            "farm_size_class",
        ],
        "training_data_summary": {
            "type": "synthetic demo for prototype",
            "notes": "Replace with real anonymised, lawfully obtained datasets with explicit licences.",
        },
        "fairness_considerations": {
            "protected_attributes": ["gender", "farm_size_class"],
            "known_limitations": "Synthetic data does not reflect real structural inequalities.",
        },
        "governance_hooks": {
            "logging": "All inferences logged to JSONL with pseudonymous user_id.",
            "fairness_monitoring": "Periodic bias metrics by protected group.",
            "consent": "User type (adult/child) and consent must be set in UI.",
        },
    }

    STRESS_MODEL_CARD_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(STRESS_MODEL_CARD_FILE, "w") as f:
        json.dump(card, f, indent=2)


def init_registry_with_stress_model() -> None:
    registry = AlgorithmRegistry()
    entry = AlgorithmEntry(
        name=SYSTEM_NAME,
        version=SYSTEM_VERSION,
        provider=PROVIDER,
        risk_level=RISK_LEVEL,
        description="Governance-first prototype for agri stress risk assessment.",
        developer_contact="jit.lab@example.org",
        data_sources=[
            {
                "name": "Synthetic farm panel (demo)",
                "licence": "internal-synthetic",
                "copyright_holder": "Jit Research Lab",
                "lawful_basis": "Research & internal prototyping only",
            }
        ],
        last_updated=datetime.utcnow().isoformat(),
        governance_refs={
            "eu_ai_act": "High-risk candidate – notified and logged.",
            "irish_oireachtas_ai_report": "Aligned with transparency, non-discrimination, copyright, and energy guidance.",
        },
    )
    registry.register_or_update(entry)
    init_stress_risk_model_card()


def verify_dataset_licensing(registry: "AlgorithmRegistry") -> Dict[str, Any]:
    """Checks that all registered datasets have explicit licence metadata."""
    issues = []
    algos = registry.list_algorithms()
    for algo in algos:
        for ds in algo.get("data_sources", []):
            if not ds.get("licence") or ds.get("licence") in ("unknown", "UNSPECIFIED"):
                issues.append(
                    {
                        "algorithm": algo["name"],
                        "dataset": ds.get("name", "unknown"),
                        "problem": "Missing or unknown licence declaration",
                    }
                )
    return {"valid": len(issues) == 0, "issues": issues}
